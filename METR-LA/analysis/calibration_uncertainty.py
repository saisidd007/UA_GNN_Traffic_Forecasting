#!/usr/bin/env python
"""Calibration and uncertainty evaluation for the PEMS-BAY TSSP model.

Outputs:
- results/calibration_uncertainty_metrics.json
- results/calibration_uncertainty_summary.csv
- results/calibration_uncertainty.png
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / 'data'
CHECKPOINT_PATH = ROOT / 'pems-bay' / 'results' / 'best_tssp_model.pt'

sys.path.insert(0, str(ROOT / 'pems-bay'))

from train_50_tssp import load_pems_bay_data, PEMSBayDataset
from models.tssp_gnn import create_tssp_model


def enable_mc_dropout(module):
    if isinstance(module, torch.nn.Dropout):
        module.train()


def load_test_loader(batch_size: int = 16):
    (train_data, val_data, test_data), edge_index, num_sensors, scaler = load_pems_bay_data(
        data_dir=str(DATA_DIR), sequence_length=12, prediction_length=12
    )
    test_dataset = PEMSBayDataset(test_data)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0), edge_index, num_sensors


def load_model(device: torch.device):
    model = create_tssp_model(
        in_channels=12,
        hidden_channels=128,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=4,
        num_attention_heads=4,
        dropout=0.15,
        sequence_length=12,
    ).to(device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def compute_metrics(predictions, uncertainties, targets):
    z_values = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    metrics = {
        'device': str(device),
        'num_samples': int(len(predictions)),
        'coverage_probabilities': {},
    }

    for conf, z in z_values.items():
        lower = predictions - z * uncertainties
        upper = predictions + z * uncertainties
        coverage = float(np.mean((targets >= lower) & (targets <= upper)))
        metrics['coverage_probabilities'][f'{int(conf*100)}%'] = coverage

    interval_widths = 2.0 * 1.96 * uncertainties
    metrics['mean_prediction_interval_width'] = float(np.mean(interval_widths))
    metrics['std_prediction_interval_width'] = float(np.std(interval_widths))
    metrics['mean_uncertainty'] = float(np.mean(uncertainties))
    metrics['std_uncertainty'] = float(np.std(uncertainties))
    metrics['min_uncertainty'] = float(np.min(uncertainties))
    metrics['max_uncertainty'] = float(np.max(uncertainties))
    metrics['mae'] = float(np.mean(np.abs(predictions - targets)))
    metrics['rmse'] = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    if np.std(uncertainties) > 0 and np.std(np.abs(predictions - targets)) > 0:
        metrics['uncertainty_error_correlation'] = float(np.corrcoef(uncertainties, np.abs(predictions - targets))[0, 1])
    else:
        metrics['uncertainty_error_correlation'] = 0.0

    try:
        from scipy.stats import spearmanr
        metrics['spearman_rank_correlation'] = float(spearmanr(uncertainties, np.abs(predictions - targets))[0])
    except Exception:
        metrics['spearman_rank_correlation'] = 0.0

    return metrics


def create_plots(predictions, uncertainties, targets, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Calibration and Uncertainty Analysis', fontsize=16, weight='bold')

    ax = axes[0, 0]
    ax.hist(uncertainties, bins=50, color='#3B82F6', alpha=0.8, edgecolor='black')
    ax.set_title('Prediction Uncertainty Distribution')
    ax.set_xlabel('Predicted Std Dev')
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    errors = np.abs(predictions - targets)
    ax.hist(errors, bins=50, color='#F97316', alpha=0.8, edgecolor='black')
    ax.set_title('Absolute Error Distribution')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    sample = min(len(predictions), 2000)
    idx = np.random.choice(len(predictions), sample, replace=False)
    ax.scatter(uncertainties[idx], errors[idx], s=8, alpha=0.3, color='#16A34A')
    ax.set_title('Uncertainty vs Absolute Error')
    ax.set_xlabel('Predicted Std Dev')
    ax.set_ylabel('Absolute Error')
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    conf_levels = [68, 90, 95, 99]
    actual = [metrics['coverage_probabilities'][f'{c}%'] for c in conf_levels]
    ax.plot(conf_levels, [c / 100 for c in conf_levels], 'k--', label='Ideal')
    ax.plot(conf_levels, actual, 'o-', label='Model')
    ax.set_title('Calibration Curve (Coverage)')
    ax.set_xlabel('Expected Coverage (%)')
    ax.set_ylabel('Actual Coverage')
    ax.set_ylim(0.5, 1.05)
    ax.set_xticks(conf_levels)
    ax.grid(alpha=0.2)
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = RESULTS / 'calibration_uncertainty.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def aggregate_uncertainty(test_loader, model, edge_index, device, mc_samples: int = 10):
    model.apply(enable_mc_dropout)
    model.eval()
    predictions = []
    uncertainties = []
    targets = []

    edge_index = edge_index.to(device)

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size_current = batch_x.shape[0]
            preds_mc = []
            var_mc = []

            for k in range(mc_samples):
                model.train()
                run_preds = []
                run_vars = []
                for sample_i in range(batch_size_current):
                    x = batch_x[sample_i]
                    pred, alea_var, epi_var = model(x, edge_index, return_uncertainty=True)
                    run_preds.append(pred.unsqueeze(0))
                    run_vars.append((alea_var + epi_var).unsqueeze(0))

run_preds = torch.cat(tuple(run_preds), dim=0)
            run_vars = torch.cat(tuple(run_vars), dim=0)
            preds_mc.append(run_preds.unsqueeze(0))
            var_mc.append(run_vars.unsqueeze(0))

        preds_mc = torch.cat(tuple(preds_mc), dim=0)
        mean_preds = preds_mc.mean(dim=0).cpu().numpy()
        mean_vars = torch.cat(tuple(var_mc), dim=0).mean(dim=0).cpu().numpy()

            predictions.append(mean_preds.reshape(-1))
            uncertainties.append(np.sqrt(mean_vars).reshape(-1))
            targets.append(batch_y.permute(0, 2, 1).cpu().numpy().reshape(-1))

    return np.concatenate(predictions), np.concatenate(uncertainties), np.concatenate(targets)


if __name__ == '__main__':
    mc_env = os.environ.get('MC_SAMPLES')
    mc_samples = int(mc_env) if mc_env and mc_env.isdigit() else 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"MC samples: {mc_samples}")

    test_loader, edge_index, num_sensors = load_test_loader(batch_size=16)
    model = load_model(device)

    predictions, uncertainties, targets = aggregate_uncertainty(test_loader, model, edge_index, device, mc_samples=mc_samples)
    metrics = compute_metrics(predictions, uncertainties, targets)

    json_path = RESULTS / 'calibration_uncertainty_metrics.json'
    with open(json_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)

    summary = []
    for name, value in metrics['coverage_probabilities'].items():
        summary.append({'metric': f'coverage_{name}', 'value': value})
    summary.extend([
        {'metric': 'mean_prediction_interval_width', 'value': metrics['mean_prediction_interval_width']},
        {'metric': 'std_prediction_interval_width', 'value': metrics['std_prediction_interval_width']},
        {'metric': 'mean_uncertainty', 'value': metrics['mean_uncertainty']},
        {'metric': 'mae', 'value': metrics['mae']},
        {'metric': 'rmse', 'value': metrics['rmse']},
        {'metric': 'uncertainty_error_correlation', 'value': metrics['uncertainty_error_correlation']},
        {'metric': 'spearman_rank_correlation', 'value': metrics['spearman_rank_correlation']},
    ])
    pd.DataFrame(summary).to_csv(RESULTS / 'calibration_uncertainty_summary.csv', index=False)
    plot_path = create_plots(predictions, uncertainties, targets, metrics)

    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved summary CSV: {RESULTS / 'calibration_uncertainty_summary.csv'}")
    print(f"Saved plot: {plot_path}")
