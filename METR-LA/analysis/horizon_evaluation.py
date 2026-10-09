#!/usr/bin/env python
"""Per-horizon evaluation using the PEMS-BAY TSSP checkpoint.

Generates per-step MAE/RMSE, calibration coverage, and interval-width statistics.
Outputs:
- results/horizon_metrics.csv
- results/horizon_metrics.json
- results/horizon_metrics.png
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


def load_test_data(batch_size: int = 16):
    (train_data, val_data, test_data), edge_index, num_sensors, scaler = load_pems_bay_data(
        data_dir=str(DATA_DIR), sequence_length=12, prediction_length=12
    )
    test_dataset = PEMSBayDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_loader, edge_index, num_sensors


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


def compute_metrics_for_horizons(predictions, targets, variances):
    horizons = predictions.shape[-1]
    rows = []
    z = 1.96

    for horizon in range(horizons):
        preds = predictions[:, horizon].reshape(-1)
        targs = targets[:, horizon].reshape(-1)
        vars_h = variances[:, horizon].reshape(-1)
        mae = float(np.mean(np.abs(preds - targs)))
        rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))
        coverage = float(np.mean(np.abs(preds - targs) <= z * np.sqrt(vars_h)))
        piw = float(np.mean(2.0 * z * np.sqrt(vars_h)))

        rows.append({
            'horizon': horizon + 1,
            'mae': mae,
            'rmse': rmse,
            'coverage95': coverage,
            'piw95_mean': piw,
        })

    return pd.DataFrame(rows)


def run_evaluation(mc_samples: int = 10, batch_size: int = 16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"MC samples: {mc_samples}, batch size: {batch_size}")

    test_loader, edge_index, num_sensors = load_test_data(batch_size=batch_size)
    model = load_model(device)
    model.apply(enable_mc_dropout)
    model.eval()

    all_preds = []
    all_targets = []
    all_vars = []

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    edge_index = edge_index.to(device)
    z = 1.96

    with torch.no_grad():
        for batch_index, (batch_x, batch_y) in enumerate(test_loader, start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_size_current = batch_x.shape[0]

            sample_preds = []
            sample_vars = []

            for k in range(mc_samples):
                model.train()
                preds_batch = []
                var_batch = []
                for sample_i in range(batch_size_current):
                    x = batch_x[sample_i]
                    preds, alea_var, epi_var = model(x, edge_index, return_uncertainty=True)
                    preds = preds.detach()
                    total_var = (alea_var + epi_var).detach()
                    preds_batch.append(preds.unsqueeze(0))
                    var_batch.append(total_var.unsqueeze(0))

                preds_batch = torch.cat(tuple(preds_batch), dim=0)
                var_batch = torch.cat(tuple(var_batch), dim=0)
                sample_preds.append(preds_batch.unsqueeze(0))
                sample_vars.append(var_batch.unsqueeze(0))

            sample_preds = torch.cat(tuple(sample_preds), dim=0)
            sample_vars = torch.cat(tuple(sample_vars), dim=0)
            mean_preds = sample_preds.mean(dim=0).cpu().numpy()
            var_preds = sample_preds.var(dim=0, unbiased=False).cpu().numpy()
            mean_alea = sample_vars.mean(dim=0).cpu().numpy()
            total_vars = var_preds + mean_alea

            all_preds.append(mean_preds)
            all_targets.append(batch_y.permute(0, 2, 1).cpu().numpy())
            all_vars.append(total_vars)

            print(f"  Batch {batch_index}/{len(test_loader)} processed", end='\r')

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_vars = np.concatenate(all_vars, axis=0)

    results_df = compute_metrics_for_horizons(all_preds, all_targets, all_vars)
    results_df.to_csv(RESULTS / 'horizon_metrics.csv', index=False)
    results_df.to_json(RESULTS / 'horizon_metrics.json', orient='records', indent=2)

    # Plot metrics
    plt.figure(figsize=(9, 5))
    plt.plot(results_df['horizon'], results_df['mae'], marker='o', label='MAE')
    plt.plot(results_df['horizon'], results_df['rmse'], marker='s', label='RMSE')
    plt.xlabel('Prediction Horizon (steps)')
    plt.ylabel('Error (mph)')
    plt.title('Per-Horizon MAE and RMSE')
    plt.xticks(results_df['horizon'])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / 'horizon_metrics.png', dpi=150)
    plt.close()

    print(f"Saved horizon metrics to {RESULTS / 'horizon_metrics.csv'}")
    print(f"Saved horizon plot to {RESULTS / 'horizon_metrics.png'}")

    return results_df


if __name__ == '__main__':
    mc_env = os.environ.get('MC_SAMPLES')
    mc_samples = int(mc_env) if mc_env and mc_env.isdigit() else 10
    run_evaluation(mc_samples=mc_samples)
