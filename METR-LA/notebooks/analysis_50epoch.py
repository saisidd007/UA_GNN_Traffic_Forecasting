"""
Analysis script for the 50-epoch run (ImprovedTrafficGNN)
Creates training/validation plots, prints/saves test metrics, and produces sensor time-series plots
Corrects sensor-ID mapping using `PEMS-BAY-META.csv` to avoid treating column 0 as "heading" or index.

Usage (PowerShell):
> .venv\Scripts\Activate.ps1
> python notebooks/analysis_50epoch.py

Outputs (saved to `results/`):
- analysis_50epoch_train_val_loss.png
- analysis_50epoch_metrics.png
- analysis_50epoch_test_summary.csv
- sensor_time_series_SENSORID.png (for example sensors)

Notes:
- This script does not assume DataFrame column order; it attempts to align columns with meta sensor IDs. If the data file uses a timestamp/index column in column 0, that column will be used as the index and not as a sensor.
- If your `PEMS-BAY.csv` file columns are already labeled with sensor IDs, the script will detect and preserve them.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

# Import dataset and model factories
from src.utils.enhanced_dataset import create_enhanced_dataset
from src.models.enhanced_gnn import create_improved_model

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DATA = ROOT / 'data'

RESULTS.mkdir(parents=True, exist_ok=True)

# Paths
json_50_path = RESULTS / 'enhanced_training_results_metrla_20260916_165902.json'
# If the exact JSON isn't present, try to detect the most recent enhanced_training_results_*.json
if not json_50_path.exists():
    import glob
    json_candidates = sorted(RESULTS.glob('enhanced_training_results_metrla_*.json'), key=lambda p: p.stat().st_mtime)
    if not json_candidates:
        json_candidates = sorted(RESULTS.glob('enhanced_training_results_*.json'), key=lambda p: p.stat().st_mtime)
    if json_candidates:
        json_50_path = json_candidates[-1]
        print(f"Using latest training JSON: {json_50_path}")
    else:
        print('No training JSON found in results/. Training history plots will be skipped.')

meta_path = DATA / 'metr-la' / 'METR-LA-META.csv'
if not meta_path.exists():
    meta_path = DATA / 'METR-LA-META.csv'

metr_csv_path = DATA / 'metr-la' / 'METR-LA.csv'
if not metr_csv_path.exists():
    metr_csv_path = DATA / 'METR-LA.csv'

# 1) Load training JSON
with open(json_50_path, 'r', encoding='utf-8') as f:
    res50 = json.load(f)

history = res50.get('training_history', res50.get('history', {}))
train_loss = history.get('train_loss', [])
val_loss = history.get('val_loss', [])
train_metrics = history.get('train_metrics', [])
val_metrics = history.get('val_metrics', [])

# Plot train/val loss curve
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train / Val Loss (50-epoch run)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / 'analysis_50epoch_train_val_loss.png', dpi=150)
plt.close()

# Optional: plot MAE trend if present in train_metrics / val_metrics
if train_metrics and val_metrics:
    train_mae = [m.get('mae') for m in train_metrics]
    val_mae = [m.get('mae') for m in val_metrics]
    plt.figure(figsize=(8,5))
    plt.plot(train_mae, label='Train MAE')
    plt.plot(val_mae, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE trend (50-epoch run)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / 'analysis_50epoch_metrics.png', dpi=150)
    plt.close()

# Save summary test metrics to CSV
test_metrics = res50.get('test_metrics', res50.get('final_test_metrics', {}))
summary_df = pd.DataFrame([test_metrics])
summary_df.index = ['enhanced_50epoch_metrla']
summary_df.to_csv(RESULTS / 'analysis_50epoch_test_summary.csv')
print('\nSaved test summary to', RESULTS / 'analysis_50epoch_test_summary.csv')
print(summary_df.T)

# 2) Load meta and METR-LA CSV with robust column alignment
meta_df = pd.read_csv(meta_path)
meta_sensor_ids = meta_df['sensor_id'].astype(str).tolist()
print(f"Loaded {len(meta_sensor_ids)} sensor IDs from meta file.")

# Read the METR-LA CSV but avoid assuming column 0 is a sensor. We'll try to detect index/timestamp column.
print('Attempting to load METR-LA data (may be large).')
# Use pandas to infer the index column: if first column name starts with 'Unnamed' or 'timestamp', set as index
try:
    df = pd.read_csv(metr_csv_path, low_memory=False)
except Exception as e:
    print('Error reading METR-LA.csv:', e)
    df = None

if df is not None:
    print('METR-LA columns (first 10):', list(df.columns[:10]))

    # If the first column is an unnamed index or time, set it as index
    first_col = df.columns[0]
    if str(first_col).lower().startswith('unnamed') or 'time' in str(first_col).lower() or 'date' in str(first_col).lower():
        df = df.set_index(first_col)
        print(f"Set column '{first_col}' as index (timestamp/index).")

    # Normalize column names to strings
    df.columns = df.columns.astype(str)

    # If columns already match meta sensor IDs, keep them. Otherwise, if number of sensor columns equals meta IDs, align by position.
    cols = list(df.columns)
    # Exclude potential non-sensor index/name columns by matching length
    if set(meta_sensor_ids).issubset(set(cols)):
        print('DataFrame columns already include meta sensor IDs; no remapping needed.')
    else:
        # If counts match, remap by position (safe when file exported without sensor IDs)
        sensor_cols = [c for c in cols if c not in df.index.names]
        if len(sensor_cols) == len(meta_sensor_ids):
            print('Remapping columns by position to meta sensor IDs (position-based).')
            df.columns = meta_sensor_ids
        else:
            print('Column count does not match meta sensor IDs. Will attempt to proceed without remapping.\n' \
                  'If plots look incorrect, inspect df.columns and meta file manually.')

    # Example sensor plots: choose a few sensors from meta (first three)
    sample_sensor_ids = meta_sensor_ids[:3]
    for sid in sample_sensor_ids:
        if sid not in df.columns:
            print(f"Sensor ID {sid} not found in data columns; skipping plot.")
            continue
        ts = pd.to_numeric(df[sid], errors='coerce')
        plt.figure(figsize=(12,3))
        plt.plot(ts[:288], label=f'Sensor {sid} (first 24h)')
        plt.xlabel('Timestep')
        plt.ylabel('Speed')
        plt.title(f'Sensor {sid} - first 24 hours')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / f'sensor_time_series_{sid}.png', dpi=150)
        plt.close()

    print('Saved sample sensor time-series plots to results/.')
else:
    print('METR-LA.csv could not be loaded; sensor plots skipped.')

    print('\nAnalysis script completed (plots).')


# ---------------------------------------------------------------------------
# Inference & per-horizon evaluation (uses checkpoint + dataset)
# ---------------------------------------------------------------------------

def run_inference_and_per_horizon_metrics(
    checkpoint_path: str = str(RESULTS / 'enhanced_best_model.pt'),
    root_dir: str = str(DATA / 'metr-la' if (DATA / 'metr-la').exists() else DATA),
    sequence_length: int = 12,
    prediction_length: int = 12,
    device: str = None,
    mc_samples: int = 30,
):
    """Load checkpoint, run MC-dropout inference on test set, compute per-horizon MAE/RMSE and 95% CI coverage.

    Saves results to `results/analysis_50epoch_per_horizon_metrics.csv`.
    """
    # Allow forcing CPU via environment variable (useful if CUDA is unavailable)
    force_cpu = os.environ.get('FORCE_CPU', '')
    if force_cpu.lower() in ('1', 'true', 'yes'):
        device = torch.device('cpu')
    else:
        device = torch.device(device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Allow overriding mc_samples via environment variable MC_SAMPLES
    mc_env = os.environ.get('MC_SAMPLES')
    if mc_env is not None:
        try:
            mc_samples = int(mc_env)
            print(f"Overriding mc_samples from environment: MC_SAMPLES={mc_samples}")
        except Exception:
            print(f"Invalid MC_SAMPLES value: {mc_env}; using default mc_samples={mc_samples}")
    print(f"Running inference on device: {device}")

    # Checkpoint exists?
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Skipping inference.")
        return None

    # Build dataset
    try:
        dataset = create_enhanced_dataset(
            root_dir=root_dir,
            sequence_length=sequence_length,
            prediction_length=prediction_length,
            dataset_name='METR-LA'
        )
    except Exception as e:
        print('Failed to create enhanced dataset for inference:', e)
        print('Make sure `data/metr-la/METR-LA.csv` and metadata are present and readable on this machine.')
        return None

    test_data = dataset.get_test_data()
    if len(test_data) == 0:
        print('No test sequences found in dataset; skipping inference.')
        return None

    # DataLoader (PyG Data objects)
    batch_size = 8
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model and load checkpoint
    model = create_improved_model(in_channels=sequence_length, out_channels=prediction_length).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Support both state-dict only and checkpoint dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)

    model.eval()

    # Containers to accumulate per-horizon arrays
    horizons = prediction_length
    all_preds_per_h = [ [] for _ in range(horizons) ]
    all_tgts_per_h = [ [] for _ in range(horizons) ]
    all_cov_flags_per_h = [ [] for _ in range(horizons) ]
    all_vars_per_h = [ [] for _ in range(horizons) ]
    # store per-sample PI widths for quantiles/plots
    all_piw_per_h = [ [] for _ in range(horizons) ]

    z95 = 1.96

    print(f"Starting MC inference with K={mc_samples} samples over {len(test_loader)} batches...")
    with torch.no_grad():
        for batch in test_loader:
            # Each batch is a PyG Data object batch
            batch = batch.to(device)

            # Collect MC samples with dropout enabled by setting model.train()
            mc_preds = []
            mc_aleas = []
            for k in range(mc_samples):
                model.train()
                preds_k, alea_k, _ = model(batch.x, batch.edge_index, batch.missing_mask, return_uncertainty=True)
                mc_preds.append(preds_k.unsqueeze(0))
                mc_aleas.append(alea_k.unsqueeze(0))

            mc_preds = torch.cat(mc_preds, dim=0)  # [K, B, N, H]
            mc_aleas = torch.cat(mc_aleas, dim=0)

            mean_pred = mc_preds.mean(dim=0).cpu().numpy()  # [B, N, H]
            epi_var = mc_preds.var(dim=0).cpu().numpy()
            alea_var = mc_aleas.mean(dim=0).cpu().numpy()
            total_var = epi_var + alea_var

            targets = batch.y.cpu().numpy()  # [B, N, H]

            # For each horizon, accumulate
            for h in range(horizons):
                pred_h = mean_pred[:, :, h].reshape(-1)
                tgt_h = targets[:, :, h].reshape(-1)
                var_h = total_var[:, :, h].reshape(-1)

                ci_half = z95 * np.sqrt(var_h)
                covered = (tgt_h >= (pred_h - ci_half)) & (tgt_h <= (pred_h + ci_half))

                piw_h = (2.0 * ci_half)

                all_preds_per_h[h].append(pred_h)
                all_tgts_per_h[h].append(tgt_h)
                all_cov_flags_per_h[h].append(covered)
                all_vars_per_h[h].append(var_h)
                all_piw_per_h[h].append(piw_h)

    # Concatenate per horizon and compute metrics
    rows = []
    for h in range(horizons):
        preds = np.concatenate(all_preds_per_h[h], axis=0)
        tgts = np.concatenate(all_tgts_per_h[h], axis=0)
        covs = np.concatenate(all_cov_flags_per_h[h], axis=0)
        vars_h = np.concatenate(all_vars_per_h[h], axis=0)
        piw_h_all = np.concatenate(all_piw_per_h[h], axis=0)

        mae = float(np.mean(np.abs(preds - tgts)))
        rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
        coverage95 = float(np.mean(covs))

        # Compute prediction-interval width stats (two-sided 95% PI)
        piw_mean = float(np.mean(piw_h_all)) if piw_h_all.size > 0 else float('nan')
        piw_median = float(np.median(piw_h_all)) if piw_h_all.size > 0 else float('nan')
        piw_q25 = float(np.percentile(piw_h_all, 25)) if piw_h_all.size > 0 else float('nan')
        piw_q75 = float(np.percentile(piw_h_all, 75)) if piw_h_all.size > 0 else float('nan')

        rows.append({
            'horizon': h+1,
            'mae': mae,
            'rmse': rmse,
            'coverage95': coverage95,
            'piw95_mean': piw_mean,
            'piw95_median': piw_median,
            'piw95_q25': piw_q25,
            'piw95_q75': piw_q75
        })

    per_h_df = pd.DataFrame(rows)
    per_h_df.to_csv(RESULTS / 'analysis_50epoch_per_horizon_metrics.csv', index=False)
    print('Saved per-horizon metrics to', RESULTS / 'analysis_50epoch_per_horizon_metrics.csv')
    print(per_h_df)

    # Create a per-horizon PIW plot (mean with 25-75% band)
    try:
        horizons_arr = per_h_df['horizon'].values
        mean_arr = per_h_df['piw95_mean'].values
        q25 = per_h_df['piw95_q25'].values
        q75 = per_h_df['piw95_q75'].values

        plt.figure(figsize=(8,4))
        plt.plot(horizons_arr, mean_arr, marker='o', color='C0', label='Mean PIW95')
        plt.fill_between(horizons_arr, q25, q75, color='C0', alpha=0.2, label='25-75% PIW')
        plt.xlabel('Horizon')
        plt.ylabel('PIW (95%)')
        plt.title('Per-horizon 95% Prediction Interval Width')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / 'analysis_50epoch_per_horizon_piw95.png', dpi=150)
        plt.close()
        print('Saved per-horizon PIW plot to', RESULTS / 'analysis_50epoch_per_horizon_piw95.png')
    except Exception as e:
        print('Failed to create PIW plot:', e)

    return per_h_df


if __name__ == '__main__':
    # Run inference block when executed directly; this will be skipped if imported
    try:
        # Support overriding MC samples and forcing CPU via env vars
        mc_env = os.environ.get('MC_SAMPLES')
        mc_val = int(mc_env) if mc_env and mc_env.isdigit() else 30
        run_inference_and_per_horizon_metrics(mc_samples=mc_val)
    except Exception as e:
        print('Inference run failed:', e)
        print('If this is due to large data files, please run the script locally where the dataset is accessible.')

    print('\nFull analysis script finished.')
