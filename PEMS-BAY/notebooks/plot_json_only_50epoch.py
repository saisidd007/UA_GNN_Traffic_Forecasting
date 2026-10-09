"""
Lightweight script to generate JSON-only figures for the 50-epoch run.
Does NOT require PyTorch or dataset files.
Generates:
 - results/analysis_50epoch_train_val_loss.png
 - results/analysis_50epoch_metrics.png (MAE trend if present)
 - results/analysis_50epoch_lr.png
 - results/analysis_50epoch_test_summary.csv

Run:
 python notebooks/plot_json_only_50epoch.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

# Locate METR-LA training results JSON
json_50 = RESULTS / 'enhanced_training_results_metrla_20260916_165902.json'
if not json_50.exists():
    candidates = sorted(RESULTS.glob('enhanced_training_results_metrla_*.json'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(RESULTS.glob('enhanced_training_results_*.json'), key=lambda p: p.stat().st_mtime)
    if candidates:
        json_50 = candidates[-1]

if not json_50.exists():
    print('METR-LA 50-epoch JSON not found in', RESULTS)
    raise SystemExit(1)

print(f"Loading training results from: {json_50.name}")
with json_50.open('r', encoding='utf-8') as f:
    js = json.load(f)

history = js.get('training_history', js.get('history', {}))
train_loss = history.get('train_loss', [])
val_loss = history.get('val_loss', [])
train_metrics = history.get('train_metrics', [])
val_metrics = history.get('val_metrics', [])
learning_rates = history.get('learning_rates', [])

# Plot loss
plt.figure(figsize=(8,5))
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
print('Saved train/val loss plot')

# Plot MAE trend if present
if train_metrics and val_metrics:
    train_mae = [m.get('mae', np.nan) for m in train_metrics]
    val_mae = [m.get('mae', np.nan) for m in val_metrics]
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
    print('Saved MAE trend plot')
else:
    print('No train/val metrics arrays found in JSON; skipped MAE plot')

# Learning rate plot
if learning_rates:
    plt.figure(figsize=(8,4))
    plt.plot(learning_rates)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.title('Learning rate schedule')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / 'analysis_50epoch_lr.png', dpi=150)
    plt.close()
    print('Saved learning rate plot')
else:
    print('No learning rate info in JSON; skipped LR plot')

# Save test summary
test_metrics = js.get('test_metrics', js.get('final_test_metrics', {}))
if test_metrics:
    df = pd.DataFrame([test_metrics])
    df.index = ['enhanced_50epoch_metrla']
    df.to_csv(RESULTS / 'analysis_50epoch_test_summary.csv')
    print('Saved test summary CSV to results/analysis_50epoch_test_summary.csv')
    print(df.T)
else:
    print('No test_metrics found in JSON')

print('\nJSON-only plotting complete.')
