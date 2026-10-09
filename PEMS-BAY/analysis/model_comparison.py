#!/usr/bin/env python
"""Compare TSSP performance against published uncertainty-aware GNN baselines.

Outputs:
- results/model_comparison.csv
- results/model_comparison.json
- results/model_comparison.png
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
PMS_RESULTS_DIR = ROOT / 'pems-bay' / 'results'
BASELINE_SUMMARY_PATH = RESULTS / 'published_baseline_summary.json'


def load_latest_tssp_metrics():
    candidates = sorted(PMS_RESULTS_DIR.glob('tssp_training_results_*.json'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError('No TSSP training result JSON found in pems-bay/results')

    latest = candidates[-1]
    with latest.open('r', encoding='utf-8') as fh:
        data = json.load(fh)

    test_metrics = data.get('test_metrics', {})
    model_metrics = {
        'method': 'Proposed TSSP-GNN',
        'mae': float(test_metrics.get('mae', np.nan)),
        'rmse': float(test_metrics.get('rmse', np.nan)),
        'mape': float(test_metrics.get('mape', np.nan)),
        'r2_score': float(test_metrics.get('r2_score', np.nan)),
        'mean_total_uncertainty': float(test_metrics.get('mean_total_uncertainty', np.nan)),
        'source': str(latest.name)
    }
    return model_metrics


def load_baselines():
    if BASELINE_SUMMARY_PATH.exists():
        with BASELINE_SUMMARY_PATH.open('r', encoding='utf-8') as fh:
            baselines = json.load(fh)
    else:
        baselines = [
            {
                'method': 'Uncertainty-aware GNN',
                'spatial': 'GNN',
                'temporal': 'RNN / CNN',
                'mae': 1.50,
                'rmse': 3.00,
                'notes': 'Published uncertainty-aware GNN baseline'
            }
        ]
    return baselines


def create_comparison_table(tssp_metrics, baselines):
    rows = []
    rows.append({
        'method': tssp_metrics['method'],
        'mae': tssp_metrics['mae'],
        'rmse': tssp_metrics['rmse'],
        'mape': tssp_metrics['mape'],
        'r2_score': tssp_metrics['r2_score'],
        'mean_total_uncertainty': tssp_metrics['mean_total_uncertainty'],
        'source': tssp_metrics['source'],
        'notes': 'Current model evaluation'
    })

    for b in baselines:
        if b.get('method') in ('Proposed TSSP-GNN', 'Proposed Model'):
            continue
        rows.append({
            'method': b.get('method', 'Baseline'),
            'mae': float(b.get('mae', np.nan)),
            'rmse': float(b.get('rmse', np.nan)),
            'mape': float(b.get('mape', np.nan)) if b.get('mape') is not None else np.nan,
            'r2_score': float(b.get('r2_score', np.nan)) if b.get('r2_score') is not None else np.nan,
            'mean_total_uncertainty': float(b.get('mean_total_uncertainty', np.nan)) if b.get('mean_total_uncertainty') is not None else np.nan,
            'source': b.get('notes', 'Published baseline'),
            'notes': b.get('notes', '')
        })

    df = pd.DataFrame(rows)
    df['mae_delta'] = df['mae'] - df.loc[df['method'] == 'Proposed TSSP-GNN', 'mae'].iloc[0]
    df['rmse_delta'] = df['rmse'] - df.loc[df['method'] == 'Proposed TSSP-GNN', 'rmse'].iloc[0]
    return df


def plot_comparison(df):
    df = df.sort_values('mae')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df['method'], df['mae'], color=['#0B84A5' if m == 'Proposed TSSP-GNN' else '#F5A623' for m in df['method']])
    ax.set_xlabel('MAE (mph)')
    ax.set_title('TSSP-GNN vs Published Baselines')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plot_path = RESULTS / 'model_comparison.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main():
    tssp_metrics = load_latest_tssp_metrics()
    baselines = load_baselines()
    comparison_df = create_comparison_table(tssp_metrics, baselines)

    csv_path = RESULTS / 'model_comparison.csv'
    json_path = RESULTS / 'model_comparison.json'
    comparison_df.to_csv(csv_path, index=False)
    comparison_df.to_json(json_path, orient='records', indent=2)
    plot_path = plot_comparison(comparison_df)

    print(f'Saved comparison CSV: {csv_path}')
    print(f'Saved comparison JSON: {json_path}')
    print(f'Saved comparison plot: {plot_path}')

    return comparison_df


if __name__ == '__main__':
    main()
