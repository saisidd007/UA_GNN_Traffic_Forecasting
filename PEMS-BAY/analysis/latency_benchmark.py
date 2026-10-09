#!/usr/bin/env python
"""Latency benchmark for the PEMS-BAY TSSP model.

Outputs:
- results/latency_benchmark.json
"""

import json
import os
import sys
import time
from pathlib import Path

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


def load_sample(device: torch.device):
    (train_data, val_data, test_data), edge_index, num_sensors, scaler = load_pems_bay_data(
        data_dir=str(DATA_DIR), sequence_length=12, prediction_length=12
    )
    if len(test_data) == 0:
        raise RuntimeError('No test sequences available')

    dataset = PEMSBayDataset(test_data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    sample_x, _ = next(iter(loader))
    return sample_x[0].to(device), edge_index.to(device)


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
    model.eval()
    return model


def sync(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark(model, sample_x, edge_index, device, warmup=20, iterations=100):
    latencies = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample_x, edge_index, return_uncertainty=False)
        sync(device)
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(sample_x, edge_index, return_uncertainty=False)
            sync(device)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
    return latencies


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    sample_x, edge_index = load_sample(device)
    model = load_model(device)

    latencies = benchmark(model, sample_x, edge_index, device)
    total_params = sum(p.numel() for p in model.parameters())
    results = {
        'device': str(device),
        'mean_latency_ms': float(np.mean(latencies)) if (latencies := latencies) else 0.0,
        'std_latency_ms': float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
        'iterations': len(latencies),
        'warmup_iterations': 20,
        'total_parameters': int(total_params),
        'params_millions': float(total_params) / 1e6,
        'checkpoint': str(CHECKPOINT_PATH),
    }

    out_path = RESULTS / 'latency_benchmark.json'
    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"Saved latency benchmark to {out_path}")


if __name__ == '__main__':
    import numpy as np
    main()
