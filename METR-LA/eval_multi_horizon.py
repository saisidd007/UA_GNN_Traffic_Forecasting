"""
Per-Horizon Evaluation for Traffic Forecasting
Following DCRNN and Graph WaveNet protocol
Evaluates MAE/RMSE at 3-step, 6-step, 12-step horizons (15, 30, 60 minutes)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader

# Setup paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
sys.path.insert(0, str(ROOT))

from src.utils.enhanced_dataset import create_enhanced_dataset
from src.models.enhanced_gnn import create_improved_model

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Mean Absolute Error"""
    return float(np.mean(np.abs(pred - true)))


def compute_rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Compute Root Mean Square Error"""
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def evaluate_multi_horizon(
    checkpoint_path: str = str(RESULTS / 'enhanced_best_model.pt'),
    batch_size: int = 8,
    device: str = None
) -> dict:
    """
    Evaluate model on multiple horizons (3, 6, 12 steps)
    
    Args:
        checkpoint_path: Path to model checkpoint
        batch_size: Batch size for evaluation
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Dictionary with MAE/RMSE for each horizon
    """
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    print(f"Device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    data_dir = str(DATA / 'metr-la') if (DATA / 'metr-la').exists() else str(DATA)
    dataset = create_enhanced_dataset(
        root_dir=data_dir,
        sequence_length=12,
        prediction_length=12,
        dataset_name='METR-LA' if 'metr-la' in data_dir else 'PEMS-BAY'
    )
    test_data = dataset.get_test_data()
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_loader)} batches, {len(test_data)} samples\n")
    
    # Create and load model
    print("Loading model...")
    model = create_improved_model(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}\n")
    
    # Collect all predictions and targets
    print("Running inference...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            preds, _, _ = model(batch.x, batch.edge_index, batch.missing_mask, return_uncertainty=False)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_loader)} batches...", end='\r')
    
    print(f"  Processed {len(test_loader)}/{len(test_loader)} batches... ✓")
    
    # Concatenate: shape [total_nodes, horizon]
    preds = np.concatenate(all_preds, axis=0)  # [N, T]
    targets = np.concatenate(all_targets, axis=0)  # [N, T]
    
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Targets shape: {targets.shape}\n")
    
    # ========================================================================
    # EVALUATE MULTI-HORIZON
    # ========================================================================
    
    horizons = {
        3: "3-step (15 min)",
        6: "6-step (30 min)",
        12: "12-step (60 min)",
    }
    
    results = {
        'overall': {},
        'per_horizon': {}
    }
    
    print("=" * 70)
    print("PER-HORIZON EVALUATION (DCRNN Protocol)")
    print("=" * 70)
    print(f"{'Horizon':<20} {'MAE':<15} {'RMSE':<15}")
    print("-" * 70)
    
    for step, label in horizons.items():
        # Extract predictions and targets at this step
        # step is 1-indexed in the paper (1, 2, ..., 12)
        # but numpy is 0-indexed, so use step-1
        step_idx = step - 1
        
        pred_at_step = preds[:, step_idx]  # [N]
        true_at_step = targets[:, step_idx]  # [N]
        
        mae = compute_mae(pred_at_step, true_at_step)
        rmse = compute_rmse(pred_at_step, true_at_step)
        
        results['per_horizon'][label] = {
            'mae': mae,
            'rmse': rmse,
            'step': step
        }
        
        print(f"{label:<20} {mae:<15.4f} {rmse:<15.4f}")
    
    # Overall metrics (average across all horizons)
    overall_mae = compute_mae(preds, targets)
    overall_rmse = compute_rmse(preds, targets)
    results['overall']['mae'] = overall_mae
    results['overall']['rmse'] = overall_rmse
    
    print("-" * 70)
    print(f"{'OVERALL (All)':<20} {overall_mae:<15.4f} {overall_rmse:<15.4f}")
    print("=" * 70)
    
    # ========================================================================
    # COMPARISON WITH OFFICIAL BASELINE
    # ========================================================================
    
    official_baseline = {
        'mae': 0.4392,
        'rmse': 1.0327
    }
    
    print("\nCOMPARISON WITH OFFICIAL BASELINE:")
    print("-" * 70)
    print(f"Official Baseline - MAE: {official_baseline['mae']:.4f}, RMSE: {official_baseline['rmse']:.4f}")
    print(f"Current Evaluation - MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}")
    
    mae_diff = overall_mae - official_baseline['mae']
    rmse_diff = overall_rmse - official_baseline['rmse']
    mae_pct = (mae_diff / official_baseline['mae']) * 100
    rmse_pct = (rmse_diff / official_baseline['rmse']) * 100
    
    print(f"Difference: MAE {mae_diff:+.4f} ({mae_pct:+.2f}%), RMSE {rmse_diff:+.4f} ({rmse_pct:+.2f}%)")
    print("-" * 70)
    
    # Save results
    results['official_baseline'] = official_baseline
    results_file = RESULTS / 'multi_horizon_eval.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    results = evaluate_multi_horizon()
