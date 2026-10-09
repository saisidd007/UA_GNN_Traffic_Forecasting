"""
CORRECTED EVALUATION - Match Official Baseline Metrics
Evaluates on the FULL TEST SET (not per-horizon average)
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

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


def evaluate_full_test_set(
    checkpoint_path: str = str(RESULTS / 'enhanced_best_model.pt'),
    batch_size: int = 8,
    device: str = None
) -> dict:
    """
    Evaluate model on FULL TEST SET to match official baseline
    Computes overall MAE/RMSE across all predictions
    THEN breaks down by horizon
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
    print("Running inference on full test set...")
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
    # OVERALL METRICS (ALL PREDICTIONS)
    # ========================================================================
    
    overall_mae = compute_mae(preds, targets)
    overall_rmse = compute_rmse(preds, targets)
    
    print("=" * 80)
    print("OVERALL METRICS (Computed on ALL Predictions - Full Test Set)")
    print("=" * 80)
    print(f"MAE:  {overall_mae:.6f}")
    print(f"RMSE: {overall_rmse:.6f}")
    print("=" * 80)
    
    # ========================================================================
    # PER-HORIZON BREAKDOWN
    # ========================================================================
    
    horizons = {
        3: "3-step (15 min)",
        6: "6-step (30 min)",
        12: "12-step (60 min)",
    }
    
    results = {
        'overall': {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'num_predictions': int(np.prod(preds.shape))
        },
        'per_horizon': {}
    }
    
    print("\nPER-HORIZON BREAKDOWN:")
    print("-" * 80)
    print(f"{'Horizon':<20} {'MAE':<15} {'RMSE':<15} {'vs Overall':<20}")
    print("-" * 80)
    
    for step, label in horizons.items():
        step_idx = step - 1
        
        pred_at_step = preds[:, step_idx]
        true_at_step = targets[:, step_idx]
        
        mae = compute_mae(pred_at_step, true_at_step)
        rmse = compute_rmse(pred_at_step, true_at_step)
        
        mae_diff_pct = ((mae - overall_mae) / overall_mae) * 100
        rmse_diff_pct = ((rmse - overall_rmse) / overall_rmse) * 100
        
        diff_str = f"{mae_diff_pct:+.2f}% / {rmse_diff_pct:+.2f}%"
        
        results['per_horizon'][label] = {
            'mae': mae,
            'rmse': rmse,
            'step': step
        }
        
        print(f"{label:<20} {mae:<15.6f} {rmse:<15.6f} {diff_str:<20}")
    
    print("-" * 80)
    
    # ========================================================================
    # COMPARISON WITH OFFICIAL BASELINE
    # ========================================================================
    
    official_baseline = {
        'mae': 0.43919557332992554,
        'rmse': 1.0326961278915405,
        'source': 'enhanced_best_model.pt (50-epoch training)',
        'r2': 0.8385,
        'correlation': 0.9158,
        'aleatoric_uncertainty': 0.5508,
        'epistemic_uncertainty': 0.2560,
        'total_uncertainty': 0.8069,
    }
    
    results['official_baseline'] = official_baseline
    
    print("\nCOMPARISON WITH OFFICIAL BASELINE:")
    print("-" * 80)
    print(f"Official Baseline:")
    print(f"  MAE:  {official_baseline['mae']:.6f}")
    print(f"  RMSE: {official_baseline['rmse']:.6f}")
    print(f"  R²:   {official_baseline['r2']:.4f}")
    print(f"  Pearson Corr: {official_baseline['correlation']:.4f}")
    
    print(f"\nCurrent Evaluation (Full Test Set):")
    print(f"  MAE:  {overall_mae:.6f}")
    print(f"  RMSE: {overall_rmse:.6f}")
    
    mae_diff = overall_mae - official_baseline['mae']
    rmse_diff = overall_rmse - official_baseline['rmse']
    mae_pct = (mae_diff / official_baseline['mae']) * 100
    rmse_pct = (rmse_diff / official_baseline['rmse']) * 100
    
    print(f"\nDifference:")
    print(f"  MAE:  {mae_diff:+.6f} ({mae_pct:+.4f}%)")
    print(f"  RMSE: {rmse_diff:+.6f} ({rmse_pct:+.4f}%)")
    
    print("-" * 80)
    
    # Check if they match
    mae_matches = abs(mae_diff) < 0.0001
    rmse_matches = abs(rmse_diff) < 0.001
    
    if mae_matches and rmse_matches:
        print("✓ BASELINE MATCHES! Evaluation is correct.")
    else:
        print("✗ Baseline does not match exactly (expected ±0.0001 for MAE, ±0.001 for RMSE)")
    
    print("-" * 80)
    
    # Save results
    results_file = RESULTS / 'evaluation_full_test_set.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    results = evaluate_full_test_set()
