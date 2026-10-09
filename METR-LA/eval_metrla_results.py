#!/usr/bin/env python
"""
Evaluate trained METR-LA model and generate final results
"""

import os
import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict

# Import modules
from src.models.enhanced_gnn import create_improved_model
from src.utils.enhanced_dataset import create_enhanced_dataset
from torch_geometric.loader import DataLoader


class MetricsCalculator:
    """Calculate comprehensive metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate MAE, RMSE, MAPE, R², Correlation"""
        # Reshape to 1D if needed
        if predictions.ndim > 1:
            predictions = predictions.reshape(-1)
        if targets.ndim > 1:
            targets = targets.reshape(-1)
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Handle NaN in correlation
        corr = np.corrcoef(predictions, targets)[0, 1]
        correlation = float(corr) if not np.isnan(corr) else 0.0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2_score),
            'correlation': float(correlation)
        }


def main():
    print("[METR-LA] Final Model Evaluation")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}\n")
    
    # Load dataset
    print("[DATA] Loading METR-LA dataset for evaluation...")
    dataset = create_enhanced_dataset(
        root_dir='data/metr-la',
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust',
        dataset_name='METR-LA'
    )
    
    # Load trained model
    print("[MODEL] Loading trained model...")
    model = create_improved_model(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,
    ).to(device)
    
    try:
        model_path = 'results/enhanced_best_model.pt' if os.path.exists('results/enhanced_best_model.pt') else 'results/best_model_metrla.pt'
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"✓ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Prepare test data
    print("[TEST] Preparing test data...")
    test_data = dataset.get_test_data()
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
    
    # Evaluate
    print("[EVAL] Evaluating on test set...\n")
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_uncertainties_alea = []
    all_uncertainties_epi = []
    
    K = 10  # MC samples
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            mc_preds = []
            mc_aleas = []
            
            for k in range(K):
                model.train()  # Keep dropout
                preds_k, alea_k, _ = model(
                    batch.x,
                    batch.edge_index,
                    batch.missing_mask,
                    return_uncertainty=True
                )
                mc_preds.append(preds_k)
                mc_aleas.append(alea_k)
            
            mc_preds = torch.stack(mc_preds, dim=0)
            mc_aleas = torch.stack(mc_aleas, dim=0)
            
            mean_pred = mc_preds.mean(dim=0)
            epi_var = mc_preds.var(dim=0)
            alea_var = mc_aleas.mean(dim=0)
            
            all_predictions.append(mean_pred.detach().cpu().numpy())
            all_targets.append(batch.y.detach().cpu().numpy())
            all_uncertainties_alea.append(alea_var.detach().cpu().numpy())
            all_uncertainties_epi.append(epi_var.detach().cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"   Processed {batch_idx + 1} batches...")
    
    # Compute metrics
    predictions_concat = np.concatenate(all_predictions, axis=0)
    targets_concat = np.concatenate(all_targets, axis=0)
    alea_concat = np.concatenate(all_uncertainties_alea, axis=0)
    epi_concat = np.concatenate(all_uncertainties_epi, axis=0)
    
    metrics = MetricsCalculator.calculate_metrics(predictions_concat, targets_concat)
    
    # Add uncertainty metrics
    mean_alea = float(np.mean(alea_concat))
    mean_epi = float(np.mean(epi_concat))
    mean_total = mean_alea + mean_epi
    
    metrics['mean_aleatoric_uncertainty'] = mean_alea
    metrics['mean_epistemic_uncertainty'] = mean_epi
    metrics['mean_total_uncertainty'] = mean_total
    
    # Print results
    print("\n" + "=" * 80)
    print("METR-LA MODEL - FINAL TEST EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nMetric                           Value")
    print("-" * 50)
    print(f"Mean Absolute Error (MAE):       {metrics['mae']:.6f}")
    print(f"Root Mean Squared Error (RMSE):  {metrics['rmse']:.6f}")
    print(f"R² Score:                        {metrics['r2_score']:.6f}")
    print(f"Pearson Correlation:             {metrics['correlation']:.6f}")
    print(f"Mean Absolute Percentage Error:  {metrics['mape']:.6f}%")
    print(f"\nUncertainty Metrics:")
    print(f"  Mean Aleatoric Uncertainty:    {metrics['mean_aleatoric_uncertainty']:.6f}")
    print(f"  Mean Epistemic Uncertainty:    {metrics['mean_epistemic_uncertainty']:.6f}")
    print(f"  Mean Total Uncertainty:        {metrics['mean_total_uncertainty']:.6f}")
    
    # Compare with PEMS-BAY
    print("\n" + "=" * 80)
    print("COMPARISON: METR-LA vs PEMS-BAY (50 epochs)")
    print("=" * 80)
    pems_metrics = {
        'MAE': 0.4392,
        'RMSE': 1.0327,
        'R² Score': 0.8385,
        'Pearson Correlation': 0.9158,
        'Mean Aleatoric Uncertainty': 0.5508,
        'Mean Epistemic Uncertainty': 0.2560,
        'Mean Total Uncertainty': 0.8069
    }
    
    print(f"\n{'Metric':<35} {'METR-LA':>15} {'PEMS-BAY':>15}")
    print("-" * 65)
    print(f"{'MAE':<35} {metrics['mae']:>15.6f} {pems_metrics['MAE']:>15.4f}")
    print(f"{'RMSE':<35} {metrics['rmse']:>15.6f} {pems_metrics['RMSE']:>15.4f}")
    print(f"{'R² Score':<35} {metrics['r2_score']:>15.6f} {pems_metrics['R² Score']:>15.4f}")
    print(f"{'Pearson Correlation':<35} {metrics['correlation']:>15.6f} {pems_metrics['Pearson Correlation']:>15.4f}")
    print(f"{'Aleatoric Uncertainty':<35} {metrics['mean_aleatoric_uncertainty']:>15.6f} {pems_metrics['Mean Aleatoric Uncertainty']:>15.4f}")
    print(f"{'Epistemic Uncertainty':<35} {metrics['mean_epistemic_uncertainty']:>15.6f} {pems_metrics['Mean Epistemic Uncertainty']:>15.4f}")
    print(f"{'Total Uncertainty':<35} {metrics['mean_total_uncertainty']:>15.6f} {pems_metrics['Mean Total Uncertainty']:>15.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/enhanced_training_results_metrla_{timestamp}.json'
    
    results = {
        'dataset': 'METR-LA',
        'num_epochs': 50,
        'num_sensors': 207,
        'test_metrics': metrics,
        'timestamp': timestamp,
        'comparison': pems_metrics
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Results saved to: {results_file}")
    print("[DONE] Evaluation complete!")


if __name__ == "__main__":
    main()
