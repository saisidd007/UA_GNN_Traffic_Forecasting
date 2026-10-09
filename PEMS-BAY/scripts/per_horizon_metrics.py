#!/usr/bin/env python
"""
Per-Horizon Metrics Analysis with MC-Dropout Uncertainty Quantification
Computes MAE, RMSE, MAPE, PICP, PIW for each prediction horizon (step 1-12)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.models.enhanced_gnn import create_improved_model
from src.utils.enhanced_dataset import create_enhanced_dataset


class HorizonMetricsCalculator:
    """Calculate per-horizon metrics with MC-dropout uncertainty"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, 
                         uncertainties: np.ndarray = None, alpha: float = 0.05) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a horizon
        
        Args:
            predictions: Model predictions [N,]
            targets: Ground truth [N,]
            uncertainties: Predicted std dev [N,]
            alpha: Confidence level (default 0.05 for 95% CI)
            
        Returns:
            Dict with MAE, RMSE, MAPE, PICP, PIW, etc.
        """
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2_score)
        }
        
        # Uncertainty metrics
        if uncertainties is not None and len(uncertainties) > 0:
            z_alpha = 1.96  # 95% confidence
            lower_bound = predictions - z_alpha * uncertainties
            upper_bound = predictions + z_alpha * uncertainties
            
            # PICP: Prediction Interval Coverage Probability
            coverage = np.sum((targets >= lower_bound) & (targets <= upper_bound)) / len(targets)
            metrics['picp'] = float(coverage)
            
            # PIW: Prediction Interval Width
            interval_width = np.mean(upper_bound - lower_bound)
            metrics['piw'] = float(interval_width)
            
            # MIS: Mean Interval Score
            lower_violations = np.maximum(0, lower_bound - targets)
            upper_violations = np.maximum(0, targets - upper_bound)
            mis = np.mean(upper_bound - lower_bound + 
                         (2/z_alpha) * (lower_violations + upper_violations))
            metrics['mis'] = float(mis)
        else:
            metrics['picp'] = 0.0
            metrics['piw'] = 0.0
            metrics['mis'] = 0.0
        
        return metrics


class PerHorizonAnalyzer:
    """Per-horizon metrics analysis with MC-dropout"""
    
    def __init__(self, model, test_loader, device, num_samples: int = 30):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.num_samples = num_samples
        self.metrics_calc = HorizonMetricsCalculator()
    
    def get_mc_predictions(self, batch, num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get MC-dropout ensemble predictions
        
        Returns:
            predictions: [num_samples, total_outputs]
            targets: [total_outputs]
            uncertainties: [total_outputs]
        """
        mc_preds = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                self.model.train()  # Enable dropout
                preds, _, _ = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.missing_mask,
                    return_uncertainty=True
                )
                mc_preds.append(preds)
        
        mc_preds = torch.stack(mc_preds, dim=0)  # [num_samples, B, N, T, C]
        pred_mean = mc_preds.mean(dim=0)  # [B, N, T, C]
        pred_std = mc_preds.std(dim=0)  # [B, N, T, C]
        
        # Flatten to [total_outputs]
        targets = batch.y.detach().cpu().numpy().reshape(-1)
        predictions = pred_mean.detach().cpu().numpy().reshape(-1)
        uncertainties = pred_std.detach().cpu().numpy().reshape(-1)
        
        return predictions, targets, uncertainties
    
    def analyze_per_horizon(self) -> pd.DataFrame:
        """Analyze metrics for each prediction horizon"""
        print(f"\n[MC-DROPOUT] Running MC-dropout inference with {self.num_samples} samples...")
        
        horizon_metrics = {h: {'mae': [], 'rmse': [], 'mape': [], 'r2_score': [], 
                               'picp': [], 'piw': [], 'mis': []} 
                          for h in range(1, 13)}
        
        self.model.eval()
        pbar = tqdm(self.test_loader, desc='Per-Horizon Analysis', disable=False)
        
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)
                
                predictions, targets, uncertainties = self.get_mc_predictions(batch, self.num_samples)
                
                # Extract per-horizon metrics
                # batch.y shape: [B, N, T] or [B*N*T]
                # Reshape predictions and targets to [B, N, T]
                num_nodes = batch.x.shape[1]  # N
                target_horizon = batch.y.shape[-1] if len(batch.y.shape) > 1 else 12  # T
                
                # Flatten and reshape
                preds_reshaped = predictions.reshape(-1, num_nodes, target_horizon)
                targets_reshaped = targets.reshape(-1, num_nodes, target_horizon)
                unce_reshaped = uncertainties.reshape(-1, num_nodes, target_horizon)
                
                for horizon in range(1, min(13, target_horizon + 1)):
                    # Extract predictions and targets for this horizon
                    h_idx = horizon - 1
                    h_preds = preds_reshaped[:, :, h_idx].flatten()
                    h_targets = targets_reshaped[:, :, h_idx].flatten()
                    h_unce = unce_reshaped[:, :, h_idx].flatten()
                    
                    metrics = self.metrics_calc.calculate_metrics(h_preds, h_targets, h_unce)
                    
                    for metric_name, value in metrics.items():
                        horizon_metrics[horizon][metric_name].append(value)
        
        # Aggregate per-horizon metrics
        results = []
        for horizon in range(1, 13):
            row = {'Horizon': horizon}
            for metric_name in horizon_metrics[horizon].keys():
                values = horizon_metrics[horizon][metric_name]
                row[f'{metric_name.upper()}'] = np.mean(values) if values else 0.0
            results.append(row)
        
        return pd.DataFrame(results)


def main():
    """Main per-horizon analysis"""
    print("\n" + "="*100)
    print("[PER-HORIZON METRICS] MC-Dropout Uncertainty Quantification Analysis")
    print("="*100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}")
    
    # Load dataset
    print("[DATA] Loading test dataset...")
    dataset = create_enhanced_dataset(
        root_dir='data',
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust'
    )
    
    test_data = dataset.get_test_data()
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
    print(f"[STATS] Test samples: {len(test_data)}")
    
    # Load best model
    print("[MODEL] Loading best trained model...")
    model = create_improved_model(
        in_channels=12,
        hidden_channels=64,
        out_channels=12,
        num_gnn_layers=4,
        num_temporal_layers=3,
        num_attention_heads=4,
    ).to(device)
    
    checkpoint = torch.load('results/enhanced_best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[LOADED] Best model from epoch {checkpoint['epoch'] + 1}")
    
    # Analyze per-horizon
    analyzer = PerHorizonAnalyzer(model, test_loader, device, num_samples=30)
    results_df = analyzer.analyze_per_horizon()
    
    # Print results
    print("\n" + "="*100)
    print("[RESULTS] PER-HORIZON METRICS")
    print("="*100)
    print(results_df.to_string(index=False))
    
    # Print summary statistics
    print("\n" + "="*100)
    print("[SUMMARY] Aggregated Metrics")
    print("="*100)
    for col in results_df.columns[1:]:
        mean_val = results_df[col].mean()
        std_val = results_df[col].std()
        print(f"{col:15s}: Mean={mean_val:.6f}, Std={std_val:.6f}")
    
    # Save results
    csv_file = 'results/per_horizon_metrics.csv'
    results_df.to_csv(csv_file, index=False)
    print(f"\n[SAVED] CSV file: {csv_file}")
    
    # Save detailed JSON
    results_dict = results_df.to_dict(orient='records')
    json_file = 'results/per_horizon_metrics.json'
    with open(json_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"[SAVED] JSON file: {json_file}")
    
    print("="*100 + "\n")
    
    return results_df


if __name__ == "__main__":
    main()
