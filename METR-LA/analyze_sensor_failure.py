#!/usr/bin/env python
"""
Sensor Failure Rate Analysis
Shows MAE, RMSE, MAPE for different sensor dropout rates (0%, 5%, 10%, 20%, 30%)
"""

import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import modules
from src.models.enhanced_gnn import create_improved_model
from src.utils.enhanced_dataset import create_enhanced_dataset


class MetricsCalculator:
    """Calculate comprehensive metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate MAE, RMSE, MAPE, R²"""
        if predictions.ndim > 2:
            predictions = predictions.reshape(-1)
        if targets.ndim > 2:
            targets = targets.reshape(-1)
        
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((targets - predictions) / (np.abs(targets) + 1e-8))) * 100
        
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2_score': float(r2_score)
        }


class SensorDropoutAnalyzer:
    """Analyze model robustness with sensor dropout at different rates"""
    
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.metrics_calc = MetricsCalculator()
    
    def apply_sensor_dropout(self, features: torch.Tensor, missing_mask: torch.Tensor, 
                             dropout_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sensor dropout to features based on rate
        
        Args:
            features: Input features [B, N, T, C]
            missing_mask: Original missing mask [B, N, T]
            dropout_rate: Dropout percentage (0-100)
            
        Returns:
            Modified features and modified mask
        """
        features_dropped = features.clone()
        mask_dropped = missing_mask.clone()
        
        if dropout_rate > 0:
            # Get valid sensor indices
            B, N, T, C = features.shape
            
            for b in range(B):
                # Find currently valid sensors (mask == 1)
                valid_sensors = []
                for n in range(N):
                    if torch.sum(missing_mask[b, n, :]) > 0:  # Has some valid timestamps
                        valid_sensors.append(n)
                
                if len(valid_sensors) > 0:
                    # Calculate how many to dropout
                    num_to_dropout = max(1, int(len(valid_sensors) * dropout_rate / 100))
                    dropout_indices = np.random.choice(valid_sensors, size=min(num_to_dropout, len(valid_sensors)), 
                                                      replace=False)
                    
                    # Set those sensors to zero
                    for n in dropout_indices:
                        features_dropped[b, n, :, :] = 0
                        mask_dropped[b, n, :] = 0
        
        return features_dropped, mask_dropped
    
    def test_with_dropout(self, dropout_rate: float = 0) -> Dict[str, float]:
        """Test model with sensor dropout"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        print(f"  Testing with {dropout_rate:.0f}% sensor dropout...")
        pbar = tqdm(self.test_loader, desc=f'Dropout {dropout_rate:.0f}%', disable=False)
        
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Apply dropout
                x_dropped, mask_dropped = self.apply_sensor_dropout(batch.x, batch.missing_mask, dropout_rate)
                
                # Forward pass with dropout
                K = 10  # MC samples
                mc_preds = []
                for k in range(K):
                    self.model.train()  # Enable dropout
                    preds_k, _, _ = self.model(
                        x_dropped,
                        batch.edge_index,
                        mask_dropped,
                        return_uncertainty=True
                    )
                    mc_preds.append(preds_k)
                
                mc_preds = torch.stack(mc_preds, dim=0)
                pred_mean = mc_preds.mean(dim=0)
                
                all_predictions.append(pred_mean.detach().cpu().numpy())
                all_targets.append(batch.y.detach().cpu().numpy())
        
        predictions_concat = np.concatenate(all_predictions, axis=0)
        targets_concat = np.concatenate(all_targets, axis=0)
        metrics = self.metrics_calc.calculate_metrics(predictions_concat, targets_concat)
        
        return metrics
    
    def analyze_all_dropout_rates(self, dropout_rates: List[float]) -> Dict[float, Dict[str, float]]:
        """Analyze model at all dropout rates"""
        results = {}
        for rate in dropout_rates:
            metrics = self.test_with_dropout(rate)
            results[rate] = metrics
        return results


def main():
    """Main analysis function"""
    print("[SENSOR FAILURE ANALYSIS] Testing robustness at different sensor dropout rates")
    print("=" * 90)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}\n")
    
    # Load dataset
    print("[DATA] Loading dataset...")
    dataset = create_enhanced_dataset(
        root_dir='data',
        sequence_length=12,
        prediction_length=12,
        preprocessing_method='robust'
    )
    
    test_data = dataset.get_test_data()
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
    print(f"[STATS] Test set: {len(test_data)} samples\n")
    
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
    print(f"[LOADED] Best model from epoch {checkpoint['epoch'] + 1}\n")
    
    # Analyze sensor dropout
    print("[ANALYSIS] Running sensor failure analysis...\n")
    analyzer = SensorDropoutAnalyzer(model, test_loader, device)
    dropout_rates = [0, 5, 10, 20, 30]
    results = analyzer.analyze_all_dropout_rates(dropout_rates)
    
    # Print results in horizontal layout
    print("\n" + "="*90)
    print("[RESULTS] SENSOR FAILURE RATE IMPACT ON MODEL PERFORMANCE")
    print("="*90)
    
    # MAE Table
    print("\n[MAE - Mean Absolute Error]")
    print("-" * 90)
    header = "Dropout Rate |"
    for rate in dropout_rates:
        header += f"  {rate:3.0f}%  |"
    print(header)
    print("-" * 90)
    mae_row = "MAE         |"
    for rate in dropout_rates:
        mae_val = results[rate]['mae']
        mae_row += f" {mae_val:6.4f} |"
    print(mae_row)
    print("-" * 90)
    
    # RMSE Table
    print("\n[RMSE - Root Mean Squared Error]")
    print("-" * 90)
    print(header)
    print("-" * 90)
    rmse_row = "RMSE        |"
    for rate in dropout_rates:
        rmse_val = results[rate]['rmse']
        rmse_row += f" {rmse_val:6.4f} |"
    print(rmse_row)
    print("-" * 90)
    
    # MAPE Table
    print("\n[MAPE - Mean Absolute Percentage Error]")
    print("-" * 90)
    print(header)
    print("-" * 90)
    mape_row = "MAPE (%)    |"
    for rate in dropout_rates:
        mape_val = results[rate]['mape']
        mape_row += f" {mape_val:6.2f}% |"
    print(mape_row)
    print("-" * 90)
    
    # R² Table
    print("\n[R² - Coefficient of Determination]")
    print("-" * 90)
    print(header)
    print("-" * 90)
    r2_row = "R² Score    |"
    for rate in dropout_rates:
        r2_val = results[rate]['r2_score']
        r2_row += f" {r2_val:6.4f} |"
    print(r2_row)
    print("-" * 90)
    
    # Detailed breakdown
    print("\n" + "="*90)
    print("[DETAILED BREAKDOWN] Per Dropout Rate Metrics")
    print("="*90)
    for rate in dropout_rates:
        print(f"\nDropout Rate: {rate}%")
        print("-" * 90)
        m = results[rate]
        print(f"  MAE:     {m['mae']:.6f}")
        print(f"  RMSE:    {m['rmse']:.6f}")
        print(f"  MAPE:    {m['mape']:.2f}%")
        print(f"  R²:      {m['r2_score']:.6f}")
    
    # Calculate degradation
    print("\n" + "="*90)
    print("[ROBUSTNESS] Performance Degradation from Baseline (0% dropout)")
    print("="*90)
    baseline_mae = results[0]['mae']
    baseline_rmse = results[0]['rmse']
    baseline_mape = results[0]['mape']
    
    print("\n[MAE Degradation (%)]")
    print("-" * 90)
    header = "Dropout Rate |"
    for rate in dropout_rates:
        header += f"  {rate:3.0f}%  |"
    print(header)
    print("-" * 90)
    mae_deg = "Degradation |"
    for rate in dropout_rates:
        if rate == 0:
            mae_deg += f"  0.00%  |"
        else:
            deg = 100 * (results[rate]['mae'] - baseline_mae) / baseline_mae
            mae_deg += f" {deg:6.2f}% |"
    print(mae_deg)
    print("-" * 90)
    
    print("\n[RMSE Degradation (%)]")
    print("-" * 90)
    print(header)
    print("-" * 90)
    rmse_deg = "Degradation |"
    for rate in dropout_rates:
        if rate == 0:
            rmse_deg += f"  0.00%  |"
        else:
            deg = 100 * (results[rate]['rmse'] - baseline_rmse) / baseline_rmse
            rmse_deg += f" {deg:6.2f}% |"
    print(rmse_deg)
    print("-" * 90)
    
    print("\n[MAPE Degradation (%)]")
    print("-" * 90)
    print(header)
    print("-" * 90)
    mape_deg = "Degradation |"
    for rate in dropout_rates:
        if rate == 0:
            mape_deg += f"  0.00%  |"
        else:
            deg = 100 * (results[rate]['mape'] - baseline_mape) / baseline_mape
            mape_deg += f" {deg:6.2f}% |"
    print(mape_deg)
    print("-" * 90)
    
    # Save results
    results_file = 'results/sensor_failure_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Analysis results: {results_file}")
    print("="*90 + "\n")
    
    return results


if __name__ == "__main__":
    main()
