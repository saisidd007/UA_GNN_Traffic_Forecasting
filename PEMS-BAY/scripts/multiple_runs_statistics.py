#!/usr/bin/env python
"""
Multiple Runs Statistical Analysis with 95% Confidence Intervals
Uses checkpoints from epochs 10, 20, 30, 40, 50 to compute mean and CI
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.models.enhanced_gnn import create_improved_model
from src.utils.enhanced_dataset import create_enhanced_dataset


class MetricsCalculator:
    """Calculate metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate MAE, RMSE, MAPE, R²"""
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


class MultipleRunsAnalyzer:
    """Analyze multiple checkpoints for statistical significance"""
    
    def __init__(self, test_loader, device):
        self.test_loader = test_loader
        self.device = device
        self.metrics_calc = MetricsCalculator()
    
    def test_checkpoint(self, checkpoint_path: str, checkpoint_name: str) -> Dict[str, float]:
        """Test a single checkpoint"""
        print(f"\n  Testing {checkpoint_name}...")
        
        # Load model
        model = create_improved_model(
            in_channels=12,
            hidden_channels=64,
            out_channels=12,
            num_gnn_layers=4,
            num_temporal_layers=3,
            num_attention_heads=4,
        ).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.test_loader, desc=checkpoint_name, disable=False)
        
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)
                
                # MC inference (K=10)
                mc_preds = []
                for k in range(10):
                    model.train()
                    preds, _, _ = model(
                        batch.x,
                        batch.edge_index,
                        batch.missing_mask,
                        return_uncertainty=True
                    )
                    mc_preds.append(preds)
                
                mc_preds = torch.stack(mc_preds, dim=0)
                pred_mean = mc_preds.mean(dim=0)
                
                all_preds.append(pred_mean.cpu().numpy())
                all_targets.append(batch.y.cpu().numpy())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        return self.metrics_calc.calculate_metrics(preds, targets)
    
    def compute_statistics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute mean and 95% CI for metrics"""
        metric_names = metrics_list[0].keys()
        stats_dict = {}
        
        for metric in metric_names:
            values = np.array([m[metric] for m in metrics_list])
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            n = len(values)
            
            # 95% CI using t-distribution
            t_crit = stats.t.ppf(0.975, df=n-1)  # 95% CI
            ci = t_crit * std / np.sqrt(n)
            
            stats_dict[metric] = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(mean - ci),
                'ci_upper': float(mean + ci),
                'ci_width': float(2 * ci),
                'n_runs': n
            }
        
        return stats_dict
    
    def run_multiple_analysis(self, checkpoint_configs: List[Tuple[str, str]]) -> Dict:
        """Run analysis on multiple checkpoints"""
        print("\n[MULTI-RUN] Analyzing multiple checkpoints...")
        
        all_metrics = []
        checkpoint_names = []
        
        for checkpoint_path, checkpoint_name in checkpoint_configs:
            if os.path.exists(checkpoint_path):
                metrics = self.test_checkpoint(checkpoint_path, checkpoint_name)
                all_metrics.append(metrics)
                checkpoint_names.append(checkpoint_name)
            else:
                print(f"  WARNING: Checkpoint not found: {checkpoint_path}")
        
        # Compute statistics
        stats = self.compute_statistics(all_metrics)
        
        return {
            'checkpoints': checkpoint_names,
            'individual_runs': all_metrics,
            'statistics': stats
        }


def format_with_ci(mean: float, ci_lower: float, ci_upper: float) -> str:
    """Format value with confidence interval"""
    return f"{mean:.6f} [{ci_lower:.6f}, {ci_upper:.6f}]"


def main():
    """Main multiple runs analysis"""
    print("\n" + "="*120)
    print("[MULTIPLE RUNS] Statistical Analysis with 95% Confidence Intervals")
    print("="*120)
    
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
    
    # Define checkpoints to analyze
    checkpoint_configs = [
        ('results/enhanced_checkpoint_epoch_10.pt', 'Epoch 10'),
        ('results/enhanced_checkpoint_epoch_20.pt', 'Epoch 20'),
        ('results/enhanced_checkpoint_epoch_30.pt', 'Epoch 30'),
        ('results/enhanced_checkpoint_epoch_40.pt', 'Epoch 40'),
        ('results/enhanced_checkpoint_epoch_50.pt', 'Epoch 50'),
    ]
    
    # Run analysis
    analyzer = MultipleRunsAnalyzer(test_loader, device)
    results = analyzer.run_multiple_analysis(checkpoint_configs)
    
    # Display results
    print("\n" + "="*120)
    print("[INDIVIDUAL RUN RESULTS]")
    print("="*120)
    
    for cp_name, metrics in zip(results['checkpoints'], results['individual_runs']):
        print(f"\n{cp_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.upper():12s}: {value:.6f}")
    
    # Display statistical summary
    print("\n" + "="*120)
    print("[STATISTICAL SUMMARY] Mean ± 95% CI across all checkpoints")
    print("="*120)
    
    stats_results = results['statistics']
    
    # Create detailed table
    table_data = []
    for metric_name, stats_vals in stats_results.items():
        table_data.append({
            'Metric': metric_name.upper(),
            'Mean': stats_vals['mean'],
            'Std': stats_vals['std'],
            'CI Lower': stats_vals['ci_lower'],
            'CI Upper': stats_vals['ci_upper'],
            'CI Width': stats_vals['ci_width'],
            'N Runs': int(stats_vals['n_runs'])
        })
    
    summary_df = pd.DataFrame(table_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Print formatted results
    print("\n" + "="*120)
    print("[FORMATTED RESULTS] For Publication")
    print("="*120)
    
    for metric_name, stats_vals in stats_results.items():
        formatted = format_with_ci(
            stats_vals['mean'],
            stats_vals['ci_lower'],
            stats_vals['ci_upper']
        )
        print(f"{metric_name.upper():12s}: {formatted}")
    
    # Save results
    json_file = 'results/multiple_runs_statistics.json'
    with open(json_file, 'w') as f:
        json.dump({
            'checkpoints': results['checkpoints'],
            'individual_runs': results['individual_runs'],
            'statistics': results['statistics']
        }, f, indent=2)
    print(f"\n[SAVED] JSON file: {json_file}")
    
    # Save summary CSV
    csv_file = 'results/multiple_runs_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"[SAVED] CSV file: {csv_file}")
    
    print("="*120 + "\n")
    
    return results


if __name__ == "__main__":
    main()
