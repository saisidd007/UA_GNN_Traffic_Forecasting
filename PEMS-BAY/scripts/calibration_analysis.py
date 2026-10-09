#!/usr/bin/env python
"""
Calibration Curves and Sharpness Analysis
Evaluates prediction interval calibration, sharpness, and uncertainty quality
"""

import os
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.models.enhanced_gnn import create_improved_model
from src.utils.enhanced_dataset import create_enhanced_dataset


class CalibrationAnalyzer:
    """Analyze calibration and uncertainty quality"""
    
    def __init__(self, model, test_loader, device, num_samples: int = 30):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.num_samples = num_samples
    
    def get_mc_predictions(self, batch, num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get MC predictions and uncertainty"""
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
        
        mc_preds = torch.stack(mc_preds, dim=0)
        pred_mean = mc_preds.mean(dim=0)
        pred_std = mc_preds.std(dim=0)
        
        return pred_mean.cpu().numpy().flatten(), pred_std.cpu().numpy().flatten(), batch.y.cpu().numpy().flatten()
    
    def compute_calibration_metrics(self) -> Dict:
        """Compute calibration metrics"""
        print("\n[CALIBRATION] Computing calibration metrics...")
        
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        all_errors = []
        
        pbar = tqdm(self.test_loader, desc='Calibration Analysis', disable=False)
        
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(self.device)
                
                pred_mean, pred_std, targets = self.get_mc_predictions(batch, self.num_samples)
                
                all_predictions.append(pred_mean)
                all_uncertainties.append(pred_std)
                all_targets.append(targets)
                all_errors.append(np.abs(pred_mean - targets))
        
        predictions = np.concatenate(all_predictions)
        uncertainties = np.concatenate(all_uncertainties)
        targets = np.concatenate(all_targets)
        errors = np.concatenate(all_errors)
        
        # Calibration metrics
        metrics = {}
        
        # 1. Coverage Probabilities at different confidence levels
        confidence_levels = [0.68, 0.90, 0.95, 0.99]
        z_values = {0.68: 1.0, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        
        coverage_dict = {}
        for conf_level in confidence_levels:
            z = z_values[conf_level]
            lower_bound = predictions - z * uncertainties
            upper_bound = predictions + z * uncertainties
            
            coverage = np.sum((targets >= lower_bound) & (targets <= upper_bound)) / len(targets)
            coverage_dict[f'{int(conf_level*100)}%'] = float(coverage)
        
        metrics['coverage_probabilities'] = coverage_dict
        
        # 2. Mean Prediction Interval Width (MPIW)
        z_95 = 1.96
        interval_widths = 2 * z_95 * uncertainties
        metrics['mean_prediction_interval_width'] = float(np.mean(interval_widths))
        metrics['std_prediction_interval_width'] = float(np.std(interval_widths))
        
        # 3. Sharpness metrics
        metrics['mean_uncertainty'] = float(np.mean(uncertainties))
        metrics['std_uncertainty'] = float(np.std(uncertainties))
        metrics['min_uncertainty'] = float(np.min(uncertainties))
        metrics['max_uncertainty'] = float(np.max(uncertainties))
        
        # 4. Reliability metrics
        metrics['mean_absolute_error'] = float(np.mean(errors))
        metrics['rmse'] = float(np.sqrt(np.mean(errors**2)))
        
        # 5. Correlation between uncertainty and error
        metrics['uncertainty_error_correlation'] = float(np.corrcoef(uncertainties, errors)[0, 1])
        
        # 6. Spearman rank correlation
        from scipy.stats import spearmanr
        metrics['spearman_rank_correlation'], _ = spearmanr(uncertainties, errors)
        metrics['spearman_rank_correlation'] = float(metrics['spearman_rank_correlation'])
        
        return metrics, predictions, uncertainties, targets, errors
    
    def compute_calibration_curve(self, errors: np.ndarray, uncertainties: np.ndarray, 
                                 num_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Compute empirical calibration curve"""
        # Bin by uncertainty
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(uncertainties, percentiles)
        
        bin_means = []
        error_means = []
        
        for i in range(len(bins) - 1):
            mask = (uncertainties >= bins[i]) & (uncertainties < bins[i + 1])
            if np.sum(mask) > 0:
                bin_means.append(np.mean(uncertainties[mask]))
                error_means.append(np.mean(errors[mask]))
        
        return np.array(bin_means), np.array(error_means)


def create_calibration_plots(metrics: Dict, predictions: np.ndarray, uncertainties: np.ndarray,
                           targets: np.ndarray, errors: np.ndarray):
    """Create calibration visualizations"""
    print("\n[PLOTS] Creating calibration visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Uncertainty Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(uncertainties, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Prediction Uncertainty (Std Dev)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Uncertainties')
    ax1.grid(True, alpha=0.3)
    
    # 2. Error Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(errors, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Absolute Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Errors')
    ax2.grid(True, alpha=0.3)
    
    # 3. Uncertainty vs Error Scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(uncertainties, errors, alpha=0.3, s=10, color='#F18F01')
    z = np.polyfit(uncertainties, errors, 1)
    p = np.poly1d(z)
    ax3.plot(uncertainties, p(uncertainties), "r--", linewidth=2, label=f'Trend line')
    ax3.set_xlabel('Prediction Uncertainty')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Uncertainty vs Error (Pearson: {:.3f})'.format(
        metrics['uncertainty_error_correlation']))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Calibration Curve
    ax4 = fig.add_subplot(gs[1, 0])
    analyzer = CalibrationAnalyzer(None, None, None)
    bin_unc, bin_err = analyzer.compute_calibration_curve(errors, uncertainties, num_bins=15)
    ax4.plot(bin_unc, bin_err, 'o-', linewidth=2, markersize=8, color='#C73E1D')
    # Perfect calibration line
    ax4.plot([0, np.max(uncertainties)], [0, np.max(uncertainties)], 'k--', label='Perfect Calibration')
    ax4.set_xlabel('Mean Uncertainty')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Calibration Curve')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Predictions vs Targets
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(targets, predictions, alpha=0.3, s=10, color='#06A77D')
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    ax5.set_xlabel('Target Values')
    ax5.set_ylabel('Predicted Values')
    ax5.set_title('Predictions vs Targets')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Prediction Intervals
    ax6 = fig.add_subplot(gs[1, 2])
    z_95 = 1.96
    indices = np.random.choice(len(targets), size=min(200, len(targets)), replace=False)
    indices = np.sort(indices)
    
    lower = predictions[indices] - z_95 * uncertainties[indices]
    upper = predictions[indices] + z_95 * uncertainties[indices]
    
    ax6.scatter(indices, targets[indices], color='red', s=20, label='Targets', zorder=3)
    ax6.plot(indices, predictions[indices], 'b-', linewidth=1, label='Predictions', zorder=2)
    ax6.fill_between(indices, lower, upper, alpha=0.3, color='blue', label='95% PI')
    ax6.set_xlabel('Sample Index')
    ax6.set_ylabel('Value')
    ax6.set_title('Prediction Intervals (95% CI)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Coverage vs Confidence Levels
    ax7 = fig.add_subplot(gs[2, 0])
    conf_levels = [float(k.rstrip('%')) for k in metrics['coverage_probabilities'].keys()]
    coverages = [metrics['coverage_probabilities'][k] for k in sorted(metrics['coverage_probabilities'].keys())]
    conf_levels_sorted = sorted(conf_levels)
    
    ax7.plot(conf_levels_sorted, [c/100 for c in conf_levels_sorted], 'k--', linewidth=2, label='Perfect Calibration')
    ax7.plot(conf_levels_sorted, coverages, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Model')
    ax7.set_xlabel('Expected Coverage (%)')
    ax7.set_ylabel('Actual Coverage')
    ax7.set_title('Calibration Curve: Expected vs Actual Coverage')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(60, 100)
    ax7.set_ylim(0.6, 1.05)
    
    # 8. Residuals Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    residuals = predictions - targets
    ax8.hist(residuals, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax8.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax8.set_xlabel('Prediction Residual')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Distribution of Residuals')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Q-Q Plot
    ax9 = fig.add_subplot(gs[2, 2])
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=ax9)
    ax9.set_title('Q-Q Plot')
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Calibration and Uncertainty Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    plot_file = 'results/calibration_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Plot: {plot_file}")
    plt.close()


def main():
    """Main calibration analysis"""
    print("\n" + "="*120)
    print("[CALIBRATION] Calibration Curves and Sharpness Analysis")
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
    
    # Run calibration analysis
    analyzer = CalibrationAnalyzer(model, test_loader, device, num_samples=30)
    metrics, predictions, uncertainties, targets, errors = analyzer.compute_calibration_metrics()
    
    # Print results
    print("\n" + "="*120)
    print("[CALIBRATION METRICS]")
    print("="*120)
    
    print("\n[Coverage Probabilities]")
    for conf_level, coverage in sorted(metrics['coverage_probabilities'].items()):
        print(f"  {conf_level:6s}: {coverage:.4f} (Expected: {int(conf_level.rstrip('%'))/100:.4f})")
    
    print("\n[Sharpness Metrics]")
    print(f"  Mean Uncertainty:             {metrics['mean_uncertainty']:.6f}")
    print(f"  Std Uncertainty:              {metrics['std_uncertainty']:.6f}")
    print(f"  Mean Prediction Interval Width: {metrics['mean_prediction_interval_width']:.6f}")
    
    print("\n[Reliability Metrics]")
    print(f"  Mean Absolute Error:          {metrics['mean_absolute_error']:.6f}")
    print(f"  RMSE:                         {metrics['rmse']:.6f}")
    print(f"  Uncertainty-Error Correlation: {metrics['uncertainty_error_correlation']:.6f}")
    print(f"  Spearman Rank Correlation:    {metrics['spearman_rank_correlation']:.6f}")
    
    # Create plots
    create_calibration_plots(metrics, predictions, uncertainties, targets, errors)
    
    # Save results
    json_file = 'results/calibration_metrics.json'
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[SAVED] JSON file: {json_file}")
    
    # Create summary CSV
    summary_data = []
    for conf_level, coverage in metrics['coverage_probabilities'].items():
        summary_data.append({
            'Metric': f'Coverage_{conf_level}',
            'Value': coverage
        })
    
    summary_data.extend([
        {'Metric': 'Mean_Uncertainty', 'Value': metrics['mean_uncertainty']},
        {'Metric': 'MPIW', 'Value': metrics['mean_prediction_interval_width']},
        {'Metric': 'MAE', 'Value': metrics['mean_absolute_error']},
        {'Metric': 'RMSE', 'Value': metrics['rmse']},
        {'Metric': 'Uncertainty_Error_Correlation', 'Value': metrics['uncertainty_error_correlation']},
    ])
    
    summary_df = pd.DataFrame(summary_data)
    csv_file = 'results/calibration_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"[SAVED] CSV file: {csv_file}")
    
    print("="*120 + "\n")
    
    return metrics


if __name__ == "__main__":
    main()
