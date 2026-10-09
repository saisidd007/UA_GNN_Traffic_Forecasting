"""
Multiple Runs Analysis: Use 5 checkpoint epochs, compute metrics + 95% CI
Usage: python scripts/multiple_runs_analysis.py
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.enhanced_dataset import create_enhanced_dataset
from src.models.enhanced_gnn import create_improved_model

RESULTS = ROOT / 'results'
DATA = ROOT / 'data'
# Allow forcing CPU via environment variable (FORCE_CPU=1)
force_cpu = os.environ.get('FORCE_CPU', '')
if str(force_cpu).lower() in ('1', 'true', 'yes'):
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_single_checkpoint(checkpoint_path, mc_samples=30):
    """Load checkpoint and compute test metrics (MAE, RMSE, MAPE, R²)."""
    
    if not Path(checkpoint_path).exists():
        print(f"  ⚠ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        model = create_improved_model(in_channels=12, out_channels=12).to(DEVICE)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        
        # Load test data
        dataset = create_enhanced_dataset(root_dir=str(DATA), sequence_length=12, prediction_length=12)
        test_data = dataset.get_test_data()
        test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(DEVICE)
                
                # MC sampling for mean prediction
                mc_preds = []
                for _ in range(mc_samples):
                    model.train()
                    pred, _, _ = model(batch.x, batch.edge_index, 
                                      batch.missing_mask, return_uncertainty=True)
                    mc_preds.append(pred.unsqueeze(0))
                
                mean_pred = torch.cat(mc_preds, dim=0).mean(dim=0).cpu().numpy()
                all_preds.append(mean_pred.flatten())
                all_targets.append(batch.y.cpu().numpy().flatten())
        
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Compute metrics
        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(np.mean((preds - targets)**2)))
        
        # MAPE with protection
        mape_vals = np.abs((preds - targets) / (np.abs(targets) + 1e-8))
        mape = float(np.mean(mape_vals[~np.isinf(mape_vals)]) * 100)
        
        # R²
        ss_res = np.sum((preds - targets)**2)
        ss_tot = np.sum((targets - targets.mean())**2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
        
    except Exception as e:
        print(f"  ❌ Error inferencing {checkpoint_path}: {e}")
        return None

def run_multiple_runs_analysis():
    """
    Use 5 available checkpoints as "runs" for diversity.
    Compute metrics + 95% CI for each metric.
    """
    print("Starting multiple runs analysis...\n")
    
    checkpoint_epochs = [10, 20, 30, 40, 50]
    results_list = []
    
    # Allow overriding MC samples via environment variable
    mc_env = os.environ.get('MC_SAMPLES')
    mc_val = int(mc_env) if mc_env and mc_env.isdigit() else 30

    for epoch in checkpoint_epochs:
        ckpt_path = RESULTS / f'enhanced_checkpoint_epoch_{epoch}.pt'
        print(f"Processing epoch {epoch}...")
        
        metrics = infer_single_checkpoint(str(ckpt_path), mc_samples=mc_val)
        if metrics is not None:
            metrics['epoch'] = epoch
            results_list.append(metrics)
            print(f"  ✓ MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"MAPE={metrics['mape']:.2f}%, R²={metrics['r2']:.4f}")
    
    if len(results_list) == 0:
        print("\n❌ No checkpoints successfully processed")
        return None, None
    
    # Create DataFrame
    runs_df = pd.DataFrame(results_list)
    print(f"\n✓ Processed {len(runs_df)} checkpoints\n")
    
    # Compute statistics (mean ± 95% CI)
    stats_list = []
    for col in ['mae', 'rmse', 'mape', 'r2']:
        values = runs_df[col].values
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        n = len(values)
        
        # 95% CI using t-distribution
        from scipy import stats as sp_stats
        ci_95 = sp_stats.t.ppf(0.975, df=n-1) * std / np.sqrt(n) if n > 1 else 0
        
        lower_ci = mean - ci_95
        upper_ci = mean + ci_95
        
        stats_list.append({
            'Metric': col.upper(),
            'Mean': f"{mean:.4f}",
            'Std Dev': f"{std:.4f}",
            'Lower 95% CI': f"{lower_ci:.4f}",
            'Upper 95% CI': f"{upper_ci:.4f}",
            '95% CI Range': f"±{ci_95:.4f}",
            'Relative Uncertainty (%)': f"{(ci_95/mean)*100:.2f}%"
        })
    
    # Create summary table
    summary_df = pd.DataFrame(stats_list)
    
    # Save results
    runs_df.to_csv(RESULTS / 'multiple_runs_metrics.csv', index=False)
    summary_df.to_csv(RESULTS / 'multiple_runs_summary.csv', index=False)
    
    print('\n' + '='*100)
    print('Multiple Runs Statistical Summary (5 checkpoints, epochs 10-50)')
    print('='*100)
    print(summary_df.to_string(index=False))
    print('='*100)
    
    # Print detailed results
    print("\nDetailed Results by Checkpoint:")
    print(runs_df.to_string(index=False))
    
    print(f"\n✓ Results saved to:")
    print(f"  - results/multiple_runs_metrics.csv (detailed)")
    print(f"  - results/multiple_runs_summary.csv (summary with CI)")
    
    return runs_df, summary_df

def create_visualization(runs_df, summary_df):
    """Create visualization of metrics across runs."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['mae', 'rmse', 'mape', 'r2']
    titles = ['MAE (Mean Absolute Error)', 'RMSE (Root Mean Squared Error)', 
              'MAPE (Mean Absolute Percentage Error %)', 'R² Score']
    
    for idx, (ax, metric, title) in enumerate(zip(axes.flat, metrics, titles)):
        values = runs_df[metric].values
        epochs = runs_df['epoch'].values
        
        # Plot line with points
        ax.plot(epochs, values, 'o-', linewidth=2.5, markersize=10, color='C0', alpha=0.7)
        
        # Add mean line
        mean_val = float(summary_df[summary_df['Metric'] == metric.upper()]['Mean'].values[0])
        ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=2, 
                   alpha=0.5, label=f'Mean: {mean_val:.4f}')
        
        # Add CI band
        ci_str = summary_df[summary_df['Metric'] == metric.upper()]['95% CI Range'].values[0]
        ci_val = float(ci_str.split('±')[1])
        ax.fill_between(epochs, mean_val - ci_val, mean_val + ci_val, 
                        alpha=0.2, color='C0', label=f'95% CI: ±{ci_val:.4f}')
        
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)
        ax.legend(fontsize=10)
        ax.set_xticks(epochs)
    
    plt.tight_layout()
    plt.savefig(RESULTS / 'multiple_runs_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Visualization saved to results/multiple_runs_visualization.png")

if __name__ == '__main__':
    print("\n" + "="*100)
    print("MULTIPLE RUNS ANALYSIS: Statistical Robustness Evaluation")
    print("="*100 + "\n")
    
    try:
        runs_df, summary_df = run_multiple_runs_analysis()
        
        if runs_df is not None and summary_df is not None:
            try:
                create_visualization(runs_df, summary_df)
            except Exception as e:
                print(f"\n⚠ Visualization creation failed: {e}")
            
            print("\n" + "="*100)
            print("✓ Multiple runs analysis completed successfully!")
            print("="*100)
        else:
            print("\n❌ Analysis failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
