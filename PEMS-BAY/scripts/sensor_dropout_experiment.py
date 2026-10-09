"""
Sensor Dropout Robustness Experiment
Tests model behavior when random sensors have missing/noisy data
Usage: python scripts/sensor_dropout_experiment.py
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
import sys
import os

# Setup paths
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

def run_dropout_experiment(dropout_rates=[0.05, 0.10, 0.20, 0.30, 0.50], mc_samples=30):
    """
    Load checkpoint, run inference on test set with varying sensor dropout levels.
    Record MAE, RMSE, MAPE, coverage95 (PICP), PIW95 for each dropout level.
    """
    
    # Load checkpoint
    checkpoint_path = RESULTS / 'enhanced_best_model.pt'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return None
    
    print(f"Device: {DEVICE}")
    model = create_improved_model(in_channels=12, out_channels=12).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("✓ Checkpoint loaded")
    
    # Load test data
    try:
        dataset = create_enhanced_dataset(root_dir=str(DATA), sequence_length=12, prediction_length=12)
        test_data = dataset.get_test_data()
        print(f"✓ Test set loaded ({len(test_data)} samples)")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return None
    
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=0)
    
    results_list = []
    z95 = 1.96
    
    for dropout_rate in dropout_rates:
        print(f"\n{'='*60}")
        print(f"Sensor Dropout: {dropout_rate*100:.0f}%")
        print(f"{'='*60}")
        
        # Run inference with dropout applied to input
        all_preds = []
        all_targets = []
        all_coverage = []
        all_piw = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = batch.to(DEVICE)
                
                # Apply random sensor dropout to input features
                # batch.x shape: [B*N, T, 1]
                dropout_mask = np.random.binomial(1, 1-dropout_rate, size=batch.x.shape)
                dropout_mask = torch.FloatTensor(dropout_mask).to(DEVICE)
                x_dropped = batch.x * dropout_mask
                
                # Run MC inference
                mc_preds = []
                mc_aleas = []
                for _ in range(mc_samples):
                    model.train()  # Enable dropout layers
                    pred, alea, _ = model(x_dropped, batch.edge_index, 
                                         batch.missing_mask, return_uncertainty=True)
                    mc_preds.append(pred.unsqueeze(0))
                    mc_aleas.append(alea.unsqueeze(0))
                
                mc_preds = torch.cat(mc_preds, dim=0)  # [K, B*N, H]
                mc_aleas = torch.cat(mc_aleas, dim=0)
                
                mean_pred = mc_preds.mean(dim=0).cpu().numpy()  # [B*N, H]
                epi_var = mc_preds.var(dim=0).cpu().numpy()
                alea_var = mc_aleas.mean(dim=0).cpu().numpy()
                total_var = epi_var + alea_var
                
                targets = batch.y.cpu().numpy()  # [B*N, H]
                
                # Flatten across batch and nodes
                pred_flat = mean_pred.flatten()
                tgt_flat = targets.flatten()
                var_flat = total_var.flatten()
                
                # Compute coverage and PIW
                ci_half = z95 * np.sqrt(var_flat + 1e-8)
                covered = (tgt_flat >= (pred_flat - ci_half)) & (tgt_flat <= (pred_flat + ci_half))
                piw = 2.0 * ci_half
                
                all_preds.append(pred_flat)
                all_targets.append(tgt_flat)
                all_coverage.append(covered)
                all_piw.append(piw)
                
                if (batch_idx + 1) % max(1, len(test_loader)//4) == 0 or batch_idx == 0:
                    print(f"  Processed batch {batch_idx+1}/{len(test_loader)}")
        
        # Aggregate metrics
        preds_concat = np.concatenate(all_preds, axis=0)
        tgts_concat = np.concatenate(all_targets, axis=0)
        cov_concat = np.concatenate(all_coverage, axis=0)
        piw_concat = np.concatenate(all_piw, axis=0)
        
        mae = float(np.mean(np.abs(preds_concat - tgts_concat)))
        rmse = float(np.sqrt(np.mean((preds_concat - tgts_concat)**2)))
        
        # MAPE with protection against div by zero
        mape_vals = np.abs((preds_concat - tgts_concat) / (np.abs(tgts_concat) + 1e-8))
        mape = float(np.mean(mape_vals[~np.isinf(mape_vals)]) * 100)
        
        coverage95 = float(np.mean(cov_concat))
        piw95_mean = float(np.mean(piw_concat))
        piw95_median = float(np.median(piw_concat))
        
        results_list.append({
            'dropout_rate_pct': f"{dropout_rate*100:.0f}%",
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'mape_pct': round(mape, 2),
            'picp95_coverage': round(coverage95, 4),
            'piw95_mean': round(piw95_mean, 4),
            'piw95_median': round(piw95_median, 4)
        })
        
        print(f"  MAE:       {mae:.4f}")
        print(f"  RMSE:      {rmse:.4f}")
        print(f"  MAPE:      {mape:.2f}%")
        print(f"  Coverage95 (PICP): {coverage95:.4f}")
        print(f"  PIW95 Mean: {piw95_mean:.4f}")
        print(f"  PIW95 Median: {piw95_median:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results_list)
    output_path = RESULTS / 'sensor_dropout_robustness.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Results saved to {output_path}")
    print(f"{'='*60}\n")
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Sensor Dropout Robustness Experiment")
    print("="*60 + "\n")
    
    try:
        # Allow overriding MC samples via environment variable
        mc_env = os.environ.get('MC_SAMPLES')
        mc_val = int(mc_env) if mc_env and mc_env.isdigit() else 30
        results_df = run_dropout_experiment(
            dropout_rates=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
            mc_samples=mc_val
        )
        if results_df is not None:
            print("\n✓ Experiment completed successfully!")
        else:
            print("\n❌ Experiment failed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
