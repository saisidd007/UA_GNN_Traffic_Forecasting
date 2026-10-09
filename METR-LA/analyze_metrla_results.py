"""
Fast METR-LA Analysis: Training Curves & Simulated Sensor Dropout
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader
from src.utils.enhanced_dataset import create_enhanced_dataset
from src.models.enhanced_gnn import EnhancedTrafficGNN

print("[ANALYSIS] Starting METR-LA Results Analysis...")

# ============================================================================
# 1. LOAD AND PLOT TRAINING/VALIDATION CURVES
# ============================================================================
print("\n[STEP 1] Loading training history and plotting curves...")

results_file = "results/enhanced_training_results_metrla_20260502_125606.json"
with open(results_file, 'r') as f:
    results = json.load(f)

train_loss = results['history']['train_loss']
val_loss = results['history']['val_loss']
epochs_completed = len(train_loss)

# Create figure with better styling
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('METR-LA 50-Epoch Training Analysis (Early Stopped at Epoch 27)', 
             fontsize=16, fontweight='bold')

# Plot 1: Training vs Validation Loss
ax1 = axes[0, 0]
ax1.plot(range(1, epochs_completed + 1), train_loss, 'b-o', linewidth=2, 
         markersize=4, label='Training Loss', alpha=0.8)
ax1.plot(range(1, epochs_completed + 1), val_loss, 'r-s', linewidth=2, 
         markersize=4, label='Validation Loss', alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=12, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Best Model (Epoch 12)')

# Plot 2: Training Loss (Zoomed)
ax2 = axes[0, 1]
ax2.plot(range(1, epochs_completed + 1), train_loss, 'b-o', linewidth=2, markersize=4)
ax2.fill_between(range(1, epochs_completed + 1), train_loss, alpha=0.2, color='blue')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
ax2.set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Validation Loss with Best Model Marker
ax3 = axes[1, 0]
ax3.plot(range(1, epochs_completed + 1), val_loss, 'r-s', linewidth=2, markersize=4)
ax3.fill_between(range(1, epochs_completed + 1), val_loss, alpha=0.2, color='red')
best_epoch = 12
best_val_loss = val_loss[best_epoch - 1]
ax3.scatter([best_epoch], [best_val_loss], color='green', s=300, marker='*', 
           zorder=5, label=f'Best (Epoch {best_epoch}, Loss={best_val_loss:.4f})')
ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax3.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
ax3.set_title('Validation Loss with Early Stopping', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Loss Improvement Rate
ax4 = axes[1, 1]
train_improvement = [0] + [train_loss[i-1] - train_loss[i] for i in range(1, len(train_loss))]
val_improvement = [0] + [val_loss[i-1] - val_loss[i] for i in range(1, len(val_loss))]
ax4.bar(np.arange(1, epochs_completed + 1) - 0.2, train_improvement, width=0.4, 
        label='Train Improvement', alpha=0.7, color='blue')
ax4.bar(np.arange(1, epochs_completed + 1) + 0.2, val_improvement, width=0.4, 
        label='Val Improvement', alpha=0.7, color='red')
ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Loss Reduction', fontsize=11, fontweight='bold')
ax4.set_title('Epoch-to-Epoch Loss Improvement', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('results/metrla_training_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: results/metrla_training_analysis.png")
plt.close()

print(f"\n[SUMMARY] Training Results:")
print(f"  Completed Epochs: {epochs_completed}/50")
print(f"  Best Epoch: 12 (Val Loss: {val_loss[11]:.6f})")
print(f"  Final Train Loss: {train_loss[-1]:.6f}")
print(f"  Final Val Loss: {val_loss[-1]:.6f}")
print(f"  Loss Reduction: {(train_loss[0] - train_loss[-1]) / train_loss[0] * 100:.2f}%")

# ============================================================================
# 2. SENSOR DROPOUT ROBUSTNESS EXPERIMENTS
# ============================================================================
print("\n[STEP 2] Running Sensor Dropout Experiments (0%, 5%, 10%, 20%, 30%)...")

device = torch.device('cpu')
dropout_rates = [0, 0.05, 0.10, 0.20, 0.30]
results_dropout = {
    'dropout_rates': dropout_rates,
    'metrics': {}
}

# Load best model
model = EnhancedTrafficGNN(
    in_channels=12,
    hidden_channels=64,
    out_channels=12,
    num_gnn_layers=4,
    num_temporal_layers=3,
    num_attention_heads=4
).to(device)

model_path = 'results/best_model_metrla.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(f"✓ Model loaded from {model_path}")

# Load METR-LA dataset
print("Loading METR-LA test dataset...")
dataset = create_enhanced_dataset('METR-LA', 'results')
test_loader = DataLoader(dataset['test'], batch_size=8, shuffle=False)

print(f"✓ Test set loaded: {len(dataset['test'])} sequences")

# Run experiments for each dropout rate
mae_results = []
rmse_results = []
r2_results = []
pearson_results = []

for dropout_rate in dropout_rates:
    print(f"\n  Testing Dropout Rate: {dropout_rate*100:.0f}%")
    
    all_preds = []
    all_targets = []
    all_aleatoric = []
    all_epistemic = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            # Apply sensor dropout to input
            if dropout_rate > 0:
                mask = torch.bernoulli(torch.ones_like(batch.x) * (1 - dropout_rate))
                batch.x = batch.x * mask
            
            # MC Dropout sampling (K=10)
            K = 10
            preds_list = []
            stds_list = []
            
            for k in range(K):
                model.train()  # Enable dropout
                with torch.enable_grad():
                    out = model(batch.x, batch.edge_index, batch.batch)
                model.eval()
                preds_list.append(out)
            
            preds_mean = torch.stack(preds_list).mean(dim=0)
            preds_std = torch.stack(preds_list).std(dim=0)
            
            all_preds.append(preds_mean.detach().cpu().numpy())
            all_targets.append(batch.y.detach().cpu().numpy())
            all_aleatoric.append(preds_std.detach().cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"    Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Compute metrics
    preds_all = np.concatenate(all_preds)
    targets_all = np.concatenate(all_targets)
    
    mae = np.mean(np.abs(preds_all - targets_all))
    rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2))
    
    # R² Score
    ss_res = np.sum((targets_all - preds_all) ** 2)
    ss_tot = np.sum((targets_all - targets_all.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Pearson Correlation
    flat_preds = preds_all.flatten()
    flat_targets = targets_all.flatten()
    pearson = np.corrcoef(flat_preds, flat_targets)[0, 1]
    
    mae_results.append(mae)
    rmse_results.append(rmse)
    r2_results.append(r2)
    pearson_results.append(pearson)
    
    results_dropout['metrics'][f'{dropout_rate*100:.0f}%'] = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'pearson': float(pearson)
    }
    
    print(f"    MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}, Pearson: {pearson:.6f}")

# Save dropout results
with open('results/metrla_sensor_dropout_results.json', 'w') as f:
    json.dump(results_dropout, f, indent=2)
print(f"\n✓ Saved: results/metrla_sensor_dropout_results.json")

# ============================================================================
# 3. PLOT SENSOR DROPOUT RESULTS
# ============================================================================
print("\n[STEP 3] Plotting Sensor Dropout Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('METR-LA Sensor Dropout Robustness Analysis (0%-30%)', 
             fontsize=16, fontweight='bold')

dropout_percentages = [d*100 for d in dropout_rates]

# Plot 1: MAE vs Dropout Rate
ax1 = axes[0, 0]
ax1.plot(dropout_percentages, mae_results, 'b-o', linewidth=2.5, markersize=8)
ax1.fill_between(dropout_percentages, mae_results, alpha=0.2, color='blue')
ax1.set_xlabel('Sensor Dropout Rate (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax1.set_title('Mean Absolute Error vs Dropout Rate', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
for i, val in enumerate(mae_results):
    ax1.annotate(f'{val:.4f}', (dropout_percentages[i], val), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# Plot 2: RMSE vs Dropout Rate
ax2 = axes[0, 1]
ax2.plot(dropout_percentages, rmse_results, 'g-s', linewidth=2.5, markersize=8)
ax2.fill_between(dropout_percentages, rmse_results, alpha=0.2, color='green')
ax2.set_xlabel('Sensor Dropout Rate (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax2.set_title('Root Mean Squared Error vs Dropout Rate', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
for i, val in enumerate(rmse_results):
    ax2.annotate(f'{val:.4f}', (dropout_percentages[i], val), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# Plot 3: R² Score vs Dropout Rate
ax3 = axes[1, 0]
ax3.plot(dropout_percentages, r2_results, 'r-^', linewidth=2.5, markersize=8)
ax3.fill_between(dropout_percentages, r2_results, alpha=0.2, color='red')
ax3.set_xlabel('Sensor Dropout Rate (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax3.set_title('R² Score vs Dropout Rate', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
for i, val in enumerate(r2_results):
    ax3.annotate(f'{val:.4f}', (dropout_percentages[i], val), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# Plot 4: Pearson Correlation vs Dropout Rate
ax4 = axes[1, 1]
ax4.plot(dropout_percentages, pearson_results, 'm-d', linewidth=2.5, markersize=8)
ax4.fill_between(dropout_percentages, pearson_results, alpha=0.2, color='magenta')
ax4.set_xlabel('Sensor Dropout Rate (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Pearson Correlation', fontsize=11, fontweight='bold')
ax4.set_title('Pearson Correlation vs Dropout Rate', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
for i, val in enumerate(pearson_results):
    ax4.annotate(f'{val:.4f}', (dropout_percentages[i], val), 
                textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/metrla_sensor_dropout_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: results/metrla_sensor_dropout_analysis.png")
plt.close()

# ============================================================================
# 4. COMBINED COMPARISON PLOT
# ============================================================================
print("\n[STEP 4] Creating Combined Comparison Plot...")

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(dropout_percentages))
width = 0.2

# Normalize metrics to 0-1 scale for comparison
mae_norm = np.array(mae_results) / max(mae_results)
rmse_norm = np.array(rmse_results) / max(rmse_results)
r2_norm = 1 - np.array(r2_results)  # Invert so lower is better
pearson_norm = 1 - np.array(pearson_results)  # Invert so lower is better

ax.bar(x - 1.5*width, mae_norm, width, label='MAE (normalized)', alpha=0.8, color='blue')
ax.bar(x - 0.5*width, rmse_norm, width, label='RMSE (normalized)', alpha=0.8, color='green')
ax.bar(x + 0.5*width, r2_norm, width, label='1-R² Score', alpha=0.8, color='red')
ax.bar(x + 1.5*width, pearson_norm, width, label='1-Pearson Corr', alpha=0.8, color='magenta')

ax.set_xlabel('Sensor Dropout Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Error / Loss', fontsize=12, fontweight='bold')
ax.set_title('METR-LA Robustness: All Metrics Comparison (Normalized)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{int(d)}%' for d in dropout_percentages])
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/metrla_dropout_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: results/metrla_dropout_comparison.png")
plt.close()

# ============================================================================
# 5. DETAILED RESULTS TABLE
# ============================================================================
print("\n" + "="*70)
print("SENSOR DROPOUT ROBUSTNESS RESULTS TABLE")
print("="*70)
print(f"{'Dropout %':<12} {'MAE':<12} {'RMSE':<12} {'R² Score':<12} {'Pearson':<12}")
print("-"*70)
for i, dropout in enumerate(dropout_percentages):
    print(f"{dropout:<12.0f} {mae_results[i]:<12.6f} {rmse_results[i]:<12.6f} {r2_results[i]:<12.6f} {pearson_results[i]:<12.6f}")
print("="*70)

# Calculate degradation rates
print("\nDegradation from 0% Dropout:")
print("-"*70)
for i in range(1, len(dropout_percentages)):
    mae_deg = ((mae_results[i] - mae_results[0]) / mae_results[0] * 100)
    rmse_deg = ((rmse_results[i] - rmse_results[0]) / rmse_results[0] * 100)
    r2_deg = ((r2_results[i] - r2_results[0]) / r2_results[0] * 100)
    pearson_deg = ((pearson_results[i] - pearson_results[0]) / pearson_results[0] * 100)
    
    print(f"\n{dropout_percentages[i]:.0f}% Dropout:")
    print(f"  MAE:      {mae_deg:+.2f}% (from {mae_results[0]:.6f} to {mae_results[i]:.6f})")
    print(f"  RMSE:     {rmse_deg:+.2f}% (from {rmse_results[0]:.6f} to {rmse_results[i]:.6f})")
    print(f"  R²:       {r2_deg:+.2f}% (from {r2_results[0]:.6f} to {r2_results[i]:.6f})")
    print(f"  Pearson:  {pearson_deg:+.2f}% (from {pearson_results[0]:.6f} to {pearson_results[i]:.6f})")

print("\n[COMPLETE] All analysis done!")
print("\nGenerated Files:")
print("  ✓ results/metrla_training_analysis.png")
print("  ✓ results/metrla_sensor_dropout_analysis.png")
print("  ✓ results/metrla_dropout_comparison.png")
print("  ✓ results/metrla_sensor_dropout_results.json")
