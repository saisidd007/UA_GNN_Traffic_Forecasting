"""
Create MAE Trend Graph for METR-LA 50-Epoch Training (like the provided example)
"""
import json
import numpy as np
import matplotlib.pyplot as plt

print("[PLOTTING] Creating MAE Trend Graph for METR-LA...")

# Load results
import glob, os
json_files = sorted(glob.glob("results/enhanced_training_results_*.json"), key=os.path.getmtime, reverse=True)
results_file = None
for jf in json_files:
    try:
        with open(jf, 'r') as f:
            data = json.load(f)
            if 'history' in data and 'train_metrics' in data['history']:
                results_file = jf
                results = data
                break
    except Exception:
        pass
if not results_file:
    raise FileNotFoundError("Could not find any training results JSON with 'history' in results/")
print(f"Loaded results from: {results_file}")

# Extract MAE values from metrics
train_metrics = results['history']['train_metrics']
val_metrics = results['history']['val_metrics']

train_mae = [m['mae'] for m in train_metrics]
val_mae = [m['mae'] for m in val_metrics]
train_rmse = [m['rmse'] for m in train_metrics]
val_rmse = [m['rmse'] for m in val_metrics]
train_r2 = [m['r2_score'] for m in train_metrics]
val_r2 = [m['r2_score'] for m in val_metrics]

epochs = range(1, len(train_mae) + 1)

print(f"Loaded {len(train_mae)} epochs of MAE data")
print(f"Train MAE range: {min(train_mae):.6f} to {max(train_mae):.6f}")
print(f"Val MAE range: {min(val_mae):.6f} to {max(val_mae):.6f}")

# Create figure matching the style
fig, ax = plt.subplots(figsize=(12, 6))

# Plot lines
ax.plot(epochs, train_mae, 'b-', linewidth=2.5, label='Train MAE', alpha=0.9)
ax.plot(epochs, val_mae, color='#FF8C00', linewidth=2.5, label='Val MAE', alpha=0.9)

# Styling
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax.set_title('MAE trend (50-epoch run)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Set x-axis to show all epochs
ax.set_xlim(0, max(epochs))

plt.tight_layout()
plt.savefig('results/metrla_mae_trend_50epoch.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: results/metrla_mae_trend_50epoch.png")
plt.close()

# Also create RMSE trend

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epochs, train_rmse, 'b-', linewidth=2.5, label='Train RMSE', alpha=0.9)
ax.plot(epochs, val_rmse, color='#FF8C00', linewidth=2.5, label='Val RMSE', alpha=0.9)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax.set_title('RMSE trend (50-epoch run)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(epochs))

plt.tight_layout()
plt.savefig('results/metrla_rmse_trend_50epoch.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: results/metrla_rmse_trend_50epoch.png")
plt.close()

# Create R² trend

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epochs, train_r2, 'b-', linewidth=2.5, label='Train R²', alpha=0.9)
ax.plot(epochs, val_r2, color='#FF8C00', linewidth=2.5, label='Val R²', alpha=0.9)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('R² trend (50-epoch run)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max(epochs))

plt.tight_layout()
plt.savefig('results/metrla_r2_trend_50epoch.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: results/metrla_r2_trend_50epoch.png")
plt.close()

print("\n[COMPLETE] All trend graphs created!")
print("\nGenerated Files:")
print("  ✓ results/metrla_mae_trend_50epoch.png")
print("  ✓ results/metrla_rmse_trend_50epoch.png")
print("  ✓ results/metrla_r2_trend_50epoch.png")
