"""
Generate synthetic per-horizon metrics for testing visualization scripts
This allows testing calibration_analysis.py without running full inference
"""
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS = Path('results')
RESULTS.mkdir(exist_ok=True)

# Create realistic synthetic per-horizon metrics based on expected values
horizons = np.arange(1, 13)  # 12 horizons
n_horizons = len(horizons)

# Synthetic data matching expected patterns
mae = 0.32 + (horizons - 1) * 0.018  # Increases with horizon
rmse = 0.78 + (horizons - 1) * 0.032
mape = 102.9 + (horizons - 1) * 1.5
coverage95 = 0.9167 - (horizons - 1) * 0.00065  # Slight decrease
piw95_mean = 3.21 + (horizons - 1) * 0.105
piw95_median = piw95_mean * 0.98
piw95_q25 = piw95_mean * 0.90
piw95_q75 = piw95_mean * 1.10

# Create DataFrame
df = pd.DataFrame({
    'horizon': horizons,
    'mae': mae,
    'rmse': rmse,
    'mape': mape,
    'coverage95': coverage95,
    'piw95_mean': piw95_mean,
    'piw95_median': piw95_median,
    'piw95_q25': piw95_q25,
    'piw95_q75': piw95_q75
})

# Save
output_path = RESULTS / 'analysis_50epoch_per_horizon_metrics.csv'
df.to_csv(output_path, index=False)

print(f"✅ Created synthetic per-horizon metrics at: {output_path}")
print(f"\nSynthetic Data Summary:")
print(df.to_string(index=False))
print(f"\nNote: This is SYNTHETIC DATA for testing visualization scripts.")
print(f"For real results, run: python notebooks/analysis_50epoch.py (in training env)")
