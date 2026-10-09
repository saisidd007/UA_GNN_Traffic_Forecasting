"""
Generate synthetic sensor dropout results for testing
"""
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS = Path('results')
RESULTS.mkdir(exist_ok=True)

# Synthetic dropout experiment results
dropout_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
results_list = []

for dropout_rate in dropout_rates:
    # Model degrades gracefully with dropout
    degradation_factor = 1 + (dropout_rate * 1.5)
    
    mae = 0.4392 * degradation_factor
    rmse = 1.0327 * degradation_factor
    mape = 102.86 * (1 + dropout_rate * 0.5)
    
    # Coverage increases slightly (uncertainty estimates increase)
    picp95 = 0.9121 + (dropout_rate * 0.02)
    piw95 = 3.5212 * (1 + dropout_rate * 0.4)
    
    results_list.append({
        'dropout_rate_pct': f"{dropout_rate*100:.0f}%",
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'mape_pct': round(mape, 2),
        'picp95_coverage': round(picp95, 4),
        'piw95_mean': round(piw95, 4),
        'piw95_median': round(piw95 * 0.98, 4)
    })

df = pd.DataFrame(results_list)
df.to_csv(RESULTS / 'sensor_dropout_robustness.csv', index=False)

print("✅ Created synthetic sensor dropout results")
print(df.to_string(index=False))
print(f"\nNote: This is SYNTHETIC DATA for demonstration.")
print(f"For real results, run: python scripts/sensor_dropout_experiment.py (in training env)")
