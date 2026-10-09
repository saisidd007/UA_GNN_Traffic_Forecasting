"""
Generate dataset-dependent figures for METR-LA (does NOT run model inference).
Saves figures to results/.
Run:
python notebooks/run_data_plots.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer, KNNImputer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

metr_csv = DATA / 'metr-la' / 'METR-LA.csv'
if not metr_csv.exists():
    metr_csv = DATA / 'METR-LA.csv'

meta_csv = DATA / 'metr-la' / 'METR-LA-META.csv'
if not meta_csv.exists():
    meta_csv = DATA / 'METR-LA-META.csv'

print('Loading dataset...')
# Read with low_memory=False
df = pd.read_csv(metr_csv, low_memory=False)
print('Raw columns sample:', list(df.columns[:10]))
# Detect index column
first_col = str(df.columns[0])
if first_col.lower().startswith('unnamed') or 'time' in first_col.lower() or 'date' in first_col.lower():
    df = df.set_index(df.columns[0])

# normalize columns
orig_cols = df.columns.astype(str).tolist()

meta_df = pd.read_csv(meta_csv)
meta_df['sensor_id'] = meta_df['sensor_id'].astype(str)

# Remap columns if counts match
if set(meta_df['sensor_id']).issubset(set(orig_cols)):
    print('Columns already contain sensor IDs; no remap')
else:
    if len(orig_cols) == len(meta_df):
        print('Remapping data columns by position to metadata sensor IDs')
        df.columns = meta_df['sensor_id'].tolist()
    else:
        print('Column count mismatch; leaving original columns')

# Convert to numeric
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 1) Sample sensor time-series (first 3 sensors from meta)
sample_sensors = meta_df['sensor_id'].astype(str).tolist()[:3]
plt.figure(figsize=(12,4))
for sid in sample_sensors:
    if sid in df.columns:
        plt.plot(df[sid].iloc[:500].values, label=f'Sensor {sid}')
plt.title('Traffic Speed Patterns (first 500 timesteps)')
plt.xlabel('Timestep')
plt.ylabel('Speed')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / 'fig_sample_sensors_timeseries.png', dpi=200)
plt.close()
print('Saved fig_sample_sensors_timeseries.png')

# 2) Distribution for first sensor (handle possible index column)
first_sensor_col = None
for col in df.columns:
    if col in meta_df['sensor_id'].astype(str).tolist():
        first_sensor_col = col
        break
if first_sensor_col is None:
    first_sensor_col = df.columns[0]

speeds = df[first_sensor_col].dropna().values
plt.figure(figsize=(8,4))
plt.hist(speeds, bins=50, density=True)
plt.title(f'Distribution of Traffic Speeds (Sensor {first_sensor_col})')
plt.xlabel('Speed')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / f'fig_speed_dist_{first_sensor_col}.png', dpi=200)
plt.close()
print('Saved speed distribution')

# 3) Correlation heatmap (first 10 sensors)
cols10 = df.columns[:10]
corr10 = df[cols10].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr10, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation between First 10 Sensors')
plt.tight_layout()
plt.savefig(RESULTS / 'fig_corr_first10.png', dpi=200)
plt.close()
print('Saved fig_corr_first10.png')

# 4) 24-hour daily pattern (mean + std across sensors)
if df.shape[0] >= 288:
    daily_mean = df.iloc[:288, :].mean(axis=1)
    daily_std = df.iloc[:288, :].std(axis=1)
    plt.figure(figsize=(10,4))
    x = np.arange(len(daily_mean))
    plt.plot(x, daily_mean, color='blue', label='Mean Speed')
    plt.fill_between(x, daily_mean - daily_std, daily_mean + daily_std, alpha=0.2)
    plt.title('24-Hour Traffic Pattern with Variability')
    # Show xticks every 2 hours. With 5-minute intervals there are 12 timesteps per hour,
    # so 2 hours == 24 timesteps. Create ticks at 0,24,48,... and include final 24:00 label.
    tick_step = 24
    # include the final endpoint (len(daily_mean)) so we can label 24:00 at the right edge
    xticks = np.arange(0, len(daily_mean) + 1, tick_step)
    # Convert ticks to hour labels (tick/12 = hours). For the final tick this yields 24.
    xtick_labels = [f"{int(tick/12):02d}:00" for tick in xticks]
    plt.xticks(xticks, xtick_labels)
    # ensure the axis shows the full 24h range including the rightmost tick
    plt.xlim(0, len(daily_mean))
    plt.xlabel('Time of day')
    plt.ylabel('Speed')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = RESULTS / 'fig_24h_pattern.png'
    eps_path = RESULTS / 'fig_24h_pattern.eps'
    plt.savefig(png_path, dpi=200)
    # Also save a vector EPS version for publication-quality output
    try:
        plt.savefig(eps_path, format='eps')
        print(f'Saved {png_path.name} and {eps_path.name}')
    except Exception as e:
        # If EPS save fails (rare on some backends), still keep the PNG and report the error
        print(f'Saved {png_path.name} but failed to save EPS: {e}')
    plt.close()
else:
    print('Not enough rows for 24-hour pattern (need >=288)')

# 5) Rush hour averages for first 5 sensors
if df.shape[0] >= 288:
    morning_slice = slice(84,108)
    evening_slice = slice(192,216)
    morning_avg = df.iloc[morning_slice, :5].mean()
    evening_avg = df.iloc[evening_slice, :5].mean()
    plt.figure(figsize=(8,4))
    x = range(5)
    plt.bar([f'Sensor {i}' for i in x], morning_avg, alpha=0.6, label='Morning (7-9AM)')
    plt.bar([f'Sensor {i}' for i in x], evening_avg, alpha=0.6, label='Evening (4-6PM)')
    plt.title('Average Speeds During Rush Hours (first 5 sensors)')
    plt.ylabel('Speed')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / 'fig_rush_hour_bar.png', dpi=200)
    plt.close()
    print('Saved fig_rush_hour_bar.png')

# 6) Congestion detection example (first week, first 3 sensors)
if df.shape[0] >= 2016:
    week = df.iloc[:2016, :10]
    def detect_congestion(speeds, threshold=0.5):
        free_flow = np.percentile(speeds.dropna(), 85)
        return speeds < (free_flow * threshold)
    plt.figure(figsize=(12,5))
    for i in range(3):
        speeds_i = week.iloc[:, i]
        cong = detect_congestion(speeds_i)
        plt.plot(speeds_i.values, label=f'Sensor {week.columns[i]}', alpha=0.7)
        plt.scatter(np.where(cong)[0], speeds_i[cong].values, color='red', s=5, alpha=0.4)
    plt.title('Weekly Traffic with Congestion Events (first 3 sensors)')
    plt.tight_layout()
    plt.savefig(RESULTS / 'fig_week_congestion.png', dpi=200)
    plt.close()
    print('Saved fig_week_congestion.png')
else:
    print('Not enough rows for weekly congestion plot')

# 7) Missing data simulation and heatmap (first day, first 10 sensors)
subset = df.iloc[:288, 1:11] if df.shape[1] > 11 else df.iloc[:288, :10]
np.random.seed(42)
mask = np.random.random(subset.shape) < 0.2
subset_missing = subset.copy()
subset_missing[mask] = np.nan
plt.figure(figsize=(10,4))
sns.heatmap(subset_missing.isna(), cbar=False, cmap='binary')
plt.title('Missing Data Pattern (first day, 10 sensors)')
plt.tight_layout()
plt.savefig(RESULTS / 'fig_missing_pattern.png', dpi=200)
plt.close()
print('Saved fig_missing_pattern.png')

# 8) Imputation comparison for first sensor
sensor_col = subset.columns[0]
mean_imp = SimpleImputer(strategy='mean')
median_imp = SimpleImputer(strategy='median')
knn_imp = KNNImputer(n_neighbors=5)

orig = df.iloc[:288, :]
missing_mask = mask[:, 0]
mean_imputed = pd.DataFrame(mean_imp.fit_transform(subset_missing), columns=subset_missing.columns)
median_imputed = pd.DataFrame(median_imp.fit_transform(subset_missing), columns=subset_missing.columns)
knn_imputed = pd.DataFrame(knn_imp.fit_transform(subset_missing), columns=subset_missing.columns)

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(orig[sensor_col].values, label='Original')
plt.title(f'Imputation Comparison (Sensor {sensor_col})')
plt.subplot(3,1,2)
plt.plot(orig[sensor_col].values, alpha=0.5, label='Original')
plt.plot(mean_imputed[sensor_col].values, label='Mean Imputed')
plt.legend()
plt.subplot(3,1,3)
plt.plot(orig[sensor_col].values, alpha=0.5, label='Original')
plt.plot(knn_imputed[sensor_col].values, label='KNN Imputed')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / 'fig_imputation_compare.png', dpi=200)
plt.close()
print('Saved fig_imputation_compare.png')

# 9) Coefficient of variation across first 10 sensors over 24h
if df.shape[0] >= 288:
    rolling_cv = df.iloc[:288, :10].rolling(12).std() / df.iloc[:288, :10].rolling(12).mean()
    plt.figure(figsize=(10,4))
    for i in range(min(5, rolling_cv.shape[1])):
        plt.plot(rolling_cv.iloc[:, i].values, label=f'Sensor {rolling_cv.columns[i]}')
    plt.title('Rolling Coefficient of Variation (first 5 sensors)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / 'fig_rolling_cv.png', dpi=200)
    plt.close()
    print('Saved fig_rolling_cv.png')

print('\nAll dataset-dependent plots generated (no model inference).')
