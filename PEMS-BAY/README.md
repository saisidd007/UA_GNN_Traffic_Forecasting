# PEMS-BAY Traffic Forecasting Pipeline (UA-GNN)

This directory contains the self-contained implementation, datasets, notebooks, training scripts, checkpoints, and evaluation results for the **PEMS-BAY** benchmark (325 sensors).

---

## 🗂 Directory Structure

```
PEMS-BAY/
├── configs/
│   └── pems_bay.yaml                                  # Dataset and model configurations
├── data/
│   └── pems-bay/
│       ├── PEMS-BAY.csv                               # Traffic speed time-series data (325 sensors)
│       ├── PEMS-BAY-META.csv                          # Sensor metadata & locations
│       └── adj_mx_bay.pkl                             # Distance-based adjacency matrix
├── src/
│   ├── models/
│   │   ├── enhanced_gnn.py                            # Enhanced UA-GNN architecture
│   │   └── gnn.py                                     # Baseline GNN
│   └── utils/
│       ├── dataset.py                                 # Standard PyG traffic dataset loader
│       └── enhanced_dataset.py                        # Memory-optimized lazy loader for 325 sensors
├── pems-bay/ & pems_bay1/                             # Additional TSSP and NewTain experimental variants
├── notebooks/
│   ├── data_analysis.ipynb                            # Sensor analysis & exploratory data analysis
│   ├── data_analysis_50epoch.ipynb                    # 50-epoch training performance visualization
│   └── run_data_plots.py                              # Automated script for exploratory plots
├── scripts/
│   ├── benchmark_inference.py                         # Latency and memory benchmarking
│   ├── sensor_dropout_robustness.py                   # Sensor failure / dropout robustness experiments
│   ├── calibration_analysis.py                        # Reliability diagrams & ECE computation
│   └── eval_per_horizon.py                            # Multi-step ahead forecasting evaluation (15, 30, 45, 60 min)
├── train_50.py                                        # Memory-optimized 50-epoch training script on PEMS-BAY
├── eval_pemsbay_results.py                            # Comprehensive evaluation script on PEMS-BAY
├── generate_final_results_pdf.py                      # ReportLab script generating publication PDF report
├── results/
│   ├── pems-bay_results.pdf                           # Final publication PDF report
│   ├── final_results_pemsbay.pdf                      # Source generated 6-page evaluation PDF
│   ├── eval/
│   │   ├── results_summary.txt                        # Formatted evaluation summary
│   │   ├── results_summary.json                       # JSON metrics
│   │   └── fig_pemsbay_reliability_diagram.pdf / .png # Reliability diagram
│   ├── figures/                                       # Publication figures (Loss curves, sensor dropouts, 24h pattern)
│   ├── checkpoints/                                   # Saved model weights (enhanced_best_model.pt)
│   └── sensor_dropout_robustness.json                 # Dropout robustness results
```

---

## 📈 Verification of Results (`pems-bay_results.pdf`)

* **Normalized Metrics:**
  * MAE: `0.439200`
  * RMSE: `1.032700`
  * $R^2$ Score: `0.838500`
  * Pearson Correlation: `0.915800`
* **Physical Speed Metrics (mph):**
  * Overall MAE: `1.6700 mph`
  * Overall RMSE: `3.7200 mph`
  * Overall MAPE: `3.85 %`
* **Horizon-Specific Breakdown (Physical mph):**
  * 15 min (Step 3): MAE = `1.3000`, RMSE = `2.7400`
  * 30 min (Step 6): MAE = `1.6400`, RMSE = `3.7000`
  * 45 min (Step 9): MAE = `1.8800`, RMSE = `4.2100`
  * 60 min (Step 12): MAE = `2.0700`, RMSE = `4.7400`
* **Calibration:**
  * Expected Calibration Error (ECE): `0.0869`
  * Mean Aleatoric Uncertainty: `0.550800`
  * Mean Epistemic Uncertainty: `0.256000`
  * Mean Total Uncertainty: `0.806800`
