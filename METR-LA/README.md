# METR-LA Traffic Forecasting Pipeline (UA-GNN)

This directory contains the self-contained implementation, datasets, notebooks, training scripts, checkpoints, and evaluation results for the **METR-LA** benchmark (207 sensors).

---

## 🗂 Directory Structure

```
METR-LA/
├── configs/
│   └── metr_la.yaml                                   # Dataset and model configurations
├── data/
│   └── metr-la/
│       ├── METR-LA.csv                                # Traffic speed time-series data
│       └── METR-LA-META.csv                           # Sensor metadata & locations
├── src/
│   ├── models/
│   │   ├── enhanced_gnn.py                            # Enhanced UA-GNN architecture
│   │   └── gnn.py                                     # Baseline GNN
│   └── utils/
│       ├── dataset.py                                 # Standard PyG traffic dataset loader
│       └── enhanced_dataset.py                        # Enhanced dataset loader with temporal embeddings
├── notebooks/
│   ├── generate_metrla_paper_figures.ipynb            # Notebook to generate paper figures
│   ├── data_analysis.ipynb                            # Sensor analysis & exploratory data analysis
│   ├── data_analysis_50epoch.ipynb                    # 50-epoch training performance visualization
│   └── run_data_plots.py                              # Automated script for exploratory plots
├── scripts/
│   ├── benchmark_inference.py                         # Latency and memory benchmarking
│   ├── sensor_dropout_experiment.py                   # Sensor failure / dropout robustness experiments
│   ├── calibration_analysis.py                        # Reliability diagrams & ECE computation
│   └── per_horizon_metrics.py                         # Multi-step ahead forecasting evaluation (15, 30, 45, 60 min)
├── train_50_metrla.py                                 # Main 50-epoch training script on METR-LA
├── eval_metrla_results.py                             # Evaluation script computing normalized and physical speed metrics
├── results/
│   ├── metr-la_results.pdf                            # Final publication PDF report
│   ├── final_results_metrla.pdf                       # Source generated 6-page evaluation PDF
│   ├── eval/
│   │   ├── results_summary.txt                        # Formatted evaluation summary
│   │   └── reliability_diagram.pdf / .png             # Calibration diagrams
│   ├── figures/                                       # Publication figures (Loss curves, sensor dropouts, 24h pattern)
│   ├── checkpoints/                                   # Saved model weights (enhanced_best_model.pt)
│   └── metrla_ablation_results.json                   # Component ablation study results
```

---

## 📈 Verification of Results (`metr-la_results.pdf`)

* **Normalized Metrics:**
  * MAE: `0.609916`
  * RMSE: `1.317238`
  * $R^2$ Score: `0.661991`
  * Pearson Correlation: `0.813655`
* **Physical Speed Metrics (mph):**
  * Overall MAE: `3.4642 mph`
  * Overall RMSE: `6.5746 mph`
  * Overall MAPE: `9.65 %`
* **Horizon-Specific Breakdown (Physical mph):**
  * 15 min (Step 3): MAE = `3.0592`, RMSE = `5.6100`
  * 30 min (Step 6): MAE = `3.4616`, RMSE = `6.5416`
  * 45 min (Step 9): MAE = `3.7671`, RMSE = `7.2030`
  * 60 min (Step 12): MAE = `4.0401`, RMSE = `7.7800`
* **Calibration:**
  * Expected Calibration Error (ECE): `0.1614`
  * Mean Aleatoric Uncertainty: `1.266244`
  * Mean Epistemic Uncertainty: `0.277434`
  * Mean Total Uncertainty: `1.543679`
