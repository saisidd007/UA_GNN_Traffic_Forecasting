# Archived 70/15/15 audited experiment results (seed 42)

The newer 70/10/20 reruns, official METR-LA graph results, and manuscript
pilot are documented in [MANUSCRIPT_RERUN_RESULTS.md](MANUSCRIPT_RERUN_RESULTS.md).

These are new results from `corrected_experiment.py`, not replacements for the
publication's numbers. Both runs used CPU, 50 epochs, batch size 32, 8,192
freshly sampled training windows per epoch, full chronological validation and
test splits under 70/15/15, and 30 Monte Carlo dropout passes at the final test evaluation.
The best checkpoint was selected by validation MSE only.

| Dataset | Best epoch | Model MAE / RMSE (mph) | Persistence MAE / RMSE (mph) | 95% coverage | Mean 95% width (mph) |
| --- | ---: | ---: | ---: | ---: | ---: |
| METR-LA | 49 | 4.043 / 7.502 | 4.182 / 8.111 | 96.7% | 32.79 |
| PEMS-BAY | 48 | 2.135 / 4.572 | 2.228 / 5.225 | 97.5% | 15.07 |

The persistence baseline repeats the last input speed for all twelve forecast
steps. It uses the same test windows, train-fitted scale, and observed-target
mask. The model beats this baseline on these two runs, but the margin in MAE
is small. Coverage above 95% and the wide intervals indicate that uncertainty
quality needs further work; coverage alone does not establish calibration.

The test periods contain 5,130 METR-LA and 7,807 PEMS-BAY windows. Missing
target readings are excluded from every metric. `metrics.json` records
horizon-specific MAE and coverage at 50%, 80%, 90%, and 95%; `training.json`
records every validation epoch. `persistence_baseline.json` holds the matching
baseline. The local `predictions.npz` arrays are omitted from Git because each
archive exceeds GitHub's per-file size limit.

## Limits on interpretation

- These are **single-seed** runs. No standard deviation or significance claim
  is available.
- The corrected temporal model is a new implementation, not the exact old
  270,340-parameter architecture. Its scores cannot be attributed solely to
  a preprocessing or temporal-axis fix.
- At the time of this run the checkout had no METR-LA adjacency file. This
  archived run used a geographic eight-nearest-neighbor proxy. The official
  DCRNN graph is now included for later runs; this old score is unchanged.
- Zero readings are treated as missing sentinels based on the observed file
  patterns. If the datasets contain genuine zero-speed observations, this
  convention needs a source-backed revision and another rerun.
- These archived runs had no diffusion or sensor-failure reruns. New stress
  tests and a limited paper-model diffusion pilot are documented in
  `MANUSCRIPT_RERUN_RESULTS.md`. Original paper tables remain provisional.
