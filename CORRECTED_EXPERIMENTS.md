# Corrected experiment path

`corrected_experiment.py` is a new, auditable experiment. Original scripts, checkpoints,
and reported results remain available for comparison. **New runs are a changed model and
preprocessing protocol; they are not a reproduction of the publication's numbers.**

## Changes supported by the code audit

- Split the raw chronological series by target timestamp (70/15/15). A training
  forecast target is wholly before the validation period; a validation target is
  wholly before the test period. Validation and test inputs may use earlier readings.
- Fill missing readings from the last observed reading. Initial missing
  readings use zero. Zero is treated as a missing sentinel in these benchmark
  files: METR-LA has 575,302 zero entries, including all-sensor zero rows,
  and PEMS-BAY has 521 zero entries; positive readings below 1 mph are nearly
  absent. This is an explicit
  dataset convention, not proof that every individual zero is missing. Fit per-sensor
  median and interquartile scale on training timestamps only.
- Exclude missing target readings from training loss, validation selection, and
  test metrics. Save the target observation mask with predictions.
- Apply graph aggregation per historical time step. The temporal convolution and
  LSTM then process the 12-step historical axis. The model returns one mean and
  positive aleatoric variance per sensor and forecast step.
- Save the checkpoint with the lowest validation MSE. Evaluate the test set once
  after training. Stop after eight epochs without validation improvement, with
  a maximum of 50 epochs by default. During MC inference, enable only the
  dropout module.
- Compute predictive variance as the population variance of MC means plus the
  average predicted aleatoric variance. Report physical-speed MAE/RMSE and
  Gaussian interval coverage and width. Predictions and variances are saved in
  normalized units in `predictions.npz`; physical metrics use the train-fitted scale.

PEMS-BAY uses its included adjacency file with a threshold of 0.1. The original
checkout lacked the METR-LA adjacency matrix, so the *earlier*
`metr_la_seed42` run used a geographic eight-nearest-neighbor proxy. The
official DCRNN METR-LA adjacency is now included and automatically used in
new runs. Its provenance and hash are in `MANUSCRIPT_AUDIT.md`. The earlier
metric remains an archival proxy-graph result and has not been relabelled.
The METR-LA CSV and metadata sensor order, and the PEMS-BAY CSV and adjacency
sensor order, are checked before training.

The earlier results used 70/15/15. `--train-ratio .7 --val-ratio .1` selects
the manuscript's 70/10/20 split. These are different test periods, so their
scores should not be directly compared as a pure model or graph effect.

## Run

From the repository root with the installed Python environment:

```bash
python -m unittest discover -s tests -v
python corrected_experiment.py --dataset METR-LA --output corrected_results/my_metr_run --epochs 50 --batch-size 32 --val-ratio .1
python corrected_experiment.py --dataset PEMS-BAY --output corrected_results/my_pems_run --epochs 50 --batch-size 32 --val-ratio .1
```

The output directory must be new, to avoid replacing any earlier result. Runs
accept `--seed`, `--train-windows`, `--val-windows`, and `--test-windows` for
reduced-budget experiments. A window limit selects evenly spaced windows over
validation and test splits; training samples a fresh seeded subset each epoch.
The latest epoch and random state are saved to `latest.pt`; after an interruption,
repeat the same command with `--resume` to continue. Limited runs and single-seed results
are exploratory, not final benchmark estimates.

The full per-window `predictions.npz` stays in the local results directory and
is ignored by Git because it can exceed GitHub's per-file size limit. The
tracked `metrics.json`, `training.json`, and checkpoints record the run.

The script uses CUDA when `torch.cuda.is_available()` is true and records the
chosen device in `metrics.json`. A GPU host also needs a CUDA-enabled PyTorch
wheel compatible with its driver; the CPU-only wheel installed during cloud
onboarding will not use a GPU.
