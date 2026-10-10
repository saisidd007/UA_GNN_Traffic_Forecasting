# Required manuscript edits before using new results

The submitted LaTeX manuscript is not part of this public repository. This
file gives concrete replacement facts and identifies claims that should be
removed or remeasured. `MANUSCRIPT_AUDIT.md` contains the supporting audit.

## Methods and dataset table

- State the **actual** 70/10/20 target-time split if using the new paper
  interpretation or new small-model reruns. Give the first validation/test
  target timestamps from `MANUSCRIPT_AUDIT.md`. Do not describe the older
  70/15/15 results as if they used this split.
- Replace “linear interpolation prior to normalization” with the actual
  causal historical forward fill and train-only per-sensor median/IQR scale.
  State that zero is treated as a missing sentinel and missing targets are
  excluded from losses and scores; this convention has uncertainty.
- Replace PEMS-BAY's claimed 1,515 directed edges with 2,369 non-self edges
  at threshold >0.1 for its supplied graph. METR-LA has 1,515 non-self
  directed edges in the official DCRNN adjacency now included in the repo.
- Correct PEMS-BAY's end date to 30 June 2017 for the supplied CSV.
- Specify the exact graph direction and row-normalization used by the code.
  If using `paper_model.py`, say it aggregates one- and two-hop weighted
  transitions in both graph directions. If using `corrected_experiment.py`,
  say it uses binarized one-hop neighborhood aggregation and is a different
  model.
- Replace the asserted 304K parameter count with the actual count for the
  checkpoint being reported, and give embedding width, batch size, mask
  probability, annealing rule, sampled training budget, and hardware.

## Results, tables, and figures

- Do not copy the manuscript's 1.67/3.72 and 3.46/6.57 mph scores into a
  corrected-results table unless a matching rerun independently recovers
  them. Use the exact new `metrics.json` files and identify model, graph,
  split, seed count, and test-window sample size with every score.
- Label DCRNN, Graph WaveNet, AGCRN, GMAN, PDFormer, and other numbers taken
  from their publications as **literature reported**, with their original
  evaluation protocol. Do not describe them as controlled matched reruns.
- Regenerate ablation tables from runs with the same target split, graph,
  train/validation/test window sample, seed, training budget, and physical
  mph metric. The extreme no-diffusion value 23.617 is not verified by the
  current source code or new short pilots.
- Replace claimed calibration ECE, coverage, reliability diagrams, and
  sensor-failure curves with outputs from `analyze_predictions.py` and
  `robustness_experiment.py` for the same trained model. State the Gaussian
  moment-matching approximation, corruption mechanisms, seed counts, and
  width alongside coverage.
- Report observed test-target counts and per-horizon errors. State clearly
  whether a result uses all test windows or an evenly spaced subset.

## Interpretation and limitations

- Delete the claim that the softplus derivative is uniformly bounded away
  from zero. It approaches zero for sufficiently negative inputs.
- Describe a BiLSTM backward state as using later steps **within the known
  input history**. It does not use future forecast targets.
- Do not assert that graph attention suppresses failed sensors, that
  intervals widen before congestion, or that robustness is smooth through
  50% failure without results from the corrected trained model and the
  specified test-time corruption setup.
- State that MC dropout is an approximation to epistemic uncertainty, and
  that high coverage with broad intervals does not by itself show useful
  calibration.
- Remove claims of RTX A2000/CUDA training and sparse operations from any
  table describing these CPU reruns. The existing small-model code uses a
  dense binary graph; the manuscript interpretation uses dense diffusion
  matrices and sparse edge indexing for attention.

The paper's full empirical conclusions should be revised after adequate
multi-seed, full-budget manuscript-model runs. A limited CPU pilot can check
implementation behavior but cannot certify the previous published tables.
