# Rerun results after manuscript and correction-plan audit

**Revision notice:** The manuscript-architecture pilot numbers below were
produced before a loss-equation correction. That training code kept MSE at
full weight while adding NLL after warm-up, whereas the manuscript replaces
MSE with NLL over ten epochs. Those architecture pilots and their ablation,
robustness, and uncertainty checks are archived observations, not current
results for the corrected implementation. The corrected small-model results
are unaffected because they use a separate training script.

All numbers below were generated on the CPU in this cloud environment. MAE,
RMSE, interval width, and CRPS use physical mph. A zero reading is treated as
missing under the stated benchmark convention. The new chronological split is
70/10/20 by complete forecast targets; preprocessing statistics come only
from the training period. The test targets were evaluated after checkpoint
selection on validation data.

## Corrected-loss manuscript interpretation, PEMS-BAY CPU pilot

The paper-style implementation now uses the manuscript's stated
`lambda*MSE + (1-lambda)*NLL` loss, with lambda falling from 1 to 0 over ten
epochs, a plateau learning-rate factor of 0.5, and the manuscript's batch
size 64. Both runs below use seed 42, 256 sampled training windows per epoch,
256 validation windows, 512 evenly spaced test windows, at most 20 epochs,
and ten MC passes. The matched persistence baseline on those test windows is
2.187 MAE / 5.184 RMSE mph. These runs are substantially under the paper's
stated training budget and are not final benchmark estimates.

| Model | MAE / RMSE (mph) | Normalized MAE | 95% coverage / width (mph) | Gaussian CRPS (mph) | Best epoch / epochs run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paper interpretation with diffusion | 4.063 / 7.592 | 1.004 | 94.35% / 20.81 | 3.088 | 16 / 20 |
| Matched no-diffusion interpretation | 4.813 / 8.396 | 1.171 | 94.20% / 21.76 | 3.560 | 11 / 19 |
| Separately trained no-dropout interpretation | 4.093 / 7.709 | 1.009 | 93.81% / 20.07 | 3.089 | 18 / 20 |

Independent saved-prediction rescoring verified identical targets and masks
across 1,996,583 observed targets. Removing diffusion raised MAE by 0.750
mph (18.46%) in this one-seed pilot. A descriptive 1,000-resample bootstrap
over 14-window blocks gave [0.650, 0.858] mph. The no-diffusion run stopped
after 19 epochs under the same early-stopping rule; the diffusion run reached
20. Both are worse than matched persistence, so this comparison does not
establish a useful deployed model or validate the manuscript's extreme
normalized-MAE ablation claim of 23.617. The new normalized MAEs use the
explicit train-only per-sensor IQR scale, which the manuscript does not
specify precisely enough for a numerical equivalence claim.
The diffusion pilot's reliability curve lies above the nominal diagonal at
most 10–95% levels (coverage ECE 0.105); its nominal 95% interval covers
94.35% at a mean width of 20.81 mph. One near-nominal coverage point does
not establish calibration across levels.
The trained no-dropout variant has zero epistemic variance by construction.
Independent rescoring matched all 1,996,583 targets and masks. Its MAE rose
0.030 mph (0.74%) relative to the dropout run; a descriptive 14-window block
bootstrap gave [-0.006, 0.071] mph. This one-seed sampled pilot does not
establish a reliable accuracy difference. Its calibration ECE is 0.088.

## Full-test corrected small model, one seed

These runs use seed 42, batch size 32, at most 50 epochs, 8,192 sampled train
windows per epoch, all validation/test windows, and 10 MC dropout passes.
METR-LA uses the official DCRNN adjacency, with CSV sensor order verified.
This model is substantially smaller and different from the manuscript model.

| Dataset | Test windows | Observed forecast targets | MAE | RMSE | Matched persistence MAE / RMSE | 95% coverage | 95% mean width | Gaussian CRPS | ECE 10–95% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| METR-LA | 6,844 | 14,940,228 | 4.079 | 7.717 | 4.243 / 8.191 | 96.61% | 32.70 | 3.385 | 0.174 |
| PEMS-BAY | 10,413 | 40,608,096 | 2.074 | 4.499 | 2.170 / 5.125 | 97.25% | 14.34 | 1.635 | 0.158 |

The paired model minus persistence MAE is −0.163 mph for METR-LA and −0.096
mph for PEMS-BAY. A 1,000-resample bootstrap over contiguous 288-window
blocks gives descriptive 95% intervals [−0.198, −0.126] and [−0.121,
−0.074] mph, respectively. Overlapping forecast windows and possible traffic
regime change limit a confidence interpretation. High 95% coverage occurs
with broad intervals; it does not show calibration at all nominal levels.
Per-horizon results, calibration plots, and exact values are saved in the
corresponding `corrected_results/*701020_seed42` folders.
For the small model, MAE at 15/30/60 minutes is 3.331/4.046/5.185 mph on
METR-LA and 1.529/2.082/2.865 mph on PEMS-BAY.

## Reduced-budget five-seed checks

For an initial variability check, the corrected small model uses seeds 7,
17, 29, 42, and 71 with 1,024 sampled training windows per epoch, 512
validation windows, 512 test windows, at most 12 epochs, and 5 MC passes.
These results are not a five-seed full benchmark; each training run is much
shorter than the 50-epoch single-seed runs above.

| Dataset | MAE mean ± SD (mph) | RMSE mean ± SD (mph) | Best validation epochs |
| --- | ---: | ---: | --- |
| METR-LA | 4.366 ± 0.109 | 7.804 ± 0.143 | 12, 12, 12, 12, 12 |
| PEMS-BAY | 2.302 ± 0.015 | 5.108 ± 0.015 | 12, 12, 12, 12, 12 |

Every reduced-budget run selected the final permitted epoch, so these
five-seed means are still training-budget limited and should not be used as
final model estimates. On the same 512 test windows, persistence MAE is
4.221 mph for METR-LA and 2.187 mph for PEMS-BAY; these short-budget model
means are worse than persistence.

## Archived manuscript architecture interpretation, superseded CPU pilot

The implemented model uses four dilated causal temporal blocks, weighted
forward/reverse two-hop diffusion, four-head graph attention, a BiLSTM over
known history, temporal embeddings, and separate mean and variance heads.
The manuscript leaves several layer and training details unspecified. The
explicit implementation has 222,480 parameters at hidden width 64, whereas
the manuscript claims approximately 304,000. These pilots sample only 256
training windows per epoch, 256 validation windows, and 512 test windows,
with a 20-epoch cap, seed 42, and 10 MC passes. They are exploratory and
cannot substantiate the manuscript's reported benchmark scores.

| Dataset / model | MAE | RMSE | Matched persistence MAE | 95% coverage | 95% width | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PEMS-BAY, manuscript interpretation | 3.881 | 7.617 | 2.187 | 95.58% | 22.38 | 20 |
| PEMS-BAY, matched no-diffusion interpretation | 4.293 | 7.965 | 2.187 | 95.76% | 25.74 | 19 |
| METR-LA, manuscript interpretation | 6.491 | 12.575 | 4.221 | 96.30% | 45.03 | 19 |

Both architecture pilots perform worse than matched persistence and the
corrected small model. The latter has a different architecture and test sample, so those
model scores should not be treated as a controlled ablation. The matched
no-diffusion pilot is the appropriate comparison for the manuscript's
diffusion claim.
Removing diffusion raises PEMS-BAY MAE by 0.413 mph, or 10.6%, in this
one-seed 20-epoch pilot. It does not reproduce the manuscript's claimed
normalized MAE of 23.617. Because the architecture and scaling cannot be
matched exactly and both variants are undertrained, this is evidence about
the explicit interpretation only, not a definitive falsification of that
published number.
The saved-prediction comparison confirms identical targets and observation
masks (1,996,583 observed targets). A descriptive 1,000-resample bootstrap
of contiguous 14-window blocks gives a 95% interval of [0.353, 0.476] mph
for no-diffusion minus diffusion MAE. The sampled windows are evenly spaced
and this interval does not address training-seed variability.
An independent rescore of saved predictions gives normalized MAE 0.937 with
diffusion and 1.046 without, under the train-fitted per-sensor IQR scaling
used here. The manuscript's normalization is not specified well enough for
a numeric comparison to its 23.617 value.
Gaussian CRPS is 4.955 mph for the METR-LA pilot and 2.906 mph for the
PEMS-BAY pilot; ECE over nominal 10–95% levels is 0.130 and 0.117.

The full-test corrected small model was also evaluated without MC sampling,
using its existing checkpoint. METR-LA gives 4.071 MAE and 96.56% nominal
95% coverage; PEMS-BAY gives 2.067 MAE and 97.11% coverage. This tests the
inference setting only, not training without dropout. For the PEMS-BAY paper
pilot, deterministic inference on its 512 test windows gives 3.879 MAE and
95.43% coverage. Removing the saved epistemic component from its Gaussian interval
leaves 95.49% coverage, whereas retaining only that component gives 21.91%.
These component checks do not substitute for a trained no-variance ablation.
Deterministic inference for the METR-LA architecture pilot gives 6.479 MAE
and 96.25% coverage on its 512 test windows.

An MC pass-count check reuses 64 fixed METR-LA test windows and dropout seed
42. At 10 passes it gives 4.008 MAE, 96.87% nominal 95% coverage, and 32.99
mph mean width. At 50 passes the values are 4.003, 96.88%, and 33.00 mph.
This is a sampling-stability check on a subset, not the full-test estimate.
The corresponding PEMS-BAY 64-window check changes from 2.068 MAE, 97.34%
coverage, and 14.57 mph width at 10 passes to 2.064, 97.39%, and 14.59
mph at 50 passes.
For the manuscript-style PEMS-BAY pilot on 16 fixed windows, 10 versus 50
passes gives 3.805 versus 3.801 MAE and 96.17% versus 96.14% coverage.
These differences are small relative to the pilot's large accuracy gap.

## CPU inference cost

Single-pass batch-16 inference on this five-CPU host, with one PyTorch thread,
gives median batch times of 39.7 ms for the 14,520-parameter METR-LA small
model and 60.4 ms for the same model on 325-node PEMS-BAY. The
222,480-parameter manuscript interpretation takes 2,268 ms on METR-LA and
3,529 ms on PEMS-BAY. Peak process RSS is approximately 385 and 497 MiB for
the small model, and 698 and 858 MiB for the manuscript interpretation,
respectively.
These are local process measurements, including imports and data structures;
they are not isolated model memory or GPU latency.
The PEMS-BAY no-diffusion pilot has 140,304 active parameters and took
1,959 ms per batch. Its model object still allocates the disabled diffusion
modules, so process RSS (approximately 854 MiB) does not measure the active
parameter savings in isolation.

## Synthetic sensor and time failures

The two full-test small-model checkpoints were stress tested on the same 128
evenly spaced test windows per dataset. Each failure type/rate uses three
corruption seeds, both train-median replacement and causal forward filling,
and three MC passes. The following 50% figures use train-median replacement;
rates mean sensor-time entries for random failure, sensors for persistent and
clustered failure, and history steps for temporal blocks. The rate definition
therefore differs by failure type. Other rates (10/20/30/40%) and the second
imputation are in the saved `robustness.json` files and plots.
Before artificial corruption, 11.5% of METR-LA historical input entries
and 0% of PEMS-BAY entries in these samples were already marked missing.

| Dataset | Clean MAE | Random 50% MAE | Persistent 50% MAE | Clustered 50% MAE | Temporal block 50% MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| METR-LA | 4.111 | 5.088 ± 0.017 | 5.353 ± 0.027 | 5.475 ± 0.080 | 4.754 ± 0.129 |
| PEMS-BAY | 2.159 | 3.437 ± 0.011 | 3.487 ± 0.069 | 3.614 ± 0.048 | 2.850 ± 0.148 |

At 50% clustered failure, mean nominal 95% coverage drops to 93.4% for
METR-LA and 79.7% for PEMS-BAY; mean interval widths are 34.99 and 10.20
mph, respectively. The PEMS-BAY intervals narrow under this severe failure.
The paper's broad claims of robustness and anticipatory uncertainty widening
are not supported by this small-model stress test. This is a synthetic
sampled-window check, not a measurement of real hardware outages or of the
fully trained manuscript architecture.

The undertrained PEMS-BAY manuscript-style pilot also ran a tiny, separate
stress-test smoke check: 16 evenly spaced windows, 10/30/50% rates, three
corruption seeds, three MC passes. Its clean MAE is 3.808 mph. At 50%
clustered failure with train-median replacement, MAE is 3.987 ± 0.082 mph,
coverage 93.4%, and width 19.51 mph. Persistent failure on this tiny sample
slightly reduces MAE, illustrating why this pilot cannot support a general
robustness conclusion. Exact rows for every condition are in its
`robustness.json`.

## Status of the original paper claims

The manuscript's PEMS-BAY 1.67/3.72 mph and METR-LA 3.46/6.57 mph results
were **not reproduced**. Published baseline values in the manuscript were
not rerun under this protocol. Its reported graph edge count for PEMS-BAY,
preprocessing description, dataset end date, parameter count, softplus
derivative statement, and hardware description need the specific fixes in
`MANUSCRIPT_REQUIRED_EDITS.md`. Robustness and ablation claims require the
matched experiments and limits documented here; they must not be copied
unchanged into a revised results section.
