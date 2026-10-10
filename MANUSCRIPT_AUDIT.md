# Manuscript and correction-plan audit

Reviewed against the uploaded LaTeX manuscript, the supplied correction PDF,
the repository code, and the benchmark files. The manuscript is source material,
not evidence that its reported experiments occurred under the protocol it states.

## Findings that change the experiment

| Claim or request | Evidence and decision |
| --- | --- |
| Chronological split | Manuscript specifies 70/10/20; the earlier corrected path used 70/15/15. Both ratios are now configurable. New runs in `corrected_results/*701020*` use 70/10/20. The split is by complete forecast targets. |
| METR-LA road graph | Repository originally lacked its adjacency. Downloaded the DCRNN benchmark `data/sensor_graph/adj_mx.pkl` from `liyaguang/DCRNN` (SHA-256 `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3`). Its sensor ID order exactly matches the CSV. New runs use it; earlier `metr_la_seed42` retains its geographic proxy result. |
| Graph edges | At threshold >0.1, METR-LA has 1,515 directed non-self edges and 207 self edges; PEMS-BAY has 2,369 directed non-self edges and 325 self edges. The manuscript's ~1,515 PEMS-BAY edge count is wrong under this convention. Edges are directed: `A[i,j]` means node `i` aggregates node `j`. |
| Missing values | The source CSVs contain 575,302 zeros in METR-LA and 521 in PEMS-BAY; no NaNs. Zero is treated as a missing sentinel based on benchmark convention and file patterns, but this cannot prove that every actual zero speed is invalid. Target zeros are excluded; historical gaps are causally forward filled. The manuscript's unrestricted linear interpolation is rejected because it can use future observations. |
| Time operations | Legacy scripts had a sensor/time axis error. `corrected_experiment.py` fixes this in a simpler model. `paper_model.py` separately implements four temporal dilation blocks over the 12 historical steps, graph diffusion/attention over nodes, and a BiLSTM over history. Shape and directed two-hop tests are included. |
| Paper architecture | The manuscript describes four blocks, two-hop bidirectional diffusion, four attention heads, 64 hidden channels, temporal embeddings, BiLSTM and dual heads. The legacy code and earlier corrected model do not implement that full specification. `paper_model.py` is an interpretation, because the manuscript omits enough layer/residual/head details to reproduce an exact 304K-parameter network. This interpretation has 222,480 parameters with 64 channels. Its scores cannot replace or verify the manuscript scores. |
| Validation and uncertainty | The corrected path chooses checkpoints by validation MSE and evaluates test after training. `paper_experiment.py` does the same and activates Dropout without switching BatchNorm to training mode for MC sampling. Gaussian intervals combine variance across MC means with the mean predicted variance. |
| Manuscript training loss | The manuscript's equation uses `lambda*MSE + (1-lambda)*NLL` and states that lambda falls from 1 to 0 over ten epochs. The first paper-style code version incorrectly used `MSE + (1-lambda)*NLL`; its saved pilots are superseded. The corrected implementation follows the equation and halves the learning rate on plateau as described. |

The thresholded graphs are genuinely directed: 2,222 METR-LA and 3,578
PEMS-BAY matrix positions differ from their transposes (counting both sides
of each asymmetric pair). The small corrected model binarizes surviving
edges; the manuscript interpretation uses their supplied weights and both
forward and reverse row-normalized transitions through two hops.

### 70/10/20 data audit

| Dataset | Raw readings | First validation target | First test target | Complete train / val / test windows | Missing sentinel entries |
| --- | ---: | --- | --- | ---: | ---: |
| METR-LA | 34,272 × 207 | 2012-05-23 07:10 | 2012-06-04 04:45 | 23,967 / 3,416 / 6,844 | 575,302 |
| PEMS-BAY | 52,116 × 325 | 2017-05-07 17:05 | 2017-05-25 19:20 | 36,458 / 5,200 / 10,413 | 521 |

METR-LA has 2,148 rows where all 207 readings are zero and only nine
positive readings below 1 mph. PEMS-BAY has no positive reading below 1 mph.
This pattern supports treating zero as a missing sentinel for these files,
while still leaving the interpretation of each individual zero uncertain.

Scaler median and interquartile range are fitted only on the first 70% of
timestamps. Every training window's 12 forecast targets precede the first
validation target, and every validation window's targets precede the first
test target. Validation and test input histories may use earlier readings.

## Claims that require revision or new evidence

- The paper's PEMS-BAY date range ends in May, but the CSV continues through 30 June 2017.
- Claimed PEMS-BAY 1.67/3.72 mph and METR-LA 3.46/6.57 mph are not reproduced by the prior corrected runs. Published baseline numbers come from other papers under potentially different protocols and are literature values, not matched reruns.
- Extreme ablation effects, including the no-diffusion value 23.617, cannot be accepted or dismissed without matched retraining and saved-prediction checks. The existing code does not support interpreting these as measured by the corrected protocol.
- The manuscript calls ECE 0.087 and 0.161 “reasonably calibrated” despite describing undercoverage and overcoverage. Coverage, width, and a reliability plot are needed together. The earlier corrected runs give ECE 0.164 (PEMS-BAY) and 0.175 (METR-LA) under 10–95% nominal levels, with Gaussian CRPS 1.689 and 3.373 mph respectively. These are results for the simpler corrected model.
- The paper states that `softplus` has a derivative bounded away from zero for all finite inputs. This is false: its derivative approaches zero as the input decreases without bound.
- Bidirectional LSTM backward states can use later *observed historical* time steps, but cannot access the future forecast targets. Wording suggesting otherwise is misleading.
- The manuscript's claims that sensor masking automatically makes graph attention suppress failed sensors, guarantees robust uncertainty, or creates anticipatory interval widening are hypotheses. They need controlled evaluation; the architecture alone does not establish them.
- The manuscript says the training mask probability should equal the sensor-dropout rate used for robustness evaluation, but the latter varies from 10% to 50%; one fixed training probability cannot equal all those rates. Its training mask probability is not specified numerically. The new paper-style implementation records its explicit 10% choice.
- “304K parameters,” “sparse tensor operations throughout,” “RTX A2000,” and “all preprocessing identical” do not describe the current corrected code or this CPU environment.
- A comparison of percentage degradation across different per-sensor normalizations is not generally scale invariant unless the same scalar multiplies every error. Use physical mph for cross-dataset comparisons.

## Scope of the new results

`corrected_experiment.py` remains the auditable small-model path. Its new
70/10/20 runs isolate the official-graph/split changes but still are **not**
paper architecture reruns. `paper_experiment.py` makes a manuscript-based model
trainable with explicit choices, and supports a matched no-diffusion switch.
Any short or sampled CPU run is exploratory. The manuscript's final claims
would require adequately trained repeated seeds, matched ablations, and a
revised manuscript that uses the actual graphs, preprocessing, and hardware.
