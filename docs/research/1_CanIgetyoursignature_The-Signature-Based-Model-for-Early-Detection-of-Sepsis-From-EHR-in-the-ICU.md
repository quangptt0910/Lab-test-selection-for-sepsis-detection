# #1 Can I get your Signature? — The Signature-Based Model for Early Detection of Sepsis From EHR in the ICU

Notes for M0-5 (top-4 review). Only what is useful for reuse here.

## Source
- Paper: Morrill, Kormilitzin, Nevado-Holgado, Swaminathan, Howison, Lyons. CinC 2019 write-up, `papers/#1-CanIgetyoursignature/CanIgetyoursignature.pdf` (4 pages).
- Code: `papers/#1-CanIgetyoursignature/physionet_submission_6/`.
- Reported score: utility **0.360** on the full test set, officially ranked 1st of 101 submissions (paper abstract and Discussion). 5-fold CV utility on the training data: **0.430** (std 0.018). The paper's AUC of 0.868 (Section 3.2) is patient-level (flagged at risk at any time vs ever septic), so it is not comparable with the other teams' hourly AUROC.
- Source tags used below: **[P]** stated in the paper, **[C]** seen in the code, **[P≠C]** paper and code disagree.

## Pipeline in one view
1. Per patient, forward-fill every column; NaN before the first value. [P, C]
2. Derived clinical features: ShockIndex, BUN/CR, PartialSOFA, SOFA deterioration. [P, C]
3. Per hour t, a look-back window of the last `k` rows gives: rolling max/min of vitals, statistical moments, measurement counts, and path signatures. [P, C]
4. All features at hour t feed a LightGBM **regressor**. Target = `U1 − U0` (utility of alarm minus utility of no alarm for that hour). [P, C]
5. Alarm if prediction > threshold. The threshold is found by a gradient-free search that maximises the official normalised utility. [P, C]
6. In the challenge, the scoring code is called hour by hour. The submission recomputes the features from all rows up to the current hour, so it is causal. [C]

## Preprocessing
- Forward-fill within patient, no backward fill. NaN stays before the first measurement; LightGBM handles it. [P, C]
- This is the same recipe as `CLAUDE.md` (ADR-0001).
- Hospital column is dropped before modelling. [C]
- No scaling. Trees do not need it.
- `RemoveExtremeValues` (percentile clipping) exists in the code, but I did not see it used in the submission path. Not verified for the final model. [C]

## Features
| Feature | Definition | Source |
|---|---|---|
| ShockIndex | HR / SBP | [P, C] |
| BUN/CR | **BUN / Creatinine** (confirmed by the project owner). The paper text says "bilirubin to creatinine", which is a wording error in the paper | [C], paper wording wrong |
| PartialSOFA | Sum of SOFA points from platelets, bilirubin (total), MAP (<70 → 1 point), creatinine | [P, C] |
| SOFA deterioration | PartialSOFA minus the minimum of the previous 23 hours, floored at 0 (code). The paper says a binary flag, 1 if PartialSOFA rose by 2 in 24 h | [P≠C] |
| SaO2/FiO2 | SaO2 / FiO2, used as a signature channel in the code | [C] only |
| Rolling max / min | Over the 7 vitals (HR, O2Sat, Temp, SBP, MAP, DBP, Resp). Window = **5 rows including the current hour** in the code (`rolling(5, min_periods=2)`); paper Table 2 says 6. No off-by-one found that explains it, see "Look-back conventions" | [P≠C] |
| Moments | 2nd and 3rd moment over an 8-row window (`moment_lookback 7`, see "Look-back conventions"), of all columns | [C] only |
| Measurement counts | Count of non-NaN values per lab, divided by ICULOS; plus counts over the last 8 rows | [P, C] |
| Signatures | See below | [P, C] |

### Signature features (the team's main idea)
- Path of the last `lookback` rows (7), padded with the first value at the start of a stay (`lookback_method: fixed`). [C]
- Transformations: lead-lag, then truncated **log-signature**, order 3, via `esig`. [C]
- Final config computes signatures **per column** (`individual: true`) for `SOFA`, `BUN/CR`, `SaO2/FiO2`. [C]
- **[P≠C]**: the paper lists PartialSOFA, MAP, BUN/CR as signature channels (with a time dimension) and adds a second set of signatures of the cumulative sum of all non-stationary columns. The final code config has `cumsum: false` and no time dimension. I could not confirm which version produced the submitted model.
- Where to check in the code: `papers/#1-CanIgetyoursignature/physionet_submission_6/models/experiments/main/new_solution_6/1/config.json` (`"columns": ["SOFA", "BUN/CR", "SaO2/FiO2"]`), the same list on the active line of `src/models/experiments/main.py` (line 53; the `['SOFA', 'MAP', 'ShockIndex']` variant is commented out), and the saved run source `models/experiments/main/new_solution_6/_sources/`. The submission reads that `config.json` in `src/submission/generate_submission/main.py` (line 14-15). The code's `SOFA` column is the PartialSOFA of the paper (platelets, MAP, creatinine, bilirubin; `DerivedFeatures.SOFA` in `src/data/transformers.py`). So two of three channels match the paper; **MAP (paper) vs SaO2/FiO2 (code)** is the difference.
- Signature gain in the paper (5-fold CV utility, Table 4): time only 0.282 → original 40 features 0.389 → + hand-crafted 0.418 → + signatures 0.430. So signatures add **about +0.012** on top of hand-crafted features, within the fold std of 0.018.

**Reference rule for this file:** where paper and code disagree, the code is the reference value. The paper's value is kept as a note (for example, max/min look-back 6 in the paper may be the real value of the submitted model, but the code says 5).

**MAP vs SaO2/FiO2 (not the same feature)**
- MAP: mean arterial pressure, a vital sign. Always available, about 10% missing in A. Also the cardiovascular component of PartialSOFA.
- SaO2/FiO2: SaO2 (arterial blood gas) divided by FiO2, both forward-filled before the division. It is an oxygenation index and a proxy for PaO2/FiO2, because PaO2 is not in the data. Both inputs are 86–98% missing, so the value is usually an old carried-forward number, and the two inputs can come from different hours.
- It is not a count feature, so it passes ADR-0002 on that rule, but it only exists after the ABG panel is revealed. Check the FiO2 unit and range (our cleaning keeps 0.21–1.0) before reusing it. Does not use O2Sat (pulse oximeter), unlike the usual SpO2/FiO2 literature.

**Oxygenation ratios and the scores (for later reuse)**
- Only the **SOFA** respiratory component uses an oxygenation ratio: PaO2/FiO2 (P/F). SIRS uses respiratory rate or PaCO2, and qSOFA uses respiratory rate, SBP and mental status. Neither SIRS nor qSOFA needs P/F, S/F or SaO2/FiO2.
- PaO2 is **not in this dataset**, so P/F cannot be computed. Candidates here: S/F = O2Sat / FiO2 (O2Sat is the pulse-oximeter vital) and SaO2/FiO2 (SaO2 is the blood-gas lab). S/F is a commonly used surrogate for P/F in the literature; I have not checked a source in this review.
- FiO2 is the limiting input for both ratios (86% missing in A, 98% in B, per our inspect output). S/F has an advantage only on the numerator side (O2Sat is 12% missing in A).
- The project's PartialSOFA has no respiratory term (platelets, bilirubin, MAP, creatinine). Adding a surrogate respiratory point is a decision for M2-2, not something this paper settles.

**How to test it later (M2-2 / M4-1)**
1. Keep everything fixed (features, LightGBM, `U1 − U0` target, split file, frozen threshold) and change only the signature channels or the added ratio. One change per experiment id.
2. Variants: (a) paper channels `PartialSOFA, MAP, BUN/CR`; (b) code channels `SOFA, BUN/CR, SaO2/FiO2`; (c) code channels with S/F (O2Sat / FiO2) in place of SaO2/FiO2; (d) no signatures (B2).
3. Compute each ratio inside the feature builder, only from labs the replay has revealed, forward-filled. Decide in config whether both inputs must be recent, or add `hours_since_ABG`, so the ratio is not built from values hours apart.
4. Run through `run-experiment` and compare on A→B, B→A, mix→A and mix→B. Keep a channel only if mix improves without hurting the cross-hospital setups (`selection.cross_hospital_tolerance`).
5. Windows, order and thresholds go in `configs/*.yaml`; unit checks of FiO2 (0.21–1.0) come from `configs/cleaning.yaml`.

### Look-back conventions (checked in code)
| Feature | Paper | Code parameter | Rows actually used | Where |
|---|---|---|---|---|
| Signature path | 7 | `lookback 7`, method `fixed` | **8 rows** (`data[i : i+lookback+1]`, 7 steps) | `MakePaths.transform`, `src/features/transformers.py` |
| Moments | not listed | `moment_lookback 7` | 8 rows (same `MakePaths`) | `AddMoments` |
| Measurement count | 8 | `num_measurements 8` | 8 rows, difference of cumulative counts = new values in 7 steps (`x[-1] - x[0]`) | `GetNumMeasurements` |
| Max / min | **6** | `max_min_lookback 5` | **5 rows** (`df.rolling(5, min_periods=2)`, no +1) | `GetStatistic` |

- The paper's 7 and 8 equal the code parameters, so the paper reports the parameter, not the row count. By that convention max/min would be 5, not 6.
- Searching the code for `max_min_lookback` finds the value 5 in `config.json`, the saved run source, `main.py` and the `feature_config` default. No `+1` or `-1` is applied on the max/min path, in training or in the submission (`last_only` branch takes the last 8 rows, then `rolling(5)`).
- So I found **no off-by-one that turns 5 into 6**. Possible reasons I cannot rule out: the shipped `clf.pickle` came from a different run than this `config.json` (only one config file is in the folder, and I do not read pickles), or the paper's Table 2 is wrong or from an earlier run.

## Model and target
- `LGBMRegressor`, objective L2, target `U1 − U0`. [P, C]
- CV config: `n_estimators 100`, `learning_rate 0.1`, tuned params (random search): `max_depth 6`, `num_leaves 41`, `min_child_samples 173`, `min_child_weight 35`, `reg_alpha 100`, `reg_lambda 20`, `subsample ≈ 0.48`, `colsample_bytree ≈ 0.59`. [C: `config.json`, `cout.txt`]
- Final-model training function: `n_estimators 1000`, `learning_rate 0.013` with the same tuned params. [C: `train_classifier`]
- Heavy L1/L2 regularisation (`reg_alpha` 100) is notable: the signal is weak and noisy.

## Threshold
- Tuned on out-of-fold predictions with a gradient-free optimiser (differential evolution, bounds [−0.2, 0.2], budget 100 in CV, 1000 for the final fit). [C]
- In CV, the threshold is fitted on the training folds' out-of-fold predictions and applied to the held-out fold. [C]
- In `train_classifier` (final fit) the threshold is optimised on predictions of a model fitted on the same data. This is in-sample. I did not confirm which script produced the shipped `threshold.pickle`. [C]
- Alarm rule in the submission: `prediction > threshold`. [C]

## Validation
- Stratified patient-grouped 5-fold CV (custom `CustomStratifiedGroupKFold`) over **hospitals A and B pooled**. [P, C] Patients never cross folds.
- Utility per fold: 0.432, 0.434, 0.437, 0.400, 0.448. [P, C]
- The input matrix in the final run is **1,552,210 rows × 276 columns** (A and B pooled). [C: `cout.txt`] The row count equals A + B patient-hours in our `inspect` output (790,215 + 761,995).

## Hidden hospital / cross-hospital
- The paper reports 0.360 on the full test set vs 0.430 in CV, a **drop of 0.07**. It gives no breakdown by hospital and no analysis of the gap. [P]
- No hospital-held-out experiment and no hospital feature in the model. [P, C]
- For this project, the CV number is closest to our `mix` setup. The test score is the only evidence of how this recipe transfers.

## ADR-0002 check (features that depend on which labs were ordered)
- **Fail:** per-lab measurement counts (`*_count` ÷ ICULOS, count over the last 8 rows). These encode ordering behaviour and break in replay.
- **Fail:** any signature or moment computed on a lab column whose NaN pattern defines the path (the paper's cumulative-sum signatures of all non-stationary columns may include labs). Not verified.
- **OK:** vitals max/min, ShockIndex, PartialSOFA and its deterioration (uses lab values only when present, forward-filled), BUN/CR, signatures of those derived channels.
- Replacement for counts: `hours_since_<panel>` (ADR-0002).

## Use / adapt / reject
| Item | Verdict | Reason | Issue |
|---|---|---|---|
| Forward-fill, NaN before first value | **use** | Same as ADR-0001; causal | M2-1 |
| Regress `U1 − U0` with LightGBM | **use** | Core of the rank #1 recipe | M1-2, M3-2 |
| ShockIndex, BUN/CR (BUN ÷ Creatinine) | **use** | Cheap, causal, already in `CLAUDE.md` | M2-2 |
| PartialSOFA and 24 h change | **use** | Already in `CLAUDE.md`. Use the code's definition (rise vs the 23-hour minimum) as the starting point and decide in M2-2 | M2-2 |
| Rolling max/min of vitals | **use** | Vitals are exempt from ADR-0002. Window goes in config (paper 6, code 5) | M2-2 |
| Threshold by gradient-free search on utility | **adapt** | Tune on the **training side only** and freeze (hard rule); search on out-of-fold predictions, not in-sample | M1-1 |
| Signature features (order 3 log-signature, lead-lag) | **adapt** | Gain was about +0.012, inside the fold std. Keep as optional B3 on derived channels only | M4-1 |
| Cumulative-sum signatures of all non-stationary columns | **reject** for now | May include lab columns and ordering patterns; unclear in the final model | M4-1 (revisit) |
| Measurement counts / lab sampling rate | **reject** | Violates ADR-0002 | — |
| Moments of all columns | **adapt** | Only on vitals and derived channels | M2-2 |
| Pooled A+B CV as the only validation | **reject as sole protocol** | Optimistic. We also run A→B and B→A (ADR-0004) | M3-1 |
| In-sample threshold in the final fit | **reject** | Leaks the training fold into the alarm threshold | M1-1 |

## Open points to check against the paper
- Which signature channels and whether cumulative-sum signatures were in the submitted model (paper vs code disagree).
- Max/min look-back: paper 6 vs code window of 5 rows (see "Look-back conventions"). Our window goes in `configs/*.yaml` anyway, so this only matters as a starting value.
- How utility labels (`utility_scores.pickle`) were generated. The generating script is not in the files I read. Our own `U1 − U0` target (M1-2) must be built from the official scorer and tested against the `rank1-features` table.
