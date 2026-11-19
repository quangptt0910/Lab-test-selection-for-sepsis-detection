# M3-3: B0–B2 baselines, B2 tuning, importance and the ordering diagnostic

Dev results only (ADR-0004); the test parts were not touched. Every number below comes from the official scorer (normalized utility) through `sepsis.eval.run_experiment`, with the threshold tuned on out-of-fold predictions of the training side and then frozen. Run folders: `experiments/runs/<id>/` (gitignored); one LOG row each in `experiments/LOG.md`.

## 1. Baselines

| id | model | A→B | B→A | mix→A | mix→B | AUROC (mean of 4) | wall |
|---|---|---|---|---|---|---|---|
| b0-001 | B0 `rule_score` (SIRS + PartialSOFA) | 0.055 | 0.078 | 0.096 | 0.055 | 0.654 | 15 min |
| b1-001 | B1 `logistic` on `sepsis_label` | 0.224 | 0.293 | 0.332 | 0.268 | 0.752 | 14 min |
| b2-001 | B2 `lightgbm_regressor` on U1 − U0 | **0.309** | **0.314** | **0.426** | **0.363** | 0.807 | 20 min |

- **M3 check: B2 beats B0 in all four columns** (by 0.24–0.33), and B1 in all four.
- No run raised a red flag (AUROC ≤ 0.95, |A→B − B→A| ≤ 0.15).
- B0 has nothing to fit, so its setups differ only in the tuned threshold: 3 everywhere except B→A (4). That is why A→B and mix→B are equal for B0.

## 2. Mix vs cross-hospital gap (b2-001)

| scored hospital | mix | cross-hospital | gap |
|---|---|---|---|
| A | mix→A 0.426 | B→A 0.314 | 0.112 |
| B | mix→B 0.363 | A→B 0.309 | 0.054 |

Hospital A loses twice as much as B when its own patients are left out of training. The M0-4 differences (`docs/eda/README.md`, Findings 1, 3, 9) fit this:
- **Record start (Finding 1).** 36.7% of A records start after ICU hour 1, against 6.4% in B, so `ICULOS`, the top feature in both directions (§4), means something different per hospital.
- **Charting (Finding 3).** Missingness and test timing differ between the hospitals. A model trained on B meets A's ordering pattern through `hours_since_<panel>` and LightGBM's NaN branches.
- **Values (Finding 9).** Blood pressures are lower in A (MAP 78.7 vs 86.3).
- **Prevalence.** A has more septic patients (8.8% vs 5.7%). Utility differences between the hospitals should be read with that in mind.

## 3. B2 tuning grid

Each run is one change from `configs/b2_lightgbm.yaml`. **Keep rule**, fixed before any run: a change is kept when the mean of mix→A and mix→B rises above b2-001's, and neither A→B nor B→A drops by more than `selection.cross_hospital_tolerance` (0.01). "OOF" is the mean out-of-fold `tuning_utility` from `thresholds.json`, a training-side figure.

| id | change | A→B | B→A | mix→A | mix→B | mix mean | OOF A→B / B→A / mix | kept |
|---|---|---|---|---|---|---|---|---|
| b2-001 | reference | 0.309 | 0.314 | 0.426 | 0.363 | 0.3945 | 0.422 / 0.384 / 0.394 | — |
| b2-002 | `subsample_freq: 1` (bagging on) | 0.305 | 0.311 | 0.425 | 0.352 | 0.3889 | 0.425 / 0.377 / 0.395 | no: mix −0.006 |
| b2-003 | 500 trees, learning rate 0.02 | 0.312 | 0.326 | 0.425 | 0.364 | 0.3944 | 0.428 / 0.387 / 0.401 | no: mix −0.0002 |
| b2-004 | `num_leaves: 15` | 0.309 | 0.328 | 0.423 | 0.355 | 0.3892 | 0.422 / 0.378 / 0.392 | no: mix −0.005 |
| b2-005 | `min_child_samples: 500` | 0.308 | 0.308 | 0.424 | 0.355 | 0.3895 | 0.424 / 0.378 / 0.395 | no: mix −0.005 |
| b2-006 | without `hours_since_panel` | 0.286 | 0.336 | 0.413 | 0.333 | 0.3728 | 0.411 / 0.346 / 0.385 | no: mix −0.022, A→B −0.023 |

**No change was kept, so b2-001 is the tuned B2.** Further findings:
- **`subsample_freq` stays off.** With bagging on, rank #1's `subsample: 0.48` acts, and it lowers mix→B by 0.011.
- **B→A likes simpler models.** b2-003, b2-004 and b2-006 raise B→A by 0.012–0.022 at little or no mix cost (except b2-006).
  - b2-003 misses the rule by 0.0002, well inside seed noise, and is the best by every OOF figure. It is the first candidate for a seed-repeat check.
- **`hours_since_panel` pulls the two directions apart.** Removing it costs A→B 0.023 and gains B→A 0.022. It carries hospital-specific ordering behaviour (see §4 and §5).
- **b2-005 coincidence.** Its A→B and B→A are both 0.3077, but they differ in the 6th decimal and in every other metric (AUROC 0.793 vs 0.776, alarm rate 12% vs 35%). It is not an error.

**Caveats.**
- The keep rule selects on eval-side dev scores (B-dev, A-dev, the mix held-out folds), so the tuned B2's dev numbers are optimistic. The honest number comes from the test parts in the final evaluation (ADR-0004).
- Differences of about 0.01 or less may be seed noise: there is no seed repeat.

## 4. What B2 learns (b2-001, `sepsis.eval.importance`)

The A→B and B→A models were refitted and checked to reproduce the run's scored predictions exactly. SHAP values come from LightGBM's `pred_contrib` on 50,000 evaluation rows. Direction is the sign of corr(feature value, SHAP): + means higher values raise the score.

| rank | A→B (scores B) | mean \|SHAP\| | dir | B→A (scores A) | mean \|SHAP\| | dir |
|---|---|---|---|---|---|---|
| 1 | ICULOS | 0.0217 | + | ICULOS | 0.0194 | + |
| 2 | hours_since_ABG | 0.0046 | − | hours_since_ABG | 0.0181 | − |
| 3 | hours_since_LACTATE | 0.0039 | + | HospAdmTime | 0.0113 | + |
| 4 | HospAdmTime | 0.0036 | − | FiO2 | 0.0050 | + |
| 5 | Temp | 0.0033 | + | MAP_min_6 | 0.0049 | − |
| 6 | shock_index | 0.0033 | + | HCO3 | 0.0047 | − |
| 7 | Temp_max_6 | 0.0032 | + | HR_max_6 | 0.0042 | + |
| 8 | MAP_min_6 | 0.0030 | − | hours_since_LACTATE | 0.0037 | − |
| 9 | Resp_min_6 | 0.0027 | + | Temp | 0.0036 | + |
| 10 | Creatinine | 0.0027 | + | Temp_max_6 | 0.0033 | + |
| 11 | FiO2 | 0.0026 | + | Alkalinephos | 0.0030 | + |
| 12 | Unit1 | 0.0022 | + | Unit1 | 0.0029 | + |
| 13 | O2Sat_min_6 | 0.0021 | − | BUN | 0.0025 | + |
| 14 | Alkalinephos | 0.0020 | − | pH | 0.0025 | − |
| 15 | hours_since_LFT | 0.0020 | + | PaCO2 | 0.0024 | − |

The clinical directions make sense: higher temperature, heart rate, shock index, FiO2 and BUN raise risk, while a lower MAP and O2Sat raise it.

**Suspicion checks:**
- **ICULOS ranks 1 in both directions, as expected** (CONTEXT: the strongest single predictor). It is about 4–5× the next feature in A→B.
- **HospAdmTime ranks 3–4 and flips direction between the hospitals.** It records hospital-to-ICU time, which partly reflects how each hospital admits patients. Unit1/Unit2 rank 12 and 19/28, which is acceptable.
- **`hours_since_ABG` is in the top 2 of both directions.** In B→A it is almost as strong as ICULOS (0.018). It encodes when blood gases were drawn, which is ordering behaviour that ADR-0002 allows as recency but which differs per hospital (EDA Finding 3).
- **Six top-15 features flip direction between the directions:** Alkalinephos, Creatinine, HospAdmTime, PaCO2, `hours_since_LACTATE` and pH. The flip in `hours_since_LACTATE` is the opposite-direction effect seen in the b2-006 ablation.

## 5. Ordering diagnostic (analysis only, ADR-0002; never a model or LOG row)

`sepsis.eval.ordering_diagnostic experiments/runs/b2-001` adds two forbidden column types:
- `n_revealed_<lab>`: the count of values revealed so far;
- `revealed_now_<lab>`: whether a value is revealed at this hour.

Both come from the as-recorded reveal-row table, so they show only what the view can show at row t. The diagnostic runs the b2-001 config with and without these columns at `lab_masking_p = 0`, under A→B and B→A. Output: `experiments/runs/b2-001-ordering/ordering.json`.

| setup | without | with ordering columns | gain | AUROC without → with |
|---|---|---|---|---|
| A→B | 0.2996 | 0.2996 | −0.0000 | 0.784 → 0.783 |
| B→A | 0.3059 | 0.3128 | +0.0069 | 0.777 → 0.775 |

- **On top of B2's existing features, explicit lab counts and just-revealed flags add almost nothing across hospitals:** 0.000 for A→B and +0.007 for B→A, which is within seed noise. Two reasons:
  - ordering behaviour does not transfer between hospitals, so a pattern learnt in one hospital buys little in the other;
  - most of the signal is already reachable through `hours_since_<panel>` and the NaN branches of the raw lab columns (§4).

  The diagnostic does not measure the within-hospital (mix) ordering signal; rank #1's setting resembles that one more.
- **Masking helps B2 transfer.** The p = 0 reference scores 0.300 / 0.306, against b2-001's 0.309 / 0.314 at `lab_masking_p = 0.3`. So the training-time masking (ADR-0002) costs nothing across hospitals and probably helps by about 0.01.

## 6. Comparison with rank #1

| | ours (b2-001) | rank #1 |
|---|---|---|
| within-distribution | mix→A 0.426, mix→B 0.363 | 0.430 (5-fold CV, A + B pooled) |
| out-of-distribution | A→B 0.309, B→A 0.314 | 0.360 (hidden test, includes unseen hospital C) |

Our mix→A matches their CV, and our mix→B is lower. The comparison is not like for like:
- **Per-hospital vs pooled.** We report mix per hospital, while they report one pooled number.
- **Lab information.** ADR-0002 forbids lab counts and missingness features, and we train with `lab_masking_p` 0.3. §5 suggests both cost little across hospitals.
- **Threshold protocol.** Our alarm threshold is tuned on out-of-fold predictions of the training side only and then frozen.
- **No hospital C.** Their 0.360 test score includes the unseen hospital C. Our cross-hospital 0.31 is the nearest analogue, and C's distribution is unknown.

## 7. Next

- Seed repeats of b2-001 and b2-003 would show whether the 0.0002 miss and the +0.012 B→A gain are real (out of scope for M3-3).
- M2-4 features and M4 signatures (B3) are judged against b2-001.
- Pass `HospAdmTime` and the `hours_since_*` effects to the leakage-reviewer before the final evaluation. They are allowed features, but they are hospital-specific.
