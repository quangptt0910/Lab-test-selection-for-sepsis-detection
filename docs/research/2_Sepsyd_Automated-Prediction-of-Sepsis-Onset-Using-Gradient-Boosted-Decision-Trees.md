# #2 Sepsyd — Automated Prediction of Sepsis Onset Using Gradient Boosted Decision Trees

Notes for M0-5 (top-4 review). Only what is useful for reuse here.

## Source
- Paper: Du, Sadr, de Chazal (University of Sydney). CinC 2019 write-up, `papers/#2-Sepsyd/Sepsyd.pdf` (4 pages).
- Code: `papers/#2-Sepsyd/submit-cinc2019/` (`get_sepsis_score.py` plus a pickled XGBoost model; no training code is included).
- Reported official utility: **0.345** on the full test set, 2nd place (paper abstract). Table 4 splits it by hospital: **A 0.409, B 0.396, C −0.042**.
- Source tags: **[P]** stated in the paper, **[C]** seen in the code, **[P≠C]** paper and code disagree. When they disagree, the code is the reference (project rule for this review).
- Table 3 (H=6 row) and Table 4 were read from the PDF by the project owner and corrected here; the text extraction of both tables was garbled.

## Pipeline in one view
1. Per patient, log-transform 12 skewed columns, then standardise with fixed training means/stds. [P, C]
2. Replace NaN by the training mean until the first valid value, then forward-fill. [P, C]
3. Per hour build 120 features: 40 values, 40 one-hour differences ("delta"), 40 present/missing flags ("mask"). [C]
4. Concatenate the current hour with the previous 5 hours (6 × 120 = **720 features**); zeros before the start of the stay. [C]
5. XGBoost classifier, one probability per hour; alarm if probability > **0.5**. [C]
6. The code recomputes everything from all rows up to the current hour, so it is causal. [C]

## Preprocessing
- Log-transformed columns (indices in `get_sepsis_score.py`): AST, BUN, Alkalinephos, Creatinine, Bilirubin_direct, Lactate, Potassium, Bilirubin_total, TroponinI, PTT, WBC, Fibrinogen. [C] The paper only says "selected variables with an exponential or long-tailed distribution". [P]
- Standardisation: `10 × (x − mean) / std`, with hard-coded means and stds (log-scale for the 12 columns). I did not check which data they were fitted on. [C]
- Missing values: mean until the first valid value, then last valid value. [P, C] ICULOS is not standardised (the loop stops at index 38). [C]
- **Effect measured in the paper (H=5, Table 1):** no preprocessing 0.3931, standardising 0.3926, log + standardising 0.3933. So preprocessing made **no real difference** (a tree model does not need it). [P]
- The code defines `varmaxes` and `varmins` arrays but does not use them: no clipping of extreme values. [C]

## Features
| Feature | Definition | Source |
|---|---|---|
| Base | 40 input columns after steps above (8 vitals, 26 labs, 6 demographic/admin columns including ICULOS) | [P, C] |
| Mask | 1 if the value was present in that hour, 0 if it was imputed. The paper says 34 flags (not the demographics); the code builds flags for all 40 columns | [P≠C] |
| Delta | Difference from the previous hour, computed **after** forward-fill, **for all 40 columns**, not standardised. The paper says delta covers only the 8 vitals, and that the submitted model did not use it | [P≠C] |
| Variance | Variance of each vital over 6 h. Described in the paper, **not in the code** | [P≠C] |
| Window | Current hour + previous 5 hours, zero-padded before t=0 | [P, C] |

- **Which version was submitted:** the paper says Base + Mask with H=5 (about 74 features per hour). The pickled booster stores **720 feature names (`f0`..`f719`)**, which equals 6 hours × (40 + 40 + 40). So the submitted model used base, delta and mask for all 40 columns, as in the code. I read only the feature-name strings with `pickletools`; I did not unpickle the model, so its tree settings are not checked. [C]
- Paper ablation (Table 3, 3-fold CV with 3 shuffles), best row **H=6**: Base+Mask 0.3932, Base+Mask+Delta 0.3935, Base+Mask+Var 0.3956, All 0.3959. So delta adds +0.0003 and variance +0.0024 over Base+Mask; other rows (H=3..9) stay within about 0.391–0.396. Differences of 0.001–0.003 are the same size as the shuffle-to-shuffle noise the authors report. [P, Table 3, confirmed by the project owner]
- The Discussion text says H=6 with mask and variance averaged 0.3981 on the hold-out fold, which does not match the table value 0.3956. The paper does not explain it (possibly a different averaging).
- The paper never reports a model **without** the mask flags, so it cannot tell us how much the flags contribute.

## Model and target
- XGBoost classifier, target is the binary label (**not** the utility difference). [P, C]
- Loss: cross-entropy with positive examples weighted **40:1**, to mimic the utility (false negatives cost far more than false positives). [P]
- Depth 4 was best in the 3-fold comparison (depth 3: 0.3912, 4: 0.3933, 5: 0.3881). [P] Up to 30 trees, 40 rounds at learning rate 0.2 for comparisons. [P]
- Final model: 10-fold CV on all training data, learning rate 0.1, early stopping on the validation **AUPRC** of each fold. [P] One booster is shipped; which fold it came from is not stated. [C]

## Threshold
- Fixed **0.5** on the predicted probability. No tuning on utility. [C]
- The 40:1 class weight pushes the probabilities up so that 0.5 is a workable alarm level. The paper does not discuss the threshold. [P, C]

## Validation
- Model selection: stratified 3-fold CV over A and B pooled, repeated with 3 shuffles because hold-out scores "varied significantly" with the split. [P]
- Final: 10-fold CV, utility 0.455 on the training folds and **0.400** on the held-out folds (A 0.416, B 0.376). [P, Table 4]
- Official test: full 0.345, A 0.409, B 0.396, **C −0.042**. [P, Table 4]
- **Submitted vs best model.** The best CV model (H=6, mask + variance, no delta) was "not able to be submitted for testing in the official phase". The paper gives **no reason**. The submitted model was H=5, Base+Mask (CV 0.3933; 10-fold 0.4004). The H=6 gain (0.3956–0.3959 vs 0.3932 for Base+Mask at H=6) is about +0.002, the same size as the split-to-split noise the authors report, so it is not evidence that the unsubmitted model would have scored better. [P]
- **Hackathon result.** The same core model with extra features (SOFA, NEWS, acceleration of time variables, multi-hour time differentials) scored 0.329 in the hackathon (third highest there), below the official 0.345. The authors conclude the extra features "don't enhance discrimination". The paper does not say whether both scores come from the same test data, so this is weak evidence: the two numbers may not be comparable. [P]

## Hidden hospital / cross-hospital
- The model scores about 0.40 on test hospitals A and B and **−0.042 on hospital C**, which was not in training. The paper does not comment on this gap. [P]
- No hospital feature and no hospital-specific handling. Training was on A and B pooled. [P, C]
- This is the clearest evidence in the four papers that a pooled-CV score overstates performance on a new hospital. It supports reporting A→B and B→A next to mix (ADR-0004).
- Held-out training score 0.400 vs official 0.345: a drop of 0.055 even on the hospitals seen in training, so part of the gap is overfitting (the authors note the 0.46 vs 0.345 difference themselves). [P]

## ADR-0002 check (features that depend on which labs were ordered)
- **Fail:** the mask flags for labs. They are missing-value indicators, exactly what ADR-0002 forbids.
- **Fail:** delta of forward-filled labs. After forward-fill, a lab's delta is non-zero only in the hour a new value arrives, so it encodes the order time. The previous 5 hours of flags and deltas also give a short history of recent ordering, which our `hours_since_<panel>` replaces.
- **OK:** flags and deltas of vitals (vitals are exempt), variance of vitals, the window of vital values.
- So part of this model's score may come from ordering behaviour. Its 0.345 cannot be taken as evidence for a replay setting.

## Use / adapt / reject
| Item | Verdict | Reason | Issue |
|---|---|---|---|
| Concatenate the previous hours (lag features) | **adapt** | Only for vitals and derived channels; window length in config (paper: 5 previous hours) | M2-2, M2-3 |
| Variance of vitals over a window | **adapt** | Small gain (about +0.002, within noise); vitals are allowed | M2-2 |
| Delta (1-hour difference) of vitals | **adapt** | Paper found no gain; test as an option | M2-2 |
| Delta of labs and mask flags of labs | **reject** | Violates ADR-0002; use `hours_since_<panel>` | — |
| Mask flags of vitals | **adapt** | Allowed, but the paper gives no evidence they help | M2-2 |
| Log transform + standardising | **adapt** | No gain for trees (Table 1); needed only for B1 logistic, with statistics fitted on the training side | M0-6, M3-2 |
| Mean fill before the first value, zero padding at t=0 | **reject** | We keep NaN before the first value (ADR-0001); B1 needs its own fill fitted on training data | M0-6 |
| Class weight 40:1 on a binary target | **adapt** | Use as a comparison for B1/B2 against our `U1 − U0` regression target | M3-2 |
| Fixed 0.5 threshold | **reject** | We tune on the training side and freeze (hard rule) | M1-1 |
| Shallow trees (depth 4), early stopping on AUPRC | **adapt** | Hyperparameter starting point for B2 (LightGBM equivalent) | M3-3 |
| Repeat CV with several shuffles to see noise | **adapt** | Use our stored folds and report the spread over folds, so small gains are not over-read | M3-3 |
| Pooled-CV score as the reported result | **reject as sole protocol** | Their own hospital C result (−0.042) shows the gap | M3-1, M8-1 |

## Open points to check against the paper
- Table 3 (resolved): H=6 row is 0.3932 / 0.3935 / 0.3956 / 0.3959. Still open: the Discussion's 0.3981 for the same configuration.
- Table 4 (resolved): columns are 10-fold CV Train (Full, A, B), Test (Full, A, B), then Official (Full, A, B, C). Official utility 0.345 / 0.409 / 0.396 / −0.042. For accuracy, F measure, AUROC and AUPRC the official "Full" value is "−"; official AUROC is A 0.811, B 0.853, C 0.805, and AUPRC A 0.105, B 0.119, C 0.065.
- Paper says the final model is Base + Mask; the pickle has 720 features (base + delta + mask for 6 hours). I trust the pickle.
- XGBoost hyperparameters of the shipped model were not read (pickle not loaded).
