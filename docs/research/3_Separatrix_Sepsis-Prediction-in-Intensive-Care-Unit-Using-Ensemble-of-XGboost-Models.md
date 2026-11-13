# #3 Separatrix — Sepsis Prediction in Intensive Care Unit Using Ensemble of XGboost Models

Notes for M0-5 (top-4 review). Only what is useful for reuse here.

## Source
- Paper: Zabihi, Kiranyaz, Gabbouj (Tampere University, Qatar University). CinC 2019 write-up, `papers/#3-Separatrix/Separatrix.pdf` (4 pages).
- Code: `papers/#3-Separatrix/CinC2019_5/` (`get_sepsis_score.py`, `Feature_MZ_PredictionMode_ver22_1.py`, two imputation files, pickled models). No training script is included.
- Reported official utility: **0.339** on the full test set, 3rd place. By hospital: **A 0.422, B 0.395, C −0.146** (paper Section 3).
- Source tags: **[P]** stated in the paper, **[C]** seen in the code, **[P≠C]** disagree. The code is the reference when they disagree.
- Table 2 was read from the PDF by the project owner and matches the per-fold numbers below; the text extraction of the table was garbled.

## Pipeline in one view
1. Impute 38 covariates (mean fitted on training data), then linearly interpolate the vitals and admin columns. [P, C]
2. Compute **407 features** per hour: 245 from the 13 imputed covariates (vitals and demographics), 162 from missingness patterns of 38 covariates (vitals **and all 26 labs**). [P, C]
3. Standardise (scaler fitted on training features). [C]
4. Five XGBoost classifiers, each with its own selected feature subset, give five probabilities. The geometric mean is the score. [P, C]
5. Alarm if score > **0.5**. [C]
6. The code recomputes all features from the rows up to the current hour. [C]

## The key point for this project
- **The model never sees a lab value.** The 13 covariates with imputation-based features are the 7 vitals, Age, Gender, Unit1, Unit2, HospAdmTime and ICULOS. The only way labs enter is through missingness features (the 38-covariate set includes the 26 labs). [P, C: `values5` has 11 columns, all vitals or admin]
- So a model that reaches 0.339 on the test set (0.422 on hospital A) uses lab **ordering behaviour** and no lab **results**. That fits the M0-4 finding that labs are ordered when clinicians are worried, and it is exactly what ADR-0002 forbids. This result cannot transfer to replay, where we control which labs are ordered.
- It also shows how strong the ordering signal is. A sensible use is as a diagnostic: how much of the utility of a model comes from ordering behaviour.

## Preprocessing
- `replace_NaN_MZ_complex`: for the 38 covariates (columns 0–33 and 36–39), missing values are replaced by the **training mean** (sklearn `SimpleImputer`, fitted on training data and stored). Age and Gender take the first valid value, or the mean if none. [C]
- `replace_NaN_MZ_interpol`: for the 7 vitals and Unit1, Unit2, HospAdmTime, ICULOS, linear interpolation between observed values (pandas default, which also carries the last value forward). Leading NaN, and series shorter than 3 rows, fall back to the mean-imputed value. [C]
- In the deployed code the series ends at hour t, so interpolation only uses values up to t. **The training script is not included**, so I cannot confirm that the training features were built from series truncated at each hour. If they were built on whole records, interpolation would have used future values at training time. [C, not verifiable]
- Labs are never imputed for features: the missingness features use the raw NaN pattern. [C]

## Features (407 in total)
| Group | Content | Count | Source |
|---|---|---|---|
| Window features | Mean, min, max, median, variance, 95/99/5/1% quantiles over the last **5** and last **11** hours, for 11 columns (7 vitals + Unit1, Unit2, HospAdmTime, ICULOS) | 2 × 99 = 198 | [P, C] |
| Non-window features | Energy (sum of squares), Shannon entropy (Gaussian kernel density), mean of first differences, all over the **whole history up to hour t** (padded to 3 rows if shorter), so energy and entropy grow with length of stay; plus the number of observations so far | 33 + 1 = 34 | [P, C] |
| Last values | Last value of the 13 covariates | 13 | [P, C] |
| Missingness (sequence abstraction) | Per covariate: mean and variance of the lengths of runs of present/missing values; fraction of hours present and its variance; plus, for each of the last 5 hours, the share of covariates present in that hour and its variance (10 values) | 162 | [P, C] |

- Short histories are padded with the first observation so the window features have a fixed length. [P, C]
- Paper reports that 102 of the 162 missingness features were commonly selected by feature selection, and that HospAdmTime, the summed valid-run length of Temp, Age, Unit1 and the variances of HR and Temp (last 11 h) were in the top 10. [P]

## Model and target
- Five XGBoost classifiers (`predict_proba`), target is the binary label (not the utility difference). [P, C]
- Training: 10% of the training data is used for feature selection and grid search (5-fold). The remaining 90% is split into five disjoint parts, one per model, each **randomly undersampled** to balance sepsis and non-sepsis rows. [P]
- Feature selection: BoostARoota (a wrapper using XGBoost split counts). [P] Each model has its own subset (`FS0`..`FS4`). [C]
- The shipped code loads only "fold 0" of the final models. Hyperparameters were not inspected (pickles not loaded).

## Threshold
- Fixed **0.5** on the geometric mean of the five probabilities. No tuning on utility. [C]
- Undersampling raises predicted probabilities, so 0.5 acts as a high-recall alarm level. The paper does not discuss the threshold. [P, C]

## Validation
- 5-fold CV on the pooled training data (Table 2, confirmed by the project owner). [P]

  | Fold | AUROC | ACC | Score |
  |---|---|---|---|
  | 0 | 0.8387 | 0.8394 | 0.4366 |
  | 1 | 0.8357 | 0.8418 | 0.4412 |
  | 2 | 0.8436 | 0.8477 | 0.4521 |
  | 3 | 0.8221 | 0.8451 | 0.3899 |
  | 4 | 0.8268 | 0.8464 | 0.4208 |
  | Mean (std) | 0.8333 (0.0078) | 0.8440 (0.0030) | **0.4281 (0.0215)** |

- Fold 3 (0.3899) is the weakest; the spread across folds (std 0.0215) is larger than most single-feature gains reported by the other teams.
- Official test: full 0.339; A 0.422 (AUROC 0.814); B 0.395 (AUROC 0.844); **C −0.146 (AUROC 0.793)**. [P]

## Hidden hospital / cross-hospital
- Utility on hospital C is strongly negative, but AUROC on C (0.793) is close to A (0.814) and B (0.844). The ranking still works; the alarm decision does not. So the fixed threshold, not the ranking, is what failed to transfer. This is my reading of the numbers; the authors do not say it.
- The authors' explanation: missingness in hospital C follows a different pattern, and missingness reflects recording behaviour, not medical information. They add that **all contestants fail on C**, even those not using missingness. [P]
- Sepsyd (#2) shows the same shape: C utility −0.042 with AUROC 0.805, close to A and B.
- For this project: tune the threshold on the training side and report A→B and B→A, because a threshold fitted on one hospital may not carry over (ADR-0004).

## ADR-0002 check (features that depend on which labs were ordered)
- **Fail:** all 162 missingness features (sequence lengths, fraction present, per-hour share of covariates present). They are the paper's main contribution and are what ADR-0002 excludes.
- **Fail (indirect):** the number of observations so far is fine, but "lengths of runs" over the 26 lab columns encode ordering rhythm.
- **OK:** window statistics, last values, energy, entropy and differences of **vitals**, and of Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS.

## Use / adapt / reject
| Item | Verdict | Reason | Issue |
|---|---|---|---|
| Window statistics (mean, min, max, median, var, quantiles) of vitals over a short and a long window (5 h and 11 h) | **adapt** | Vitals are exempt from ADR-0002. Windows go in config (our default is 6 h); test a long window beside the short one | M2-2 |
| Energy, Shannon entropy, mean first difference of vitals | **adapt** | Cheap except the kernel-density entropy. Their versions run over the whole stay and so track length of stay; compute them over a fixed window from config, and add only if it helps in an ablation | M2-2 |
| Last observed values, ICULOS, HospAdmTime, Unit1/Unit2, Age, Gender | **use** | Already in the recipe (raw values); HospAdmTime appears in their top 10 | M2-3 |
| Missingness / sequence-length features | **reject** | Violates ADR-0002. Keep only as an optional diagnostic of how much utility comes from ordering behaviour | — |
| Mean imputation fitted on the training side | **adapt** | For B1 logistic only; trees keep NaN (ADR-0001) | M0-6, M3-2 |
| Linear interpolation of vitals | **reject** | Forward-fill is enough in replay; interpolation is only safe if built from rows ≤ t, and the training code is not available to confirm that | M0-6 |
| BoostARoota feature selection | **adapt (optional)** | Fit only on the training side and never on rows used for evaluation; unclear in the paper whether the 10% subset was excluded from CV | M3-3 |
| Ensemble of 5 models on disjoint parts with undersampling, geometric mean | **adapt (optional)** | Possible variance reduction for B2 later; our `U1 − U0` regression target does not need class balancing | M3-3 |
| Fixed 0.5 threshold | **reject** | Tune on the training side, then freeze (hard rule); their C result shows a fixed threshold does not carry across hospitals | M1-1 |
| Pooled CV as the reported result | **reject as sole protocol** | C utility −0.146 vs 0.42 on A | M3-1, M8-1 |

## Open points to check against the paper
- Table 2 (resolved): per-fold scores and means confirmed by the project owner.
- Whether the training features were built on series truncated at each hour (training script not included).
- Number of models and feature subsets in the pickles (paper says five; not read).
- Whether the 10% subset used for feature selection and tuning was held out from the 5-fold CV.
