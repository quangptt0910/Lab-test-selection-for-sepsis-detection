# #0 SailOcean — Early Prediction of Sepsis Using Multi-Feature Fusion Based XGBoost Learning and Bayesian Optimization

Notes for M0-5 (top-4 review). Only what is useful for reuse here. `0` marks the team with the highest **unofficial** score (not in the official ranking).

## Source
- Paper: Yang, Wang, Gao, Li, Liu, Li, Liu (Southeast University). CinC 2019 write-up, `papers/##-SailOcean/SailOcean.pdf` (4 pages).
- Code: `papers/##-SailOcean/PhysioNet-CinC-2019-Challenge-Submit10/` (`get_sepsis_score.py` plus five XGBoost model files). No training script is included.
- Reported hidden-test U-Score: full **0.364** (24,819 patients, three hospital systems); by hospital **A 0.430, B 0.422, C −0.048** (Table 4). The paper calls this the highest unofficial score.
- **The threshold 0.522 was chosen using feedback from the hidden test set** (see Threshold). The 0.364 is therefore not comparable with the other teams' numbers.
- Source tags: **[P]** stated in the paper, **[C]** seen in the code, **[P≠C]** disagree. The code is the reference when they disagree.
- Tables 1 and 3 were read from the PDF by the project owner and are used as given below; the text extraction of both tables was garbled.

## Pipeline in one view
1. Drop 3 columns that are more than 99% missing (Bilirubin_direct, TroponinI, Fibrinogen); forward-fill the rest; NaN stays before the first value. [P, C]
2. Per hour build **168 features** from the rows up to the current hour: 37 last values, 62 missing-value features, 31 differences, 30 window statistics, 8 clinical scores. [P, C]
3. Five XGBoost classifiers (one per CV fold) each output a probability. The final score is their mean. [P, C]
4. Alarm if score > **0.522**. [P, C]
5. The code recomputes everything from all rows up to hour t, so it is causal. [C]

## Preprocessing
- Step 1: remove Bilirubin_direct, TroponinI, Fibrinogen (missing more than 99% in the paper). [P, C: columns 20, 27, 32]
- Step 2: forward-fill within patient. Leading NaN are not filled. [P, C]
- No scaling, no clipping. [C]
- Paper's data notes: 2,932 septic patients of 40,336 (7.27%); septic patients have longer records (58.6 vs 38.9 h); Temp 66% missing, DBP 31%, EtCO2 96%, Glucose 83%, other labs over 90%. [P] Our inspect output gives the same picture for Temp (66%) and the lab sparsity.

## Features (168)
| Group | Content | Count | Source |
|---|---|---|---|
| Last values | Last forward-filled row after the 3 columns are dropped: 8 vitals, 23 labs, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS | 37 | [P, C] |
| Missing-value features | For each of 31 non-demographic variables: (1) number of measurements so far, (2) hours since the last measurement (−1 or 0 count if none) | 62 | [P, C] |
| Differences | For the same 31 variables: last recorded value minus the previous recorded value (NaN if fewer than two) | 31 | [P, C] |
| Window statistics | HR, O2Sat, SBP, MAP, Resp: max, min, mean, median, std, std of first differences over the window | 30 | [P, C] |
| Scores | 8 scores from the current row (below) | 8 | [P, C] |

- Check: 37 + 62 + 31 + 30 + 8 = 168. The count from the code is certain. The model files appear to agree: the value 168 sits where the feature count is stored in the file header. That is inferred from raw header bytes; the models were not loaded. [C]
- **Window length:** the paper says "six-hour window", and the example fills a 4-hour record to `t = [0,1,2,3,3,3,3]` (7 entries). The code uses the last **7 rows** (t−6..t) and pads short records by repeating the last row. So "6 hours" means 6 intervals, 7 rows. [P, C]
- The 8 scores [C]:
  - NEWS-style points for HR, Temp and Resp. (The paper names only HR and Temp for NEWS.) [P≠C]
  - SOFA-style points for Creatinine (0–3), MAP (1 point if < 70), Platelets (0–3), Bilirubin (0–3).
  - qSOFA flag, set only when **both** SBP ≤ 100 and Resp ≥ 22 (the clinical rule needs two of three criteria, one of which is mental status).
- **Latent bug in the code:** the scores test `value == np.nan`, which is always false in Python. A missing value therefore falls through to the final `else` of each chain instead of giving NaN, and that branch differs per score: HR, Temp, Resp, Platelets and qSOFA give **0** (normal), MAP gives **1**, and Creatinine and Bilirubin give **3** (the worst score). Only hours before the first measurement are affected, because the series is forward-filled first. [C]
- Consequence: until the first creatinine or bilirubin value arrives, those two scores sit at 3, so they act as "lab never measured" flags. Bilirubin is never measured for most patients, so this is a large share of hours.

## Model and target
- XGBoost, objective `binary:logistic`; target is the binary label (not `U1 − U0`). [P]
- Hyperparameters tuned by Bayesian optimisation (Tree-structured Parzen Estimator): max depth, learning rate, colsample_bytree, regularisation. Values are not given, and the paper does not say which objective the tuner minimised. [P]
- 5-fold CV produces five models. The majority class is **down-sampled** in each fold; early stopping is used. The five probabilities are averaged. [P]
- The final submitted models were trained "on the entire 40,336 patients via 5-fold cross validation", so the local test patients were merged back in. [P]

## Threshold
- Local experiments use 0.50. A sweep from 0.49 to 0.55 on the local test set (Figure 3, not recoverable from the text) showed an effect on utility. [P]
- The submitted threshold **0.522 was "determined … after testing our model on a subset of the hidden test set during the official phase"**, found "through the official submissions". [P] The code uses `score > 0.522`. [C]
- That is tuning on test feedback, which our rules forbid.

## Validation
- Patient-level partition, stratified by septic status, A and B pooled (Table 1, confirmed by the project owner). [P]

  | Set | Septic | Non-septic | Total (as printed) |
  |---|---|---|---|
  | 5-fold train | 1,994 | 25,435 | 27,428 |
  | 5-fold validation | 498 | 6,358 | 6,856 |
  | Local test | 440 | 5,611 | 6,051 |

- **Arithmetic check:** the train row adds up to 27,429 (1,994 + 25,435), not the printed 27,428. With 27,429 the three sets sum to the stated 40,336 patients and 2,932 septic (7.27%); with 27,428 they sum to 40,335. So the printed train total is an off-by-one typo in the paper. The local test is 15% of patients, the validation fold 17%, train 68%.
- Local test (threshold 0.50, Table 3, confirmed by the project owner): individual models ACC 0.812, AUROC 0.840, U-Score 0.411 on average (0.406–0.419); ensemble ACC 0.818, AUROC 0.847, U-Score **0.425**. [P]
- Hidden test: A 0.430, B 0.422, **C −0.048**, full 0.364. [P]

## Hidden hospital / cross-hospital
- The local test (0.425) and hospitals A and B (0.430, 0.422) agree, so the pooled split is a fair estimate **for hospitals seen in training**.
- Hospital C, not seen in training, gives −0.048. The paper does not comment on it. [P]
- Same pattern as the other two teams that report by hospital: Sepsyd (C −0.042) and Separatrix (C −0.146). Pooled-CV utility does not predict utility on a new hospital.

## ADR-0002 check (features that depend on which labs were ordered)
- **Fail:** number of measurements so far, for all 26 labs (and vitals). It is a count and encodes how often labs were ordered. The paper's own motivation is that septic patients have fewer missing values because clinicians order more tests.
- **Allowed, and close to our plan:** hours since the last measurement. This is the same idea as `hours_since_<panel>` (ADR-0002), provided it is computed per panel from revealed labs only.
- **Allowed with care:** the difference between the last two recorded values uses lab values, not counts. It depends on how often a lab is revealed in replay, so it must be computed on revealed values only.
- **Fail as implemented:** the Creatinine, Bilirubin and MAP scores, because a missing value gets a fixed non-zero score (see the bug above) and so marks "never measured". The idea passes if a missing input gives NaN.
- **OK:** window statistics of vitals, last values, and the HR, Temp, Resp, Platelets and qSOFA scores.

## Use / adapt / reject
| Item | Verdict | Reason | Issue |
|---|---|---|---|
| Forward-fill, NaN before the first value | **use** | Same as ADR-0001 | M2-1 |
| Window statistics of 5 vitals (max, min, mean, median, std, std of differences) | **adapt** | Vitals are exempt. Take the window from config; note this paper's "6 h" is 7 rows | M2-2 |
| Hours since last measurement | **adapt** | Becomes `hours_since_<panel>` from revealed labs only | M2-2, M2-3 |
| Measurement count | **reject** | Violates ADR-0002 | — |
| Difference between the last two recorded values | **adapt** | On revealed values only; test as an option | M2-2 |
| Individual clinical component scores (NEWS-style HR/Temp/Resp, SOFA-style Creatinine/MAP/Platelets/Bilirubin, qSOFA) as separate features | **adapt** | Complements PartialSOFA, SIRS count and shock index. Give NaN when the input is missing; their code gives a fixed score instead (3 for Creatinine and Bilirubin), which turns the score into a "never measured" flag | M2-2 |
| Bayesian optimisation (TPE) for XGBoost/LightGBM hyperparameters | **adapt** | Tune on the training side's CV folds only, with the official utility as the objective | M3-3 |
| Down-sampling the majority class | **adapt (optional)** | Our `U1 − U0` regression target does not need it; test only with a classification baseline | M3-2 |
| Average of the five fold models | **adapt (optional)** | Fits our nested CV: the fold models are already trained; averaging reduces variance | M3-3 |
| Threshold 0.522 chosen on hidden-test feedback | **reject** | Tune on the training side, then freeze (hard rule) | M1-1 |
| Final model fitted on all data including the local test set | **reject for development** | Test parts stay locked until the final evaluation (ADR-0004) | M8-1 |

## Window conventions across the four teams (for M2-2)
Different teams count look-back differently. Fix one definition in `CONTEXT.md` before writing config values.
- #1: max/min `rolling(5)`, 5 rows including the current hour; signature path look-back 7 = 8 rows.
- #2: current hour + previous 5 hours = 6 rows.
- #3: windows of 5 and 11 rows.
- #0: "6-hour window" = 7 rows (t−6..t).

## Open points to check against the paper
- Tables 1 and 3 (resolved): confirmed by the project owner; only the printed train total 27,428 is off by one (components add to 27,429).
- Figure 3 (utility vs threshold from 0.49 to 0.55) is not in the extracted text.
- Hyperparameter values and the objective of the Bayesian optimiser are not stated.
- Tree settings of the shipped models were not read (only the header with the feature count).
- Hidden test size: this paper says 24,819 patients, the Separatrix paper says 22,761. Not resolved.
