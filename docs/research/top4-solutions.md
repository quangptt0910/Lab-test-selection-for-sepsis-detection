# Review of four PhysioNet/CinC 2019 solutions (M0-5)

Date: 2026-11-13. Inputs to M0-6 (preprocessing recipe and panels) and M2-2 (clinical features).

Per-team notes, with all details and sources:
- [#1 Can I get your Signature?](1_CanIgetyoursignature_The-Signature-Based-Model-for-Early-Detection-of-Sepsis-From-EHR-in-the-ICU.md)
- [#2 Sepsyd](2_Sepsyd_Automated-Prediction-of-Sepsis-Onset-Using-Gradient-Boosted-Decision-Trees.md)
- [#3 Separatrix](3_Separatrix_Sepsis-Prediction-in-Intensive-Care-Unit-Using-Ensemble-of-XGboost-Models.md)
- [#0 SailOcean](0_SailOcean_Early-Prediction-of-Sepsis-Using-Multi-Feature-Fusion-Based-XGBoost-Learning-and-Bayesian-Optimization.md) (highest unofficial score)

## How to read this
- Scores and ranks are each team's own claim. They were **not** checked against the official challenge paper (Reyna et al.), which is not in `papers/`.
- Where a paper and its code disagree, the code is the reference and the paper's value is kept as a note.
- The model files were never loaded. Only feature names and file headers were read.
- The papers and the teams' code are in `papers/`, which is local only and not committed (it contains model files and a patient file).
- ADR-0002 test applied to every feature: does it depend on which labs a clinician ordered?

## 1. Comparison of the four solutions
| | #1 Can I get your Signature? | #2 Sepsyd | #3 Separatrix | #0 SailOcean |
|---|---|---|---|---|
| Official / hidden test utility (full) | **0.360** (101 submissions) | **0.345** | **0.339** | 0.364 (unofficial; threshold tuned on test feedback) |
| By hospital A / B / C | not reported | 0.409 / 0.396 / **−0.042** | 0.422 / 0.395 / **−0.146** | 0.430 / 0.422 / **−0.048** |
| Pooled CV or local test | 5-fold CV 0.430 (std 0.018) | 10-fold held-out 0.400 (A 0.416, B 0.376) | 5-fold CV 0.4281 (std 0.0215) | local test 0.425 (ensemble) |
| Pooled CV minus official full | 0.070 | 0.055 | 0.089 | 0.061 |
| Missing values | Forward-fill; NaN before first value | Mean until first value, then forward-fill; log + standardise | Training mean, then linear interpolation of vitals | Drop 3 columns over 99% missing; forward-fill; NaN before first value |
| Features | ShockIndex, BUN/CR, PartialSOFA + deterioration, rolling max/min of vitals, moments, lab counts, log-signatures (order 3) | Per hour: 40 values, 40 one-hour differences, 40 present flags; current + 5 previous hours (720 features) | 407: windows (5 h, 11 h) of vitals and admin columns, energy, entropy, **162 missingness features** | 168: 37 last values, 62 count + time-since-last, 31 differences, 30 window stats of 5 vitals, 8 clinical scores |
| Lab **values** used | Yes | Yes | **No** (only whether and when labs were present) | Yes |
| Model | LightGBM regressor, heavy regularisation | XGBoost classifier, depth 4, early stopping on AUPRC | 5 XGBoost classifiers on different feature subsets, geometric mean | 5 XGBoost classifiers (fold models), mean |
| Training target | **U1 − U0** (utility difference) | Binary label, positives weighted 40:1 | Binary label, undersampled | Binary label, majority class down-sampled |
| Threshold | Gradient-free search on utility (out-of-fold in CV; in-sample in the final-fit function) | Fixed 0.5 | Fixed 0.5 | **0.522, chosen from hidden-test feedback** |
| Hyperparameter tuning | Random search | Grid over depth; 3-fold, 3 shuffles | Grid search + BoostARoota feature selection | Bayesian optimisation (TPE) |
| ADR-0002 | **Fail** (count features); rest OK | **Fail** (lab flags, lab differences) | **Fail by design** (all 162 missingness features) | **Fail** (count); time-since-last is the allowed analogue |
| Hospital shift handling | None | None | None (authors blame missingness pattern in C) | None |

## 2. Critical analysis: #1 and #3

### #1 — the basis of our recipe (ADR-0001)
What it gives us: the `U1 − U0` regression target, forward-fill, ShockIndex, BUN/CR, PartialSOFA, rolling max/min.

Limits of the evidence:
- **Its score includes ordering behaviour.** The paper's "hand-crafted" row (0.418) and the full model (0.430) contain measurement counts. Those numbers are therefore not what a replay-valid B2 should be expected to reach. M3-3 already asks to explain the gap to the published numbers; ADR-0002 is the first reason.
- **Signatures add little.** In the paper's own ablation they add about +0.012 on top of hand-crafted features (0.418 → 0.430), inside the fold std of 0.018. B3 should stay optional (M4-1).
- **Paper and code disagree** on the signature channels (paper: PartialSOFA, MAP, BUN/CR; code: SOFA, BUN/CR, SaO2/FiO2), on whether cumulative-sum signatures were used, and on the max/min window (paper 6, code 5 rows). The `rank1-features` skill and M4-1 follow the paper's channels. M4-1 should test both sets (how-to in the #1 file).
- **Threshold.** In CV it is tuned on out-of-fold predictions, which is what we do. In the final-fit function it is tuned on in-sample predictions, which we reject.
- **No hospital breakdown.** The 0.070 drop from CV (0.430) to the test set (0.360) is unexplained in the paper. The other teams' tables (below) suggest hospital C.

### #3 — the extreme case for ADR-0002
What it shows: a model with **no lab values** reaches 0.339 overall and 0.422 on hospital A. The signal from when labs were ordered is, by itself, enough for a top-three result.

Limits of the evidence:
- **It cannot transfer to replay.** In replay we decide which labs are ordered, so the 162 missingness features are not available in any usable form. Nothing from this paper's headline score should be used as a target for our results.
- **The explanation for hospital C is untested.** The authors say missingness in C follows a different pattern, but they run no experiment on it. They also note that all contestants fail on C, including those who use no missingness features.
- **Ranking versus decision.** On C the utility is −0.146 while AUROC is 0.793 (A 0.814, B 0.844). Sepsyd shows the same shape (C −0.042, AUROC 0.805 vs 0.811 on A). The ranking survives; the fixed alarm threshold does not. This is my reading of the numbers, not a claim by the authors.
- **Training code missing.** I cannot confirm that training features were built from series cut at each hour (interpolation would use future values otherwise).

### What the four teams show together
- **Pooled CV is close to the official score on the hospitals seen in training**, and the gap is hospital C:

  | Team | Pooled CV / local | Official A | Official B | Official C |
  |---|---|---|---|---|
  | #2 | 0.400 (A 0.416, B 0.376) | 0.409 | 0.396 | −0.042 |
  | #3 | 0.428 | 0.422 | 0.395 | −0.146 |
  | #0 | 0.425 | 0.430 | 0.422 | −0.048 |

  So our `mix→A` and `mix→B` should resemble "seen hospital" scores, and `A→B` and `B→A` are our stand-in for a new hospital (ADR-0004).
- **No team tuned its threshold on a held-out hospital.** Thresholds were fixed at 0.5, tuned in-sample, or tuned on test feedback. Whether a threshold transfers is exactly what our cross-hospital setups measure (M1-1).
- **Differences below about 0.01 are noise.** #1's fold std is 0.018, #3's 0.0215, and #2 reports that its hold-out score varies with the shuffle. Single-feature gains of 0.002–0.012 in these papers should not be read as real effects.
- **Look-back conventions differ:** #1 max/min 5 rows (signature path 7 = 8 rows); #2 current + 5 previous rows; #3 5 and 11 rows; #0 "6 h" = 7 rows. Fix one definition before writing config values (M2-2).

## 3. Use / adapt / reject
Issues are named as in `scripts/create_github_issues.py`; numbers are those listed in the project handoff.

| Item | Teams | Verdict | Reason | Issue |
|---|---|---|---|---|
| Forward-fill within patient, NaN before first value | #1, #0 | **use** | ADR-0001; causal | M2-1 (#14) |
| Regress `U1 − U0` | #1 | **use** | Core of the recipe; weights hours by utility | M1-2 (#9), M3-2 |
| Regularised gradient-boosted trees (LightGBM/XGBoost, shallow, heavy L1/L2) | #1, #2 | **use** | B2 baseline; start from depth about 4–6 and strong regularisation | M3-2, M3-3 |
| ShockIndex, BUN/CR, PartialSOFA and its 24 h change | #1 | **use** | Already in `CLAUDE.md`; vitals and revealed labs only | M2-2 (#15) |
| Individual clinical component scores (NEWS-style HR/Temp/Resp, SOFA-style Creatinine/MAP/Platelets/Bilirubin, qSOFA) as separate features | #0 | **adapt** | Complements PartialSOFA and SIRS; give NaN when the input is missing (their code gives a fixed score instead, 3 for Creatinine and Bilirubin, which makes the score a "never measured" flag and fails ADR-0002) | M2-2 (#15) |
| Rolling max/min of vitals | #1 | **use** | Vitals exempt from ADR-0002; window from config | M2-2 (#15) |
| More window statistics (mean, median, std, quantiles, variance, std of differences) of vitals; short and long window | #2, #3, #0 | **adapt** | Vitals allowed; evidence of gain is within noise, so add as an ablation | M2-2 (#15) |
| Lagged values of the previous hours (vitals and derived channels) | #2 | **adapt** | Lab flags excluded; window length in config | M2-2, M2-3 (#16) |
| One-hour or last-two-values differences | #2, #0 | **adapt** | Vitals and revealed lab values only; never as a "new lab arrived" signal | M2-2 (#15) |
| Hours since last measurement | #0 | **adapt** | Becomes `hours_since_<panel>` from revealed labs only (ADR-0002) | M2-2 (#15) |
| Signature features (order 3 log-signatures, lead-lag) | #1 | **adapt, optional** | Gain +0.012, inside fold std. Test both channel sets (paper vs code) | M4-1 |
| Cumulative-sum signatures | #1 | **adapt for vitals only; reject for labs** | May include ordering patterns of labs; unclear in the final model | M4-1 |
| Measurement counts, present/missing flags, sequence lengths, sampling rate | #1, #2, #3, #0 | **reject** | Violates ADR-0002. Optional diagnostic only: how much utility comes from ordering behaviour | — |
| Differences of forward-filled labs | #2 | **reject** | Non-zero only when a new value arrives, so it encodes the order time | — |
| Log transform and standardising | #2 | **adapt (B1 only)** | No gain for trees (0.3931 vs 0.3933); needed for logistic regression, fitted on the training side | M0-6 (#6), M3-2 |
| Mean imputation before the first value | #2, #3 | **adapt (B1 only)**; reject for trees | Trees keep NaN (ADR-0001) | M0-6 (#6), M3-2 |
| Linear interpolation of vitals | #3 | **reject** | Forward-fill suffices; interpolation uses later values unless truncated at t | M0-6 (#6) |
| Class weights or undersampling for a binary target | #2, #3, #0 | **adapt, optional** | Only for a binary comparison model; our `U1 − U0` target needs neither | M3-2 |
| Ensemble of fold or disjoint-subset models (mean or geometric mean) | #3, #0 | **adapt, optional** | Fits our nested CV; variance reduction for the final B2 | M3-3 |
| Hyperparameter search (random, grid, Bayesian TPE) | #1, #2, #0 | **adapt** | On the training side's CV folds only, with the official utility as objective | M3-3 |
| Feature selection (BoostARoota) | #3 | **adapt, optional** | Fit on the training side only; the paper does not say whether its 10% selection subset was held out from CV | M3-3 |
| Threshold search by gradient-free optimiser on utility | #1 | **adapt** | Grid from config, training-side out-of-fold predictions only | M1-1 (#8) |
| Fixed 0.5 threshold | #2, #3 | **reject** | Their hospital C results show it does not transfer | M1-1 (#8) |
| Threshold chosen from test feedback; final fit including the local test set | #0 | **reject** | Test parts stay locked until the final evaluation (ADR-0004) | M1-1, M8-1 |
| In-sample threshold in the final fit | #1 | **reject** | Leaks the training fold into the alarm level | M1-1 (#8) |
| Pooled CV as the only reported result | all | **reject as sole protocol** | The gap to hospital C is large for every team that reports it | M3-1 |

## 4. Inputs for M0-6 (preprocessing recipe)
- Forward-fill, NaN before the first value: confirmed by #1 and #0 (the best two scores by the teams' own claims). No fill or scaling is needed for trees (#2's Table 1).
- B1 logistic needs its own fill values and scaler, fitted on the training side only.
- No interpolation, no mean-before-first-value for trees.
- Lab timing features are only `hours_since_<panel>`. The reveal window is not informed by these papers.
- Windows: decide one convention (rows vs hours, inclusive or not) in `CONTEXT.md`.

## 5. What these papers cannot tell us
- How a model behaves when it controls which labs are ordered (all four were built on records as charted).
- How to handle hospital shift. None of the four tries, and all three that report by hospital collapse on C.
- Whether any gain below about 0.01 is real.

## 6. Unverified or unresolved
- Ranks and scores: team claims only (official paper not used).
- #1: which signature channels and whether cumulative-sum signatures were in the submitted model; the max/min window (paper 6 vs code 5); the utility-label generation; which script produced the shipped threshold.
- #2: the paper's Discussion gives 0.3981 for H=6 while Table 3 gives 0.3956; the paper does not explain why the best CV model was not submitted. The 0.329 hackathon result may not be comparable to 0.345.
- #3: training script not included; number of models in the pickles and the 10% selection subset not confirmed.
- #0: hyperparameters and tuner objective not stated; Figure 3 (threshold sweep) not extracted; printed train total in Table 1 is off by one.
- Tree settings of all shipped models were not read.
- Hidden test size differs between papers: 24,819 patients (#0) vs 22,761 (#3).
