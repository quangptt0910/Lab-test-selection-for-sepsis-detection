# M4-1: Signature features (B3) on top of b2-001

**Decision: B3 is not kept. `b2-001` stays the reference risk model.** Over three seeds, the best B3 run (b3-002) raises the mix mean by 0.0026, less than the seed spread (0.0054) the keep rule asks it to beat.

Dev results only (ADR-0004); the test parts were not touched. Every utility comes from the official scorer, unmodified (`third_party/evaluate_sepsis_score.py`, hash-checked by `sepsis.eval.scorer`). The threshold is tuned on out-of-fold predictions of the training side and then frozen. One LOG row per run in `experiments/LOG.md`; run folders under `experiments/runs/<id>/` (gitignored). The six new runs are at commit `725518f`; b2-001 is the M3 run at `13837f8`.

As a spot check, b3-002's A→B predictions (16,000 B-dev patients) were re-scored with the scorer's own command line on labels read straight from the raw `.psv` files: all five values match `scores.json` exactly (utility 0.3096010194553721).

## 1. What was tested

Rank #1's signatures, in the form decided for this project (issue #37, ADR-0002 note, `rank1-features` skill §3):
- **Code:** `sepsis.features.signatures`, own NumPy code to depth 3. Per channel, a 2-D path over the last 8 rows ending at t (padded with the first row at the stay start); features are the log-signature coefficients at the Lyndon words `1, 2, 12, 112, 122`.
- **Blocks** (the `signatures` section of `configs/features.yaml`, picked with `features.signatures`):

| block | channels | path | columns |
|---|---|---|---|
| `sig_vitals` | MAP, HR, Resp, shock_index | `time` (row position, value) | 20 |
| `sig_vitals_cumsum` | HR, O2Sat, Temp, SBP, MAP, Resp | `cumsum_lead_lag` (running sum, then lead-lag) | 30 |
| `sig_partial_sofa` | partial_sofa | `time` | 5 |

- **Deviations from issue #37's wording:** lead-lag is used only in `sig_vitals_cumsum`; the `time` blocks use the (row position, value) path, as the paper's "time channel" does. b3-002 is not the paper's channel set (`partial_sofa, MAP, bun_cr`): `bun_cr` is refused under ADR-0002, so b3-002 adds `sig_partial_sofa` on top of b3-001 instead. b3-002 is therefore two block changes from b2-001, and the keep result is for the three blocks together.

- **Left out (ADR-0002):** `bun_cr`, FiO2 ratios and every other lab channel, because a forward-filled lab's path steps are lab differences. The code refuses them.

## 2. Runs

| id | change from b2-001 | seed | A→B | B→A | mix→A | mix→B | mix mean | AUROC (mean of 4) |
|---|---|---|---|---|---|---|---|---|
| b2-001 | — | 0 | 0.309 | 0.314 | 0.426 | 0.363 | 0.3945 | 0.807 |
| b2-011 | — | 1 | 0.308 | 0.283 | 0.422 | 0.369 | 0.3957 | 0.805 |
| b2-012 | — | 2 | 0.303 | 0.309 | 0.419 | 0.361 | 0.3904 | 0.807 |
| b3-001 | + sig_vitals, sig_vitals_cumsum | 0 | 0.304 | 0.350 | 0.424 | 0.361 | 0.3927 | 0.808 |
| b3-002 | b3-001 + sig_partial_sofa | 0 | 0.310 | 0.339 | 0.425 | 0.370 | 0.3979 | 0.810 |
| b3-011 | b3-002 | 1 | 0.303 | 0.304 | 0.419 | 0.375 | 0.3973 | 0.809 |
| b3-012 | b3-002 | 2 | 0.305 | 0.323 | 0.418 | 0.369 | 0.3933 | 0.811 |

- No run raised a red flag: AUROC ≤ 0.843, |A→B − B→A| ≤ 0.046, no cross-hospital result above mix for the same hospital, no column moved by more than 0.05 from one change.
- b3-002 was the better of the two channel sets on seed 0, so it got the seed repeats. The optional b3-003 (depth or window change) was not run; see §5.

## 3. Keep rule (eval side, 3-seed means)

The rule settled before the runs (issue #37): B3's 3-seed mean of (mix→A + mix→B)/2 must beat b2-001's by more than the larger of the two seed spreads (max − min), and neither A→B nor B→A may drop by more than 0.01 (`selection.cross_hospital_tolerance`).

| | A→B | B→A | mix→A | mix→B | mix mean | mix-mean spread |
|---|---|---|---|---|---|---|
| b2-001 (seeds 0–2) | 0.3068 | 0.3023 | 0.4225 | 0.3646 | 0.3935 | 0.0054 |
| b3-002 (seeds 0–2) | 0.3061 | 0.3224 | 0.4207 | 0.3715 | 0.3961 | 0.0046 |
| change | −0.0007 | +0.0201 | −0.0018 | +0.0069 | **+0.0026** | margin 0.0054 |

Seed spread (max − min) per column:

| | A→B | B→A | mix→A | mix→B |
|---|---|---|---|---|
| b2-001 | 0.0059 | 0.0307 | 0.0065 | 0.0077 |
| b3-002 | 0.0062 | 0.0349 | 0.0075 | 0.0065 |

- **Mix: fails.** +0.0026 against a required +0.0054.
- **Cross-hospital: passes.** A→B −0.0007, B→A +0.020.
- **B→A is the noisiest column:** its seed spread is 0.031 for b2-001 and 0.035 for b3-002. The B→A rise of single runs (+0.036 in b3-001) is within that, and the rule does not decide on it.

**Training-side OOF figures** (the tuning utility at the chosen threshold, mean over the 3 seeds; reported alongside, as agreed):

| | A→B (CV on A-dev) | B→A (CV on B-dev) | mix |
|---|---|---|---|
| b2-001 | 0.4196 | 0.3776 | 0.3948 (spread 0.0056) |
| b3-002 | 0.4235 | 0.3797 | 0.3991 (spread 0.0068) |
| change | +0.0039 | +0.0021 | +0.0043 |

The same rule on these figures also fails (+0.0043 < 0.0068). Both sides agree.

## 4. What the model does with the signatures (b3-002, `sepsis.eval.importance`)

- **The signatures take a share of the model but add little.** The 55 signature columns carry 14.5% (A→B) and 12.8% (B→A) of the mean |SHAP|, yet utility barely moves. They mostly take over what other columns already carried.
- **The strongest are near-copies of existing columns.** `sig_vitals_cumsum_Resp_1` (#15 in A→B) is the sum of the last 8 Resp values, close to a window mean. `sig_partial_sofa_partial_sofa_2` (#11 in B→A) is the 8-row change of PartialSOFA, next to the existing `partial_sofa_delta_24`.
- **By block** (share of mean |SHAP|, A→B / B→A): `sig_vitals_cumsum` 10.6% / 7.1%, `sig_partial_sofa` 2.7% / 3.1%, `sig_vitals` 1.2% / 2.5%. 11 signature columns have zero gain in both setups (17 in at least one).
- **8 signature columns flip direction** between A→B and B→A, all in `sig_vitals_cumsum` (O2Sat, Resp, SBP and Temp terms), none in the top 15.
- **The hospital-specific columns are unchanged** from b2-001 (M3-3 §4): `ICULOS` #1 and `HospAdmTime` #3 in both setups, `hours_since_ABG` #2. The direction flips of Creatinine, HospAdmTime, PaCO2 and `hours_since_LACTATE` are also the same as before.

## 5. Why b3-003 was not run

A depth or window change cannot be ruled out with certainty, but the evidence points against it. Two channel sets and three seeds all land within ±0.005 of b2-001. The signatures already hold about 13–15% of the model and add almost nothing. Their strongest columns overlap `vital_minmax` and `partial_sofa_delta`. A b3-003 would need three seeds (about 2 hours) to be judged, and a gain of the size seen here would not pass the margin.

## 6. Cost

| | b2-001 (61 columns) | b3-002 (116 columns) |
|---|---|---|
| Feature tables, first build (wall − slowest setup, 3 parallel setups) | ≈ 130 s (b2-012) | ≈ 325 s (b3-002); ≈ 230 s for b3-001 |
| Full run, 3 setups in parallel (`setup_jobs: 3`) | 10–11 min | 12–16 min |
| Full run, setups in turn (`--jobs 1`) | — | 19–23 min |
| ALL-policy replay, 200 dev patients (8,555 rows) | 57.6 s | 67.7 s (+18%) |

- **Memory:** the first b3-011 run, with 3 setups in parallel, was stopped because the machine was critically low on memory. It wrote nothing and was rerun with `--jobs 1`. With signatures on, run with `--jobs 1` or `2` on this machine.
- **Replay:** ALL-policy replay gave the batch scores in both cases. The replay check's toy score does not read signature columns; batch = hour by hour for the signature blocks themselves is tested by `tests/test_feature_leakage.py`.
- The replay engine rebuilds the features of every view row at each step (`replay/engine.py`). The signature blocks add 18% to that, so they would not block M5/M6 if they were ever kept.

## 7. Leakage review

The `leakage-reviewer` checked the branch diff and the six configs and found nothing that invalidates a result:
- The windows never read a row after t or another patient's rows, and the blocks read the view only. The leakage tests parametrise over every registered block, the three signature blocks included.
- The channel guard keeps labs, DBP, SaO2 and EtCO2 out. The cache keys include the selected blocks, and the resolved `features.yaml` in each run folder shows them.
- One note was added to ADR-0002. The 8-row `partial_sofa` window places a change of the score at the row the lab joined the view. In replay that row follows the request, so a late request shifts the step: a small difference between training and replay, not leakage.

## 8. Next

- B3 is dropped; M5/M6 use `b2-001`. The signature code and blocks stay in the repo, off by default.
- The noise figures above (B→A seed spread about 0.03, mix-mean spread about 0.005) apply to every later keep decision on B2: a single-seed change smaller than that should not be read as a gain.
