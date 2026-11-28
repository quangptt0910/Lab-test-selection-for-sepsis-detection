# M5-2: Baseline policies NONE, ALL and AS_RECORDED on b2-001

**Result: the replay reproduces b2-001 exactly, and the M5 check passed.** ALL and AS_RECORDED give b2-001's utilities in every setup. AS_RECORDED does so with 9.1 (B) and 14.1 (A) panel requests per patient-day, against ALL's 168. NONE (no labs at all) loses 0.041 to 0.148 utility. These are the reference points M6's VoI policy has to place itself between.

Dev results only (ADR-0004); the test parts were not touched. Utilities come from the official scorer, unmodified (`third_party/evaluate_sepsis_score.py`, sha256 `26b8b262…`, hash-checked). Each replayed patient is scored by a refit of b2-001's model that never trained on them, with that setup's frozen threshold from the b2-001 run folder (ADR-0009). The threshold is **not** re-tuned per policy.

Command, at commit `13a22af` (clean tree):

```
uv run python -m sepsis.replay.run_policies configs/replay_b2.yaml
```

It wrote the run folders `r-none-001`, `r-all-001` and `r-asrec-001` and one LOG row each in `experiments/LOG.md`. Versions: Python 3.14.5, pandas 3.1.1, numpy 2.5.4, LightGBM 4.7.0, scikit-learn 1.9.1.

## 1. Policies

| policy | requests | module |
|---|---|---|
| NONE | nothing, ever (WAIT every hour) | `sepsis.action.baselines` |
| ALL | every panel, every patient, every hour | `sepsis.action.baselines` |
| AS_RECORDED | panel P at row t iff a lab of P was recorded at row t | `sepsis.replay.baselines` (reads the record through `bind`, ADR-0009 amendment) |

With a reveal delay of 1 row, ALL and AS_RECORDED both reveal each value at its recorded row + 1: the as-recorded view the model was trained on.

## 2. Results

Utility per setup (frozen b2-001 thresholds):

| id | policy | A→B | B→A | mix→A | mix→B | AUROC (mean of 4) | labs/pt-day (mean of 4) |
|---|---|---|---|---|---|---|---|
| b2-001 | (source run) | 0.309 | 0.314 | 0.426 | 0.363 | 0.807 | — |
| r-all-001 | ALL | 0.309 | 0.314 | 0.426 | 0.363 | 0.807 | 168.0 |
| r-asrec-001 | AS_RECORDED | 0.309 | 0.314 | 0.426 | 0.363 | 0.807 | 11.6 |
| r-none-001 | NONE | 0.268 | 0.166 | 0.337 | 0.262 | 0.761 | 0.0 |
| | NONE − ALL | −0.041 | −0.148 | −0.088 | −0.102 | −0.046 | |

Burden and request status per setup:

| policy | | A→B (B patients) | B→A (A patients) | mix→A | mix→B |
|---|---|---|---|---|---|
| ALL | labs/pt-day | 168.0 | 168.0 | 168.0 | 168.0 |
| | unavailable rate | 0.642 | 0.525 | 0.525 | 0.642 |
| | nothing-new rate | 0.304 | 0.392 | 0.392 | 0.304 |
| AS_RECORDED | labs/pt-day | 9.1 | 14.1 | 14.1 | 9.1 |
| | unavailable / nothing-new rate | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| NONE | labs/pt-day | 0.0 | 0.0 | 0.0 | 0.0 |
| | unavailable / nothing-new rate | — | — | — | — |

(Rates are `None`, shown as —, when no request was made.)

## 3. What the numbers say

- **The replay is exact.** ALL's predictions equal b2-001's stored scored predictions (scores within 1e-9, alarms equal), and its utilities equal b2-001's `scores.json` exactly: the hard M5 check built into the CLI, which would have refused to write the LOG rows otherwise. So the replay engine, the reveal rules and the model refits add no difference of their own.
- **AS_RECORDED == ALL at 5–8% of the requests.** Every AS_RECORDED request revealed new values (both rates 0.000), as it should. Its burden is the hospitals' own ordering rate: 9.1 panels per patient-day in B and 14.1 in A.
- **Cross-check of the two:** ALL's revealed share times 168 gives AS_RECORDED's burden. For B: 168 × (1 − 0.642 − 0.304) = 9.1; for A: 168 × (1 − 0.525 − 0.392) = 14.1. Of ALL's requests, 94% (B) and 92% (A) are wasted: either nothing was recorded in the look-back window or nothing new.
- **Labs are worth 0.04–0.15 utility to b2-001.** Without labs the model still ranks fairly well (AUROC 0.72–0.80), mostly on vitals and `ICULOS`. The largest loss is B→A (−0.148). A small part of NONE's loss is the threshold: it was tuned on as-recorded scores and stays frozen, so NONE alarms less (B→A accuracy 0.903 against ALL's 0.667). That is the intended comparison: every policy runs with the same model and threshold.
- **The rates repeat across setups** because the same patients are replayed: A→B and mix→B both replay B-dev, so their request tables have the same statuses.

## 4. What this means for M6

A VoI policy has to land between NONE and AS_RECORDED. It is worth something if it keeps most of the 0.04–0.15 utility labs bring, at a burden at or below the hospitals' 9–14 panels per patient-day. ALL is no target: it buys nothing over AS_RECORDED for 12–18× the requests.

## 5. Cost

One run, three policies, three setups, on this machine: 103 min wall. Replay time summed over the setups: NONE 13.5 min, ALL 30.0 min, AS_RECORDED 27.7 min (`replay_seconds` in each `meta.json`). The rest is loading, the feature cache, the model refits and scoring. The setups run one after the other; running them in parallel would cut this, but it is not needed for M5-2.
