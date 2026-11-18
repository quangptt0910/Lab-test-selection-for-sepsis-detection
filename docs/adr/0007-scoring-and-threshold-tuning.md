# ADR-0007: Scoring and alarm-threshold tuning

Status: accepted (M1-1). Settings in `configs/eval.yaml`, code in `sepsis.eval`.

## Context
Every experiment, replay and the final evaluation must be scored the same way, with the official scorer in `third_party/`, unmodified. The alarm threshold is a fitted statistic, so it comes from the training side of a setup only (ADR-0004). The top-4 review (M0-5) found that no team did this: thresholds were fixed at 0.5, tuned in-sample or tuned on test feedback.

Four points were open:
- The scorer's AUROC assumes values in [0, 1]. For larger values it only warns, but it puts a threshold of 1 at the top of its list, so every score above 1 falls into one ROC point. The U1 − U0 regressor outputs about −0.05 to 2.
- B0 (rule counts), B1 (probabilities) and B2 (utility differences) have different score scales, so one grid of absolute thresholds cannot serve all of them.
- The alarm rule at equality and the choice between tied thresholds were not defined.
- Nothing stopped an edit to the scorer from going unnoticed.

## Decision
1. **One scoring path.** `sepsis.eval.scorer` writes a label folder and a prediction folder (`PredictedProbability|PredictedLabel`, one `<patient_id>.psv` per patient, rows in `hour` order) and calls the scorer's `evaluate_sepsis_score` on them. It returns AUROC, AUPRC, accuracy, F-measure and utility by name.
2. **Scorer integrity.** The SHA-256 of `third_party/evaluate_sepsis_score.py` (line endings normalised to LF) is stored in `configs/eval.yaml`. The loader refuses a file that differs.
3. **Raw score for alarms, logistic value for the file.** The alarm is `score > threshold` on the raw model output, strictly greater. `PredictedProbability` is the logistic function of the raw score. The function is fixed and increasing, so it keeps the ranking and fits nothing on the scored patients. Scores already in [0, 1] go through the same transform.
4. **Threshold grid from training-side scores.** Candidates are observed out-of-fold scores of the training side at evenly spaced quantiles (`threshold.grid`). Only observed scores are used, because a threshold between two neighbouring scores gives the same alarms as the lower one.
5. **Tuning uses the scorer's own utility.** Every number comes from the scorer's `compute_prediction_utility`: per label record, the inaction utility, the best utility, and the gain of an alarm at each hour (the utility of alarming at that hour alone minus inaction). The scorer sums per-hour terms that depend only on that hour's prediction, so a candidate's utility is inaction plus the gains of its alarmed hours, normalised as the scorer does. A test checks that this equals the folder path for every candidate. Ties go to the highest threshold (fewer alarms).
6. **Tuning is guarded.** `tune_threshold` takes the split table, the setup and, for mix, the outer fold. It works out the training side itself and raises `ThresholdLeakError` for any other patient: the evaluated hospital, the held-out fold, or a test patient.
   A split table that lists a patient twice, or has a dev patient without a fold, is refused.
7. **Complete records only.** The scorer finds onset by row position, so scoring and tuning require every patient's full record (integer hours 0..n−1), labels of 0 or 1 and finite scores.
8. **Per-hospital scores.** `score_by_hospital` calls the scorer once per hospital. Mix → A and mix → B are each normalised over that hospital's held-out patients pooled across the five outer folds, not averaged over folds.

## Consequences
- `PredictedProbability` is not a calibrated probability. Accuracy and F-measure use the alarms and are not affected.
- Very large raw scores (above about 36) reach 1.0 after the logistic function and would tie. No baseline produces such scores; a model that does must rescale its output on the training side.
- A set with no septic patient cannot be scored (the normalisation is 0/0), so scoring raises an error.
- The scorer's utility is a Python loop, so calling it per patient and candidate took 281 s for one tuning on A-dev (16,268 patients, 200 candidates). The per-hour gains are computed once per distinct label record and cached, after which every candidate is one vectorised sum. The same tuning then takes 50 s, most of it filling the cache, which later tunings in the same process reuse.
- Replacing the scorer file needs a new hash in `configs/eval.yaml`, which makes the change visible in review.
