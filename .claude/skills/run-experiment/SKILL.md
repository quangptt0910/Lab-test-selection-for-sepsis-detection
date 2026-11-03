---
name: run-experiment
description: Run one risk-model or policy experiment from a config, score it with the official PhysioNet scorer under both split protocols (pooled_cv and cross_hospital), and log both to experiments/LOG.md.
disable-model-invocation: true
argument-hint: configs/<name>.yaml
---

Run the experiment defined in `$ARGUMENTS`.

Two protocols are always scored and logged side by side (`split.protocol` in the config names the headline one, it does not skip the other):
- `pooled_cv` (optimistic): 80/20 dev/test split stratified by hospital × septic, 5-fold patient-level CV on dev. Hospitals are mixed in train and validation.
- `cross_hospital` (pessimistic): train A → test B and train B → test A.

1. **Pre-flight**
   - The config exists and has `id`, `hypothesis`, `model`, `features`, `split` (with `split.protocol` set to `pooled_cv` or `cross_hospital`).
   - `git status` is clean, or the user agreed to log a dirty tree. Record `git rev-parse --short HEAD`.
   - `make check` passes. If not, stop and report.
   - The `id` is not already in `experiments/LOG.md`. Never reuse an id.
2. **Pooled CV.** Patient ids are hospital-prefixed (`A_p000001`) so they never collide.
   - Train on 4 folds of dev, predict the 5th. Fit every fill statistic, scaler and calibration on the training folds only.
   - Tune the alarm threshold on the out-of-fold dev predictions with the official scorer, then freeze it.
   - Score the out-of-fold predictions **per hospital** (A patients, B patients separately). The 20% test set is not touched here; it is for the final report only.
3. **Cross-hospital.** For each direction (A → B, B → A):
   - Train on the training hospital. Fit statistics on training data only.
   - Tune the alarm threshold on validation patients (`val_fraction`, a held-out slice of the *training* hospital), then freeze it.
   - Score on the other hospital with the frozen threshold.
4. **Score** each run with `third_party/evaluate_sepsis_score.py`, unmodified.
   - Write per-patient prediction files in the challenge format (`PredictedProbability|PredictedLabel`).
   - Read AUROC and utility (5th value). Save each threshold in the run folder.
   - Replay experiments also report labs per patient-day and the unavailable-request rate.
5. **Log**: append one row to `experiments/LOG.md`:
   `| id | date | sha | config | pooled-CV utility A | pooled-CV utility B | A→B utility | B→A utility | AUROC pooled-CV | AUROC cross-hospital (mean) | labs/pt-day | note |`
   - The pooled-CV columns are the optimistic numbers and the A→B / B→A columns are the pessimistic ones. Never merge them into one number.
   - If a protocol was not run, write `—`, never an estimate.
6. **Compare** with the best previous row of the same kind, on both protocols. Say plainly whether the hypothesis held, and report the pooled-CV vs cross-hospital gap.
7. **Report** to the user: the commands run, the scorer output lines, and the new LOG row. No metric without its command output.

Red flags. Stop and suspect leakage, then ask the leakage-reviewer subagent:
- Utility jumps by more than 0.05 from one small change.
- AUROC above 0.95.
- A→B and B→A differ by more than 0.15.
- Cross-hospital utility is higher than pooled-CV utility for the same hospital. Pooled CV should be the easier setting, so this points to a bug.
