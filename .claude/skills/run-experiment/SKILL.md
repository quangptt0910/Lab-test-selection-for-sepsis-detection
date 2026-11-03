---
name: run-experiment
description: Run one risk-model or policy experiment from a config, score it with the official PhysioNet scorer on both hospital-held-out directions, and log it to experiments/LOG.md.
disable-model-invocation: true
argument-hint: configs/<name>.yaml
---

Run the experiment defined in `$ARGUMENTS`.

1. **Pre-flight**
   - The config exists and has `id`, `hypothesis`, `model`, `features`, `split`.
   - `git status` is clean, or the user agreed to log a dirty tree. Record `git rev-parse --short HEAD`.
   - `make check` passes. If not, stop and report.
   - The `id` is not already in `experiments/LOG.md`. Never reuse an id.
2. **Train** on the training hospital. Fit any fill statistic, scaler or calibration on training data only.
3. **Tune the alarm threshold** on the validation patients (a held-out slice of the *training* hospital), using the official scorer. Save it in the run folder.
4. **Score** with the frozen threshold on the other hospital. Do this for both directions, A → B and B → A, unless the config says otherwise.
   - Write per-patient prediction files in the challenge format (`PredictedProbability|PredictedLabel`).
   - Run `third_party/evaluate_sepsis_score.py` and read AUROC, AUPRC and utility (5th value).
   - Replay experiments also report labs per patient-day and the unavailable-request rate.
5. **Log**: append one row to `experiments/LOG.md`:
   `| id | date | sha | config | A→B utility | B→A utility | AUROC (mean) | labs/pt-day | note |`
6. **Compare** with the best previous row of the same kind. Say plainly whether the hypothesis held.
7. **Report** to the user: the commands run, the scorer output lines, and the new LOG row. No metric without its command output.

Red flags. Stop and suspect leakage, then ask the leakage-reviewer subagent:
- Utility jumps by more than 0.05 from one small change.
- AUROC above 0.95.
- A→B and B→A differ by more than 0.15.
