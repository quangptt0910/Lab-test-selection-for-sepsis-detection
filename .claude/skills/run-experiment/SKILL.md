---
name: run-experiment
description: Run one risk-model or policy experiment from a config under the three setups (A → B, B → A, mix) from ADR-0004, score each with the official PhysioNet scorer, and log them to experiments/LOG.md.
disable-model-invocation: true
argument-hint: configs/<name>.yaml
---

Run the experiment defined in `$ARGUMENTS`.

Every experiment runs all three setups on **dev data only** (ADR-0004). Each hospital was split once into dev (80%) and test (20%); A-test and B-test are never read here.
- `A_to_B` (pessimistic): train and tune on A-dev, evaluate on B-dev.
- `B_to_A` (pessimistic): train and tune on B-dev, evaluate on A-dev.
- `mix` (optimistic): 5-fold patient CV on A-dev + B-dev, reported separately for A and B patients.

1. **Pre-flight**
   - The config exists and has `id`, `hypothesis`, `model`, `features`, `split` (with `split.setups`).
   - `git status` is clean, or the user agreed to log a dirty tree. Record `git rev-parse --short HEAD`.
   - `make check` passes (on Windows without `make`, the three `uv run` commands in `CLAUDE.md`). If not, stop and report.
   - The `id` is not already in `experiments/LOG.md`. Never reuse an id.
   - Nothing in the config or code path loads A-test or B-test, and the dev/test patient lists come from the stored split (`configs/protocol.yaml`), not from a new split.
2. **A → B and B → A.** For each direction:
   - Run 5-fold patient CV inside the training hospital's dev part. Fit every fill statistic, scaler and calibration on the training folds only.
   - Tune the alarm threshold on those out-of-fold predictions with the official scorer, then freeze it.
   - Fit on the whole training dev part and score the other hospital's dev part with the frozen threshold. Never tune anything on the evaluated hospital.
3. **Mix (nested CV).** Patient ids are hospital-prefixed (`A_p000001`) so they never collide.
   - Run an outer 5-fold patient CV on A-dev + B-dev (folds stratified by hospital × septic).
   - In each outer fold, run an inner CV (`n_inner_folds`) on that fold's training patients only, tune the alarm threshold on the inner out-of-fold predictions, then fit on all of the fold's training patients and score the held-out fold with that threshold. A held-out fold is never used to tune its own threshold.
   - Fit every statistic on the training patients of the fold only.
   - Score the held-out predictions **separately** for A patients (mix → A) and B patients (mix → B).
4. **Score** each run with `third_party/evaluate_sepsis_score.py`, unmodified.
   - Write per-patient prediction files in the challenge format (`PredictedProbability|PredictedLabel`).
   - Read AUROC and utility (5th value). Save each threshold in the run folder.
   - Replay experiments also report labs per patient-day and the unavailable-request rate.
5. **Log**: append one row to `experiments/LOG.md`:
   `| id | date | sha | config | eval | A→B utility | B→A utility | mix→A utility | mix→B utility | AUROC (mean of 4) | labs/pt-day | note |`
   - `eval` is `dev` here. Only the final evaluation writes `test` rows.
   - Mix columns are the optimistic numbers and A→B / B→A the pessimistic ones. Never merge them into one number.
   - If a setup was not run, write `—`, never an estimate.
6. **Compare** with the best previous row of the same kind, on all four columns. Say plainly whether the hypothesis held. A change is kept only if mix improves without A→B or B→A dropping by more than `selection.cross_hospital_tolerance` in `configs/protocol.yaml`. Report the gap between mix and cross-hospital results.
7. **Report** to the user: the commands run, the scorer output lines, and the new LOG row. No metric without its command output.

Red flags. Stop and suspect leakage, then ask the leakage-reviewer subagent:
- Utility jumps by more than 0.05 from one small change.
- AUROC above 0.95.
- A→B and B→A differ by more than 0.15.
- A cross-hospital result beats the mix result for the same evaluated hospital (A→B above mix→B, or B→A above mix→A). Mix trains on that hospital, so it should be the easier setting; this points to a bug.
