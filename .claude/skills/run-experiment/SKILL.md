---
name: run-experiment
description: Run one risk-model or policy experiment from a config under the three setups (A → B, B → A, mix) from ADR-0004, score each with the official PhysioNet scorer, and log them to experiments/LOG.md.
disable-model-invocation: true
argument-hint: configs/<name>.yaml
---

Run the experiment defined in `$ARGUMENTS`.

The runner (ADR-0008) does the work: `uv run python -m sepsis.eval.run_experiment $ARGUMENTS`. It runs every setup listed in the config on **dev data only** (ADR-0004); A-test and B-test are never read.
- `A_to_B` (pessimistic): train and tune on A-dev, evaluate on B-dev.
- `B_to_A` (pessimistic): train and tune on B-dev, evaluate on A-dev.
- `mix` (optimistic): nested CV over the stored folds on A-dev + B-dev, reported separately for A and B patients.

Your job is the checks around it and the judgment after it. Never run experiments with ad-hoc scripts.

1. **Pre-flight**
   - The config exists. The runner checks its schema strictly (unknown keys are refused) and refuses an `id` already in `experiments/LOG.md` or an existing `experiments/runs/<id>/`.
   - `git status` is clean, or the user agreed to log a dirty tree (`--allow-dirty`; the SHA then ends in `-dirty`). `experiments/LOG.md` and untracked configs don't count.
   - `make check` passes (on Windows without `make`, the three `uv run` commands in `CLAUDE.md`). If not, stop and report.
   - The model type is registered in `sepsis.risk.base.MODELS`.
2. **Run** `uv run python -m sepsis.eval.run_experiment $ARGUMENTS` (add `--no-cache` only to force a rebuild of the feature tables). Expect about 37 model fits for a full run. Partial runs (fewer setups) are logged with `—` and the note "partial run: not comparable".
3. **Read the outputs.** The run folder `experiments/runs/<id>/` holds `scores.json`, per-setup `thresholds.json`, `threshold_curve.csv`, `folds.csv` and `predictions.parquet`, and `meta.json`. The runner prints the scores, the new LOG row and any red flags.
4. **Compare** with the best previous row of the same kind, on all four columns. Say plainly whether the hypothesis held. A change is kept only if mix improves without A→B or B→A dropping by more than `selection.cross_hospital_tolerance` in `configs/protocol.yaml`, and never on a partial run. Report the gap between mix and cross-hospital results.
5. **Report** to the user: the command run, the scores it printed, and the new LOG row. No metric without its command output.

Red flags. Stop and suspect leakage, then ask the leakage-reviewer subagent. The limits are in the `red_flags` section of `configs/protocol.yaml`. The runner writes the first three into the LOG note; the fourth needs your comparison:
- AUROC above `max_auroc` (0.95) in any setup.
- A→B and B→A differ by more than `max_direction_gap` (0.15).
- A cross-hospital result beats the mix result for the same evaluated hospital (A→B above mix→B, or B→A above mix→A). Mix trains on that hospital, so it should be the easier setting; this points to a bug.
- Utility jumps by more than `max_utility_jump` (0.05) from one small change.
