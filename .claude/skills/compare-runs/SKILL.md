---
name: compare-runs
description: Decide keep or drop for a change from its finished run folders, against a reference, over seed repeats. Use before claiming a change helps, before writing numbers into a docs/experiments report, and to plan seed repeats.
---

Compare finished runs with `sepsis.eval.compare`. Never compute means, spreads, gains or the keep rule by hand or with ad-hoc scripts: every number in a keep claim or an experiment report comes from this tool's output.

1. **Have the runs.** The reference (usually the current best, e.g. `b2-001`) and the candidate each need `min_seeds` runs (`selection` in `configs/protocol.yaml`, 3). They must be full runs of the `run-experiment` skill, with distinct seeds.
   - A seed repeat is a copy of the config with only `id`, `seed`, `hypothesis` and `note` changed. Its resolved configs (switched-off blocks aside) and feature columns must match too. The tool refuses anything else, partial runs and `.partial` folders, and runs scored by different scorer files.
   - Give both sides the same seeds (0, 1, 2), so the same-seed jump check covers every candidate run.
   - Done when every run id has a folder under `experiments/runs/`.
2. **Run** `uv run python -m sepsis.eval.compare --reference <ids> --candidate <ids>`.
3. **Read the output, all of it.**
   - **Decision:** `keep` only if the candidate's mix mean (mean of mix→A and mix→B, over its seeds) beats the reference by more than the larger of the two mix-mean seed spreads, **and** neither A→B nor B→A drops by more than `cross_hospital_tolerance`.
   - `undecided` means too few seeds, or a seed spread of 0 (seeds that change nothing leave no margin). It is a screening result, never a keep: run the seed repeats, or report the change as not tested.
   - **Flags and warnings:** a red flag (AUROC, direction gap, cross-hospital beating mix, a utility jump from the reference run with the same seed) means stop and ask the `leakage-reviewer` subagent before anything else. Every warning (several commits, uncommitted code, different seed sets) goes into the report next to the decision.
   - **OOF rule:** information only. Its mix figure pools A and B patients, so it is not on the decision's scale. Say so when it disagrees with the decision.
   - Done when you can state the decision, the gain against the margin, and every flag or warning.
4. **Report.** Paste the printed tables into `docs/experiments/<issue>.md` unchanged, with the command above them. Retype nothing, and round nothing yourself.
   - State that a keep is optimistic. The seed spread measures masking and model noise only (patients and folds are fixed), and the decision selects on the same dev patients it scores. A gain that only just clears the margin is weak evidence; only the one-time final evaluation on the test parts is unbiased.

## Noise reference (M4-1, b2-001 over seeds 0–2)

- Mix-mean seed spread is about 0.005. B→A is the noisiest column, with a spread of about 0.03; A→B and the two mix columns are about 0.006–0.008.
- So one run that moves B→A by +0.03 says nothing on its own. Compare single runs only to choose which change gets seed repeats.
- Each run takes 10–23 minutes. With the signature blocks on, run with `--jobs 1` or `2` on this machine: three parallel setups ran out of memory.
