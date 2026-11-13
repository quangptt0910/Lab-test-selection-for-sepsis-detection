# Project handoff: M0-1 to M0-4

Nov 13, 2026 · @Cld

M0-1 to M0-3 are merged into `develop`, and M0-4 (EDA) is ready for review in PR #75. The data can now be loaded, split, cleaned and described; the next decision is the preprocessing recipe and lab panels (M0-6), which the EDA findings feed into.

## Done so far

Every step is merged into `develop` except M0-4. `main` is still at the old setup commit; PR #76 brings `develop` into `main`.

| Issue | What it delivered | Main files | PR |
| --- | --- | --- | --- |
| M0-4 · EDA of both hospitals | Script that writes tables and figures from dev patients after cleaning, plus 9 hand-written findings. Leakage review passed | `src/sepsis/eda/`, `configs/eda.yaml`, `docs/eda/README.md`, `docs/eda/figures/` | #75, open into `develop` |
| M0-3 · Data quality and cleaning | Data-quality report and cleaning rules: plausible ranges, units, DBP ≥ SBP blanks all pressures, EtCO2 and ionized calcium dropped, FiO2 kept to 0.21–1.0 | `src/sepsis/data/quality.py`, `clean.py`, `configs/cleaning.yaml`, `docs/eda/data_quality.md` | #74, merged |
| M0-2 · Load data and split | Per-hospital parquet cache, one fixed dev/test split per hospital (80/20, stratified by septic, folds 0–4 on dev) with a fingerprint check | `src/sepsis/data/splits.py`, `data/splits/split.csv` | #73, merged |
| M0-1 · Repo and evaluation setup | Repo structure, configs, ADR-0004: A→B, B→A and mix (nested CV), locked test parts | `docs/adr/0004-cross-hospital-evaluation-setup.md`, `configs/protocol.yaml` | #72, merged |

The GitHub issues for M0-1 to M0-4 are still open; close them once their PRs reach `main`. M0-7 (A→B and B→A splits) is closed because ADR-0004 covers it.

## Getting set up

After one install and one cache build (several minutes), everything else runs in seconds. The README has the full version.

1. Check out the branch: `git fetch` then `git checkout develop` (or `m04_explore_hospitals` to review PR #75).
2. Install: `uv sync`, or with your own env `pip install -e .` plus `pip install pytest ruff matplotlib`.
3. Put the PhysioNet files in `data/training_setA/training_setA/` and `data/training_setB/training_setB/`. Data is never committed.
4. Build the cache: `python -m sepsis.data.build_cache`.
5. Check the split: `python -m sepsis.data.make_split`. The split file is in the repo; this only verifies it against the fingerprint.
6. Optional: `python -m sepsis.data.quality` (data-quality report) and `python -m sepsis.eda.run` (EDA, about 20 s). Their outputs are already committed in `docs/eda/`, so you only need to run them after changing their code or config.
7. After every change: `ruff check src tests`, `ruff format --check src tests`, `pytest -q -m "not slow"`. With uv, prefix each command with `uv run`.

Code gets patient-hours only through `load_project_split()` and `load_patient_hours(hospitals, patient_ids)` in `sepsis.data.splits`. These refuse test patients and check the split fingerprint.

## Rules to know before writing code

Breaking any of these invalidates a result. The full list is in `CLAUDE.md`; terms are defined in `CONTEXT.md`.

- **Dev patients only.** Experiments load the stored split and never re-split. A-test and B-test are used once, in the final evaluation (ADR-0004).
- **Three setups, logged apart:** A→B, B→A and mix (nested 5-fold CV on A-dev + B-dev, reported as mix→A and mix→B).
- **Fit on the training side only:** fill values, scalers and the alarm threshold.
- **Hours are rows, not ICULOS.** `SepsisLabel` is already shifted 6 h early, so t\_sepsis = first label-1 row + 6. Never shift again.
- **Features at hour t** use only rows ≤ t and only labs the replay has revealed.
- **No lab count or missingness features** (ADR-0002); use `hours_since_<panel>` instead. Vitals are exempt.
- **Never commit data** or model files. The only exception is `data/splits/split.csv`.
- **Thresholds and windows go in `configs/*.yaml`**, not in code. Tests use small hand-built tables, never real patient files.
- **Workflow:** after a change, run the checks, then have the leakage reviewer (a Claude Code subagent) review the diff. Ask Claude Code: "run the leakage-reviewer on the changes".

## What the data showed

The hospitals look alike in stay length and onset timing, but differ in prevalence, in how often things are charted and in baseline values. Expect a gap between mix and the cross-hospital setups. Dev part: A has 16,268 patients (8.8% septic), B has 16,000 (5.7% septic). Full tables, figures and all 9 findings are in `docs/eda/README.md`.

**Cleaning (M0-3)**, applied before any analysis by `sepsis.data.clean.clean()`:

- Values outside fixed plausible ranges are blanked. The ranges mark impossible readings, not abnormal ones.
- When DBP ≥ SBP in an hour, SBP, MAP and DBP of that hour are all blanked.
- EtCO2 is dropped (empty in A), and so is ionized calcium. FiO2 must be 0.21–1.0.
- Cleaning never drops or reorders rows and never changes ids, ICULOS or SepsisLabel.

**Main EDA findings (M0-4)**:

- **Records start differently.** 36.7% of A records start after ICU hour 1 (B: 6.4%), so ICULOS behaves differently per hospital.
- **Charting differs.** 36.6% of A patients have no DBP. Blood gases are 4–6 times rarer in B. Troponin is recorded for 2.4% of A patients vs 32.8% of B.
- **Panels.** A stable chemistry/CBC/liver core and a blood-gas group appear in both hospitals. HCO3 and Chloride move between groups, so panels must be one fixed clinical mapping for both hospitals.
- **Lab timing.** Blood gases come every 2–3 rows; chemistry and CBC every 5–12 rows in A and mostly 11–19 in B. Lab windows must span many hours.
- **No single strong marker.** The largest septic vs non-septic difference is SMD 0.42 (HR, B). Gains will come from combinations and changes over time.
- **Labs are ordered when worried.** Never-septic patients with a measured lactate have a higher median than septic patients, which supports ADR-0002.
- **Redundant pairs:** Hct–Hgb, the two bilirubins, BaseExcess–HCO3 (r ≥ 0.95).
- **Baseline shift.** MAP is 78.7 in A vs 86.3 in B (SMD −0.48).

The variable and panel choices in the findings are proposals; they are decided in M0-6 and M2.

## Open items and next steps

M0 is finished once PR #75 is reviewed and merged and M0-5 and M0-6 are done.

- [ ] Review PR #75 (M0-4 EDA, `m04_explore_hospitals` → `develop`): read the Findings section of `docs/eda/README.md` against the figures.
- [ ] Merge PR #76 (`develop` → `main`) once #75 is in, then close issues #1–#4.
- [ ] M0-5 (#5): review four top PhysioNet 2019 solutions (the top three and the highest unofficial entry).
- [ ] M0-6 (#6): decide the preprocessing recipe and lab panels. It takes these inputs from the EDA:
  - one fixed panel mapping for both hospitals, with FiO2 in ABG and a decision on HCO3 and Chloride;
  - the reveal window, from the time between lab values (print `p90_hours` in the EDA first);
  - whether TroponinI stays as a panel, since A almost never records it.
- [ ] Then M1 (scoring toolkit #8, U1 − U0 target #9) and M2 (view layer #14, clinical features #15, feature builder #16). Feature proposals from the EDA for M2: MAP and SBP rather than DBP; drop Hct, Bilirubin\_direct and SaO2; decide BaseExcess vs HCO3.
- [ ] Optional, at M2: a test that fails if `features/`, `action/` or `risk/` import `sepsis.eda` or `sepsis.data.patients`, which read the whole record.
