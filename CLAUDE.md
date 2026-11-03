# Sepsis decision support (PhysioNet/CinC 2019)

Hourly ICU decision-support prototype with three parts:
- a **risk model** (will this patient become septic soon?)
- an **action model** (which lab panel to order next, or WAIT)
- a **guideline layer** (Surviving Sepsis Campaign 2026 rules, compiled offline into YAML)

It is validated by **replaying** patient records with labs hidden until the system requests them.

Domain vocabulary: @CONTEXT.md · Decisions: `docs/adr/` · How we work: `docs/WORKFLOW.md`

## Commands
- Setup: `uv sync` (add `--extra signatures` for signature features)
- Fast check, run after every change: `make check` (ruff + `pytest -m "not slow"`)
- Full tests: `make test`
- Official score: `uv run python third_party/evaluate_sepsis_score.py <labels_dir> <pred_dir> <out.psv>`. Utility is the 5th value.
- Experiments: use the `run-experiment` skill, not ad-hoc scripts.

## Layout
`src/sepsis/` holds `data/` (load .psv, splits), `features/` (builder, vitals, labs, clinical, signatures), `labels/` (utility targets), `risk/`, `action/`, `guideline/` (engine + `rules/*.yaml`), `replay/`, `eval/`.
Other top-level folders: `configs/`, `experiments/`, `tests/`. `data/raw/` is read-only.

## Hard rules (breaking any one invalidates a result)
- IMPORTANT: features at hour t may use only rows ≤ t **and** only labs the replay has revealed. Code in `features/` and `action/` must never read the full record.
- Split by patient. Report hospital-held-out results (train A → test B, and B → A). One patient's hours never cross folds.
- Fit every statistic (fill means, scalers, alarm threshold) on the training/validation fold only.
- `SepsisLabel` is already shifted 6 h early, so t_sepsis = first label-1 row + 6. Never shift again.
- Utility comes only from the official scorer in `third_party/`, unmodified. Tune the alarm threshold on validation, then freeze it for all replay comparisons.
- No lab measurement-count or missing/present-pattern features, because they encode ordering behaviour (ADR-0002). Use `hours_since_<panel>` instead. Vitals are exempt.
- Never commit data, patient files or model artifacts.

## Modelling defaults (rank #1 recipe, ADR-0001)
- Forward-fill within patient. Leave NaN before the first value; LightGBM handles it.
- Features: `ICULOS`, raw values, PartialSOFA and its 24 h change, shock index (HR/SBP), BUN/Creatinine, rolling max/min of vitals, `hours_since_<panel>`. Signatures are optional (B3).
- Target: regress U1 − U0 from the official utility. Alarm when the prediction exceeds the tuned threshold.
- Baselines: B0 rules → B1 logistic → B2 LightGBM → B3 B2 + signatures. Keep B3 only if it beats B2 hospital-held-out.
- Details and formulas live in the `rank1-features` skill.

## Code style
- Python 3.11, type hints on public functions, pandas
- Feature, label and guideline-rule functions are pure. I/O stays at the edges (CLI modules).
- Windows, thresholds and costs go in `configs/*.yaml`, never as literals in code.
- Tests use small hand-built patient tables, never real patient files.

## Workflow
- Any change touching more than one file starts in plan mode. Deterministic code (features, labels, rules, replay masking) is test-first.
- After implementing: run `make check`, then have the `leakage-reviewer` subagent review the diff.
- When you claim a metric or a passing test, show the command and its output.
- Every experiment appends one row to `experiments/LOG.md`. Never edit past rows.
- When compacting, preserve: modified files, current experiment id, latest metrics, open leakage concerns.
