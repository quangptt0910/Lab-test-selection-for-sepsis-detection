# ruff: noqa: E501
"""Sync this project's labels, milestones, issues and project board on GitHub with the plan below.

Preview (reads GitHub if `gh` is logged in, changes nothing):
    python scripts/create_github_issues.py --dry-run

Real run (needs the GitHub CLI, logged in with the project scope):
    gh auth login
    gh auth refresh -s project
    python scripts/create_github_issues.py

What a run does:
- creates or updates the labels and milestones
- edits planned issues that already exist (matched by id, e.g. "M0-1"), so their numbers stay the same
- creates planned issues that don't exist yet
- closes open "M<n>-<n>" issues that are no longer in the plan as "not planned", and removes them from the board
Safe to re-run: anything already in sync is left alone. Nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

DEFAULT_REPO = "quangptt0910/Lab-test-selection-for-sepsis-detection"
DEFAULT_PROJECT_TITLE = "Sepsis ML Project"
DEFAULT_PROJECT_OWNER = "quangptt0910"
SECONDS_BETWEEN_WRITES = 1.0  # stays under GitHub's limit on creating content quickly
ISSUE_ID = re.compile(r"M\d+-\d+")

LABELS = {
    # Area
    "data": ("1d76db", "Loading, cleaning, splitting and exploring patient data"),
    "eval": ("0e8a16", "Official scoring, protocols and metrics"),
    "labels": ("5319e7", "t_sepsis, dt and the U1 - U0 target"),
    "features": ("006b75", "Feature builder and view"),
    "risk-model": ("b60205", "B0-B3 risk models"),
    "replay": ("d93f0b", "Replay simulator and baseline policies"),
    "action-model": ("c2410c", "VoI models and the WAIT decision"),
    "guideline": ("0052cc", "Guideline engine and YAML rules"),
    "report": ("7057ff", "Research, final evaluation and write-up"),
    "infra": ("6a737d", "Tooling, configs and repo setup"),
    # Type
    "test-first": ("bfd4f2", "Deterministic code: write the tests first"),
    "experiment": ("fbca04", "Model run scored by the official scorer and logged"),
    "decision": ("f9d0c4", "A choice to make and record, usually in an ADR"),
    "chore": ("e4e669", "Docs, analysis, config or manual work"),
    # Extra
    "optional": ("ededed", "Not required to complete the project"),
    "needs-input": ("d876e3", "Needs a person or material Claude cannot supply"),
}

# Added to every issue marked "Explain: evaluation" (ADR-0004).
EVALUATION_SETUP = """\
## How the evaluation setup works (ADR-0004)
1. Each hospital is split once by patient, stratified by septic: **80% dev** and **20% test**, using `configs/protocol.yaml`. The split is stored as patient-id lists and every experiment loads them, so it can never change. A-test and B-test stay locked until the final evaluation (M8-1). The hospitals are loaded separately and only combined for the mix setup.
2. During development, every experiment runs three setups on dev data only:

| Setup | Train and tune on | Evaluate on |
|---|---|---|
| A → B | A-dev, with 5-fold CV inside it for tuning and the threshold | B-dev |
| B → A | B-dev, with 5-fold CV inside it | A-dev |
| Mix | A-dev + B-dev, outer 5-fold CV; each outer fold tunes its threshold with an inner CV on its own training patients (nested CV) | Each held-out fold, reported separately for A and B patients (mix → A, mix → B) |

3. The final evaluation runs once with the frozen setup: A-dev → B-test, B-dev → A-test, and A-dev + B-dev → A-test and B-test.
4. Every fitted statistic (fill values, scalers, the alarm threshold) comes from the training side only, never from the hospital or patients being evaluated. In mix, a held-out fold is never used to tune its own threshold.
5. Mix is the optimistic number and A → B / B → A are the pessimistic ones. They are logged side by side and never merged into one number.
"""

DEFINITION_OF_DONE = [
    "`make check` is green.",
    "`leakage-reviewer` has reviewed the diff (for features, labels, models, replay or guidelines).",
    "The command output is pasted in the issue, and experiments also have a new `experiments/LOG.md` row.",
]

# Format:
#   ## <milestone title> | <milestone description>
#   ### <id> | <title> | <space-separated labels>
#   Goal line(s)
#   Depends on: <ids>        (optional)
#   Tasks:
#   - task
#   Acceptance:
#   - criterion
ISSUES_TEXT = r"""
## M0: Data understanding | Both hospitals are loaded, split into dev and locked test parts, cleaned and explored; the evaluation setup, preprocessing recipe and lab panels are decided with evidence

### M0-1 | Set up the repo and record the evaluation setup | infra decision
Make the docs match the real setup, and record how models are trained and evaluated across the two hospitals (A → B, B → A and a mix of both), so every later issue follows the same rules.
Explain: evaluation
Tasks:
- Use one data path everywhere (`CLAUDE.md`, `README.md`, `docs/WORKFLOW.md`), and make the deny rule in `.claude/settings.json` cover the real data folders.
- Document a check command that works on Windows without `make`, and make the ruff hook run on Windows (it calls `python3`, which is the Microsoft Store placeholder there).
- Point `docs/WORKFLOW.md` at GitHub Issues and the installed `mattpocock-skills` plugin, and update its milestone table to this plan.
- Write ADR-0004 describing the evaluation setup below. It replaces the protocol in `CLAUDE.md` and the "both directions" rule in ADR-0001.
- Update `CLAUDE.md`, `CONTEXT.md`, the `run-experiment` skill, the `leakage-reviewer` agent and `configs/b2_lightgbm.yaml` to match ADR-0004.
- Change the `experiments/LOG.md` columns to `A→B | B→A | mix→A | mix→B | AUROC (mean) | labs/pt-day` (it has no rows yet).
Acceptance:
- A teammate can follow `README.md` on Windows and get a passing check, and the hook formats a Python file after an edit.
- `CLAUDE.md`, ADR-0004, `run-experiment` and the config describe the same setup.
- `leakage-reviewer` checks that nothing reads A-test or B-test outside the final evaluation.

### M0-2 | Load both hospitals and create the splits | data test-first
Load each hospital into its own table, split each one into dev and a locked test part, and create the CV folds for the three training setups. The loader never merges the hospitals: the mix setup combines A-dev and B-dev only where it needs them.
Depends on: M0-1
Explain: evaluation
Tasks:
- Write `load_hospital(name)`, which returns one table with `patient_id` (`A_p000001`), `hospital` and rows in hour order. Reuse `load_patients` from `sepsis.data.inspect`.
- Add a CLI that saves each hospital as parquet in a gitignored folder, and a loader that reads it back.
- Split each hospital 80/20 by patient into dev and test, stratified by septic, using `configs/protocol.yaml`. Save the patient-id lists in a gitignored folder, and refuse to overwrite them once they exist.
- Create patient-level CV folds for each training side, from the experiment config's `cv_seed`: A-dev and B-dev (stratified by septic), and A-dev + B-dev (stratified by hospital × septic), including the inner folds that mix needs for nested CV.
- Add a guard that raises if anything outside the final evaluation asks for A-test or B-test.
Acceptance:
- Tests on tiny hand-made `.psv` files cover the ids, hour order and the hospital column.
- A `slow` test confirms 20,336 patients for Hospital A and 20,000 for Hospital B.
- No patient is in both dev and test, or in two validation folds, and the same seed gives the same splits.
- Changing `cv_seed` in an experiment config changes only the CV folds, never which patients are in dev or test.
- Reloading the parquet cache gives tables equal to the originals.

### M0-3 | Check data quality and write cleaning rules | data test-first
Find and handle problems in the raw data before anything is computed from it. Work on A-dev and B-dev only, never on the test parts.
Depends on: M0-2
Tasks:
- Check every column against plausible physiological ranges and expected units (for example Temp in °C, FiO2 as a fraction).
- Check `ICULOS` for gaps, duplicates and patients who don't start at 1, and check that `SepsisLabel` never goes from 1 back to 0.
- Look for constant or empty columns, and patients with very few hours.
- Decide how to handle each problem (drop, clip, set to NaN or keep), and write the decisions as pure cleaning functions with limits in `configs/`. The same rules apply to both hospitals.
Acceptance:
- `docs/eda/data_quality.md` has a findings table: problem, rows and patients affected in each hospital, decision.
- Each cleaning rule has a test on a hand-built table, and its limits come from config.

### M0-4 | Explore both hospitals (EDA) | data chore
Understand the data well enough to make the preprocessing, panel and feature decisions with evidence, and show how different the two hospitals are. Work on A-dev and B-dev only, never on the test parts.
Depends on: M0-2, M0-3
Tasks:
- Missingness per variable, and the time between measurements for each lab.
- Which labs are measured together (co-measurement matrix), as candidate panels.
- Cohort: septic share, length of stay, onset-time distribution, and patients already septic in their first hours.
- Septic vs non-septic distributions, and how each variable changes in the hours before onset.
- Correlations and redundant variables (MAP vs SBP/DBP, Hct vs Hgb, the two bilirubins), with PCA as a plot only.
- Hospital A vs B: compare all of the above side by side. Differences here explain later gaps between the mix and cross-hospital results.
- A short "finding → what it means for the project" list.
Acceptance:
- `docs/eda/README.md` has the plots and the findings list.
- Only aggregates and plots are committed: no patient rows, and no notebook outputs showing raw data.
- The whole analysis re-runs from one script or notebook.

### M0-5 | Review the top-3 PhysioNet 2019 solutions | report chore
Learn from the three best challenge entries, and decide which of their techniques fit this project.
Tasks:
- Confirm the top three teams and their scores from the official challenge paper (Reyna et al., 2020).
- For each team, summarise their preprocessing, features, model, training target and threshold choice.
- Sort every technique into use, adapt or reject. For each, check whether it depends on which labs a clinician ordered (ADR-0002), since that breaks in replay.
- Note what each team reports about performance on the hidden hospital, and how (or whether) they handled differences between hospitals.
Acceptance:
- `docs/research/top3-solutions.md` has a comparison table and the use/adapt/reject list, with a reason for each item and citations.
- Every "use" or "adapt" item names the issue that will build it.

### M0-6 | Decide the preprocessing recipe and lab panels | features decision
Turn the data checks, EDA and top-3 review into fixed decisions for the feature builder.
Depends on: M0-3, M0-4, M0-5
Tasks:
- ADR on preprocessing: forward-fill with NaN before the first value, cleaning, what is fitted on training data only, why trees need no scaling or PCA, and what B1 logistic needs.
- Write `configs/panels.yaml`, mapping all 26 labs to exactly one panel (LACTATE, ABG, CHEM, CBC, LFT, COAG, TROPONIN), based on the co-measurement analysis.
- ADR on reveal rules: the look-back window, and the hour a requested result becomes visible. No look-forward window, because it leaks the future.
- Update `CONTEXT.md` and the `rank1-features` skill if anything changes.
Acceptance:
- A test confirms every lab maps to exactly one panel.
- Each decision cites a finding from M0-3, M0-4 or M0-5.

## M1: Scoring and target | All-zero predictions score 0.0; best predictions score 1.0; U1 - U0 tests match the `rank1-features` table

### M1-1 | Build the scoring toolkit | eval test-first
One way to score predictions with the official scorer and to tune the alarm threshold. The threshold is always tuned on the training side of a setup, never on the hospital being evaluated: whether it transfers is part of what the cross-hospital setups test.
Depends on: M0-2
Explain: evaluation
Tasks:
- Write label and prediction folders (`PredictedProbability|PredictedLabel`) for any patient subset.
- Wrap the unmodified scorer and return AUROC, AUPRC, accuracy, F-measure and utility by name.
- Tune the threshold over a grid from config, using the scorer's own `compute_prediction_utility` on the training side's out-of-fold predictions only (for mix, the inner-CV predictions of each outer fold).
- Score a patient set split by hospital, so the mix setup can report A and B patients separately.
Acceptance:
- All-zero predictions score 0.0, and best predictions score 1.0.
- Tuning returns the best threshold on a toy set, and it can't receive test patients or patients from the evaluated hospital.
- `third_party/` is unchanged.

### M1-2 | Build the U1 − U0 training target | labels test-first
The per-hour regression target from the official utility, as in the rank 1 recipe.
Depends on: M0-2
Tasks:
- Compute `t_sepsis` as the first `SepsisLabel` = 1 row + 6 (by row index, not `ICULOS`), and `dt` = t − `t_sepsis`. Never shift labels again.
- Compute U1, U0 and U1 − U0 with constants from config, and give hours with `dt` > 3 target 0 and weight 0.
- Cross-check against the official scorer.
Acceptance:
- All values in the `rank1-features` table pass.
- On a toy patient, flipping one hour's prediction changes raw utility by exactly U1 − U0.

## M2: Features on a view | Changing any row after t leaves features at t identical; an unrevealed lab never appears

### M2-1 | Build the view and data access layer | features test-first
The only way feature and action code may see data: vitals ≤ t plus revealed labs ≤ t.
Depends on: M0-6
Tasks:
- Build a `View` object from a record and a reveal mask, with rows after t unreachable.
- Add an "as recorded" reveal mask for training, and random lab masking by panel (`lab_masking_p`, seeded) as in ADR-0002.
- Forward-fill within each patient using revealed labs only, keeping NaN before the first value.
- Apply the cleaning rules from M0-3.
Acceptance:
- Unrevealed labs are NaN, masking never touches vitals, and masking is off for evaluation.
- No backfill is used, and values never cross patients.

### M2-2 | Add the clinical features | features test-first
The rank 1 hand-crafted features adapted to replay, plus the techniques kept in M0-5.
Depends on: M2-1
Tasks:
- `shock_index` and `bun_cr` (a zero or missing denominator gives NaN).
- PartialSOFA with `partial_sofa_known_<comp>` flags, and `partial_sofa_delta_24h`.
- SIRS count (0–4), with missing-value handling documented.
- Rolling vital max/min over windows from config.
- `hours_since_<panel>` (NaN if the panel was never revealed).
- The "use" and "adapt" items from M0-5.
Acceptance:
- Every feature has a hand-computed example, and every PartialSOFA threshold boundary is tested.
- There are no lab counts, missing flags or "measured this hour" features.

### M2-3 | Assemble the feature builder and leakage tests | features test-first
One function that turns a view into model-ready rows, guarded by tests that every feature must pass.
Depends on: M2-2
Tasks:
- `build_features(view, config)` returns one row per patient-hour with `ICULOS`, raw values and the enabled features.
- Add a batch mode for training that gives the same rows as building hour by hour.
- Write a test suite (no-future, masking, forbidden columns) that runs on every feature automatically.
Acceptance:
- Changing rows after t leaves features at t identical, and an unrevealed lab never affects any feature.
- Batch output equals hour-by-hour output on a toy patient.
- The time to build features for all dev patients is recorded in the issue.

## M3: Risk models B0-B2 | LOG rows for B0-B2 under all three setups (A → B, B → A, mix); B2 beats B0

### M3-1 | Build the experiment runner | eval test-first
Run any model config under the three setups (A → B, B → A, mix), score it and log it, so every experiment is compared the same way.
Depends on: M1-1, M1-2, M2-3
Explain: evaluation
Tasks:
- Load and validate configs (`id`, `hypothesis`, `model`, `features`, `split`).
- A → B and B → A: run 5-fold CV inside the training hospital's dev part to get out-of-fold predictions and tune the threshold. Then fit on that whole dev part and score the other hospital's dev part with the frozen threshold.
- Mix: run nested CV on A-dev + B-dev. In each outer fold, tune the threshold with an inner CV on the fold's training patients, then score the held-out fold with it. Report A and B patients separately.
- Fit every statistic (fill values, scalers, threshold) on the training side only.
- Save a run folder (predictions, thresholds, config, git SHA) in the gitignored `runs/`.
- Append one row to `experiments/LOG.md` with A→B, B→A, mix→A and mix→B: refuse duplicate ids, write `—` for setups not run, and print the `run-experiment` red-flag warnings.
Acceptance:
- A toy end-to-end run with a dummy model produces a run folder and a LOG row with all four numbers.
- Tests prove that no statistic is fitted on validation patients or on the evaluated hospital, and that A-test and B-test are never loaded.

### M3-2 | Implement baseline models B0, B1 and B2 | risk-model test-first
The three baseline risk models behind one common interface.
Depends on: M2-3
Tasks:
- B0: a rule score from SIRS and PartialSOFA, with thresholds from config.
- B1: logistic regression with fill values and scaling fitted on training folds. The docs don't define its target; suggested `SepsisLabel` with `dt` > 3 weighted 0.
- B2: LightGBM regressor on U1 − U0 with sample weights, NaN passed through, parameters from config.
- One model interface the runner can use for all three.
Acceptance:
- Each model has a toy test, and fitting is reproducible with a fixed seed.
- A test proves B1's fill values and scaler see training data only.

### M3-3 | Run and tune B0–B2 | risk-model experiment
Run the baselines, tune the best one, and check that it learns sensible things.
Depends on: M3-1, M3-2
Explain: evaluation
Tasks:
- Run the B0, B1 and B2 configs under all three setups.
- Tune B2 hyperparameters with CV on the training side only. Keep a change only if the mix result improves and A → B and B → A don't drop by more than `selection.cross_hospital_tolerance` in `configs/protocol.yaml`.
- Check feature importance (or SHAP) for anything suspicious.
- Report the gap between the mix and cross-hospital results, and link it to the hospital differences from M0-4.
- Compare with the published rank 1 numbers, and explain the gap (ADR-0002, no hospital C).
Acceptance:
- LOG rows exist for B0, B1, B2 and the tuned B2, and B2 beats B0 in all three setups (the M3 check).
- Any red flag is investigated with `leakage-reviewer`.

## M4: Signatures (optional) | LOG row; B3 is kept only if it improves the mix result without hurting A → B or B → A

### M4-1 | Try signature features (B3) | features experiment optional
Test whether rank 1's signature features help on top of B2.
Depends on: M3-3
Tasks:
- Install `iisignature` on Windows (`uv sync --extra signatures`), with a fallback plan if it fails.
- Add signature features over windows ending at t: `[partial_sofa, MAP, bun_cr]` with a time channel, and vitals after cumulative sum and lead-lag, as in `rank1-features`.
- Run B3 and compare with the tuned B2.
Acceptance:
- The no-future and masking tests pass for the new features.
- A LOG row with all three setups exists, and the keep/drop decision is recorded. Keep B3 only if it beats B2 on mix without hurting A → B or B → A (the rank 1 team blamed their hospital C failure partly on overfitting local patterns).

## M5: Replay | ALL reproduces B2's full-information utility; NONE uses zero labs

### M5-1 | Build the replay engine | replay test-first
Re-run patients hour by hour with labs hidden until a policy requests them.
Depends on: M2-3, M3-1, M3-2
Explain: evaluation
Tasks:
- `reveal_panel` follows the M0-6 reveal rules, and logs a panel with no value as an unavailable request without imputing it.
- Define a policy interface that receives the view only.
- Build the hourly loop: view → risk → alarm → policy → reveal.
- Score each patient with a model that didn't train on them, using that setup's frozen threshold: the other hospital's model for A → B and B → A, and the fold model for mix.
Acceptance:
- A test confirms a requested result appears only when the reveal rules allow, and never a value after t.
- A test confirms policies can't access the record.

### M5-2 | Add baseline policies and replay metrics | replay experiment
The reference policies every learned policy is compared with.
Depends on: M5-1
Tasks:
- Add NONE, ALL and AS_RECORDED (AS_RECORDED reads the record by design, as a baseline only).
- Compute labs per patient-day and the unavailable-request rate, and document the counting unit (lab or panel).
- Score replay predictions through M1-1.
- Run all three policies under all three setups, on dev data only.
Acceptance:
- In every setup, ALL reproduces B2's utility and NONE uses zero labs (the M5 check).
- LOG rows exist for all three policies.

## M6: Action model | Utility vs labs per patient-day beats AS_RECORDED at equal lab count

### M6-1 | Design VoI and build its training targets | action-model decision
Decide what "value of information" means here, then build the data to learn it.
Depends on: M5-2
Tasks:
- ADR: the VoI definition (change in risk or in the alarm decision), the training data, and how the bias is handled (a panel's counterfactual is only known when the record has a value).
- Add panel costs to config, each with its source or assumption.
- Build the VoI targets from out-of-fold risk scores.
Acceptance:
- The ADR is merged, and the target matches its definition on a hand-built patient.

### M6-2 | Train VoI models and add the VoI policy | action-model test-first
Learn VoI per panel and turn it into a policy that can also WAIT.
Depends on: M6-1
Tasks:
- Train one VoI model per panel on view features only, never on validation patients.
- `select_action` returns the best panel, or WAIT when no VoI exceeds its cost.
- Plug the VoI policy into replay through the M5-1 interface.
Acceptance:
- A leakage test confirms VoI inputs come from the view only.
- `select_action` is pure, with deterministic ties, and the policy runs end to end on a toy patient.

### M6-3 | Sweep costs and compare with AS_RECORDED | action-model experiment
Show the trade-off between utility and how many labs are ordered.
Depends on: M6-2
Tasks:
- Sweep a cost scale, replaying under all three setups, each with its own frozen threshold.
- Plot utility vs labs per patient-day for VoI, AS_RECORDED, ALL and NONE.
Acceptance:
- A LOG row exists for each sweep point.
- The issue states plainly whether VoI beats AS_RECORDED at equal lab count (the M6 check).

## M7: Guideline layer | Every rule's `tests:` pass; the action model never suppresses a FORCE

### M7-1 | Collect and triage the guideline statements | guideline chore needs-input
Get the source text that rules must be written from.
Tasks:
- Obtain the Surviving Sepsis Campaign 2026 adult guideline (plus Sepsis-3 and, optionally, CMS SEP-1), and check the copyright before committing anything.
- Put the source material in `docs/guidelines/`.
- Triage every statement in `docs/guidelines/triage.md`: A computable (rule), B applies but not computable (gap record), C out of scope.
Acceptance:
- Every statement has a bin and a one-line reason.

### M7-2 | Build the guideline engine | guideline test-first
Run YAML rules deterministically against a view.
Depends on: M2-3
Tasks:
- Define the rule schema and a loader that requires provenance and loads only `active` rules for replay.
- Build a safe condition evaluator over view features, without Python `eval`, with defined behaviour for missing values.
- Add `definitions.yaml` with `DEF-SIRS` and `DEF-SCREEN-POS`.
- `evaluate_rules` returns FORCE, BLOCK, PRIOR and gap reminders, each traceable to a `rule_id`.
- Run every rule's `tests:` block as part of `make check`.
Acceptance:
- Invalid rules and missing provenance give clear errors.
- A failing rule test fails `make check`, and the engine is made of pure functions.

### M7-3 | Write and review the rules | guideline chore needs-input
Turn the triaged statements into reviewed rules and gap records.
Depends on: M7-1, M7-2
Tasks:
- Write the lactate rules first with the `add-guideline-rule` skill, then the remaining bin A statements.
- Write a gap record for every bin B statement.
- Have a person review each rule from `draft` to `reviewed` to `active`, including every `project_assumption`.
Acceptance:
- Every rule has page numbers and three tests: fires, blocked by its condition, and false trigger.
- No rule is `active` without a review.

### M7-4 | Combine guidelines with the action model | guideline experiment
Make the policy follow the guideline, and measure how often it does.
Depends on: M6-2, M7-3
Tasks:
- Decide FORCE vs BLOCK precedence for the same panel, and write it down.
- Apply the rules: BLOCK masks a panel, PRIOR adds its bonus to VoI, FORCE always goes through.
- Define and compute guideline adherence.
- Replay VoI + guidelines and compare with VoI alone.
Acceptance:
- A property test confirms the action model never removes a FORCE (the M7 check).
- A LOG row reports utility, labs per patient-day and adherence.

## M8: Final evaluation and report | One table: policy × {utility, AUROC, labs/patient-day, guideline adherence}

### M8-1 | Run the final evaluation | report experiment
Score the frozen system once on the locked test parts, which nothing has seen during the project.
Depends on: M3-3, M5-2, M6-3, M7-4
Explain: evaluation
Tasks:
- Freeze the model, threshold and policy configs, and tag the commit before opening A-test and B-test.
- A → B: fit on all of A-dev and score B-test. B → A: fit on all of B-dev and score A-test. Each uses the threshold tuned on its own training side.
- Mix: fit on A-dev + B-dev and score A-test and B-test separately (mix → A, mix → B).
- Replay every policy on A-test and B-test with the same models.
Acceptance:
- A-test and B-test were used exactly once, by the tagged commit.
- LOG rows report A→B, B→A, mix→A and mix→B on the test parts.

### M8-2 | Write the report | report chore
Present the results honestly and make them reproducible.
Depends on: M8-1
Tasks:
- Build the policy × {utility, AUROC, labs/patient-day, adherence} table and the figures.
- Optionally, compare with and without count features at full information, as analysis only (ADR-0002).
- Write methods, results and limitations: the drop from mix to cross-hospital results, the gap to rank 1, and that the challenge's hidden hospital C isn't available.
- Check the project reproduces from a fresh clone by following the README.
Acceptance:
- Every number traces to a LOG row.
- The fresh-clone reproduction works, and no data or artifacts are committed.
"""


@dataclass
class Issue:
    id: str
    name: str
    labels: list[str]
    milestone: str
    goal: list[str] = field(default_factory=list)
    depends_on: str = ""
    explain: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.id} · {self.name}"

    @property
    def body(self) -> str:
        parts = ["## Goal", " ".join(self.goal), ""]
        if self.depends_on:
            parts += [f"**Depends on:** {self.depends_on}", ""]
        if "evaluation" in self.explain:
            parts += [EVALUATION_SETUP]
        parts += ["## Tasks", *[f"- [ ] {t}" for t in self.tasks], ""]
        parts += ["## Acceptance criteria", *[f"- [ ] {c}" for c in self.criteria], ""]
        parts += ["## Definition of done", *[f"- [ ] {d}" for d in DEFINITION_OF_DONE]]
        return "\n".join(parts) + "\n"


def parse_plan(text: str) -> tuple[dict[str, str], list[Issue]]:
    """Return milestones as {title: description}, and the issues in order."""
    milestones: dict[str, str] = {}
    issues: list[Issue] = []
    section = "goal"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("### "):
            issue_id, name, labels = (part.strip() for part in line[4:].split(" | "))
            issues.append(Issue(issue_id, name, labels.split(), list(milestones)[-1]))
            section = "goal"
        elif line.startswith("## "):
            title, description = (part.strip() for part in line[3:].split(" | ", 1))
            milestones[title] = description
        elif line.startswith("Depends on:"):
            issues[-1].depends_on = line.split(":", 1)[1].strip()
        elif line.startswith("Explain:"):
            topic = line.split(":", 1)[1].strip()
            if topic != "evaluation":
                raise ValueError(f"{issues[-1].id}: unknown explanation {topic!r}")
            issues[-1].explain.append(topic)
        elif line in ("Tasks:", "Acceptance:"):
            section = line[:-1].lower()
        elif line.startswith("- ") and section == "tasks":
            issues[-1].tasks.append(line[2:])
        elif line.startswith("- ") and section == "acceptance":
            issues[-1].criteria.append(line[2:])
        elif section == "goal":
            issues[-1].goal.append(line)
        else:
            raise ValueError(f"{issues[-1].id}: unexpected line {line!r}")
    return milestones, issues


def validate_plan(milestones: dict[str, str], issues: list[Issue]) -> None:
    """Fail before touching GitHub if the plan has a mistake."""
    ids = [issue.id for issue in issues]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate issue id")
    for issue in issues:
        unknown = set(issue.labels) - set(LABELS)
        if unknown:
            raise ValueError(f"{issue.id}: unknown labels {sorted(unknown)}")
        if not (issue.goal and issue.tasks and issue.criteria):
            raise ValueError(f"{issue.id}: needs a goal, tasks and acceptance criteria")
        if not ISSUE_ID.fullmatch(issue.id) or not issue.milestone.startswith(
            issue.id.split("-")[0] + ":"
        ):
            raise ValueError(f"{issue.id}: bad id or wrong milestone {issue.milestone!r}")
        for dep in re.findall(r"M\d+-\d+", issue.depends_on):
            if dep not in ids[: ids.index(issue.id)]:
                raise ValueError(f"{issue.id}: depends on {dep}, which is not an earlier issue")
    if set(milestones) != {issue.milestone for issue in issues}:
        raise ValueError("a milestone has no issues")


def issue_id(title: str) -> str:
    return title.split(" · ")[0].strip()


def run_gh(*args: str, stdin: str | None = None) -> str:
    """Run the GitHub CLI and return its output, raising with its error message on failure."""
    result = subprocess.run(
        ["gh", *args], input=stdin, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {result.stderr.strip()}")
    return result.stdout.strip()


class GitHub:
    """Reads always run. Writes run, or are only printed in a dry run."""

    def __init__(self, repo: str, dry_run: bool) -> None:
        self.repo = repo
        self.dry_run = dry_run
        self.writes = 0

    def read(self, *args: str) -> str:
        return run_gh(*args)

    def write(self, summary: str, *args: str, stdin: str | None = None, pause: bool = False) -> str:
        if self.dry_run:
            self.writes += 1
            print(f"  would {summary}")
            return ""
        output = run_gh(*args, stdin=stdin)
        self.writes += 1
        print(f"  {summary}")
        if pause:
            time.sleep(SECONDS_BETWEEN_WRITES)
        return output


def check_access(gh: GitHub) -> None:
    permission = json.loads(gh.read("repo", "view", gh.repo, "--json", "viewerPermission"))
    level = permission["viewerPermission"]
    if level not in {"ADMIN", "MAINTAIN", "WRITE"}:
        raise RuntimeError(f"you have {level} access to {gh.repo}, but editing issues needs WRITE")


def sync_labels(gh: GitHub) -> None:
    print("Labels")
    existing = json.loads(
        gh.read(
            "label", "list", "--repo", gh.repo, "--limit", "200", "--json", "name,color,description"
        )
    )
    current = {label["name"]: (label["color"].lower(), label["description"]) for label in existing}
    for name, (color, description) in LABELS.items():
        if current.get(name) == (color, description):
            continue
        gh.write(
            f"create or update label {name}",
            "label", "create", name, "--repo", gh.repo,
            "--color", color, "--description", description, "--force",
        )  # fmt: skip


def sync_milestones(gh: GitHub, milestones: dict[str, str]) -> None:
    """Match milestones by their "M<n>" prefix, so a renamed milestone is updated, not duplicated."""
    print("Milestones")
    listing = json.loads(gh.read("api", f"repos/{gh.repo}/milestones?state=all&per_page=100"))
    by_prefix = {m["title"].split(":")[0]: m for m in listing}
    for title, description in milestones.items():
        existing = by_prefix.get(title.split(":")[0])
        if existing is None:
            gh.write(
                f"create milestone {title}",
                "api", f"repos/{gh.repo}/milestones",
                "-f", f"title={title}", "-f", f"description={description}",
            )  # fmt: skip
        elif (existing["title"], existing.get("description") or "") != (title, description):
            gh.write(
                f"update milestone {existing['title']!r} -> {title!r}",
                "api", "-X", "PATCH", f"repos/{gh.repo}/milestones/{existing['number']}",
                "-f", f"title={title}", "-f", f"description={description}",
            )  # fmt: skip


def normalise(text: str | None) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def sync_issues(
    gh: GitHub, issues: list[Issue], overwrite_progress: bool = False
) -> tuple[list[str], list[str]]:
    """Edit, create and close issues. Return (planned issue URLs, closed issue URLs).

    An issue with ticked checkboxes is not edited (that would untick them) unless
    overwrite_progress is set.
    """
    print("Issues")
    listing = json.loads(
        gh.read(
            "issue",
            "list",
            "--repo",
            gh.repo,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number,title,body,state,labels,milestone,url",
        )  # fmt: skip
    )
    existing = {}
    for item in listing:
        key = issue_id(item["title"])
        # If an id is used twice, prefer the open issue.
        if ISSUE_ID.fullmatch(key) and (key not in existing or item["state"] == "OPEN"):
            existing[key] = item

    planned_urls: list[str] = []
    counts = {"created": 0, "edited": 0, "unchanged": 0, "reopened": 0, "closed": 0}
    for issue in issues:
        item = existing.get(issue.id)
        if item is None:
            output = gh.write(
                f"create {issue.title}",
                "issue", "create", "--repo", gh.repo, "--title", issue.title,
                "--milestone", issue.milestone, "--body-file", "-",
                *[arg for label in issue.labels for arg in ("--label", label)],
                stdin=issue.body, pause=True,
            )  # fmt: skip
            planned_urls.append(output.splitlines()[-1] if output else "")
            counts["created"] += 1
            continue

        planned_urls.append(item["url"])
        number = str(item["number"])
        if item["state"] != "OPEN":
            gh.write(
                f"reopen #{number} {issue.title}", "issue", "reopen", number, "--repo", gh.repo
            )
            counts["reopened"] += 1

        current_labels = {label["name"] for label in item["labels"]}
        add = sorted(set(issue.labels) - current_labels)
        remove = sorted((current_labels & set(LABELS)) - set(issue.labels))
        milestone = (item.get("milestone") or {}).get("title")
        same = (
            item["title"] == issue.title
            and normalise(item["body"]) == normalise(issue.body)
            and milestone == issue.milestone
            and not add
            and not remove
        )
        if same:
            counts["unchanged"] += 1
            continue
        if not overwrite_progress and re.search(r"- \[[xX]\]", item["body"] or ""):
            print(f"  skip #{number} {issue.title}: it has ticked boxes (use --overwrite-progress)")
            counts["skipped (progress)"] = counts.get("skipped (progress)", 0) + 1
            continue
        args = ["issue", "edit", number, "--repo", gh.repo, "--title", issue.title]
        args += ["--milestone", issue.milestone, "--body-file", "-"]
        if add:
            args += ["--add-label", ",".join(add)]
        if remove:
            args += ["--remove-label", ",".join(remove)]
        gh.write(f"edit #{number} -> {issue.title}", *args, stdin=issue.body, pause=True)
        counts["edited"] += 1

    planned_ids = {issue.id for issue in issues}
    closed_urls: list[str] = []
    for key, item in sorted(existing.items(), key=lambda kv: kv[1]["number"]):
        if key in planned_ids or item["state"] != "OPEN":
            continue
        gh.write(
            f"close #{item['number']} {item['title']}",
            "issue", "close", str(item["number"]), "--repo", gh.repo, "--reason", "not planned",
            "--comment", "Closed when the plan was reworked into fewer, larger issues. "
            "Its work is now part of the open issues in the same milestone.",
            pause=True,
        )  # fmt: skip
        closed_urls.append(item["url"])
        counts["closed"] += 1

    print("  " + ", ".join(f"{n} {what}" for what, n in counts.items()))
    return planned_urls, closed_urls


def sync_project(
    gh: GitHub, owner: str, title: str, planned_urls: list[str], closed_urls: list[str]
) -> None:
    """Find or create the project, link it, add planned issues and remove closed ones."""
    print("Project board")
    projects = json.loads(
        gh.read("project", "list", "--owner", owner, "--limit", "100", "--format", "json")
    )["projects"]
    project = next((p for p in projects if p["title"] == title), None)
    if project is None:
        print(f"  no project called {title!r} under {owner}")
        output = gh.write(
            f"create project {title!r}",
            "project", "create", "--owner", owner, "--title", title, "--format", "json",
        )  # fmt: skip
        if gh.dry_run:
            return
        project = json.loads(output)
    number = str(project["number"])

    items = json.loads(
        gh.read(
            "project", "item-list", number, "--owner", owner, "--limit", "1000", "--format", "json"
        )
    )["items"]
    on_board = {(item.get("content") or {}).get("url"): item["id"] for item in items}

    try:
        gh.write(
            "link project to the repo",
            "project", "link", number, "--owner", owner, "--repo", gh.repo.split("/")[1],
        )  # fmt: skip
    except RuntimeError as error:
        # The link only lists the project on the repo's Projects tab; issues can still be added.
        print(f"  warning: could not link the project to the repo: {error}")

    for url in planned_urls:
        if url and url not in on_board:
            gh.write(
                f"add {url} to the board",
                "project",
                "item-add",
                number,
                "--owner",
                owner,
                "--url",
                url,
            )
    for url in closed_urls:
        if url in on_board:
            gh.write(
                f"remove {url} from the board",
                "project", "item-delete", number, "--owner", owner, "--id", on_board[url],
            )  # fmt: skip
    if not gh.dry_run:
        print(f"Open the board: {project['url']}")


def print_plan(milestones: dict[str, str], issues: list[Issue]) -> None:
    for title in milestones:
        group = [issue for issue in issues if issue.milestone == title]
        print(f"\n{title} ({len(group)} issues)")
        for issue in group:
            print(f"  {issue.title}  [{', '.join(issue.labels)}]")
    print(f"\nPlan: {len(LABELS)} labels, {len(milestones)} milestones, {len(issues)} issues\n")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/name of the repository")
    parser.add_argument("--project-owner", default=DEFAULT_PROJECT_OWNER, help="who owns the board")
    parser.add_argument("--project-title", default=DEFAULT_PROJECT_TITLE)
    parser.add_argument("--no-project", action="store_true", help="skip the project board")
    parser.add_argument(
        "--overwrite-progress",
        action="store_true",
        help="also edit issues with ticked checkboxes (this unticks them)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would change, change nothing"
    )
    args = parser.parse_args(argv)

    milestones, issues = parse_plan(ISSUES_TEXT)
    validate_plan(milestones, issues)
    print_plan(milestones, issues)

    if shutil.which("gh") is None:
        if args.dry_run:
            print("GitHub CLI not found, so only the plan is shown. Nothing was changed.")
            return 0
        print("GitHub CLI not found. Install it with: winget install GitHub.cli", file=sys.stderr)
        return 1

    gh = GitHub(args.repo, args.dry_run)
    try:
        check_access(gh)
        sync_labels(gh)
        sync_milestones(gh, milestones)
        planned_urls, closed_urls = sync_issues(gh, issues, args.overwrite_progress)
        if not args.no_project:
            sync_project(gh, args.project_owner, args.project_title, planned_urls, closed_urls)
    except RuntimeError as error:
        print(f"\nStopped: {error}", file=sys.stderr)
        print("Fix the problem and run the script again. Finished work is skipped.")
        return 1

    if args.dry_run:
        print(f"\nDry run: {gh.writes} changes would be made. Nothing was changed.")
    else:
        print(f"\nDone: {gh.writes} changes made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
