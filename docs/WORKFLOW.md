# Workflow

How this project is built with Claude Code. It combines Anthropic's Claude Code best practices, Matt Pocock's engineering skills, and an ML experiment loop for the model work.

## 1. Principles (from Anthropic's best practices)

1. **Give Claude a check it can run.** Every milestone below ends in a command that passes or fails. "Looks done" is not done.
2. **Explore → plan → implement → commit.** Use plan mode for anything touching several files. Skip it when you could describe the diff in one sentence.
3. **Context is the scarce resource.** Run `/clear` between unrelated tasks. After two failed corrections, `/clear` and rewrite the prompt. Use subagents for wide investigation.
4. **CLAUDE.md stays short.** It holds only rules that apply every session. Occasional knowledge goes into skills (`.claude/skills/`).
5. **Deterministic guarantees go in hooks and permissions, not prose.** Ruff runs after every edit; the raw data folders (`data/training_setA/`, `data/training_setB/`) and `third_party/` can't be edited.
6. **The writer doesn't grade its own work.** A fresh-context reviewer (`leakage-reviewer`, `/code-review`) checks every diff.

## 2. One-time setup

```bash
# Python env
uv sync                       # add --extra signatures later for B3

# Official scorer (vendored, never edited)
# copy evaluate_sepsis_score.py from github.com/physionetchallenges/evaluation-2019 into third_party/

# Matt Pocock's engineering skills (Claude Code plugin)
claude plugin install mattpocock-skills@claude-plugins-official

# GitHub CLI, for the issues and the project board
winget install GitHub.cli     # Windows; see cli.github.com for other systems
gh auth login
gh auth refresh -s project
```

Work is tracked in **GitHub Issues** on the "Sepsis ML Project" board, one issue per milestone stage (`M0-1`, `M0-2`, …). The plan lives in `scripts/create_github_issues.py`; change it there and run it to sync GitHub:

```bash
python scripts/create_github_issues.py --dry-run   # preview, changes nothing
python scripts/create_github_issues.py             # apply
```

It skips issues with ticked checkboxes unless you pass `--overwrite-progress`.

Matt Pocock's skill names change between plugin versions. If a command below doesn't exist, check the installed list with `/` in Claude Code.

## 3. Two kinds of work, two loops

ML projects mix code that must be **exactly right** with models that must be **good enough**. They need different loops.

| | Deterministic code | Model experiments |
|---|---|---|
| Examples | data loader, feature builder, U1−U0 target, PartialSOFA, guideline engine, replay masking | B0–B3 risk models, signatures, action model, thresholds |
| "Correct" means | tests pass | metric improves in the mix setup without getting worse in A → B or B → A (ADR-0004) |
| Loop | `/tdd` (red → green → refactor) | `run-experiment` skill: config → train → score → log |
| Evidence | test output | a new row in `experiments/LOG.md` |

### Loop A: a feature or component (one GitHub issue)

| Step | Command | What happens |
|---|---|---|
| 1. Sharpen | `/grilling` | Claude interviews you about the issue until the plan has no gaps |
| 2. Record | `/domain-modeling` | New terms go into `CONTEXT.md`, decisions into `docs/adr/` |
| 3. Build | `/clear`, then `/tdd` with the issue text | Fresh context; test-first, red → green → refactor |
| 4. Review | `/code-review` | Checks the diff against the repo's standards and the issue |
| 5. Domain check | "use the leakage-reviewer subagent on the diff" | Fresh-context check for leakage and clinical logic |
| 6. Commit | "commit with a descriptive message" | One issue, one or a few commits; the PR or last commit says `Closes #<n>` |

Use `/codebase-design` when a module's interface feels wrong, and `/pr` to write a pull request description.

### Loop B: a model experiment

1. Write or copy a config in `configs/` (one hypothesis per config, e.g. "add PartialSOFA 24 h delta").
2. `/run-experiment configs/<name>.yaml`
3. Read the new `LOG.md` row. Keep the change only if the mix result improves and A → B and B → A don't drop by more than `selection.cross_hospital_tolerance` (`configs/protocol.yaml`), without a large AUROC drop.
4. Record the decision in the config's `note` field or an ADR if it changes the default recipe.

### Other skills worth knowing

- `/diagnosing-bugs`: a metric suddenly jumps or collapses. Suspect leakage first.
- `/prototype`: throwaway spike, e.g. "do signatures help at all?", before productionizing.
- `/research`: literature questions saved as a cited markdown file, e.g. the top-3 solutions review in `docs/research/`.
- Project skills: `rank1-features` (formulas, loaded automatically), `run-experiment`, `add-guideline-rule`.

## 4. Milestones

Each milestone is done only when its **check** passes and the evidence is pasted into the issue. Evaluation setups are defined in ADR-0004.

| # | Milestone | Check (pass/fail) |
|---|---|---|
| M0 | Data understanding: load both hospitals, dev/test split per hospital, data quality, EDA, top-3 review, preprocessing and panel decisions | Patient counts match the raw files (20,336 A / 20,000 B); no patient in two splits; EDA and decisions documented; `make check` green |
| M1 | Official scorer wrapper, threshold tuning, U1−U0 target | All-zero predictions score 0.0; "best" predictions score 1.0; U1−U0 unit tests match the table in `rank1-features` |
| M2 | Feature builder on a *view* (vitals ≤ t + revealed labs) | Leakage test: changing any row > t leaves features at t identical. Masking test: an unrevealed lab never appears. |
| M3 | Experiment runner; B0 rules, B1 logistic, B2 LightGBM | LOG rows for B0–B2 with A→B, B→A, mix→A and mix→B; B2 > B0 in all three setups |
| M4 | B3 signatures (optional) | LOG row; keep only if it improves mix without hurting A→B or B→A |
| M5 | Replay simulator + baseline policies ALL / NONE / AS_RECORDED | ALL reproduces B2's full-information utility in every setup; NONE uses zero labs |
| M6 | Action model (VoI) + WAIT | Utility vs. labs per patient-day curve beats AS_RECORDED at equal lab count |
| M7 | Guideline engine + YAML rules | Every rule's `tests:` pass; the action model never suppresses a FORCE (property test) |
| M8 | Final evaluation on A-test and B-test, and the report | One table: policy × {utility, AUROC, labs/patient-day, guideline adherence}, for A→B, B→A, mix→A and mix→B |

## 5. Session hygiene

- One issue per session. Name it with `/rename m0-2-load-and-split`.
- Start implementation sessions with the issue text, not a vague prompt.
- Use `/btw` for side questions that shouldn't fill context.
- `Esc Esc` or `/rewind` to undo a bad turn. Git is still the real safety net.
- Long unattended run: set a `/goal`, e.g. "`make check` passes and the M2 leakage test exists and passes".

## 6. Prompt templates

**Start an issue**

```text
/grilling We're starting issue M2-1 (view and data access layer). Read CLAUDE.md, CONTEXT.md,
ADR-0004 and the rank1-features skill, then the issue text below. Interview me on anything the
issue leaves open: reveal windows, masking, and which features survive the no-measurement-count rule.
```

**Debug a suspicious metric**

```text
/diagnosing-bugs B2 mix→A utility jumped from 0.38 to 0.61 after the last commit. Treat it as
leakage until proven otherwise. Check the diff against the hard rules in CLAUDE.md, and
reproduce with the M2 leakage test before changing anything.
```

**Review before commit**

```text
Use the leakage-reviewer subagent to review the current diff against CLAUDE.md's hard rules.
Report only findings that would invalidate a result or break a stated requirement.
```

## 7. Optional ML skills

From `k-dense-ai/claude-scientific-skills`, install only what you use, since every skill adds to the list Claude scans:
- `shap`: explain B2 per patient-hour
- `exploratory-data-analysis`: first look at missingness and distributions
- `scikit-learn`: B1 and calibration
- `polars`: only if pandas gets slow

## 8. Maintenance

- After each milestone, ask: "Which CLAUDE.md lines did you not need? Which mistakes repeated?" Prune or move lines into skills. Run `/doctor` for suggested cuts.
- If Claude ignores a rule, the file is probably too long. Shorten it before adding emphasis.
- A decision that changes the default recipe gets an ADR, not a CLAUDE.md line.
