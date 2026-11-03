# Workflow

How this project is built with Claude Code. It combines Anthropic's Claude Code best practices, Matt Pocock's engineering skills, and an ML experiment loop for the model work.

## 1. Principles (from Anthropic's best practices)

1. **Give Claude a check it can run.** Every milestone below ends in a command that passes or fails. "Looks done" is not done.
2. **Explore → plan → implement → commit.** Use plan mode for anything touching several files. Skip it when you could describe the diff in one sentence.
3. **Context is the scarce resource.** Run `/clear` between unrelated tasks. After two failed corrections, `/clear` and rewrite the prompt. Use subagents for wide investigation.
4. **CLAUDE.md stays short.** It holds only rules that apply every session. Occasional knowledge goes into skills (`.claude/skills/`).
5. **Deterministic guarantees go in hooks and permissions, not prose.** Ruff runs after every edit; `data/raw/` and `third_party/` can't be edited.
6. **The writer doesn't grade its own work.** A fresh-context reviewer (`leakage-reviewer`, `/code-review`) checks every diff.

## 2. One-time setup

```bash
# Python env
uv sync                       # add --extra signatures later for B3

# Official scorer (vendored, never edited)
# copy evaluate_sepsis_score.py from github.com/physionetchallenges/evaluation-2019 into third_party/

# Matt Pocock's engineering skills
npx skills@latest add mattpocock/skills
claude                        # then, inside Claude Code:
/setup-matt-pocock-skills     # choose local markdown files as the tracker, e.g. docs/tickets/
```

Skill names in that repo have changed between versions (e.g. `to-prd` → `to-spec`, `to-issues` → `to-tickets`). If a command below doesn't exist, run `/ask-matt` or check the installed list.

## 3. Two kinds of work, two loops

ML projects mix code that must be **exactly right** with models that must be **good enough**. They need different loops.

| | Deterministic code | Model experiments |
|---|---|---|
| Examples | data loader, feature builder, U1−U0 target, PartialSOFA, guideline engine, replay masking | B0–B3 risk models, signatures, action model, thresholds |
| "Correct" means | tests pass | metric beats the previous baseline on hospital-held-out data |
| Loop | `/tdd` (red → green → refactor) | `run-experiment` skill: config → train → score → log |
| Evidence | test output | a new row in `experiments/LOG.md` |

### Loop A: a feature or component (Matt Pocock flow)

| Step | Command | What happens |
|---|---|---|
| 1. Sharpen | `/grill-with-docs` | Claude interviews you about the milestone and updates `CONTEXT.md` and `docs/adr/` |
| 2. Spec | `/to-spec` | The conversation becomes a self-contained spec with an end-to-end verification step |
| 3. Tickets | `/to-tickets` | The spec becomes small tickets with blockers |
| 4. Build | `/clear`, then `/implement <ticket>` | Fresh context; test-first via `/tdd`, ends with `/code-review` |
| 5. Domain check | "use the leakage-reviewer subagent on the diff" | Fresh-context check for leakage and clinical logic |
| 6. Commit | "commit with a descriptive message" | One ticket, one commit |

Use `/grill-me` instead of step 1 for a quick sanity check of a plan without docs updates. Use `/wayfinder` when a milestone is too big for one session.

### Loop B: a model experiment

1. Write or copy a config in `configs/` (one hypothesis per config, e.g. "add PartialSOFA 24 h delta").
2. `/run-experiment configs/<name>.yaml`
3. Read the new `LOG.md` row. Keep the change only if utility improves on **both** hospital-held-out directions without a large AUROC drop.
4. Record the decision in the config's `note` field or an ADR if it changes the default recipe.

### Other skills worth knowing

- `/diagnosing-bugs`: a metric suddenly jumps or collapses. Suspect leakage first.
- `/prototype`: throwaway spike, e.g. "do signatures help at all?", before productionizing.
- `/research`: literature questions saved as a cited markdown file in `docs/research/`.
- `/teach`: learn a concept across sessions (signatures, off-policy evaluation).
- `/handoff`: end of a long session; the next session starts from its summary.
- `/improve-codebase-architecture`: once per milestone, to keep modules deep and interfaces small.
- Project skills: `rank1-features` (formulas, loaded automatically), `run-experiment`, `add-guideline-rule`.

## 4. Milestones

Each milestone is done only when its **check** passes and the evidence is pasted into the ticket.

| # | Milestone | Check (pass/fail) |
|---|---|---|
| M0 | Scaffold, loader for sets A and B into one table with `patient_id`, `hospital` | Patient counts per hospital match the raw file counts; `make check` green |
| M1 | Official scorer wrapper + U1−U0 target | All-zero predictions score utility 0.0; "best" predictions score 1.0; U1−U0 unit tests match the table in `rank1-features` |
| M2 | Feature builder on a *view* (vitals ≤ t + revealed labs) | Leakage test: changing any row > t leaves features at t identical. Masking test: an unrevealed lab never appears. |
| M3 | B0 rules, B1 logistic, B2 LightGBM | LOG rows for B0–B2 in both hospital directions; B2 > B0 |
| M4 | B3 signatures (optional) | LOG row; keep only if it beats B2 in both directions |
| M5 | Replay simulator + baseline policies ALL / NONE / AS_RECORDED | ALL policy reproduces B2's full-information utility; NONE uses zero labs |
| M6 | Action model (VoI) + WAIT | Utility vs. labs per patient-day curve beats AS_RECORDED at equal lab count |
| M7 | Guideline engine + YAML rules | Every rule's `tests:` pass; the action model never suppresses a FORCE (property test) |
| M8 | Report | One table: policy × {utility, AUROC, labs/patient-day, guideline adherence} |

## 5. Session hygiene

- One milestone ticket per session. Name it with `/rename m2-feature-builder`.
- Start implementation sessions with a spec or ticket, not a vague prompt.
- Use `/btw` for side questions that shouldn't fill context.
- `Esc Esc` or `/rewind` to undo a bad turn. Git is still the real safety net.
- Long unattended run: set a `/goal`, e.g. "`make check` passes and the M2 leakage test exists and passes".

## 6. Prompt templates

**Start a milestone**

```text
/grill-with-docs We're starting M2 (feature builder). Read CLAUDE.md, CONTEXT.md and the
rank1-features skill. The builder takes a view (vitals ≤ t + revealed labs) and returns one
feature row per patient-hour. Interview me on window sizes, panel windows for "revealed",
and which rank-1 features survive the no-measurement-count rule.
```

**Debug a suspicious metric**

```text
/diagnosing-bugs B2 utility jumped from 0.38 to 0.61 after the last commit. Treat it as
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
