# ADR-0003: Guideline rules are compiled offline into reviewed YAML

Status: accepted

## Context
The guideline layer must apply the Surviving Sepsis Campaign 2026 adult guidelines (plus Sepsis-3 definitions and, optionally, the CMS SEP-1 repeat-lactate rule). Options were runtime retrieval with an LLM, or rules extracted once and executed deterministically.

## Decision
- Rules are extracted offline with the `add-guideline-rule` skill, reviewed by a person, and stored in `src/sepsis/guideline/rules/*.yaml` under version control.
- No LLM or retrieval call in the hourly loop. Retrieval may be used only to show the source text of a rule that fired.
- Every rule field carries provenance: `guideline_statement`, `guideline_remark`, `external_spec` or `project_assumption`.
- Statements that apply but can't be computed from this data are stored as gap records.
- Rule actions: FORCE (cannot be suppressed by the action model), BLOCK (masks a panel), PRIOR (adds a bonus to a panel's VoI, scaled by strength and certainty).

## Consequences
- Replays are reproducible and every recommendation can be traced to a statement and page.
- Updating to a new guideline version means re-running the extraction and reviewing the YAML diff.
