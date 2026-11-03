---
name: add-guideline-rule
description: Turn one Surviving Sepsis Campaign (or Sepsis-3 / SEP-1) statement into a reviewed YAML rule or gap record with provenance and tests.
disable-model-invocation: true
argument-hint: <statement number or pasted statement text>
---

Add a guideline rule for: $ARGUMENTS

The user supplies the statement text, and ideally its rationale and remarks with page numbers. Never write a rule from memory of the guideline: if the source text isn't in the conversation or in `docs/guidelines/`, ask for it.

1. **Triage** the statement into one bin:
   - A: computable with PhysioNet 2019 columns → rule
   - B: applies, but not computable (cultures, antibiotics, fluids) → gap record
   - C: out of scope (pediatric, health-system level) → add to `docs/guidelines/triage.md` only
2. **Fill the template** from the statement, rationale and remarks. Set any field the text doesn't state to `null`.
   - population, trigger, action, timing, exceptions, strength, certainty, page
3. **Operationalize** each null or vague field as a PhysioNet column condition. Tag every such decision `project_assumption` and give a one-line reason.
4. **Write the YAML** in `src/sepsis/guideline/rules/` using the schema below. Reuse definitions from `definitions.yaml` (e.g. `DEF-SIRS`, `DEF-SCREEN-POS`) instead of repeating logic.
5. **Write tests** in the rule's `tests:` block, at minimum:
   - one case that fires
   - one blocked by the condition (e.g. recent lactate)
   - one where the trigger is false
6. Run `make check`. The engine test runs every rule's `tests:` block.
7. Summarize the rule for the user. Mark `status: draft` until a person reviews it.

```yaml
- rule_id: SSC26-<TOPIC>-<NN>
  status: draft                       # draft -> reviewed -> active
  source:
    doc: SSC-2026-adult               # or Sepsis-3, CMS-SEP-1
    statement: "<number>"
    page: "<page>"
    strength: strong | conditional
    certainty: high | moderate | low | very_low
    paraphrase: "<own words, no long quotes>"
  applies_to: "Age >= 18"
  trigger: DEF-SCREEN-POS
  condition: "<expression over view features>"
  condition_provenance: {type: project_assumption, notes: "<why>"}
  action: {type: FORCE | BLOCK | PRIOR, panel: LACTATE, bonus: null}
  explanation: "<one sentence shown to the clinician>"
  tests:
    - {given: {...}, expect: "FORCE LACTATE"}

- gap_id: SSC26-GAP-<TOPIC>
  source: {doc: SSC-2026-adult, statement: "<number>", strength: strong}
  paraphrase: "<own words>"
  reason_not_computable: "<missing data>"
  runtime_behavior: "show reminder text when <definition> fires"
```

Invariant (tested in the engine): the action model can never remove a FORCE.
