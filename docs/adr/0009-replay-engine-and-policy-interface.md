# ADR-0009: Replay engine and policy interface

Status: accepted (M5-1). Code in `sepsis.replay.{reveal,engine,models}` and `sepsis.action.policy`.

## Context
M3 scored the risk models with labs as recorded. The project question is which panel to order and when, so patients must be re-run hour by hour with labs hidden until a policy requests them: view → risk → alarm → policy → reveal (ADR-0006 §4). The pieces existed: the view and its reveal-row format (`sepsis.data.view`), the feature builder with an hour-by-hour path (`features_at`), the fold plans and frozen thresholds of a run (ADR-0008). Open were the loop itself, what a policy may see and answer, what a request logs, how lab burden is counted, and which model scores each replayed patient.

Building one patient's view and features hour by hour (`view_at` + `features_at`) costs about 0.2 s per patient-hour, which is days for the 1.24M dev patient-hours.

## Decision
1. **All patients in step.** The engine steps every patient through row t together. At each t it builds the view of the patients still in the ICU from their rows ≤ t and the reveals visible by t, in one `view_table` call. It then builds their feature rows, takes row t, scores it, applies the alarm, asks the policy and applies its requests.
   - The record is cleaned once up front. Cleaning judges each hour on its own, so this equals cleaning the rows ≤ t.
   - A test checks the engine against a slow reference written independently: one patient at a time, `view_at` + `features_at`, and the reveal rule as plain loops. Scores and requests must be equal.
2. **What a policy sees: a `Step`.** It holds:
   - `t`;
   - the view (`ViewTable`, rows ≤ t of the patients in the ICU, revealed labs only);
   - their feature rows at t;
   - the risk scores;
   - the alarms (`score > threshold`, strictly).

   The record, labels and hospital are never in it. Policies live in `sepsis.action`, which may import only the view modules (import guard).

   **Anonymous ids.** Patient ids name the hospital (`A_…`, `B_…`), so the engine gives the policy per-replay codes (`p000000`, … in the order of the real ids) and maps its answers back. Outputs carry the real ids. A test checks that relabelled ids change no decision.
3. **What a policy answers.** Any set of panels per patient (`patient_id -> frozenset[panel]`). A missing patient or an empty set means WAIT. Unknown panel names, and patients not in the ICU at t, are refused.
4. **Reveal rule** (ADR-0006). A request for panel P at row t reveals every recorded value of P's labs in rows t − lookback_P + 1 … t, with reveal row t + `reveal_delay_rows`. A value already revealed keeps its earlier reveal row. The record is never written. The final reveal-row table of every replay is checked with `validate_reveal_rows`.
5. **Request log.** One row per (patient, row, panel): `patient_id, hour, panel, status, n_values`, where `status` is one of:
   - `revealed`: n_values new values;
   - `nothing_new`: values in the window, all already revealed;
   - `unavailable`: no value in the window; logged, never imputed.
6. **Labs per patient-day** = panel requests ÷ (scored patient-hours / 24). Every request counts, unavailable and nothing-new included, because each one is an order. Mix replays pool both hospitals, so its result is split with `by_hospital` before scoring or counting: mix → A and mix → B are reported apart (ADR-0004).
7. **Models and thresholds.** No models are saved (ADR-0008 §9), so replay refits each fold plan of a finished run (`replay_models`) and refuses a refit whose scores differ from the run's `predictions.parquet`.
   - A → B / B → A: one model, fitted on the training hospital's dev patients, scores the other hospital.
   - Mix: each outer fold's model scores its own fold.
   - The threshold is the run's frozen one from `thresholds.json`, never re-tuned.
8. **Output.** A prediction table (`patient_id, hour, score, alarm`, ADR-0007) ready for the scorer, the request log, and the final reveal-row table.

## Consequences
- Replaying ALL (every panel every hour) gives exactly the as-recorded reveal table. So ALL reproduces the run's scored predictions and alarms, which is tested on a toy run for A → B, B → A and every mix fold.
- **Cost:** every step rebuilds the view from row 0, so work grows with the square of stay length (about 33M view rows for all dev).
  - Measured on 2,000 dev patients: NONE 35 s, ALL 92 s, mostly fixed cost per hour step.
  - Estimated for all dev patients: about 5 min (NONE) to 12 min (ALL) per replay.
  - If M6 policies need it, view rows can be built incrementally, because a view row never changes once its hour has passed. That would need a second implementation of the view, checked by the same equivalence test.
- Policies that order in a pattern (ALL, NONE, AS_RECORDED) could skip the loop and use a fixed reveal table, but they go through the same engine, so every policy is measured the same way.
