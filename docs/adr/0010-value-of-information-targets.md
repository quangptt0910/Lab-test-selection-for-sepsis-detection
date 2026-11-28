# ADR-0010: Value-of-information targets and panel costs

Status: accepted (M6-1). Code in `sepsis.labels.voi` and the CLI `sepsis.labels.build_voi`; settings in `configs/voi.yaml`.

## Context
M5-2 set the reference points on b2-001: ALL and AS_RECORDED give the same utilities (0.309 / 0.314 / 0.426 / 0.363), AS_RECORDED at 9.1 (B) and 14.1 (A) panel requests per patient-day, and NONE loses 0.041 to 0.148. The M6 policy orders a panel when its value of information (VoI) exceeds its cost, and WAITs otherwise. M6-2 learns one VoI model per panel from the view. Open were what VoI measures, what data it is learned from, and how to handle the fact that a panel's counterfactual is only known where the record has a value.

## Decision
1. **What VoI measures: utility from changed alarms.** For patient i, decision row t and panel P, compare one fitted risk model with one frozen threshold on two views of the same record (§3):

   gain_h = Σ over s = t+1 … min(t+h, n−1) of (U1 − U0)_s × (alarm_with,s − alarm_without,s)

   - U1 − U0 is the official per-row utility difference (`utility_targets`, the B2 target). Rows the scorer ignores (septic, dt > 3) have U1 − U0 = 0 and add nothing.
   - Alarm = score > frozen threshold, strictly, as everywhere (ADR-0007).
   - A change in risk that flips no alarm is worth nothing, because the frozen-threshold utility is what M6 is judged on.
   - Also stored, as a dense diagnostic and fallback: `dscore_t1` = |score_with − score_without| at t + 1.
2. **Horizon.** A revealed value stays in the view, so it can flip alarms for many rows. The gain is stored for h = 1, 6 and 12 (`gain_h1`, `gain_h6`, `gain_h12`); 12 is the default. 12 covers the scorer's early-reward window (dt −12 … +3) and is the look-back of LACTATE and ABG. One horizon for every panel keeps gains comparable across panels. The sum stops at the end of the stay.
3. **The two views.** Both are built from the cleaned record **as recorded, without masking**: the view b2-001's thresholds were tuned on (out-of-fold scores on `eval_X`, ADR-0008). Only panel P differs:
   - **without:** P's values recorded in its look-back window (t − L_P, t] and every P value recorded after t are hidden. P's older values are shown as recorded. "P was not checked within its window."
   - **with:** the same, plus every P value in (t − L_P, t] revealed at row t + `reveal_delay_rows`, exactly what a request at t reveals in replay (ADR-0006, ADR-0009 §4). Values the training masking would have hidden are revealed too: a request reveals what the record has.
   - P's values after t stay hidden in both views. In the policy's world a value appears only when requested, so the gain is the value of this one request if P is not requested again within h. A later natural draw would cut the gain short by an order the policy never made.
   - The other six panels are as recorded at every row.

   Both reveal-row tables pass `validate_reveal_rows`.
4. **Availability and the counterfactual bias.** `available_P(t)` = the record has a value of P in (t − L_P, t]. Only then does "with" differ from "without". Elsewhere a request would be `unavailable` in replay, so its value there is **exactly 0**: a fact of the replay, not an imputation.
   - The table keeps the two parts apart: `available` (0/1), and the gains, NaN where not available.
   - Availability is where clinicians drew labs, which is ordering behaviour (ADR-0002). A model of the product learns it implicitly; keeping it apart lets M6-2 learn P(available) × E[gain | available] and lets the report show how much of the policy is availability and how much is value. A → B against B → A tests whether the availability part transfers across hospitals.
   - With this "without" view nothing in the window is visible, so `nothing_new` cannot occur at a decision row. States like "requested 2 rows ago, nothing new" exist in replay but not in the targets; their value is 0 anyway.
5. **Decision rows.** Per patient, rows t = o, o + 12, o + 24, … with t ≤ n − 2 (t + 1 must exist). The offset o in [0, 12) is drawn per patient from `rng([seed, crc32(patient_id)])`, the masking scheme, so a patient's rows do not depend on which other patients are in the table. Spacing 12 = the default horizon, so one patient's horizons do not overlap; the random offset keeps decision rows off fixed ICULOS hours.
6. **Who scores a row: out-of-fold only, training side only.** No models are saved (ADR-0008), so the source run's inner fold models are refit and checked against its stored out-of-fold predictions (atol 1e-9).
   - A → B: A-dev patients, each scored by the inner fold model that held it out, with the setup's frozen threshold. B → A likewise on B-dev.
   - Mix: for each outer fold k, every patient not in fold k, by the inner model of k's plan that held it out, with fold k's threshold. A patient therefore has 4 rows per decision row and panel, one per outer fold whose training side it is in.
   - So no target is computed with a model that trained on that patient, and the VoI model of a setup (M6-2) is trained only on that setup's training side, never on the patients it is replayed on.
7. **Targets are labels.** They read the whole record, the future and U1 − U0, so they live in `sepsis.labels` and are never features. A test checks that no module in `sepsis.action` imports `sepsis.labels`. VoI models (M6-2) read view features only.
8. **Panel costs.** `configs/voi.yaml` holds a relative cost per panel, all 1 (`provenance: project_assumption`). M6-3 sweeps one scale λ over them, so only the ratios matter. Price-based ratios are a later variant and need a cited fee schedule. The costs are not in `panels.yaml`, because that file is part of every feature-cache key and of the resolved configs `compare` checks for seed repeats.
9. **Output.** One long table per setup, one row per (setup, outer_fold, patient_id, t, panel): `available`, `gain_h1`, `gain_h6`, `gain_h12`, `dscore_t1`, `score_without_t1`, `alarm_without_t1`, plus `inner_fold`. Written as parquet under `data/cache/voi/<key>/` with a `meta.json`; the key hashes `voi.yaml`, the source run's config, the split and the source code. A dataset, not a scored experiment, so no LOG row.

## Consequences
- The targets measure one request against "not checked in its window", with no later requests. The policy in replay acts in states the targets do not cover (after its own requests); M6-3's replay is the test of whether that matters.
- The gain is sparse: it is non-zero only where an alarm flips. If it is too sparse to learn, `dscore_t1` is the fallback, decided in M6-2 from the summary counts.
- Cost: each decision row becomes a copy of the record up to t + 12, and each copy is built for 7 panels × 2 views. Features are built once per chunk of patients and shared by every model that scores them.
