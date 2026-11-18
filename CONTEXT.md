# Domain glossary

Shared vocabulary for code, docs, tickets and prompts. If a term changes, change it here first.

## Data
- **Patient-hour**: one row of a `.psv` file. Predictions are made for every patient-hour.
- **Hospital A / B**: `training_setA` / `training_setB`. They come from different hospital systems. Their patient-hour tables are stored per hospital and can be loaded together or apart. The hidden hospital C is not available.
- **Dev / test part**: each hospital's 80% / 20% patient split (A-dev, A-test, B-dev, B-test), stratified by septic. Test parts are used once, in the final evaluation (ADR-0004).
- **Split file**: the one stored split, one row per patient: `patient_id, hospital, septic, split (dev|test), fold` (0–4 on dev). Every setup takes its patients and folds from it.
- **Patient table**: one row per patient (`n_hours`, `septic`, `t_sepsis`), computed from the whole record. Metadata for splits, labels and EDA; never a model feature.
- **hour**: 0-based row number within a patient's file. Labels and the scorer count rows, not `ICULOS`.
- **Window**: w rows ending at row t, t included (rows t − w + 1 … t). Every window in configs counts rows (ADR-0005).
- **ICULOS**: hours since ICU admission. On its own it is the strongest single predictor.
- **SepsisLabel**: 1 from 6 h *before* clinical onset onward (septic patients only). It is already shifted.
- **t_sepsis**: clinical onset hour = first SepsisLabel=1 row + 6.
- **dt**: t − t_sepsis for a septic patient-hour.
- **Vitals**: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2. Recorded regardless of our policy. EtCO2 is dropped by cleaning (empty in hospital A).
- **Labs**: the 26 laboratory columns. In replay they are hidden until revealed.
- **Plausible range**: fixed clinical limits per column in `configs/cleaning.yaml`. They mark impossible readings and artefacts, not abnormal values (a hypertensive crisis is kept). Never fitted on data.
- **Cleaning rule**: a pure function from `sepsis.data.clean` that applies the cleaning config: drop unused columns, convert units, blank (or clip) values outside the plausible range, and blank all pressures of an hour where DBP >= SBP. It judges each value only within its own hour, and never drops or reorders rows or changes ids, `ICULOS` or `SepsisLabel`.

## Replay
- **Replay**: re-running a patient record hour by hour. Labs stay hidden until the system requests them.
- **Panel**: a group of labs requested together: LACTATE, ABG, CHEM, CBC, LFT, COAG, TROPONIN. Each of the 26 labs is in exactly one panel, the same for both hospitals (`configs/panels.yaml`, ADR-0006).
- **Look-back window**: the rows a panel request at row t may reveal values from: the panel's `lookback_rows` ending at t. Never rows after t.
- **Reveal delay**: rows between a request and its values joining the view (1: requested at t, visible from t + 1). The row they join is the **reveal row**.
- **Revealed**: a lab value the system requested and that exists in the record within the panel's look-back window. It stays revealed from its reveal row on, and is never written into earlier rows.
- **Unavailable request**: a panel was requested, but the record has no value in the panel's look-back window (rows ≤ t). It is logged, never imputed.
- **View**: everything the system is allowed to see at hour t: all vitals ≤ t, plus revealed labs whose reveal row is ≤ t (so requested at t − 1 or earlier). Per lab it shows the revealed value recorded latest, and the row it was recorded. Built only by `sepsis.data.view`; `SepsisLabel` and `hospital` are never in it.
- **Reveal-row table**: the record's rows, one column per lab, holding the reveal row of each recorded value (NaN: never revealed). One format for training (as recorded: recorded row + 1) and replay (request row + 1). `validate_reveal_rows` refuses a reveal sooner than the delay or older than the panel's look-back window.
- **View table**: the view at every row of every patient at once (`ViewTable`: `values` and `lab_rows`), for training; `build_view_at` gives one patient's view at one row, for replay (cleaned and checked the same way).
- **Panel-draw masking**: in training only, each panel draw (one patient, row and panel) is hidden with probability `lab_masking_p`, seeded per patient so a patient's mask is the same in every fold (ADR-0002, ADR-0006).
- **Policy**: rule deciding which panel (or WAIT) to request each hour. Baselines: ALL, NONE, AS_RECORDED.

## Models
- **Risk model**: maps the view at t to a risk score (higher means more likely to alarm).
- **Alarm**: the risk score exceeds the frozen threshold (`score > threshold`, strictly). This is what the utility scores.
- **Action model / VoI**: estimates the value of information of each panel: how much revealing it would change the risk or the alarm decision.
- **WAIT**: no panel's VoI exceeds its cost.
- **B0–B3**: baseline ladder. B0 rules (SIRS, PartialSOFA), B1 logistic, B2 LightGBM, B3 B2 + signatures.

## Clinical features
- **PartialSOFA**: SOFA computed only from components available here: platelets, bilirubin, MAP, creatinine.
- **Shock index**: HR / SBP.
- **SIRS count**: 0–4 from Temp, HR, Resp/PaCO2, WBC (bands not available). A criterion whose inputs are missing counts as not met.
- **Partial qSOFA**: 0–2 from Resp ≥ 22 and SBP ≤ 100; altered mentation is not in the data.
- **Component scores**: `sofa_<component>` and `news_<vital>` (NEWS2) score one input each and are NaN exactly when that input is NaN, so they add no missingness signal beyond the raw value. The sums (PartialSOFA, SIRS, qSOFA) score a missing input exactly like a normal one (ADR-0002).
- **hours_since_<panel>**: rows since the most recent revealed value of the panel was recorded (not since it was requested); NaN if none was ever revealed.
- **Signature**: summary of a path over a window. Level 1 = net changes; level 2 = signed areas (order of events).

## Evaluation
- **Utility (U)**: the official per-hour points. **Normalized utility**: (observed − inaction) / (best − inaction) over the whole dataset.
- **U1 / U0**: utility of predicting alarm / no alarm for one patient-hour. **U1 − U0** (`utility_diff`) is the rank #1 regression target, built by `sepsis.labels.utility` with the scorer's constants in `configs/labels.yaml`. It uses the whole record, so it is a label, never a feature. The same holds for every other column of the target table (`dt, u1, u0, weight`; `LABEL_COLUMNS`).
- **Target weight**: 1 for every scored hour, 0 for septic hours with dt > 3, which the scorer ignores (their target is 0).
- **Labs per patient-day**: test burden metric reported next to utility.
- **Setup**: how a model is trained and evaluated across hospitals (ADR-0004). **A → B**: train on A-dev, evaluate on B. **B → A**: the reverse. **Mix**: train on A-dev + B-dev with CV, report A and B patients separately (**mix → A**, **mix → B**). Mix is the optimistic number; A → B and B → A are the pessimistic ones.
- **Training side**: the patients a setup trains and tunes on. Every fitted statistic, including the alarm threshold, comes from here.
- **Eval side**: the dev patients a setup is evaluated on during development (B-dev for A → B). Not the same as the locked test part.
- **Prediction table**: one row per patient-hour: `patient_id, hour, score` (the raw risk score, any scale), plus `alarm` (0/1) once a threshold is applied. It is what the scoring toolkit takes (ADR-0007).
- **Out-of-fold predictions**: scores for training-side patients, each from a model that was not fitted on that patient's fold. The threshold is tuned on these only.
- **Threshold grid**: the candidate thresholds for tuning: observed out-of-fold scores of the training side at evenly spaced quantiles (`configs/eval.yaml`).

## Guideline layer
- **Rule**: YAML record with trigger, condition, action, source and provenance.
- **FORCE / BLOCK / PRIOR**: must recommend / must not recommend / bonus added to a panel's VoI.
- **Gap record**: a guideline statement that applies but cannot be computed from this data (e.g. blood cultures).
- **Provenance**: where a rule field comes from: `guideline_statement`, `guideline_remark`, `external_spec`, `project_assumption`.
