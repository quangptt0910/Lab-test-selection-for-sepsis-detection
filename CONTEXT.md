# Domain glossary

Shared vocabulary for code, docs, tickets and prompts. If a term changes, change it here first.

## Data
- **Patient-hour**: one row of a `.psv` file. Predictions are made for every patient-hour.
- **Hospital A / B**: `training_setA` / `training_setB`. They come from different hospital systems and are loaded separately. The hidden hospital C is not available.
- **Dev / test part**: each hospital's 80% / 20% patient split (A-dev, A-test, B-dev, B-test), stratified by septic. Test parts are used once, in the final evaluation (ADR-0004).
- **ICULOS**: hours since ICU admission. On its own it is the strongest single predictor.
- **SepsisLabel**: 1 from 6 h *before* clinical onset onward (septic patients only). It is already shifted.
- **t_sepsis**: clinical onset hour = first SepsisLabel=1 row + 6.
- **dt**: t − t_sepsis for a septic patient-hour.
- **Vitals**: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2. Recorded regardless of our policy.
- **Labs**: the 26 laboratory columns. In replay they are hidden until revealed.

## Replay
- **Replay**: re-running a patient record hour by hour. Labs stay hidden until the system requests them.
- **Panel**: a group of labs requested together: LACTATE, ABG, CHEM, CBC, LFT, COAG, TROPONIN.
- **Revealed**: a lab value the system requested and that exists in the record within the matching window.
- **Unavailable request**: a panel was requested, but the record has no value nearby. It is logged, never imputed.
- **View**: everything the system is allowed to see at hour t (all vitals ≤ t + revealed labs ≤ t).
- **Policy**: rule deciding which panel (or WAIT) to request each hour. Baselines: ALL, NONE, AS_RECORDED.

## Models
- **Risk model**: maps the view at t to a risk score (higher means more likely to alarm).
- **Alarm**: the risk score exceeds the frozen threshold. This is what the utility scores.
- **Action model / VoI**: estimates the value of information of each panel: how much revealing it would change the risk or the alarm decision.
- **WAIT**: no panel's VoI exceeds its cost.
- **B0–B3**: baseline ladder. B0 rules (SIRS, PartialSOFA), B1 logistic, B2 LightGBM, B3 B2 + signatures.

## Clinical features
- **PartialSOFA**: SOFA computed only from components available here: platelets, bilirubin, MAP, creatinine.
- **Shock index**: HR / SBP.
- **SIRS count**: 0–4 from Temp, HR, Resp/PaCO2, WBC (bands not available).
- **hours_since_<panel>**: hours since that panel was last revealed (NaN if never).
- **Signature**: summary of a path over a window. Level 1 = net changes; level 2 = signed areas (order of events).

## Evaluation
- **Utility (U)**: the official per-hour points. **Normalized utility**: (observed − inaction) / (best − inaction) over the whole dataset.
- **U1 / U0**: utility of predicting alarm / no alarm for one patient-hour. **U1 − U0** is the rank #1 regression target.
- **Labs per patient-day**: test burden metric reported next to utility.
- **Setup**: how a model is trained and evaluated across hospitals (ADR-0004). **A → B**: train on A-dev, evaluate on B. **B → A**: the reverse. **Mix**: train on A-dev + B-dev with CV, report A and B patients separately (**mix → A**, **mix → B**). Mix is the optimistic number; A → B and B → A are the pessimistic ones.
- **Training side**: the patients a setup trains and tunes on. Every fitted statistic, including the alarm threshold, comes from here.

## Guideline layer
- **Rule**: YAML record with trigger, condition, action, source and provenance.
- **FORCE / BLOCK / PRIOR**: must recommend / must not recommend / bonus added to a panel's VoI.
- **Gap record**: a guideline statement that applies but cannot be computed from this data (e.g. blood cultures).
- **Provenance**: where a rule field comes from: `guideline_statement`, `guideline_remark`, `external_spec`, `project_assumption`.
