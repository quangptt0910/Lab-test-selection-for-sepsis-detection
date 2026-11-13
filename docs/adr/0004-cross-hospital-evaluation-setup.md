# ADR-0004: Evaluate every model in three setups: A → B, B → A and mix

Status: accepted. Replaces the split protocol previously stated in `CLAUDE.md` (pooled 80/20 split as primary, cross-hospital A → B and B → A as secondary). Replaces the "beats B2 in both hospital-held-out directions" rule in ADR-0001.

## Context
The rank 1 team scored 0.360 utility overall but −0.123 on the hidden hospital C (ADR-0001). Generalising to a new hospital is the known weakness of this recipe, so it has to be measured on every experiment, not only at the end.

The previous protocol had a flaw: cross-hospital runs trained on all of one hospital and scored all of the other. That included the 20% test patients, so the test set was used during development.

## Decision
1. **Split once.** Each hospital is split by patient into **dev (80%)** and **test (20%)**, stratified by septic. In practice one split is made over both hospitals, stratified by hospital × septic, which gives each hospital its own 80/20 split stratified by septic. The settings live in `configs/protocol.yaml` and are never changed after the first split. The split is created once (`python -m sepsis.data.make_split`) and stored as one **split file** with one row per patient: `patient_id, hospital, septic, split (dev|test), fold`. The 5 CV folds on dev (also stratified by hospital × septic) are stored in the same file, so every experiment uses identical folds and nothing in an experiment config can move patients between dev and test. A fingerprint of the file in `configs/protocol.yaml` makes the loader refuse any split that differs. The split file holds patient ids and per-patient labels only (no patient-hours) and is committed at `data/splits/split.csv`. Once a fingerprint is saved, `make_split` never creates a new split: if the file exists it only checks it, and if the file is missing it rebuilds it from the cache and writes it only when the result matches the fingerprint. Patient-hour tables are stored per hospital and can be loaded together or apart; outside the final evaluation, code only gets dev patients.
2. **Three setups during development.** Every experiment runs all three, on dev data only:

   | Setup | Train and tune on | Evaluate on |
   |---|---|---|
   | A → B | A-dev. The threshold is tuned on out-of-fold predictions from the stored folds restricted to A-dev, then the model is refit on all of A-dev | B-dev |
   | B → A | B-dev, the same way | A-dev |
   | Mix | A-dev + B-dev. Outer CV over the 5 stored folds; for outer fold k, the threshold is tuned by an inner CV over the other 4 stored folds only | Each outer held-out fold, with the threshold from its own inner CV; reported separately for A and B patients (mix → A, mix → B) |

3. **Training side only.** Every fitted statistic (fill values, scalers, the alarm threshold, hyperparameters) comes from the training side of a setup, never from the evaluated hospital or patients. In mix, a held-out fold is never used to tune the threshold it is scored with (nested CV). Whether the threshold transfers is part of what A → B and B → A test.
4. **Locked test parts.** A-test and B-test are read only by the final evaluation, once, from a tagged commit with frozen configs: A-dev → B-test, B-dev → A-test, and A-dev + B-dev → A-test and B-test. Each final model uses a threshold tuned by CV on its own dev training side.
5. **Reporting.** Mix is the optimistic number and A → B / B → A are the pessimistic ones. All four (A→B, B→A, mix→A, mix→B) are logged side by side and never merged into one number.
6. **Keeping a change** (for example B3 over B2): it must improve the mix result without making A → B or B → A worse by more than `selection.cross_hospital_tolerance` in `configs/protocol.yaml`. The tolerance is fixed before the first comparison and is not changed per experiment.

## Consequences
- The test parts remove patient-level overfitting: no model, threshold or development decision is fitted on a test patient.
- They do not remove hospital-level selection bias. A-dev and B-dev, which come from the same hospitals as A-test and B-test, are used for EDA and for choosing what to keep. So the final A → B and B → A estimate how the system transfers between *these two* hospitals, not to a new one. Hospital C, the only true transfer test, is not available, and the report must say so.
- B-dev is used to evaluate A → B many times during development, so the dev A → B number drifts optimistic. The final A → B on B-test is the number to report.
- Each experiment trains three setups plus their CV folds, and mix uses nested CV (5 outer folds, each with a 4-fold inner CV). This is several times the compute of a single protocol.
- EDA and data-quality work uses A-dev and B-dev only, and compares the hospitals side by side.
