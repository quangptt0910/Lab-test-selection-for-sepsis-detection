# ADR-0004: Evaluate every model in three setups: A → B, B → A and mix

Status: accepted. Replaces the split protocol previously stated in `CLAUDE.md` (pooled 80/20 split as primary, cross-hospital A → B and B → A as secondary). Replaces the "beats B2 in both hospital-held-out directions" rule in ADR-0001.

## Context
The rank 1 team scored 0.360 utility overall but −0.123 on the hidden hospital C (ADR-0001). Generalising to a new hospital is the known weakness of this recipe, so it has to be measured on every experiment, not only at the end.

The previous protocol had a flaw: cross-hospital runs trained on all of one hospital and scored all of the other. That included the 20% test patients, so the test set was used during development.

## Decision
1. **Split once.** Each hospital is split by patient into **dev (80%)** and **test (20%)**, stratified by septic. The split settings live in `configs/protocol.yaml` and are never changed after the first split. The split is created once and stored as patient-id lists (gitignored), and every experiment loads those lists instead of re-splitting. CV folds use a separate seed in the experiment config, so changing it can never move patients between dev and test. The hospitals are loaded and stored separately; the mix setup combines A-dev and B-dev only where it needs them.
2. **Three setups during development.** Every experiment runs all three, on dev data only:

   | Setup | Train and tune on | Evaluate on |
   |---|---|---|
   | A → B | A-dev. The threshold is tuned on out-of-fold predictions from 5-fold patient CV inside A-dev, then the model is refit on all of A-dev | B-dev |
   | B → A | B-dev, the same way | A-dev |
   | Mix | A-dev + B-dev, outer 5-fold patient CV stratified by hospital × septic. In each outer fold, the threshold is tuned by an inner 5-fold CV on that fold's training patients only | Each outer held-out fold, with the threshold from its own inner CV; reported separately for A and B patients (mix → A, mix → B) |

3. **Training side only.** Every fitted statistic (fill values, scalers, the alarm threshold, hyperparameters) comes from the training side of a setup, never from the evaluated hospital or patients. In mix, a held-out fold is never used to tune the threshold it is scored with (nested CV). Whether the threshold transfers is part of what A → B and B → A test.
4. **Locked test parts.** A-test and B-test are read only by the final evaluation, once, from a tagged commit with frozen configs: A-dev → B-test, B-dev → A-test, and A-dev + B-dev → A-test and B-test. Each final model uses a threshold tuned by CV on its own dev training side.
5. **Reporting.** Mix is the optimistic number and A → B / B → A are the pessimistic ones. All four (A→B, B→A, mix→A, mix→B) are logged side by side and never merged into one number.
6. **Keeping a change** (for example B3 over B2): it must improve the mix result without making A → B or B → A worse by more than `selection.cross_hospital_tolerance` in `configs/protocol.yaml`. The tolerance is fixed before the first comparison and is not changed per experiment.

## Consequences
- The test parts remove patient-level overfitting: no model, threshold or development decision is fitted on a test patient.
- They do not remove hospital-level selection bias. A-dev and B-dev, which come from the same hospitals as A-test and B-test, are used for EDA and for choosing what to keep. So the final A → B and B → A estimate how the system transfers between *these two* hospitals, not to a new one. Hospital C, the only true transfer test, is not available, and the report must say so.
- B-dev is used to evaluate A → B many times during development, so the dev A → B number drifts optimistic. The final A → B on B-test is the number to report.
- Each experiment trains three setups plus their CV folds, and mix uses nested CV (5 outer × 5 inner). This is several times the compute of a single protocol.
- EDA and data-quality work uses A-dev and B-dev only, and compares the hospitals side by side.
