# Experiment log

Append-only. One row per run of the `run-experiment` skill. A changed result gets a new id.

Setups are defined in ADR-0004. `eval` is `dev` for every development run and `test` only for the final evaluation.

| id | date | sha | config | eval | A→B utility (pessimistic) | B→A utility (pessimistic) | mix→A utility (optimistic) | mix→B utility (optimistic) | AUROC (mean of 4) | labs/pt-day | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b0-001 | 2026-11-19 | 0f8fe74 | configs/b0_rules.yaml | dev | 0.055 | 0.078 | 0.096 | 0.055 | 0.654 | — |  |
| b1-001 | 2026-11-19 | 13837f8 | configs/b1_logistic.yaml | dev | 0.224 | 0.293 | 0.332 | 0.268 | 0.752 | — |  |
| b2-001 | 2026-11-19 | 13837f8 | configs/b2_lightgbm.yaml | dev | 0.309 | 0.314 | 0.426 | 0.363 | 0.807 | — |  |
| b2-002 | 2026-11-19 | 13837f8 | configs/b2_002_bagging.yaml | dev | 0.305 | 0.311 | 0.425 | 0.352 | 0.809 | — |  |
| b2-003 | 2026-11-19 | 13837f8 | configs/b2_003_slower.yaml | dev | 0.312 | 0.326 | 0.425 | 0.364 | 0.811 | — |  |
| b2-004 | 2026-11-19 | 13837f8 | configs/b2_004_leaves15.yaml | dev | 0.309 | 0.328 | 0.423 | 0.355 | 0.808 | — |  |
| b2-005 | 2026-11-19 | 13837f8 | configs/b2_005_minchild500.yaml | dev | 0.308 | 0.308 | 0.424 | 0.355 | 0.807 | — |  |
| b2-006 | 2026-11-19 | 13837f8 | configs/b2_006_no_hours_since.yaml | dev | 0.286 | 0.336 | 0.413 | 0.333 | 0.800 | — |  |
