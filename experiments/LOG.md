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
| b3-001 | 2026-11-22 | 725518f | configs/b3_001_sig_vitals.yaml | dev | 0.304 | 0.350 | 0.424 | 0.361 | 0.808 | — | b2-001 + sig_vitals + sig_vitals_cumsum (M4-1) |
| b3-002 | 2026-11-22 | 725518f | configs/b3_002_sig_psofa.yaml | dev | 0.310 | 0.339 | 0.425 | 0.370 | 0.810 | — | b3-001 + sig_partial_sofa (M4-1) |
| b2-011 | 2026-11-22 | 725518f | configs/b2_011_seed1.yaml | dev | 0.308 | 0.283 | 0.422 | 0.369 | 0.805 | — | b2-001 with seed 1 (seed repeat, M4-1) |
| b2-012 | 2026-11-22 | 725518f | configs/b2_012_seed2.yaml | dev | 0.303 | 0.309 | 0.419 | 0.361 | 0.807 | — | b2-001 with seed 2 (seed repeat, M4-1) |
| b3-011 | 2026-11-22 | 725518f | configs/b3_011_seed1.yaml | dev | 0.303 | 0.304 | 0.419 | 0.375 | 0.809 | — | b3-002 with seed 1 (seed repeat, M4-1) |
| b3-012 | 2026-11-22 | 725518f | configs/b3_012_seed2.yaml | dev | 0.305 | 0.323 | 0.418 | 0.369 | 0.811 | — | b3-002 with seed 2 (seed repeat, M4-1) |
| r-none-001 | 2026-11-28 | 13a22af | configs/replay_b2.yaml | dev | 0.268 | 0.166 | 0.337 | 0.262 | 0.761 | 0.0 | M5-2 baseline policies; source b2-001; labs/pt-day A→B 0.0, B→A 0.0, mix→A 0.0, mix→B 0.0; unavailable rate A→B —, B→A —, mix→A —, mix→B — |
| r-all-001 | 2026-11-28 | 13a22af | configs/replay_b2.yaml | dev | 0.309 | 0.314 | 0.426 | 0.363 | 0.807 | 168.0 | M5-2 baseline policies; source b2-001; labs/pt-day A→B 168.0, B→A 168.0, mix→A 168.0, mix→B 168.0; unavailable rate A→B 0.642, B→A 0.525, mix→A 0.525, mix→B 0.642 |
| r-asrec-001 | 2026-11-28 | 13a22af | configs/replay_b2.yaml | dev | 0.309 | 0.314 | 0.426 | 0.363 | 0.807 | 11.6 | M5-2 baseline policies; source b2-001; labs/pt-day A→B 9.1, B→A 14.1, mix→A 14.1, mix→B 9.1; unavailable rate A→B 0.000, B→A 0.000, mix→A 0.000, mix→B 0.000 |
