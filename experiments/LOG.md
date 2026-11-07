# Experiment log

Append-only. One row per run of the `run-experiment` skill. A changed result gets a new id.

Setups are defined in ADR-0004. `eval` is `dev` for every development run and `test` only for the final evaluation.

| id | date | sha | config | eval | A→B utility (pessimistic) | B→A utility (pessimistic) | mix→A utility (optimistic) | mix→B utility (optimistic) | AUROC (mean of 4) | labs/pt-day | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
