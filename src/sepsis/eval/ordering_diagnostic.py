"""Ordering-signal diagnostic: how much utility lab-ordering behaviour would add (M3-3).

    uv run python -m sepsis.eval.ordering_diagnostic experiments/runs/<id>

ANALYSIS ONLY. ADR-0002 forbids lab counts and missing/present patterns in every model, because
they encode ordering behaviour. This module adds exactly those columns to measure the signal, so
it lives outside `sepsis.features`, its columns are in no config or feature block, and its result
goes into the M3-3 report, never into `experiments/LOG.md`.

Setting: full information (`lab_masking_p = 0`, which ADR-0002 allows for this ablation), so the
training counts match the training lab values. Columns come from the as-recorded reveal-row table
(reveal delay from configs/panels.yaml), so they show only what the view can show at row t:
- `n_revealed_<lab>`: values of the lab revealed by row t (reveal row <= t);
- `revealed_now_<lab>`: 1 if a value of the lab is revealed at row t.
The same config runs twice under A_to_B and B_to_A, without and with the columns; the difference
is the ordering signal.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sepsis.data.clean import clean
from sepsis.data.view import evaluation_reveal_rows
from sepsis.eval.refit import CROSS_HOSPITAL
from sepsis.eval.tables import KEYS, Tables

PREFIXES = ("n_revealed_", "revealed_now_")


def ordering_columns(
    hours: pd.DataFrame, cleaning: Mapping[str, Any], panels: Mapping[str, Any]
) -> pd.DataFrame:
    """`patient_id, hour` plus `n_revealed_<lab>` and `revealed_now_<lab>`, sorted by KEYS."""
    reveal = evaluation_reveal_rows(clean(hours, cleaning), panels)
    labs = [c for c in reveal.columns if c not in KEYS]
    n_rows = reveal.groupby("patient_id", sort=False)["hour"].transform("size").to_numpy()
    start = np.arange(len(reveal)) - reveal["hour"].to_numpy()  # first row of each patient
    out = reveal[KEYS].copy()
    for lab in labs:
        at = reveal[lab].to_numpy()  # reveal row of the value recorded at this row, or NaN
        keep = np.isfinite(at) & (at < n_rows)  # revealed within the record
        now = np.zeros(len(reveal))
        np.add.at(now, (start[keep] + at[keep]).astype(int), 1.0)
        out[f"revealed_now_{lab}"] = (now > 0).astype(float)
        out[f"n_revealed_{lab}"] = pd.Series(now).groupby(reveal["patient_id"].to_numpy()).cumsum()
    return out


def with_ordering(tables: Tables, columns: pd.DataFrame) -> Tables:
    """The tables with the ordering columns added to both views, same rows and order."""
    if not columns[KEYS].reset_index(drop=True).equals(tables.targets[KEYS]):
        raise ValueError("ordering columns must be aligned with the tables")
    extra = [c for c in columns.columns if c not in KEYS]
    added = columns[extra].reset_index(drop=True)
    return replace(
        tables,
        train_X=pd.concat([tables.train_X, added], axis=1),
        eval_X=pd.concat([tables.eval_X, added], axis=1),
        feature_columns=[*tables.feature_columns, *extra],
    )


def main(argv: list[str] | None = None) -> int:
    from sepsis.config import EVAL_CONFIG, load_yaml
    from sepsis.eval.experiment_config import model_factory
    from sepsis.eval.refit import load_run
    from sepsis.eval.run_experiment import load_dev_tables
    from sepsis.eval.runner import run_setup

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    cfg, preprocessing = load_run(args.run_dir)
    dev = load_dev_tables(cfg, lab_masking_p=0.0)
    tables, labels, split = dev.tables, dev.labels, dev.split
    columns = ordering_columns(dev.hours, dev.settings.cleaning, dev.settings.panels)
    grid = load_yaml(EVAL_CONFIG)["threshold"]["grid"]
    new_model = model_factory(cfg, preprocessing)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, variant in (("without", tables), ("with_ordering", with_ordering(tables, columns))):
        for setup in CROSS_HOSPITAL:
            print(f"[ordering] {name} {setup} ...", flush=True)
            result = run_setup(
                variant, labels, split, setup, new_model=new_model, target=cfg["target"], grid=grid
            )
            s = result.scores["all"]
            results.setdefault(setup, {})[name] = {"utility": s.utility, "auroc": s.auroc}
    out_dir = Path(str(args.run_dir) + "-ordering")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run": str(args.run_dir),
        "lab_masking_p": 0.0,
        "columns": [c for c in columns.columns if c not in KEYS],
        "results": results,
        "utility_gain": {
            s: r["with_ordering"]["utility"] - r["without"]["utility"] for s, r in results.items()
        },
    }
    (out_dir / "ordering.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"results": results, "utility_gain": summary["utility_gain"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
