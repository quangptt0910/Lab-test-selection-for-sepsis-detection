"""Feature importance of a finished B2 run: gain and SHAP from LightGBM itself (M3-3).

    uv run python -m sepsis.eval.importance experiments/runs/<id>

For A_to_B and B_to_A the scored model is refitted (`sepsis.eval.refit`, checked against the
run's predictions). Per feature: gain, mean |SHAP| on a seeded sample of evaluation rows, and the
direction (sign of the correlation between the feature's value and its SHAP value; mean SHAP is
near 0 by construction and says little). Writes `<run>/importance.csv` and prints the suspicion
checks.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.eval.refit import CROSS_HOSPITAL, Refit, check_against_run, load_run
from sepsis.eval.refit import refit_cross_hospital as refit
from sepsis.risk.gbm import LightGBMRegressor

SHAP_ROWS = 50_000
# Columns that can mark the hospital or the length of stay rather than the patient's state.
WATCHED = ("ICULOS", "HospAdmTime", "Unit1", "Unit2")
TOP = 15


def _direction(values: np.ndarray, shap: np.ndarray) -> float:
    keep = np.isfinite(values)
    if keep.sum() < 2 or np.std(values[keep]) == 0 or np.std(shap[keep]) == 0:
        return float("nan")
    return float(np.sign(np.corrcoef(values[keep], shap[keep])[0, 1]))


def shap_importance(refitted: Refit, *, max_rows: int = SHAP_ROWS, seed: int = 0) -> pd.DataFrame:
    """`feature, gain, mean_abs_shap, direction, rank` for one refitted LightGBM model."""
    if not isinstance(refitted.model, LightGBMRegressor):
        raise ValueError("importance needs a lightgbm_regressor run")
    booster = refitted.model.model_.booster_
    X = refitted.X_eval
    if len(X) > max_rows:
        X = X.sample(n=max_rows, random_state=seed)
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]  # last column: the bias
    values = X.to_numpy(dtype=float)
    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "gain": booster.feature_importance("gain"),
            "mean_abs_shap": np.abs(contrib).mean(axis=0),
            "direction": [_direction(values[:, j], contrib[:, j]) for j in range(X.shape[1])],
        }
    )
    table = table.sort_values("mean_abs_shap", ascending=False, kind="stable")
    return table.assign(rank=range(1, len(table) + 1)).reset_index(drop=True)


def suspicion_checks(by_setup: Mapping[str, pd.DataFrame]) -> list[str]:
    """Findings to read before trusting the model (printed and copied into the report)."""
    notes = []
    for setup, table in by_setup.items():
        ranks = table.set_index("feature")["rank"]
        watched = {c: int(ranks[c]) for c in WATCHED if c in ranks}
        notes.append(f"{setup}: rank of hospital/stay columns {watched}")
        top5 = [f for f in table["feature"].head(5) if f.startswith("hours_since_")]
        if top5:
            notes.append(f"{setup}: hours_since in the top 5: {top5}")
    if set(CROSS_HOSPITAL) <= set(by_setup):
        a, b = (by_setup[s].set_index("feature") for s in CROSS_HOSPITAL)
        top = set(a.index[:TOP]) | set(b.index[:TOP])
        flips = sorted(
            f
            for f in top
            if np.isfinite(a.loc[f, "direction"])
            and np.isfinite(b.loc[f, "direction"])
            and a.loc[f, "direction"] != b.loc[f, "direction"]
        )
        if flips:
            notes.append(f"direction flips between A_to_B and B_to_A: {flips}")
    return notes


def main(argv: list[str] | None = None) -> int:
    from sepsis.eval.run_experiment import load_dev_tables

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    cfg, preprocessing = load_run(args.run_dir)
    if cfg["model"]["type"] != "lightgbm_regressor":
        raise ValueError("importance needs a lightgbm_regressor run")
    tables, _, split, _, _ = load_dev_tables(cfg)
    by_setup = {}
    for setup in CROSS_HOSPITAL:
        refitted = refit(cfg, preprocessing, tables, split, setup)
        check_against_run(refitted, args.run_dir)
        by_setup[setup] = shap_importance(refitted, seed=int(cfg["seed"]))
    out = pd.concat([t.assign(setup=s) for s, t in by_setup.items()], ignore_index=True)
    out.to_csv(Path(args.run_dir) / "importance.csv", index=False)
    for setup, table in by_setup.items():
        print(f"\n{setup}: top {TOP} by mean |SHAP|")
        print(table.head(TOP).to_string(index=False))
    print()
    for note in suspicion_checks(by_setup):
        print(note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
