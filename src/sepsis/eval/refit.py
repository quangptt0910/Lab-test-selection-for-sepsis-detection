"""Refit the scored models of a finished run, for analysis (M3-3) and replay (M5-1).

No models are saved (ADR-0008), so an analysis refits: the config, seed, preprocessing and
training rows of the run, through the same `plan_setup` and `PatientIndex` as the runner. The
refit's scores must equal the run's scored predictions, or `check_against_run` raises: an analysis
of a different model would be worthless.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.eval.experiment_config import (
    TARGET_COLUMN,
    model_factory,
    validate_experiment_config,
)
from sepsis.eval.runner import FoldPlan, PatientIndex, plan_setup
from sepsis.eval.tables import KEYS, Tables
from sepsis.risk.base import RiskModel

CROSS_HOSPITAL = ("A_to_B", "B_to_A")


@dataclass
class Refit:
    """The refitted model of one fold plan (a cross-hospital setup, or one mix outer fold) and
    what it scored."""

    setup: str
    model: RiskModel
    X_eval: pd.DataFrame  # evaluation-view feature rows of the scored patients
    keys: pd.DataFrame  # patient_id, hour of those rows
    scores: np.ndarray
    outer_fold: int | None = None  # mix only


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The run's checked config and the `b1` preprocessing it recorded (None if absent)."""
    run_dir = Path(run_dir)
    cfg = validate_experiment_config(load_yaml(run_dir / "config.yaml"), load_yaml(FEATURES_CONFIG))
    path = run_dir / "resolved" / "preprocessing.yaml"
    return cfg, (load_yaml(path) if path.exists() else None)


def refit_cross_hospital(
    cfg: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None,
    tables: Tables,
    split: pd.DataFrame,
    setup: str,
) -> Refit:
    """Fit the run's model on the setup's whole training dev part and score the other hospital."""
    if setup not in CROSS_HOSPITAL:
        raise ValueError(f"setup must be one of {CROSS_HOSPITAL}")
    return refit_plan(cfg, preprocessing, tables, plan_setup(split, setup)[0])


def refit_plan(
    cfg: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None,
    tables: Tables,
    plan: FoldPlan,
) -> Refit:
    """Fit the run's model on one plan's refit patients and score its scored patients.

    Cross-hospital setups have one plan; mix has one per outer fold (replay, M5-1).
    """
    index = PatientIndex.of(tables)
    fit_rows, scored_rows = index.rows(plan.refit), index.rows(plan.scored)
    model = model_factory(cfg, preprocessing)()
    targets = tables.targets
    model.fit(
        tables.train_X.loc[fit_rows, tables.feature_columns],
        targets.loc[fit_rows, TARGET_COLUMN[cfg["target"]]].to_numpy(dtype=float),
        targets.loc[fit_rows, "weight"].to_numpy(dtype=float),
    )
    X_eval = tables.eval_X.loc[scored_rows, tables.feature_columns]
    scores = np.asarray(model.predict(X_eval), dtype=float)
    keys = targets.loc[scored_rows, KEYS].reset_index(drop=True)
    return Refit(plan.setup, model, X_eval.reset_index(drop=True), keys, scores, plan.outer_fold)


def check_against_run(refit: Refit, run_dir: Path) -> None:
    """Raise unless the refit scores equal the run's scored predictions for that setup (and
    outer fold, for mix)."""
    predictions = pd.read_parquet(Path(run_dir) / refit.setup / "predictions.parquet")
    scored = predictions[predictions["kind"] == "scored"]
    if refit.outer_fold is not None:
        scored = scored[scored["outer_fold"] == refit.outer_fold]
    scored = scored.sort_values(KEYS)
    mine = refit.keys.assign(score=refit.scores).sort_values(KEYS)
    if not scored[KEYS].reset_index(drop=True).equals(mine[KEYS].reset_index(drop=True)):
        raise ValueError(f"{refit.setup}: the refit scores other rows than the run")
    if not np.allclose(scored["score"].to_numpy(), mine["score"].to_numpy(), rtol=0, atol=1e-9):
        raise ValueError(f"{refit.setup}: the refit does not reproduce the run's scores")
