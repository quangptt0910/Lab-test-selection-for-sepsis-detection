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
    inner_fold: int | None = None  # an inner (out-of-fold) model only, `refit_inner`


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
    return _fit(cfg, preprocessing, tables, plan.refit, plan.scored, plan, None)


def refit_inner(
    cfg: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None,
    tables: Tables,
    plan: FoldPlan,
    inner_fold: int,
) -> Refit:
    """Fit the run's inner model of `inner_fold` and score the patients it held out: the
    out-of-fold scores the run tuned its threshold on (VoI targets, M6-1)."""
    inner = {int(fold): (fit, oof) for fold, fit, oof in plan.inner}
    if int(inner_fold) not in inner:
        raise ValueError(f"{plan.setup}: no inner fold {inner_fold} in this plan")
    fit_ids, oof_ids = inner[int(inner_fold)]
    return _fit(cfg, preprocessing, tables, fit_ids, oof_ids, plan, int(inner_fold))


def _fit(
    cfg: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None,
    tables: Tables,
    fit_ids: tuple[str, ...],
    scored_ids: tuple[str, ...],
    plan: FoldPlan,
    inner_fold: int | None,
) -> Refit:
    """The runner's fit: training-view rows of `fit_ids`, evaluation-view rows scored."""
    index = PatientIndex.of(tables)
    fit_rows, scored_rows = index.rows(fit_ids), index.rows(scored_ids)
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
    return Refit(
        plan.setup,
        model,
        X_eval.reset_index(drop=True),
        keys,
        scores,
        plan.outer_fold,
        inner_fold,
    )


def check_against_run(refit: Refit, run_dir: Path) -> None:
    """Raise unless the refit scores equal the run's scored predictions for that setup (and
    outer fold, for mix)."""
    predictions = pd.read_parquet(Path(run_dir) / refit.setup / "predictions.parquet")
    scored = predictions[predictions["kind"] == "scored"]
    if refit.outer_fold is not None:
        scored = scored[scored["outer_fold"] == refit.outer_fold]
    _compare(refit, scored)


@dataclass(frozen=True)
class RefitTolerance:
    """How far an inner refit may differ from the run's stored scores and still count as the
    run's model (ADR-0010 §6): float-level split ties across machines, never another decision.

    Rows differing by more than `atol` are "off"; they may be at most `max_share_off` of the rows,
    none may differ by more than `max_abs_diff`, and every alarm under the frozen threshold must
    be identical.
    """

    atol: float
    max_share_off: float
    max_abs_diff: float


def check_oof_against_run(
    refit: Refit,
    run_dir: Path,
    *,
    threshold: float | None = None,
    tolerance: RefitTolerance | None = None,
) -> dict[str, float]:
    """Raise unless an inner refit's scores equal the run's stored out-of-fold predictions of
    that inner fold (and outer fold, for mix): within 1e-9, or within `tolerance` with identical
    alarms under `threshold`. Returns `rows`, `rows_off` and `max_abs_diff`."""
    if refit.inner_fold is None:
        raise ValueError("check_oof_against_run needs an inner refit (`refit_inner`)")
    if tolerance is not None and threshold is None:
        raise ValueError("a tolerance needs the frozen threshold, to compare the alarms")
    predictions = pd.read_parquet(Path(run_dir) / refit.setup / "predictions.parquet")
    oof = predictions[
        (predictions["kind"] == "oof") & (predictions["inner_fold"] == refit.inner_fold)
    ]
    if refit.outer_fold is not None:
        oof = oof[oof["outer_fold"] == refit.outer_fold]
    if tolerance is None:
        _compare(refit, oof)
        return {"rows": len(oof), "rows_off": 0, "max_abs_diff": 0.0}
    stored, mine = _aligned(refit, oof)
    where = f"{refit.setup} outer {refit.outer_fold} inner {refit.inner_fold}"
    diff = np.abs(stored - mine)
    off = int((diff > tolerance.atol).sum())
    largest = float(diff.max()) if len(diff) else 0.0
    report = {"rows": len(diff), "rows_off": off, "max_abs_diff": largest}
    flips = int(((stored > threshold) != (mine > threshold)).sum())
    if flips:
        raise ValueError(
            f"{where}: the refit does not reproduce the run's scores: {flips} alarm(s) differ"
        )
    if off > tolerance.max_share_off * len(diff):
        raise ValueError(
            f"{where}: the refit does not reproduce the run's scores: {off} of {len(diff)} rows "
            f"differ by more than {tolerance.atol}"
        )
    if largest > tolerance.max_abs_diff:
        raise ValueError(
            f"{where}: the refit does not reproduce the run's scores: max |diff| {largest:.3g} > "
            f"{tolerance.max_abs_diff}"
        )
    return report


def _aligned(refit: Refit, stored: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The stored and refit scores on the same rows, in the same order; raise if the rows differ."""
    scored = stored.sort_values(KEYS)
    mine = refit.keys.assign(score=refit.scores).sort_values(KEYS)
    if not scored[KEYS].reset_index(drop=True).equals(mine[KEYS].reset_index(drop=True)):
        raise ValueError(f"{refit.setup}: the refit scores other rows than the run")
    return scored["score"].to_numpy(dtype=float), mine["score"].to_numpy(dtype=float)


def _compare(refit: Refit, stored: pd.DataFrame) -> None:
    scored, mine = _aligned(refit, stored)
    if not np.allclose(scored, mine, rtol=0, atol=1e-9):
        raise ValueError(f"{refit.setup}: the refit does not reproduce the run's scores")
