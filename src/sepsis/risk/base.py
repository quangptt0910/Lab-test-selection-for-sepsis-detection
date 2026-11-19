"""The interface every risk model follows, and the registry the experiment runner builds from.

A subset of scikit-learn's estimator API with three rules (ADR-0008):
- `fit` always receives `sample_weight` (the target weight is 0 for hours the scorer ignores);
- `predict` returns a raw score on any scale, never class labels: the alarm is
  `score > threshold`, with the threshold tuned on the training side;
- a fresh model is built per fold by `make_model(type, params, seed)`, so no fitted state can
  move between folds. Every fitted statistic (fill values, scalers) lives inside `fit`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from sepsis.risk.gbm import LightGBMRegressor
from sepsis.risk.logistic import LogisticBaseline
from sepsis.risk.rules import RuleScore


@runtime_checkable
class RiskModel(Protocol):
    """Maps feature rows to a risk score (higher means more likely to alarm)."""

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> RiskModel: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class DummyModel:
    """Scores one feature column as it is. For runner tests and smoke runs, not a baseline.

    `fit` stores the column's minimum over the training rows, and `predict` fills NaN with it, so
    the dummy also has a fitted statistic that must come from the training side.
    """

    def __init__(self, column: str) -> None:
        self.column = column
        self.fill_: float | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> DummyModel:
        values = X[self.column].to_numpy(dtype=float)
        self.fill_ = float(np.nanmin(values)) if np.isfinite(values).any() else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.fill_ is None:
            raise RuntimeError("call fit before predict")
        values = X[self.column].to_numpy(dtype=float)
        return np.where(np.isnan(values), self.fill_, values)


Preprocessing = Mapping[str, Any] | None  # the `b1` section of configs/preprocessing.yaml
Factory = Callable[[Mapping[str, Any], int, Preprocessing], RiskModel]


def _dummy(params: Mapping[str, Any], seed: int, preprocessing: Preprocessing) -> DummyModel:
    if "column" not in params:
        raise ValueError("the dummy model needs params.column")
    return DummyModel(str(params["column"]))


def _rule_score(params: Mapping[str, Any], seed: int, preprocessing: Preprocessing) -> RuleScore:
    """B0. params: `weights` {feature column: weight}."""
    _only(params, {"weights"}, "rule_score")
    if not params.get("weights"):
        raise ValueError("rule_score needs params.weights")
    return RuleScore(params["weights"])


def _logistic(
    params: Mapping[str, Any], seed: int, preprocessing: Preprocessing
) -> LogisticBaseline:
    """B1. params: `C`, `max_iter`. Fills and log labs from `preprocessing` (ADR-0005)."""
    _only(params, {"C", "max_iter"}, "logistic")
    missing = sorted({"C", "max_iter"} - set(params))
    if missing:
        raise ValueError(f"logistic needs params {missing} in the experiment config")
    if preprocessing is None:
        raise ValueError("logistic needs the b1 section of configs/preprocessing.yaml")
    # Keys the model implements one way only: refuse any other value instead of ignoring it.
    for key, implemented in (("vital_fill", "training_median"), ("scaler", "standard")):
        if preprocessing.get(key) != implemented:
            raise ValueError(f"preprocessing {key} must be {implemented!r} for logistic")
    return LogisticBaseline(
        lab_fill={lab: entry["value"] for lab, entry in preprocessing["lab_fill"].items()},
        log_labs=preprocessing["log_labs"],
        C=params["C"],
        max_iter=params["max_iter"],
        seed=seed,
    )


def _lightgbm(
    params: Mapping[str, Any], seed: int, preprocessing: Preprocessing
) -> LightGBMRegressor:
    """B2. params: any `LGBMRegressor` parameter except the seed and determinism ones.

    Unlike B0 and B1 there is no allow-list: LightGBM takes many parameters (M3-3 tunes them), and
    it rejects unknown ones itself at fit.
    """
    return LightGBMRegressor(params, seed)


def _only(params: Mapping[str, Any], allowed: set[str], model_type: str) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"unknown {model_type} params {unknown}; allowed: {sorted(allowed)}")


# model.type in an experiment config -> factory(params, seed, preprocessing).
MODELS: dict[str, Factory] = {
    "dummy": _dummy,
    "rule_score": _rule_score,  # B0
    "logistic": _logistic,  # B1
    "lightgbm_regressor": _lightgbm,  # B2
}
# Models that need a particular target; any model not listed takes either target.
MODEL_TARGETS: dict[str, tuple[str, ...]] = {"logistic": ("sepsis_label",)}


def make_model(
    model_type: str,
    params: Mapping[str, Any],
    seed: int,
    *,
    preprocessing: Preprocessing = None,
) -> RiskModel:
    """A fresh, unfitted model of the given type. `preprocessing`: the `b1` config section."""
    if model_type not in MODELS:
        raise ValueError(f"unknown model type {model_type!r}; registered: {sorted(MODELS)}")
    return MODELS[model_type](params, seed, preprocessing)
