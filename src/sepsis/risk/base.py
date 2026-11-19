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


def _dummy(params: Mapping[str, Any], seed: int) -> DummyModel:
    if "column" not in params:
        raise ValueError("the dummy model needs params.column")
    return DummyModel(str(params["column"]))


# model.type in an experiment config -> factory(params, seed). M3-2 adds B0, B1 and B2.
MODELS: dict[str, Callable[[Mapping[str, Any], int], RiskModel]] = {"dummy": _dummy}


def make_model(model_type: str, params: Mapping[str, Any], seed: int) -> RiskModel:
    """A fresh, unfitted model of the given type."""
    if model_type not in MODELS:
        raise ValueError(f"unknown model type {model_type!r}; registered: {sorted(MODELS)}")
    return MODELS[model_type](params, seed)
