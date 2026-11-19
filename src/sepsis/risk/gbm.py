"""B2: LightGBM regressor on the configured target, NaN passed through (ADR-0001, M3-2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Set by the runner, never by an experiment config: the seed comes from the config's `seed`
# (ADR-0008), and runs are deterministic. Every LightGBM alias is listed: an explicit seed such as
# `bagging_seed` or `extra_seed` would override the one LightGBM derives from `random_state`
# (a test checks this list against LightGBM's own alias table).
REFUSED_PARAMS = frozenset(
    {
        "random_state",
        "seed",
        "random_seed",
        "bagging_seed",
        "bagging_fraction_seed",
        "feature_fraction_seed",
        "data_random_seed",
        "data_seed",
        "extra_seed",
        "drop_seed",
        "objective_seed",
        "deterministic",
        "verbose",
        "verbosity",
        "force_row_wise",
        "force_col_wise",
    }
)


class LightGBMRegressor:
    """`LGBMRegressor(**params)` with the run's seed; `predict` returns the raw regression output.

    Rows with weight 0 still shape the feature bins but not the trees' gradients.
    """

    def __init__(self, params: Mapping[str, Any], seed: int) -> None:
        refused = sorted(REFUSED_PARAMS & set(params))
        if refused:
            raise ValueError(f"{refused} are set by the runner, not the experiment config")
        self.params = dict(params)
        self.seed = int(seed)
        self.model_: LGBMRegressor | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> LightGBMRegressor:
        self.model_ = LGBMRegressor(
            **self.params,
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        ).fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit before predict")
        return np.asarray(self.model_.predict(X), dtype=float)
