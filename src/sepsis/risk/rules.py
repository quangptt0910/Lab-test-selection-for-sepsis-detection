"""B0: a rule score from clinical scores, with a threshold tuned like every model's (M3-2)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


class RuleScore:
    """`score = Σ weight_c · X[c]`, by default `sirs_count + partial_sofa`.

    Nothing is fitted: the rule is fixed and only the alarm threshold is tuned on the training side
    (ADR-0007). The score takes few distinct integer values, so the threshold grid (quantiles of
    out-of-fold scores) has few distinct points; `score > k` alarms at k + 1 and above.
    """

    def __init__(self, weights: Mapping[str, float]) -> None:
        if not weights:
            raise ValueError("the rule score has no weights")
        self.weights = {str(c): float(w) for c, w in weights.items()}

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> RuleScore:
        missing = sorted(set(self.weights) - set(X.columns))
        if missing:
            raise ValueError(f"rule score columns missing from the features: {missing}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = X[list(self.weights)].to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError("rule score columns must never be NaN (missing inputs count as 0)")
        return values @ np.array(list(self.weights.values()))
