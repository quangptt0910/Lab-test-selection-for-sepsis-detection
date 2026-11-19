"""B1: logistic regression with the ADR-0005 preprocessing, fitted inside `fit` (M3-2).

Fills, per column (no missing flags, ADR-0002):
- a lab in `lab_fill`: its fixed normal adult value (never fitted on data, ADR-0005);
- `bun_cr`: BUN normal / Creatinine normal;
- `hours_since_<panel>`: the training rows' maximum, so a panel never revealed looks like the
  longest gap seen, never like a recent test;
- any other column: the training rows' median.
A column with no value in the training rows gets 0 (it is then constant and carries no weight).
Then `log1p` on the right-skewed labs and a standard scaler, both fitted on the training rows.

The score is the log-odds (`decision_function`): same ranking and alarms as the probability, and
the scorer's `expit` (ADR-0007) turns it back into the model's own probability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HOURS_SINCE_PREFIX = "hours_since_"
# Lab-derived columns with no fixed fill yet. A median would give an untested patient the value of
# tested (sicker) patients, which ADR-0005 forbids, so B1 refuses them until a fill is defined.
UNFILLED_LAB_DERIVED = frozenset({"sofa_coagulation", "sofa_liver", "sofa_renal"})


class LogisticBaseline:
    def __init__(
        self,
        lab_fill: Mapping[str, float],
        log_labs: Sequence[str],
        C: float,
        max_iter: int,
        seed: int,
    ) -> None:
        self.lab_fill = {str(k): float(v) for k, v in lab_fill.items()}
        self.log_labs = list(log_labs)
        self.C, self.max_iter, self.seed = float(C), int(max_iter), int(seed)
        self.columns_: list[str] | None = None
        self.fill_: dict[str, float] | None = None

    def _fitted_fill(self, X: pd.DataFrame) -> dict[str, float]:
        fill = {}
        for column in X.columns:
            values = X[column].to_numpy(dtype=float)
            if column in self.lab_fill:
                fill[column] = self.lab_fill[column]
            elif column == "bun_cr":
                fill[column] = self.lab_fill["BUN"] / self.lab_fill["Creatinine"]
            elif not np.isfinite(values).any():
                fill[column] = 0.0
            elif column.startswith(HOURS_SINCE_PREFIX):
                fill[column] = float(np.nanmax(values))
            else:
                fill[column] = float(np.nanmedian(values))
        return fill

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        """Fill, then log1p the skewed labs, in one float64 array (no scaling)."""
        if self.columns_ is None or self.fill_ is None:
            raise RuntimeError("call fit before predict")
        if list(X.columns) != self.columns_:
            raise ValueError("predict needs the same columns, in the same order, as fit")
        values = X.to_numpy(dtype=np.float64, copy=True)
        for j, column in enumerate(self.columns_):
            np.copyto(values[:, j], self.fill_[column], where=np.isnan(values[:, j]))
            if column in self.log_labs:
                np.log1p(values[:, j], out=values[:, j])
        return values

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> LogisticBaseline:
        y = np.asarray(y, dtype=float)
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("the logistic baseline needs a 0/1 target (sepsis_label)")
        unfilled = sorted(UNFILLED_LAB_DERIVED & set(X.columns))
        if unfilled:
            raise ValueError(f"no fixed B1 fill for lab-derived columns {unfilled} (ADR-0005)")
        self.columns_ = list(X.columns)
        self.fill_ = self._fitted_fill(X)
        values = self._prepare(X)
        self.scaler_ = StandardScaler().fit(values)
        self.classifier_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, random_state=self.seed
        ).fit(self.scaler_.transform(values), y, sample_weight=sample_weight)
        self.coef_ = self.classifier_.coef_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Log-odds of sepsis_label = 1."""
        values = self._prepare(X)  # raises if not fitted
        return self.classifier_.decision_function(self.scaler_.transform(values))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of sepsis_label = 1 (for analysis; the runner uses `predict`)."""
        values = self._prepare(X)  # raises if not fitted
        return self.classifier_.predict_proba(self.scaler_.transform(values))[:, 1]
