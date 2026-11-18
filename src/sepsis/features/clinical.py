"""Clinical features (M2-2): ratios and scores from the view's values at each row.

Thresholds come from `configs/features.yaml`. Conventions:
- Ratios give NaN for a zero or missing denominator.
- Sums (`partial_sofa`, `sirs_count`, `qsofa_partial`) count a missing input as normal (0 / not
  met), so no score carries a hidden "was measured" flag (ADR-0002).
- Component scores (`sofa_<component>`, `news_<vital>`) are NaN when their input is missing.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sepsis.features._frame import require_sorted

OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def _column(values: pd.DataFrame, name: str) -> pd.Series:
    """The column, or all-NaN if the view has no such column."""
    if name in values.columns:
        return values[name].astype(float)
    return pd.Series(np.nan, index=values.index)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator != 0)


def shock_index(values: pd.DataFrame) -> pd.DataFrame:
    """HR / SBP."""
    return pd.DataFrame({"shock_index": _ratio(_column(values, "HR"), _column(values, "SBP"))})


def bun_cr(values: pd.DataFrame) -> pd.DataFrame:
    """BUN / Creatinine."""
    return pd.DataFrame({"bun_cr": _ratio(_column(values, "BUN"), _column(values, "Creatinine"))})


# --- SOFA ------------------------------------------------------------------------------------


def _sofa_score(x: pd.Series, rule: Mapping[str, Any]) -> pd.Series:
    cutoffs = [float(c) for c in rule["cutoffs"]]
    if rule["direction"] == "below":
        score = sum((x < c).astype(float) for c in cutoffs)
    elif rule["direction"] == "at_or_above":
        score = sum((x >= c).astype(float) for c in cutoffs)
    else:
        raise ValueError(f"unknown SOFA direction {rule['direction']!r}")
    return score.where(x.notna())


def sofa_components(values: pd.DataFrame, sofa: Mapping[str, Any]) -> pd.DataFrame:
    """`sofa_<component>`: one SOFA score per component; NaN when its input is missing."""
    return pd.DataFrame(
        {
            f"sofa_{name}": _sofa_score(_column(values, rule["column"]), rule)
            for name, rule in sofa.items()
        },
        index=values.index,
    )


def partial_sofa(values: pd.DataFrame, sofa: Mapping[str, Any]) -> pd.DataFrame:
    """Sum of the SOFA components; a missing component scores 0."""
    total = sofa_components(values, sofa).fillna(0.0).sum(axis=1)
    return pd.DataFrame({"partial_sofa": total})


def partial_sofa_delta(
    values: pd.DataFrame, sofa: Mapping[str, Any], window_rows: int
) -> pd.DataFrame:
    """`partial_sofa_delta_<w>`: partial_sofa now minus its minimum over the last w rows."""
    require_sorted(values)
    score = partial_sofa(values, sofa)["partial_sofa"]
    lowest = (
        score.groupby(values["patient_id"], sort=False)
        .rolling(int(window_rows), min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )
    return pd.DataFrame({f"partial_sofa_delta_{window_rows}": score - lowest})


# --- Criteria counts: SIRS, qSOFA -------------------------------------------------------------


def _criteria_count(
    values: pd.DataFrame, criteria: Mapping[str, Sequence[Mapping[str, Any]]]
) -> pd.Series:
    """Number of criteria met; a criterion is met if any condition holds (NaN never holds)."""
    count = pd.Series(0.0, index=values.index)
    for conditions in criteria.values():
        met = pd.Series(False, index=values.index)
        for c in conditions:
            met |= OPS[c["op"]](_column(values, c["column"]), float(c["value"])).fillna(False)
        count += met.astype(float)
    return count


def sirs_count(values: pd.DataFrame, sirs: Mapping[str, Any]) -> pd.DataFrame:
    """SIRS criteria met (0-4). A criterion whose inputs are missing counts as not met."""
    return pd.DataFrame({"sirs_count": _criteria_count(values, sirs)})


def qsofa_partial(values: pd.DataFrame, qsofa: Mapping[str, Any]) -> pd.DataFrame:
    """qSOFA without altered mentation (0-2). A missing input counts as not met."""
    return pd.DataFrame({"qsofa_partial": _criteria_count(values, qsofa)})


# --- NEWS2 components -------------------------------------------------------------------------


def news_components(values: pd.DataFrame, news2: Mapping[str, Any]) -> pd.DataFrame:
    """`news_<vital>`: the NEWS2 score of each vital; NaN when the vital is missing.

    Values are rounded half up to the band resolution first, as NEWS2 scores rounded readings.
    Comparisons are made in whole steps of the resolution, so no float error moves a band edge.
    """
    out = pd.DataFrame(index=values.index)
    for vital, band in news2.items():
        step = float(band["resolution"])
        upper = np.round(np.asarray(band["upper"], dtype=float) / step)
        scores = np.asarray(band["scores"], dtype=float)
        if len(scores) != len(upper) + 1:
            raise ValueError(f"NEWS2 {vital}: need one more score than upper bounds")
        x = _column(values, vital).to_numpy()
        steps = np.floor(x / step + 0.5 + 1e-9)  # half up; 1e-9 absorbs float error at .5
        score = scores[np.searchsorted(upper, np.nan_to_num(steps), side="left")]
        out[f"news_{vital}"] = np.where(np.isnan(x), np.nan, score)
    return out
