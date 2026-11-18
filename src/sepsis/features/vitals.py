"""Windowed vital features (M2-2): rolling max/min, and the M0-5 extras (mean/std, lags, diffs).

Vitals are recorded every hour regardless of our policy, so they are exempt from ADR-0002. Every
window counts rows and ends at the current row (ADR-0005), within the patient; nothing looks at
later rows. Input is the view's `values` (already forward-filled), sorted by patient and hour.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from sepsis.data.columns import present
from sepsis.features._frame import require_sorted


def _rolling(values: pd.DataFrame, column: str, window: int, stat: str) -> pd.Series:
    groups = values.groupby("patient_id", sort=False)[column]
    result = getattr(groups.rolling(window, min_periods=1), stat)()
    return result.reset_index(level=0, drop=True).reindex(values.index)


def rolling_minmax(
    values: pd.DataFrame, vitals: Sequence[str], windows_rows: Sequence[int]
) -> pd.DataFrame:
    """`<vital>_max_<w>` and `<vital>_min_<w>` over the last w rows (rank #1)."""
    return window_stats(values, vitals, windows_rows, stats=["max", "min"])


def window_stats(
    values: pd.DataFrame,
    vitals: Sequence[str],
    windows_rows: Sequence[int],
    stats: Sequence[str],
) -> pd.DataFrame:
    """`<vital>_<stat>_<w>` over the last w rows; stats from max, min, mean, std, median."""
    require_sorted(values)
    allowed = {"max", "min", "mean", "std", "median"}
    if not set(stats) <= allowed:
        raise ValueError(f"unknown stats {sorted(set(stats) - allowed)}")
    out = pd.DataFrame(index=values.index)
    for vital in present(list(vitals), values.columns):
        for w in windows_rows:
            for stat in stats:
                out[f"{vital}_{stat}_{w}"] = _rolling(values, vital, int(w), stat)
    return out


def lags(values: pd.DataFrame, vitals: Sequence[str], lags: Sequence[int]) -> pd.DataFrame:
    """`<vital>_lag_<k>`: the value k rows earlier in the same patient (NaN before row k)."""
    require_sorted(values)
    groups = values.groupby("patient_id", sort=False)
    out = pd.DataFrame(index=values.index)
    for vital in present(list(vitals), values.columns):
        for k in lags:
            if int(k) < 1:
                raise ValueError("lags must be at least 1 row (no look-forward)")
            out[f"{vital}_lag_{k}"] = groups[vital].shift(int(k))
    return out


def diffs(values: pd.DataFrame, vitals: Sequence[str]) -> pd.DataFrame:
    """`<vital>_diff_1`: change since the previous row of the same patient. Vitals only."""
    require_sorted(values)
    groups = values.groupby("patient_id", sort=False)
    out = pd.DataFrame(index=values.index)
    for vital in present(list(vitals), values.columns):
        out[f"{vital}_diff_1"] = groups[vital].diff(1)
    return out
