"""The U1 - U0 training target of the rank #1 recipe (ADR-0001), from the official utility.

For every patient-hour, U1 is the utility the official scorer gives if we alarm at that hour and U0
if we don't. The scorer sums per-hour terms, so U1 - U0 is exactly how much one alarm at that hour
changes a patient's utility. Septic hours after `dt_late` are not scored: target 0, weight 0.

Every column except `patient_id` and `hour` (see `LABEL_COLUMNS`) is computed from the whole record,
future labels included. They are label-side and must never be joined into a feature matrix: `dt` is
NaN exactly for never-septic patients and `weight` is 0 only after onset, so either one gives away
the class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sepsis.labels.onset import LABEL_LEAD_HOURS, t_sepsis

LABELS_CONFIG = Path("configs/labels.yaml")
KEYS = ["patient_id", "hour"]
LABEL_COLUMNS = ["dt", "u1", "u0", "utility_diff", "weight"]  # never features
COLUMNS = [*KEYS, *LABEL_COLUMNS]


@dataclass(frozen=True)
class UtilityConstants:
    """The official scorer's utility constants (`configs/labels.yaml`)."""

    dt_early: float
    dt_optimal: float
    dt_late: float
    max_u_tp: float
    min_u_fn: float
    u_fp: float
    u_tn: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> UtilityConstants:
        """Read the `utility` section; raise ValueError if the constants are inconsistent."""
        values = config["utility"]
        constants = cls(**{f.name: values[f.name] for f in fields(cls)})
        if not constants.dt_early < constants.dt_optimal < constants.dt_late:
            raise ValueError("need dt_early < dt_optimal < dt_late")
        if constants.dt_optimal != -LABEL_LEAD_HOURS:
            raise ValueError(
                f"dt_optimal must be -{LABEL_LEAD_HOURS}: SepsisLabel is already shifted by "
                f"{LABEL_LEAD_HOURS} rows, and t_sepsis is defined from that shift"
            )
        return constants


def patient_utility(labels: Sequence[int] | np.ndarray, c: UtilityConstants) -> pd.DataFrame:
    """`dt, u1, u0, utility_diff, weight` for one patient's complete record, in row order.

    Follows the scorer's `compute_prediction_utility` term by term. `dt` is NaN for a patient who is
    never septic.
    """
    y = np.asarray(labels)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("SepsisLabel must be 0 or 1")
    n = len(y)
    onset = t_sepsis(y)
    if np.isnan(onset):
        u1 = np.full(n, float(c.u_fp))
        u0 = np.full(n, float(c.u_tn))
        dt = np.full(n, np.nan)
        weight = np.ones(n)
    else:
        dt = np.arange(n) - onset
        m1 = c.max_u_tp / (c.dt_optimal - c.dt_early)
        b1 = -m1 * c.dt_early
        m2 = -c.max_u_tp / (c.dt_late - c.dt_optimal)
        b2 = -m2 * c.dt_late
        m3 = c.min_u_fn / (c.dt_late - c.dt_optimal)
        b3 = -m3 * c.dt_optimal
        early = dt <= c.dt_optimal
        window = (dt > c.dt_optimal) & (dt <= c.dt_late)
        u1 = np.where(early, np.maximum(m1 * dt + b1, c.u_fp), np.where(window, m2 * dt + b2, 0.0))
        u0 = np.where(window, m3 * dt + b3, 0.0)
        weight = (dt <= c.dt_late).astype(float)
    return pd.DataFrame({"dt": dt, "u1": u1, "u0": u0, "utility_diff": u1 - u0, "weight": weight})


def _check(hours: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_integer_dtype(hours["hour"]):
        raise ValueError("hour must be an integer row number")
    if not hours["SepsisLabel"].isin([0, 1]).all():
        raise ValueError("SepsisLabel must be 0 or 1")
    if hours.duplicated(KEYS).any():
        raise ValueError("duplicate (patient_id, hour) rows")
    ordered = hours.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if not (ordered.groupby("patient_id").cumcount() == ordered["hour"]).all():
        raise ValueError(
            "each patient needs its complete record, hours 0..n-1: onset is found by row position"
        )
    return ordered


def utility_targets(hours: pd.DataFrame, c: UtilityConstants) -> pd.DataFrame:
    """Per patient-hour targets: `patient_id, hour, dt, u1, u0, utility_diff, weight`.

    `hours` needs `patient_id, hour, SepsisLabel` with every patient's complete record. Rows come
    back sorted by patient and hour; join on `patient_id, hour`.

    The check refuses gaps and duplicates, but a record cut off at the end looks complete. Pass
    whole records as `load_patient_hours` returns them, never a view or rows up to some hour t.
    """
    ordered = _check(hours)
    parts = [
        patient_utility(rows["SepsisLabel"].to_numpy(), c)
        for _, rows in ordered.groupby("patient_id", sort=False)
    ]
    values = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS[2:])
    return pd.concat([ordered[KEYS], values], axis=1)[COLUMNS]
