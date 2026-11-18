"""Shared checks for feature functions."""

from __future__ import annotations

import pandas as pd

from sepsis.data.view import KEYS


def require_sorted(values: pd.DataFrame) -> None:
    """Raise unless each patient's rows form one block in hour order, one row per hour.

    Windowed features look back by row position, so rows must be in the view's order.
    """
    pid, hour = values["patient_id"], values["hour"]
    same = pid.eq(pid.shift())
    blocks = int((~same).sum())
    if blocks != pid.nunique() or not (hour.diff()[same] == 1).all():
        raise ValueError(f"values must be sorted by {KEYS}, each patient's rows in hour order")
