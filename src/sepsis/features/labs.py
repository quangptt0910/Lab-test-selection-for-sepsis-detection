"""Lab timing feature (M2-2): `hours_since_<panel>` (ADR-0002, ADR-0006 section 7).

Rows since the most recent revealed value of any lab in the panel was recorded, read from the
view's `lab_rows`. NaN if the panel has never been revealed. This is the only lab timing feature:
no counts, no missing flags, no differences of lab values (ADR-0002).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from sepsis.data.columns import present


def hours_since_panel(lab_rows: pd.DataFrame, panels_config: Mapping[str, Any]) -> pd.DataFrame:
    """`hours_since_<PANEL>` for every panel with at least one lab column in the view."""
    out = pd.DataFrame(index=lab_rows.index)
    for name, panel in panels_config["panels"].items():
        columns = present(panel["labs"], lab_rows.columns)
        if columns:
            latest = lab_rows[columns].max(axis=1, skipna=True)
            out[f"hours_since_{name}"] = lab_rows["hour"] - latest
    return out
