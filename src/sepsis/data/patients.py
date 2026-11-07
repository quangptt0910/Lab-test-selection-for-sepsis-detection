"""One row per patient: metadata for splits, labels and EDA.

WARNING: `septic` and `t_sepsis` are computed from the whole record, including the future.
This table must never be joined into model features.
"""

from __future__ import annotations

import pandas as pd

from sepsis.labels.onset import t_sepsis

PATIENT_COLUMNS = ["patient_id", "hospital", "n_hours", "septic", "t_sepsis"]


def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize a patient-hour table into `patient_id, hospital, n_hours, septic, t_sepsis`."""
    ordered = df.sort_values(["patient_id", "hour"], kind="stable")
    grouped = ordered.groupby("patient_id", sort=True)
    table = pd.DataFrame(
        {
            "hospital": grouped["hospital"].first(),
            "n_hours": grouped.size(),
            "septic": grouped["SepsisLabel"].max().eq(1).astype(int),
            "t_sepsis": grouped["SepsisLabel"].agg(t_sepsis).astype(float),
        }
    )
    return table.reset_index()[PATIENT_COLUMNS]
