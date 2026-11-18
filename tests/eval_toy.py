"""Hand-built patients shared by the eval tests. No real patient data."""

from __future__ import annotations

import numpy as np
import pandas as pd

# patient_id -> (hospital, rows, first SepsisLabel == 1 row or None, split, fold)
PATIENTS = {
    "A_s": ("A", 30, 14, "dev", 0),  # t_sepsis = 20, best alarm window rows 8..23
    "A_n": ("A", 10, None, "dev", 1),
    "B_s": ("B", 25, 10, "dev", 1),  # t_sepsis = 16, best alarm window rows 4..19
    "B_n": ("B", 8, None, "dev", 0),
    "A_t": ("A", 20, 5, "test", None),  # locked test patient
}
DEV = ["A_s", "A_n", "B_s", "B_n"]
BEST_WINDOW = {"A_s": (8, 23), "B_s": (4, 19)}


def hours(patient_ids: list[str] = DEV) -> pd.DataFrame:
    """Patient-hour rows (`patient_id, hospital, hour, SepsisLabel`) of the given toy patients."""
    frames = []
    for pid in patient_ids:
        hospital, n, first, _, _ = PATIENTS[pid]
        label = np.zeros(n, dtype=int)
        if first is not None:
            label[first:] = 1
        frames.append(
            pd.DataFrame(
                {"patient_id": pid, "hospital": hospital, "hour": range(n), "SepsisLabel": label}
            )
        )
    return pd.concat(frames, ignore_index=True)


def split_table() -> pd.DataFrame:
    """A split file for the toy patients, test patient included."""
    rows = [
        (pid, hospital, int(first is not None), part, fold)
        for pid, (hospital, _, first, part, fold) in PATIENTS.items()
    ]
    table = pd.DataFrame(rows, columns=["patient_id", "hospital", "septic", "split", "fold"])
    table["fold"] = table["fold"].astype("Int64")
    return table


def in_best_window(table: pd.DataFrame) -> pd.Series:
    """True for the rows the official scorer's best predictions alarm on."""
    inside = pd.Series(False, index=table.index)
    for pid, (first, last) in BEST_WINDOW.items():
        inside |= (table["patient_id"] == pid) & table["hour"].between(first, last)
    return inside


def predictions(table: pd.DataFrame, score: np.ndarray | float, alarm: np.ndarray | int):
    """A prediction table for the rows of `table`."""
    return table[["patient_id", "hour"]].assign(score=score, alarm=alarm)
