"""Hand-built records shared by the feature-builder tests. No real patient data.

Every column the view can show (all demographics, vitals and the 26 labs) is in the record, so a
feature block that reads any of them is exercised by the leakage suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sepsis.data.columns import DEMOGRAPHICS, LABS, VITALS

NAN = np.nan
# Ranges that cross the score thresholds; other columns use 1-100.
RANGES = {
    "HR": (35, 140),
    "O2Sat": (85, 100),
    "Temp": (34.5, 40.0),
    "SBP": (80, 230),
    "MAP": (50, 100),
    "DBP": (40, 80),
    "Resp": (6, 30),
    "Lactate": (0.8, 5.0),
    "pH": (7.2, 7.5),
    "PaCO2": (28, 50),
    "SaO2": (85, 100),
    "WBC": (2, 18),
    "Platelets": (15, 300),
    "Bilirubin_total": (0.3, 13),
    "Creatinine": (0.5, 5.5),
    "BUN": (8, 60),
}


def _patient(pid: str, n: int, seed: int, first_lab_row: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def column(name: str, p: float) -> np.ndarray:
        low, high = RANGES.get(name, (1, 100))
        return np.where(rng.random(n) < p, np.round(rng.uniform(low, high, n), 2), NAN)

    record = pd.DataFrame(
        {
            "patient_id": pid,
            "hospital": pid[0],
            "hour": range(n),
            "ICULOS": range(3, n + 3),
            "SepsisLabel": [0] * (n - 4) + [1] * 4,
        }
    )
    for name, value in zip(DEMOGRAPHICS, [64.0, 1.0, 0.0, 1.0, -12.0], strict=True):
        record[name] = value
    for vital in VITALS:
        record[vital] = column(vital, 0.4 if vital in ("Temp", "EtCO2") else 0.9)
    for lab in LABS:
        values = column(lab, 0.25)
        values[:first_lab_row] = NAN  # no lab before this row: a backward fill would show
        record[lab] = values
    return record


def toy_record() -> pd.DataFrame:
    """Two patients of different lengths; B_2 has no labs before row 7."""
    return pd.concat(
        [_patient("A_1", 30, 1, first_lab_row=0), _patient("B_2", 18, 2, first_lab_row=7)],
        ignore_index=True,
    )
