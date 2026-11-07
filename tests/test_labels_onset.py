import math

import pandas as pd

from sepsis.data.patients import patient_table
from sepsis.labels.onset import t_sepsis


def test_t_sepsis_is_first_label_row_plus_6() -> None:
    assert t_sepsis([0, 0, 0, 1, 1]) == 9
    assert t_sepsis(pd.Series([1, 1])) == 6


def test_t_sepsis_nan_when_never_septic() -> None:
    assert math.isnan(t_sepsis([0, 0, 0]))
    assert math.isnan(t_sepsis([]))


def _patients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["A_p1"] * 4 + ["A_p2"] * 2,
            "hospital": ["A"] * 6,
            "hour": [0, 1, 2, 3, 0, 1],
            # A_p1's ICULOS starts at 10: t_sepsis must use the row index, not ICULOS.
            "ICULOS": [10, 11, 12, 13, 1, 2],
            "SepsisLabel": [0, 0, 1, 1, 0, 0],
        }
    )


def test_patient_table_columns_and_values() -> None:
    table = patient_table(_patients())
    assert list(table.columns) == ["patient_id", "hospital", "n_hours", "septic", "t_sepsis"]
    p1, p2 = table.set_index("patient_id").loc["A_p1"], table.set_index("patient_id").loc["A_p2"]
    assert (p1["n_hours"], p1["septic"], p1["t_sepsis"]) == (4, 1, 8)
    assert (p2["n_hours"], p2["septic"]) == (2, 0)
    assert math.isnan(p2["t_sepsis"])
