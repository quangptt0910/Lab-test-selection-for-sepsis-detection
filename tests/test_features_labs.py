import numpy as np
import pandas as pd
import pytest

from sepsis.config import load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.view import as_recorded_reveal_rows, view_table
from sepsis.features.labs import hours_since_panel

NAN = np.nan


@pytest.fixture(scope="module")
def panels() -> dict:
    return load_yaml(PANELS_CONFIG)


def test_hours_since_panel_from_lab_rows(panels: dict) -> None:
    lab_rows = pd.DataFrame(
        {
            "patient_id": "A_1",
            "hour": range(5),
            "pH": [NAN, NAN, 1.0, 1.0, 1.0],
            "PaCO2": [NAN, NAN, 1.0, 1.0, 3.0],  # a newer PaCO2 recorded at row 3, shown from 4
            "Lactate": [NAN] * 5,
        }
    )
    out = hours_since_panel(lab_rows, panels)
    assert out["hours_since_ABG"].tolist()[2:] == [1.0, 2.0, 1.0]  # latest of the panel's labs
    assert out["hours_since_ABG"].iloc[:2].isna().all()
    assert out["hours_since_LACTATE"].isna().all()  # never revealed
    assert "hours_since_CHEM" not in out.columns  # no CHEM lab in the view at all


def test_hours_since_counts_from_the_recorded_row(panels: dict) -> None:
    record = pd.DataFrame(
        {
            "patient_id": "A_1",
            "hour": range(8),
            "Lactate": [NAN, 2.0, NAN, NAN, NAN, NAN, NAN, NAN],
        }
    )
    reveal = as_recorded_reveal_rows(record, delay=1)
    reveal.loc[1, "Lactate"] = 5  # requested at row 4, revealed at 5 (replay-style)
    view = view_table(record, reveal)
    out = hours_since_panel(view.lab_rows, panels)["hours_since_LACTATE"]
    assert out.iloc[:5].isna().all()
    assert out.iloc[5:].tolist() == [4.0, 5.0, 6.0]  # age of the value, not of the request


def test_hours_since_keeps_the_index(panels: dict) -> None:
    lab_rows = pd.DataFrame(
        {"patient_id": "A_1", "hour": [0, 1], "Lactate": [NAN, 0.0]}, index=[10, 11]
    )
    out = hours_since_panel(lab_rows, panels)
    assert list(out.index) == [10, 11]
