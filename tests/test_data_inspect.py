from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sepsis.data.inspect import format_report, load_patients, summarize


def _patient(n: int, septic: bool = False, hr_nan: int = 0) -> pd.DataFrame:
    hr = [80.0] * n
    for i in range(hr_nan):
        hr[i] = np.nan
    label = [0] * n
    if septic:
        label[-1] = 1
    return pd.DataFrame({"HR": hr, "Lactate": [np.nan] * n, "SepsisLabel": label})


def test_hours_and_septic_share() -> None:
    s = summarize([_patient(2), _patient(4, septic=True), _patient(6)])
    assert s["n_patients"] == 3
    assert (s["hours_min"], s["hours_median"], s["hours_max"]) == (2, 4, 6)
    assert s["pct_septic"] == pytest.approx(100 / 3)


def test_missing_percent_per_column() -> None:
    s = summarize([_patient(4, hr_nan=3), _patient(8)])  # 3 NaN of 12 rows
    assert s["pct_missing"]["HR"] == pytest.approx(25.0)
    assert s["pct_missing"]["Lactate"] == pytest.approx(100.0)


def test_load_patients_reads_psv(tmp_path: Path) -> None:
    for name in ("p000001.psv", "p000002.psv"):
        (tmp_path / name).write_text("HR|SepsisLabel\nNaN|0\n90|1\n")
    frames = list(load_patients(tmp_path))
    assert len(frames) == 2
    assert frames[0]["HR"].isna().sum() == 1


def test_format_report_lists_columns() -> None:
    text = format_report("A", summarize([_patient(3)]))
    assert "HR" in text and "Lactate" in text and "A" in text
