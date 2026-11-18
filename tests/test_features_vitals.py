import numpy as np
import pandas as pd
import pytest

from sepsis.features.vitals import diffs, lags, rolling_minmax, window_stats

NAN = np.nan


def _two_patients() -> pd.DataFrame:
    a = pd.DataFrame({"patient_id": "A_1", "hour": range(5), "HR": [NAN, 80.0, 100.0, 90.0, 70.0]})
    b = pd.DataFrame({"patient_id": "B_1", "hour": range(3), "HR": [200.0, 60.0, 65.0]})
    return pd.concat([a, b], ignore_index=True)


def test_rolling_minmax_hand_example() -> None:
    out = rolling_minmax(_two_patients(), ["HR"], windows_rows=[2, 3])
    a = out.iloc[:5]
    assert np.isnan(a["HR_max_2"].iloc[0])  # NaN before the first value
    assert a["HR_max_2"].iloc[1:].tolist() == [80.0, 100.0, 100.0, 90.0]
    assert a["HR_min_3"].iloc[1:].tolist() == [80.0, 80.0, 80.0, 70.0]
    b = out.iloc[5:]
    assert b["HR_max_3"].tolist() == [200.0, 200.0, 200.0]
    assert b["HR_min_2"].tolist() == [200.0, 60.0, 60.0]  # never sees A_1's rows


def test_window_stats_hand_example() -> None:
    out = window_stats(_two_patients(), ["HR"], windows_rows=[3], stats=["mean", "std"])
    a = out.iloc[:5]
    assert a["HR_mean_3"].iloc[1:].tolist() == pytest.approx([80.0, 90.0, 90.0, 260 / 3])
    assert np.isnan(a["HR_std_3"].iloc[1])  # one value: no spread
    assert a["HR_std_3"].iloc[3] == pytest.approx(10.0)


def test_lags_and_diffs() -> None:
    values = _two_patients()
    lagged = lags(values, ["HR"], lags=[1, 2])
    assert lagged["HR_lag_1"].iloc[:5].tolist()[2:] == [80.0, 100.0, 90.0]
    assert np.isnan(lagged["HR_lag_1"].iloc[5])  # B_1's first row has no previous row
    assert np.isnan(lagged["HR_lag_2"].iloc[6])
    diff = diffs(values, ["HR"])["HR_diff_1"]
    assert diff.iloc[2:5].tolist() == [20.0, -10.0, -20.0]
    assert np.isnan(diff.iloc[1]) and np.isnan(diff.iloc[5])


@pytest.mark.parametrize("t", [0, 1, 2, 3])
def test_rows_after_t_never_change_rows_up_to_t(t: int) -> None:
    values = _two_patients()
    changed = values.copy()
    later = (changed["patient_id"] == "A_1") & (changed["hour"] > t)
    changed.loc[later, "HR"] = 999.0
    keep = (values["patient_id"] != "A_1") | (values["hour"] <= t)
    for function, kwargs in (
        (rolling_minmax, {"windows_rows": [2, 6]}),
        (window_stats, {"windows_rows": [3], "stats": ["mean", "std"]}),
        (lags, {"lags": [1, 3]}),
        (diffs, {}),
    ):
        before = function(values, ["HR"], **kwargs)
        after = function(changed, ["HR"], **kwargs)
        pd.testing.assert_frame_equal(before[keep], after[keep])


def test_unsorted_input_is_refused() -> None:
    with pytest.raises(ValueError, match="sorted"):
        rolling_minmax(_two_patients().iloc[::-1], ["HR"], windows_rows=[2])


def test_missing_vital_column_is_skipped() -> None:
    out = rolling_minmax(_two_patients(), ["HR", "DBP"], windows_rows=[2])
    assert list(out.columns) == ["HR_max_2", "HR_min_2"]
