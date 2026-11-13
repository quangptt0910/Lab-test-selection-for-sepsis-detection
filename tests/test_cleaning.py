import numpy as np
import pandas as pd
import pytest

from sepsis.config import load_yaml
from sepsis.data.clean import (
    apply_bp_consistency,
    apply_ranges,
    clean,
    convert_units,
    drop_columns,
)

NAN = np.nan


def _table(**columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    ids = {
        "patient_id": ["A_p1"] * n,
        "hospital": ["A"] * n,
        "hour": list(range(n)),
        "ICULOS": list(range(1, n + 1)),
        "SepsisLabel": [0] * n,
    }
    return pd.DataFrame({**ids, **columns})


def test_range_action_nan_sets_values_outside_to_nan() -> None:
    df = _table(HR=[10.0, 20.0, 150.0, 300.0, 301.0, NAN])
    out = apply_ranges(df, {"HR": {"low": 20, "high": 300, "action": "nan"}})
    assert np.isnan(out["HR"][0]) and np.isnan(out["HR"][4]) and np.isnan(out["HR"][5])
    assert out["HR"][1:4].tolist() == [20.0, 150.0, 300.0]  # limits are inclusive


def test_range_action_clip_and_keep() -> None:
    df = _table(HR=[10.0, 400.0], Age=[5.0, 120.0])
    out = apply_ranges(
        df,
        {
            "HR": {"low": 20, "high": 300, "action": "clip"},
            "Age": {"low": 14, "high": 100, "action": "keep"},
        },
    )
    assert out["HR"].tolist() == [20.0, 300.0]
    assert out["Age"].tolist() == [5.0, 120.0]


def test_range_unknown_action_is_an_error() -> None:
    with pytest.raises(ValueError, match="action"):
        apply_ranges(_table(HR=[80.0]), {"HR": {"low": 20, "high": 300, "action": "drop"}})


def test_convert_units_only_converts_values_in_the_window() -> None:
    df = _table(Temp=[37.0, 98.6, NAN], FiO2=[0.4, 40.0, 0.5])
    suspects = {
        "Temp": {
            "low": 80,
            "high": 115,
            "conversion": "fahrenheit_to_celsius",
            "action": "convert",
        },
        "FiO2": {"low": 1.01, "high": 100, "conversion": "percent_to_fraction", "action": "keep"},
    }
    out = convert_units(df, suspects)
    assert out["Temp"][0] == 37.0 and out["Temp"][1] == pytest.approx(37.0)
    assert np.isnan(out["Temp"][2])
    assert out["FiO2"].tolist() == [0.4, 40.0, 0.5]  # action keep: unchanged


def test_bp_consistency_blanks_all_pressures_of_that_hour_only() -> None:
    df = _table(
        SBP=[120.0, 80.0, 290.0, 100.0, NAN],
        MAP=[90.0, 85.0, 290.0, 70.0, 60.0],
        DBP=[70.0, 90.0, 290.0, 60.0, 50.0],
        HR=[80.0, 81.0, 82.0, 83.0, 84.0],
    )
    out = apply_bp_consistency(df, {"columns": ["SBP", "MAP", "DBP"], "action": "nan"})
    for col in ("SBP", "MAP", "DBP"):
        assert out[col][[1, 2]].isna().all()  # DBP > SBP and DBP == SBP
        assert out[col][0] == df[col][0] and out[col][3] == df[col][3]
    assert out["MAP"][4] == 60.0  # SBP missing: nothing to compare, kept
    assert out["HR"].tolist() == df["HR"].tolist()


def test_drop_columns() -> None:
    df = _table(HR=[80.0], EtCO2=[NAN])
    assert "EtCO2" not in drop_columns(df, ["EtCO2"]).columns
    assert drop_columns(df, ["NotThere"]).columns.tolist() == df.columns.tolist()


def test_rules_may_not_touch_protected_columns() -> None:
    df = _table(HR=[80.0])
    with pytest.raises(ValueError, match="ICULOS"):
        clean(df, {"ranges": {"ICULOS": {"low": 1, "high": 2, "action": "nan"}}})
    with pytest.raises(ValueError, match="SepsisLabel"):
        clean(df, {"drop_columns": {"SepsisLabel": "no"}})


def test_clean_keeps_rows_order_and_input() -> None:
    df = _table(
        HR=[80.0, 400.0, 90.0],
        SBP=[120.0, 80.0, 110.0],
        MAP=[90.0, 85.0, 80.0],
        DBP=[70.0, 90.0, 60.0],
        EtCO2=[NAN, NAN, NAN],
    ).iloc[[2, 0, 1]]  # stored order must survive
    before = df.copy()
    config = {
        "ranges": {"HR": {"low": 20, "high": 300, "action": "nan"}},
        "bp_consistency": {"columns": ["SBP", "MAP", "DBP"], "action": "nan"},
        "drop_columns": {"EtCO2": "empty in A"},
    }
    out = clean(df, config)
    pd.testing.assert_frame_equal(df, before)  # input untouched
    assert out.index.tolist() == df.index.tolist()
    assert out[["patient_id", "hour", "ICULOS", "SepsisLabel"]].equals(
        df[["patient_id", "hour", "ICULOS", "SepsisLabel"]]
    )
    assert np.isnan(out.loc[1, "HR"]) and np.isnan(out.loc[1, "SBP"])
    assert "EtCO2" not in out.columns


def test_clean_is_row_local() -> None:
    """Changing one hour never changes the cleaned values of any other hour."""
    config = load_yaml("configs/cleaning.yaml")
    base = _table(HR=[80.0, 85.0, 90.0], SBP=[120.0, 118.0, 115.0], DBP=[70.0, 72.0, 71.0])
    base["MAP"] = [85.0, 87.0, 86.0]
    changed = base.copy()
    changed.loc[2, ["HR", "DBP"]] = [999.0, 200.0]
    a, b = clean(base, config), clean(changed, config)
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])


def test_project_config_is_valid() -> None:
    config = load_yaml("configs/cleaning.yaml")
    assert (
        clean(_table(HR=[80.0], Calcium=[1.1], FiO2=[0.04]), config)[["Calcium", "FiO2"]]
        .isna()
        .all(axis=None)
    )


def test_project_config_blanks_flush_artefacts_even_when_dbp_is_out_of_range() -> None:
    """The DBP >= SBP check sees the values before the range rule blanks DBP."""
    config = load_yaml("configs/cleaning.yaml")
    df = _table(
        HR=[80.0, 81.0, 82.0],
        SBP=[290.0, 180.0, 120.0],
        MAP=[290.0, 150.0, 90.0],
        DBP=[290.0, 250.0, 70.0],  # flush artefact; DBP > 200 and DBP >= SBP
    )
    out = clean(df, config)
    assert out.loc[[0, 1], ["SBP", "MAP", "DBP"]].isna().all(axis=None)
    assert out.loc[2, ["SBP", "MAP", "DBP"]].tolist() == [120.0, 90.0, 70.0]
    assert out["HR"].tolist() == [80.0, 81.0, 82.0]
