import numpy as np
import pandas as pd
import pytest

from sepsis.data.quality import (
    bp_inconsistent,
    constant_columns,
    empty_columns,
    format_report,
    iculos_problems,
    label_reversals,
    missing_values,
    quality_findings,
    range_violations,
    short_stays,
    unit_suspects,
)

RANGES = {
    "HR": {"low": 20, "high": 300, "action": "nan"},
    "Temp": {"low": 25, "high": 45, "action": "nan"},
}
SUSPECTS = {"Temp": {"low": 80, "high": 115, "meaning": "degF"}}


def _table(rows: list[tuple]) -> pd.DataFrame:
    """rows: (patient_id, HR, Temp, ICULOS, SepsisLabel), hours numbered per patient."""
    df = pd.DataFrame(rows, columns=["patient_id", "HR", "Temp", "ICULOS", "SepsisLabel"])
    df.insert(1, "hospital", df["patient_id"].str[0])
    df.insert(2, "hour", df.groupby("patient_id").cumcount())
    return df


@pytest.fixture
def table() -> pd.DataFrame:
    return _table(
        [
            ("A_p1", 80.0, 37.0, 1, 0),
            ("A_p1", 400.0, 98.6, 2, 0),  # HR too high; Temp looks like degF
            ("A_p1", 10.0, np.nan, 3, 1),  # HR too low
            ("A_p2", 90.0, 36.5, 2, 1),  # starts at 2
            ("A_p2", 95.0, 36.6, 2, 0),  # duplicate ICULOS; label goes 1 -> 0
            ("A_p2", 99.0, 36.7, 5, 0),  # gap 2 -> 5
        ]
    )


def test_range_violations_counts_rows_and_patients(table: pd.DataFrame) -> None:
    out = range_violations(table, RANGES).set_index(["column", "side"])
    assert out.loc[("HR", "low"), ["rows", "patients"]].tolist() == [1, 1]
    assert out.loc[("HR", "high"), ["rows", "patients"]].tolist() == [1, 1]
    assert out.loc[("Temp", "high"), ["rows", "patients"]].tolist() == [1, 1]
    assert ("Temp", "low") not in out.index  # no zero rows; NaN is never a violation


def test_range_violations_skip_columns_not_in_table(table: pd.DataFrame) -> None:
    out = range_violations(table, {**RANGES, "Lactate": {"low": 0, "high": 40}})
    assert "Lactate" not in set(out["column"])


def test_unit_suspects(table: pd.DataFrame) -> None:
    out = unit_suspects(table, SUSPECTS)
    assert out[["column", "meaning", "rows", "patients"]].values.tolist() == [
        ["Temp", "degF", 1, 1]
    ]


def test_iculos_problems(table: pd.DataFrame) -> None:
    out = iculos_problems(table)
    assert out == {"not_starting_at_1": 1, "duplicates": 1, "gaps": 1, "decreasing": 0}


def test_iculos_uses_row_order_not_iculos_order() -> None:
    shuffled = _table([("A_p1", 80.0, 37.0, 1, 0), ("A_p1", 80.0, 37.0, 3, 0)])
    shuffled = shuffled.iloc[::-1]  # stored order differs from hour order
    assert iculos_problems(shuffled)["gaps"] == 1


def test_label_reversals(table: pd.DataFrame) -> None:
    assert label_reversals(table) == 1


def test_empty_and_constant_columns() -> None:
    df = _table([("A_p1", 80.0, np.nan, 1, 0), ("A_p2", 80.0, np.nan, 1, 0)])
    df["Unit1"] = 1.0
    assert empty_columns(df, ["HR", "Temp", "Unit1"]) == ["Temp"]
    assert constant_columns(df, ["HR", "Temp", "Unit1"]) == ["HR", "Unit1"]


def test_short_stays(table: pd.DataFrame) -> None:
    assert short_stays(table, min_hours=3) == 0
    assert short_stays(table, min_hours=4) == 2


def test_quality_findings_one_column_per_hospital(table: pd.DataFrame) -> None:
    b = table.assign(patient_id=table["patient_id"].str.replace("A_", "B_"), hospital="B", HR=80.0)
    both = pd.concat([table, b], ignore_index=True)
    config = {"ranges": RANGES, "unit_suspects": SUSPECTS, "min_hours": 4}
    out = quality_findings(both, config).set_index(["check", "column"])
    assert out.loc[("out of range (high)", "HR"), ["A_rows", "A_patients"]].tolist() == [1, 1]
    assert out.loc[("out of range (high)", "HR"), ["B_rows", "B_patients"]].tolist() == [0, 0]
    assert out.loc[("label goes 1 -> 0", ""), ["A_patients", "B_patients"]].tolist() == [1, 1]
    assert out.loc[("out of range (high)", "HR"), "decision"] == "nan"


def test_quality_findings_fills_zero_and_groups_by_check(table: pd.DataFrame) -> None:
    # B has a Temp violation that A lacks, and an EtCO2 column that A lacks.
    b = table.assign(
        patient_id=table["patient_id"].str.replace("A_", "B_"), hospital="B", Temp=10.0, EtCO2=30.0
    )
    both = pd.concat([table.assign(EtCO2=np.nan), b], ignore_index=True)
    config = {"ranges": RANGES, "unit_suspects": SUSPECTS, "min_hours": 4}
    out = quality_findings(both, config)
    indexed = out.set_index(["check", "column"])
    assert indexed.loc[("out of range (low)", "Temp"), ["A_rows", "A_patients"]].tolist() == [0, 0]
    assert indexed.loc[("empty column", "EtCO2"), ["A_patients", "B_patients"]].tolist() == [2, 0]
    kinds = out["check"].str.split(" ").str[0].tolist()
    first_iculos = kinds.index("ICULOS")
    assert all(k == "out" for k in kinds[: kinds.index("looks")])  # all range rows come first
    assert "out" not in kinds[first_iculos:]


def test_format_report_is_markdown_with_totals(table: pd.DataFrame) -> None:
    config = {"ranges": RANGES, "unit_suspects": SUSPECTS, "min_hours": 4}
    text = format_report(quality_findings(table, config), table)
    assert text.startswith("# Data quality")
    assert "| check | column |" in text
    assert "A: 2 patients, 6 patient-hours" in text


def test_bp_inconsistent_counts_hours_where_dbp_reaches_sbp() -> None:
    df = pd.DataFrame(
        {
            "patient_id": ["A_p1", "A_p1", "A_p2", "A_p2"],
            "SBP": [120.0, 80.0, 90.0, np.nan],
            "DBP": [70.0, 90.0, 90.0, 95.0],
        }
    )
    assert bp_inconsistent(df) == (2, 2)  # DBP > SBP and DBP == SBP; missing SBP is skipped


def test_findings_show_decision_and_reason_from_config(table: pd.DataFrame) -> None:
    df = table.assign(SBP=[120.0, 80.0, 110.0, 100.0, 100.0, 100.0], DBP=70.0, EtCO2=np.nan)
    df.loc[1, "DBP"] = 90.0
    config = {
        "ranges": {**RANGES, "HR": {**RANGES["HR"], "reason": "artefact"}},
        "unit_suspects": SUSPECTS,
        "min_hours": 4,
        "bp_consistency": {"columns": ["SBP", "DBP"], "action": "nan", "reason": "impossible"},
        "drop_columns": {"EtCO2": "empty in A"},
    }
    out = quality_findings(df, config).set_index(["check", "column"])
    assert out.loc[("out of range (high)", "HR"), ["decision", "reason"]].tolist() == [
        "nan",
        "artefact",
    ]
    assert out.loc[("DBP >= SBP", "SBP, DBP"), ["A_rows", "A_patients"]].tolist() == [1, 1]
    assert out.loc[("DBP >= SBP", "SBP, DBP"), "reason"] == "impossible"
    assert out.loc[("empty column", "EtCO2"), ["decision", "reason"]].tolist() == [
        "drop column",
        "empty in A",
    ]


def test_format_report_prints_notes(table: pd.DataFrame) -> None:
    config = {"ranges": RANGES, "unit_suspects": SUSPECTS, "min_hours": 4}
    text = format_report(quality_findings(table, config), table, notes=["ICULOS is kept."])
    assert "## Notes" in text and "- ICULOS is kept." in text


def test_missing_protected_values_leading_and_interior() -> None:
    df = pd.DataFrame(
        {
            "patient_id": ["A_p1"] * 3 + ["A_p2"] * 3 + ["A_p3"] * 2,
            "hour": [0, 1, 2, 0, 1, 2, 0, 1],
            "ICULOS": [np.nan, 2, 3, 1, np.nan, 3, 1, 2],  # leading null, interior null
            "SepsisLabel": [0, 0, 0, 1, np.nan, 0, 0, 0],  # interior null hides 1 -> 0
        }
    )
    out = missing_values(df, ["ICULOS", "SepsisLabel"]).set_index("column")
    assert out.loc["ICULOS", ["rows", "patients"]].tolist() == [2, 2]
    assert out.loc["SepsisLabel", ["rows", "patients"]].tolist() == [1, 1]


def test_findings_always_report_missing_ids_and_labels(table: pd.DataFrame) -> None:
    config = {"ranges": RANGES, "unit_suspects": SUSPECTS, "min_hours": 4}
    clean_out = quality_findings(table, config).set_index(["check", "column"])
    assert clean_out.loc[("missing value", "ICULOS"), ["A_rows", "A_patients"]].tolist() == [0, 0]
    assert clean_out.loc[("missing value", "SepsisLabel"), ["A_rows", "A_patients"]].tolist() == [
        0,
        0,
    ]
    gappy = table.copy()
    gappy.loc[4, "ICULOS"] = np.nan  # A_p2's repeated ICULOS is now hidden behind a NaN
    gappy.loc[4, "SepsisLabel"] = np.nan
    out = quality_findings(gappy, config).set_index(["check", "column"])
    assert out.loc[("missing value", "ICULOS"), ["A_rows", "A_patients"]].tolist() == [1, 1]
    assert out.loc[("missing value", "SepsisLabel"), ["A_rows", "A_patients"]].tolist() == [1, 1]
