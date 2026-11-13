import numpy as np
import pandas as pd
import pytest

from sepsis.eda.stats import (
    candidate_panels,
    co_measurement,
    cohort_summary,
    correlations,
    hospital_shift,
    measurement_gaps,
    missingness,
    onset_trajectories,
    patient_pca,
    redundant_pairs,
    septic_vs_nonseptic,
)

NAN = np.nan


def _patient(pid: str, labels: list[int], **columns) -> pd.DataFrame:
    n = len(labels)
    data = {
        "patient_id": [pid] * n,
        "hospital": [pid[0]] * n,
        "hour": list(range(n)),
        "ICULOS": list(range(1, n + 1)),
        "SepsisLabel": labels,
    }
    for name, values in columns.items():
        data[name] = values
    return pd.DataFrame(data)


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.concat(
        [
            # septic: first label row 2 -> t_sepsis 8
            _patient("A_p1", [0, 0, 1, 1], HR=[80.0, NAN, 90.0, 100.0], WBC=[NAN, 10.0, NAN, 14.0]),
            _patient("A_p2", [0, 0, 0], HR=[70.0, 72.0, 74.0], WBC=[NAN, NAN, NAN]),
            # septic on arrival: label 1 in the first row
            _patient("B_p1", [1, 1], HR=[110.0, 120.0], WBC=[20.0, NAN]),
            _patient("B_p2", [0, 0, 0, 0, 0], HR=[60.0, 62.0, 64.0, 66.0, 68.0], WBC=[5.0] * 5),
        ],
        ignore_index=True,
    )


def test_cohort_summary(df: pd.DataFrame) -> None:
    out = cohort_summary(df, septic_on_arrival_rows=1)
    assert out.loc["A", "patients"] == 2 and out.loc["A", "patient_hours"] == 7
    assert out.loc["A", "septic_pct"] == 50.0
    assert out.loc["A", "onset_hour_median"] == 8.0  # first label row 2 + 6
    assert out.loc["A", "septic_on_arrival_pct"] == 0.0
    assert out.loc["B", "septic_on_arrival_pct"] == 100.0
    assert out.loc["B", "hours_median"] == 3.5


def test_cohort_summary_counts_late_starting_records(df: pd.DataFrame) -> None:
    late = df.copy()
    late.loc[late["patient_id"] == "A_p2", "ICULOS"] += 10
    assert cohort_summary(late, 1).loc["A", "iculos_start_after_1_pct"] == 50.0


def test_missingness(df: pd.DataFrame) -> None:
    out = missingness(df, ["HR", "WBC"]).set_index(["hospital", "column"])
    assert out.loc[("A", "HR"), "pct_hours"] == pytest.approx(100 * 6 / 7)
    assert out.loc[("A", "WBC"), "pct_hours"] == pytest.approx(100 * 2 / 7)
    assert out.loc[("A", "WBC"), "pct_patients"] == 50.0  # A_p2 never has WBC


def test_measurement_gaps_use_rows_between_values(df: pd.DataFrame) -> None:
    out = measurement_gaps(df, ["WBC"]).set_index(["hospital", "column"])
    assert out.loc[("A", "WBC"), "n_gaps"] == 1  # A_p1: rows 1 and 3
    assert out.loc[("A", "WBC"), "median_hours"] == 2.0
    assert out.loc[("B", "WBC"), "median_hours"] == 1.0  # B_p2 every hour; B_p1 one value


def test_measurement_gaps_follow_hour_order_not_stored_order(df: pd.DataFrame) -> None:
    shuffled = df.sample(frac=1, random_state=0)
    pd.testing.assert_frame_equal(
        measurement_gaps(shuffled, ["WBC"]), measurement_gaps(df, ["WBC"])
    )


def test_co_measurement_is_conditional_probability() -> None:
    df = _patient(
        "A_p1",
        [0, 0, 0, 0],
        Lactate=[1.0, NAN, 2.0, NAN],
        pH=[7.4, NAN, 7.3, 7.2],
        WBC=[NAN, 9.0, NAN, NAN],
    )
    m = co_measurement(df, ["Lactate", "pH", "WBC"])
    assert m.loc["Lactate", "pH"] == 1.0  # every lactate hour also has pH
    assert m.loc["pH", "Lactate"] == pytest.approx(2 / 3)
    assert m.loc["Lactate", "WBC"] == 0.0
    assert m.loc["WBC", "WBC"] == 1.0


def test_candidate_panels_group_labs_measured_together() -> None:
    df = _patient(
        "A_p1",
        [0] * 6,
        Lactate=[1.0, NAN, 2.0, NAN, 1.5, NAN],
        pH=[7.4, NAN, 7.3, NAN, 7.35, NAN],
        WBC=[NAN, 9.0, NAN, 8.0, NAN, 7.0],
        Hgb=[NAN, 12.0, NAN, 11.0, NAN, 10.0],
        Fibrinogen=[NAN] * 6,  # never measured: left out
    )
    panels = candidate_panels(
        co_measurement(df, ["Lactate", "pH", "WBC", "Hgb", "Fibrinogen"]), 0.5
    )
    assert panels["Lactate"] == panels["pH"]
    assert panels["WBC"] == panels["Hgb"]
    assert panels["Lactate"] != panels["WBC"]
    assert "Fibrinogen" not in panels


def test_onset_trajectories_align_on_rows(df: pd.DataFrame) -> None:
    out = onset_trajectories(df, ["HR"], before=8, after=0, min_patients=1)
    a = out[(out["hospital"] == "A")].set_index("rel_hour")
    # A_p1: t_sepsis 8, rows 0..3 -> rel_hour -8..-5
    assert a["median"].to_dict() == {-8: 80.0, -6: 90.0, -5: 100.0}
    assert set(out["hospital"]) == {"A", "B"}  # non-septic patients never appear


def test_onset_trajectories_drop_sparse_points(df: pd.DataFrame) -> None:
    out = onset_trajectories(df, ["HR"], before=8, after=0, min_patients=2)
    assert out.empty  # each point has a single septic patient


def test_septic_vs_nonseptic_uses_label_hours_against_never_septic(df: pd.DataFrame) -> None:
    out = septic_vs_nonseptic(df, ["HR"]).set_index(["hospital", "column"])
    assert out.loc[("A", "HR"), "septic_median"] == 95.0  # label-1 hours of A_p1
    assert out.loc[("A", "HR"), "nonseptic_median"] == 72.0  # all hours of A_p2
    assert out.loc[("A", "HR"), "smd"] > 0


def test_correlations_and_redundant_pairs() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    df = pd.DataFrame({"SBP": x, "MAP": 2 * x + 1, "HR": rng.normal(size=50)})
    corr = correlations(df, ["SBP", "MAP", "HR"])
    pairs = redundant_pairs(corr, threshold=0.9)
    assert pairs[["a", "b"]].values.tolist() == [["SBP", "MAP"]]
    assert pairs["r"].iloc[0] == pytest.approx(1.0)


def test_hospital_shift_ranks_by_absolute_smd(df: pd.DataFrame) -> None:
    out = hospital_shift(df, ["HR", "WBC"])
    assert list(out.columns[:3]) == ["column", "mean_A", "mean_B"]
    assert out["smd"].abs().is_monotonic_decreasing
    hr = out.set_index("column").loc["HR"]
    assert hr["mean_A"] == pytest.approx(np.mean([80, 90, 100, 70, 72, 74]))
    assert hr["pct_hours_A"] == pytest.approx(100 * 6 / 7)


def test_patient_pca_one_row_per_patient(df: pd.DataFrame) -> None:
    points, explained = patient_pca(df, ["HR", "WBC"])
    assert sorted(points["patient_id"]) == ["A_p1", "A_p2", "B_p1", "B_p2"]
    assert list(points.columns) == ["patient_id", "hospital", "pc1", "pc2"]
    assert len(explained) == 2 and 0.99 <= sum(explained) <= 1.0 + 1e-9


def test_correlations_need_enough_shared_hours() -> None:
    x = np.arange(10, dtype=float)
    sparse = np.where(x < 3, x, np.nan)  # only 3 hours shared with SBP
    df = pd.DataFrame({"SBP": x, "MAP": 2 * x, "pH": sparse})
    corr = correlations(df, ["SBP", "MAP", "pH"], min_periods=5)
    assert corr.loc["SBP", "MAP"] == pytest.approx(1.0)
    assert np.isnan(corr.loc["SBP", "pH"])


def test_septic_vs_nonseptic_reports_counts(df: pd.DataFrame) -> None:
    out = septic_vs_nonseptic(df, ["HR", "WBC"]).set_index(["hospital", "column"])
    assert out.loc[("A", "HR"), ["n_septic", "n_nonseptic"]].tolist() == [2, 3]
    assert out.loc[("A", "WBC"), ["n_septic", "n_nonseptic"]].tolist() == [1, 0]


def test_onset_trajectories_forward_fill_uses_only_earlier_hours(df: pd.DataFrame) -> None:
    out = onset_trajectories(df, ["WBC"], before=8, after=0, min_patients=1, forward_fill=True)
    a = out[out["hospital"] == "A"].set_index("rel_hour")["median"].to_dict()
    # A_p1 WBC: [NaN, 10, NaN, 14] -> filled [NaN, 10, 10, 14]; row 0 stays NaN (no earlier value)
    assert a == {-7: 10.0, -6: 10.0, -5: 14.0}
