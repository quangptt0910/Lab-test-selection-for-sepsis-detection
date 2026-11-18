import numpy as np
import pandas as pd
import pytest

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.features.clinical import (
    bun_cr,
    news_components,
    partial_sofa,
    partial_sofa_delta,
    qsofa_partial,
    shock_index,
    sirs_count,
    sofa_components,
)

NAN = np.nan


@pytest.fixture(scope="module")
def config() -> dict:
    return load_yaml(FEATURES_CONFIG)


def _values(pid: str = "A_1", **columns) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    return pd.DataFrame({"patient_id": pid, "hour": range(n), **columns})


# --- Ratios ------------------------------------------------------------------------------------


def test_shock_index() -> None:
    values = _values(HR=[100.0, 90.0, 80.0, NAN], SBP=[125.0, 0.0, NAN, 100.0])
    out = shock_index(values)["shock_index"]
    assert out.iloc[0] == pytest.approx(0.8)
    assert out.iloc[1:].isna().all()  # zero SBP, missing SBP, missing HR


def test_bun_cr() -> None:
    values = _values(BUN=[20.0, 20.0, NAN, 30.0], Creatinine=[1.0, 0.0, 1.0, NAN])
    out = bun_cr(values)["bun_cr"]
    assert out.iloc[0] == pytest.approx(20.0)
    assert out.iloc[1:].isna().all()


# --- SOFA: every boundary ---------------------------------------------------------------------

SOFA_BOUNDARIES = {
    "coagulation": (
        "Platelets",
        [
            (200, 0),
            (150, 0),
            (149.9, 1),
            (100, 1),
            (99.9, 2),
            (50, 2),
            (49.9, 3),
            (20, 3),
            (19.9, 4),
        ],
    ),
    "liver": (
        "Bilirubin_total",
        [
            (0.5, 0),
            (1.19, 0),
            (1.2, 1),
            (1.99, 1),
            (2.0, 2),
            (5.99, 2),
            (6.0, 3),
            (11.99, 3),
            (12.0, 4),
        ],
    ),
    "cardiovascular": ("MAP", [(80, 0), (70, 0), (69.9, 1), (40, 1)]),
    "renal": (
        "Creatinine",
        [
            (0.8, 0),
            (1.19, 0),
            (1.2, 1),
            (1.99, 1),
            (2.0, 2),
            (3.49, 2),
            (3.5, 3),
            (4.99, 3),
            (5.0, 4),
        ],
    ),
}


@pytest.mark.parametrize("component", SOFA_BOUNDARIES)
def test_sofa_component_boundaries(config: dict, component: str) -> None:
    column, cases = SOFA_BOUNDARIES[component]
    values = _values(**{column: [float(v) for v, _ in cases]})
    scores = sofa_components(values, config["thresholds"]["sofa"])[f"sofa_{component}"]
    assert scores.tolist() == [float(s) for _, s in cases]


def test_partial_sofa_hand_example(config: dict) -> None:
    sofa = config["thresholds"]["sofa"]
    values = _values(
        Platelets=[90.0, NAN],  # 2
        Bilirubin_total=[2.5, NAN],  # 2
        MAP=[65.0, 75.0],  # 1, 0
        Creatinine=[1.3, NAN],  # 1
    )
    assert partial_sofa(values, sofa)["partial_sofa"].tolist() == [6.0, 0.0]


def test_a_missing_component_scores_like_a_normal_one(config: dict) -> None:
    sofa = config["thresholds"]["sofa"]
    missing = _values(Platelets=[NAN], Bilirubin_total=[0.5], MAP=[80.0], Creatinine=[3.0])
    normal = missing.assign(Platelets=[250.0])
    assert partial_sofa(missing, sofa).equals(partial_sofa(normal, sofa))
    assert np.isnan(sofa_components(missing, sofa).loc[0, "sofa_coagulation"])
    # A column absent from the view behaves like a missing value.
    absent = missing.drop(columns="Platelets")
    assert partial_sofa(absent, sofa).equals(partial_sofa(normal, sofa))


def test_partial_sofa_delta_uses_a_24_row_window(config: dict) -> None:
    sofa = config["thresholds"]["sofa"]
    n = 30
    creat = [1.0] * n  # renal 0
    creat[3] = 2.5  # renal 2 at row 3 only
    creat[10:] = [3.6] * (n - 10)  # renal 3 from row 10
    values = _values(Creatinine=creat, MAP=[80.0] * n)
    delta = partial_sofa_delta(values, sofa, window_rows=24)["partial_sofa_delta_24"]
    assert delta.iloc[0] == 0  # one row of history
    assert delta.iloc[3] == 2 and delta.iloc[4] == 0
    assert delta.iloc[10] == 3  # min over rows 0..10 is 0
    assert delta.iloc[29] == 3  # window rows 6..29 still holds rows 6-9 (score 0)


def test_partial_sofa_delta_forgets_after_the_window(config: dict) -> None:
    sofa = config["thresholds"]["sofa"]
    creat = [1.0] * 5 + [3.6] * 30  # renal 0 for rows 0-4, then 3
    values = _values(Creatinine=creat, MAP=[80.0] * 35)
    delta = partial_sofa_delta(values, sofa, window_rows=24)["partial_sofa_delta_24"]
    assert delta.iloc[27] == 3  # window rows 4..27 includes row 4 (score 0)
    assert delta.iloc[28] == 0  # window rows 5..28 are all 3


@pytest.mark.parametrize("t", [0, 3, 9, 20])
def test_partial_sofa_delta_ignores_rows_after_t(config: dict, t: int) -> None:
    sofa = config["thresholds"]["sofa"]
    a = _values("A_1", Creatinine=[1.0, 2.5] * 15, MAP=[80.0] * 30)
    b = _values("B_1", Creatinine=[3.6] * 5, MAP=[80.0] * 5)
    values = pd.concat([a, b], ignore_index=True)
    changed = values.copy()
    later = (changed["patient_id"] == "A_1") & (changed["hour"] > t)
    changed.loc[later, ["Creatinine", "MAP"]] = [0.1, 40.0]
    keep = (values["patient_id"] != "A_1") | (values["hour"] <= t)
    before = partial_sofa_delta(values, sofa, window_rows=24)
    after = partial_sofa_delta(changed, sofa, window_rows=24)
    pd.testing.assert_frame_equal(before[keep], after[keep])
    assert not before.equals(after)  # the change after t does show up later


def test_partial_sofa_delta_never_mixes_patients(config: dict) -> None:
    sofa = config["thresholds"]["sofa"]
    a = _values("A_1", Creatinine=[1.0, 1.0], MAP=[80.0, 80.0])
    b = _values("B_1", Creatinine=[3.6, 3.6], MAP=[80.0, 80.0])
    delta = partial_sofa_delta(pd.concat([a, b], ignore_index=True), sofa, window_rows=24)
    assert delta["partial_sofa_delta_24"].tolist() == [0, 0, 0, 0]  # B_1 does not see A_1's 0


# --- SIRS and qSOFA ----------------------------------------------------------------------------


def test_sirs_boundaries_and_missing(config: dict) -> None:
    sirs = config["thresholds"]["sirs"]
    values = _values(
        Temp=[38.0, 38.1, 36.0, 35.9, NAN],
        HR=[90.0, 91.0, 90.0, 90.0, NAN],
        Resp=[20.0, 21.0, 20.0, 20.0, NAN],
        PaCO2=[32.0, NAN, 31.9, NAN, NAN],
        WBC=[12.0, 12.1, 4.0, 3.9, NAN],
    )
    # row 0: all at the boundary, none met. row 1: all four met. row 2: PaCO2 < 32 only.
    # row 3: Temp < 36 and WBC < 4. row 4: everything missing -> 0.
    assert sirs_count(values, sirs)["sirs_count"].tolist() == [0, 4, 1, 2, 0]


def test_qsofa_partial(config: dict) -> None:
    qsofa = config["thresholds"]["qsofa"]
    values = _values(Resp=[21.9, 22.0, 30.0, NAN], SBP=[100.1, 100.0, 120.0, 90.0])
    assert qsofa_partial(values, qsofa)["qsofa_partial"].tolist() == [0, 2, 1, 1]


# --- NEWS2 components --------------------------------------------------------------------------

NEWS_CASES = {
    "Resp": [(8, 3), (9, 1), (11, 1), (12, 0), (20, 0), (21, 2), (24, 2), (25, 3)],
    "O2Sat": [(91, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0), (100, 0)],
    "SBP": [(90, 3), (91, 2), (100, 2), (101, 1), (110, 1), (111, 0), (219, 0), (220, 3)],
    "HR": [
        (40, 3),
        (41, 1),
        (50, 1),
        (51, 0),
        (90, 0),
        (91, 1),
        (110, 1),
        (111, 2),
        (130, 2),
        (131, 3),
    ],
    "Temp": [
        (35.0, 3),
        (35.1, 1),
        (36.0, 1),
        (36.1, 0),
        (38.0, 0),
        (38.1, 1),
        (39.0, 1),
        (39.1, 2),
    ],
}


def test_news2_rounds_readings_before_banding(config: dict) -> None:
    values = _values(
        Temp=[38.04, 38.05, 35.95, 36.04],
        O2Sat=[91.5, 91.4, 95.49, 95.5],
        HR=[90.4, 90.5, 50.5, 40.4],
    )
    out = news_components(values, config["thresholds"]["news2"])
    assert out["news_Temp"].tolist() == [0.0, 1.0, 1.0, 1.0]  # 38.0, 38.1, 36.0, 36.0
    assert out["news_O2Sat"].tolist() == [2.0, 3.0, 1.0, 0.0]  # 92, 91, 95, 96
    assert out["news_HR"].tolist() == [0.0, 1.0, 0.0, 3.0]  # 90, 91, 51, 40


@pytest.mark.parametrize("vital", NEWS_CASES)
def test_news2_bands(config: dict, vital: str) -> None:
    cases = NEWS_CASES[vital]
    values = _values(**{vital: [float(v) for v, _ in cases] + [NAN]})
    scores = news_components(values, config["thresholds"]["news2"])[f"news_{vital}"]
    assert scores.iloc[:-1].tolist() == [float(s) for _, s in cases]
    assert np.isnan(scores.iloc[-1])  # missing input -> NaN, not a score


# --- Shape -------------------------------------------------------------------------------------


def test_outputs_keep_the_index(config: dict) -> None:
    values = _values(HR=[80.0, 90.0], SBP=[120.0, 100.0], MAP=[80.0, 60.0]).set_axis([7, 3])
    sofa = config["thresholds"]["sofa"]
    for out in (shock_index(values), partial_sofa(values, sofa), sofa_components(values, sofa)):
        assert list(out.index) == [7, 3]
