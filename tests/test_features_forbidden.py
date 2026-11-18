"""ADR-0002: no lab counts, missing flags or "measured this hour" features.

Every feature function runs with all flags on. Two records show the same lab values at every row,
but one has extra draws of the same value: every feature except `hours_since_<panel>` (the value's
age, allowed by ADR-0002) must be identical, so no feature counts or flags draws.
"""

import re

import numpy as np
import pandas as pd
import pytest

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.view import as_recorded_reveal_rows, view_table
from sepsis.features import clinical, labs, vitals

NAN = np.nan
FORBIDDEN_NAME = re.compile(
    r"(^|_)(count|n|num|missing|measured|known|flag|present|indicator|is)(_|$)", re.I
)
ALLOWED_NAMES = {"sirs_count"}  # a clinical score: how many SIRS criteria are met


@pytest.fixture(scope="module")
def config() -> dict:
    return load_yaml(FEATURES_CONFIG)


def _all_features(values: pd.DataFrame, lab_rows: pd.DataFrame, config: dict) -> pd.DataFrame:
    t, extras = config["thresholds"], config["extras"]
    vital_list = config["core"]["vital_minmax"]["vitals"]
    parts = [
        clinical.shock_index(values),
        clinical.bun_cr(values),
        clinical.sofa_components(values, t["sofa"]),
        clinical.partial_sofa(values, t["sofa"]),
        clinical.partial_sofa_delta(
            values, t["sofa"], window_rows=config["core"]["partial_sofa_delta"]["window_rows"]
        ),
        clinical.sirs_count(values, t["sirs"]),
        clinical.qsofa_partial(values, t["qsofa"]),
        clinical.news_components(values, t["news2"]),
        vitals.rolling_minmax(
            values, vital_list, windows_rows=config["core"]["vital_minmax"]["windows_rows"]
        ),
        vitals.window_stats(
            values,
            vital_list,
            windows_rows=extras["vital_window_stats"]["windows_rows"],
            stats=extras["vital_window_stats"]["stats"],
        ),
        vitals.lags(values, vital_list, lags=extras["vital_lags"]["lags"]),
        vitals.diffs(values, vital_list),
        labs.hours_since_panel(lab_rows, load_yaml(PANELS_CONFIG)),
    ]
    return pd.concat(parts, axis=1)


def _record(extra_draws: bool) -> pd.DataFrame:
    n = 10
    lactate = [NAN, 2.0, NAN, NAN, NAN, NAN, 4.0, NAN, NAN, NAN]
    platelets = [NAN, NAN, 120.0, NAN, NAN, NAN, NAN, NAN, NAN, NAN]
    if extra_draws:  # same values drawn again: shown values never change
        lactate[3] = lactate[4] = 2.0
        platelets[5] = platelets[8] = 120.0
    return pd.DataFrame(
        {
            "patient_id": "A_1",
            "hour": range(n),
            "HR": np.linspace(80, 110, n),
            "SBP": [120.0] * n,
            "MAP": [75.0] * n,
            "Resp": [18.0] * n,
            "Temp": [37.0] * n,
            "O2Sat": [97.0] * n,
            "Lactate": lactate,
            "Platelets": platelets,
            "Creatinine": [NAN] * n,
            "BUN": [NAN] * n,
            "WBC": [NAN] * n,
            "PaCO2": [NAN] * n,
            "Bilirubin_total": [NAN] * n,
        }
    )


def _features(extra_draws: bool, config: dict) -> pd.DataFrame:
    record = _record(extra_draws)
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    return _all_features(view.values, view.lab_rows, config)


def test_no_feature_name_counts_or_flags_labs(config: dict) -> None:
    names = [c for c in _features(False, config).columns if c not in ALLOWED_NAMES]
    assert [c for c in names if FORBIDDEN_NAME.search(c)] == []


def test_the_name_check_catches_flag_names() -> None:
    bad = ["n_lactate", "lactate_count", "lactate_missing", "is_measured_pH", "CBC_known"]
    assert all(FORBIDDEN_NAME.search(name) for name in bad)
    assert not any(FORBIDDEN_NAME.search(n) for n in ["bun_cr", "HR_min_6", "news_SBP"])


def test_extra_draws_of_the_same_value_change_nothing_but_the_age(config: dict) -> None:
    few, many = _features(False, config), _features(True, config)
    age = [c for c in few.columns if c.startswith("hours_since_")]
    pd.testing.assert_frame_equal(few.drop(columns=age), many.drop(columns=age))
    assert not few[age].equals(many[age])  # the age does change, as intended


def test_extras_are_off_by_default(config: dict) -> None:
    assert all(not block["enabled"] for block in config["extras"].values())
    assert all(block["enabled"] for block in config["core"].values())
