"""ADR-0002: no lab counts, missing flags or "measured this hour" features.

Every feature block runs through the builder with all blocks on. Two records show the same lab
values at every row, but one has extra draws of the same value: every feature except
`hours_since_<panel>` (the value's age, allowed by ADR-0002) must be identical, so no feature
counts or flags draws.
"""

import re

import numpy as np
import pandas as pd
import pytest

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.view import ViewTable, as_recorded_reveal_rows, view_table
from sepsis.features.builder import BLOCKS, build_features, with_blocks

NAN = np.nan
FORBIDDEN_NAME = re.compile(
    r"(^|_)(count|n|num|missing|measured|known|flag|present|indicator|is)(_|$)", re.I
)
ALLOWED_NAMES = {"sirs_count"}  # a clinical score: how many SIRS criteria are met


@pytest.fixture(scope="module")
def config() -> dict:
    return load_yaml(FEATURES_CONFIG)


def all_features(view: ViewTable, config: dict) -> pd.DataFrame:
    """Every feature block on, through the builder."""
    cfg = with_blocks(config, list(BLOCKS))
    return build_features(view, cfg, panels_config=load_yaml(PANELS_CONFIG))


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
    return all_features(view_table(record, as_recorded_reveal_rows(record, delay=1)), config)


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


def test_experiment_configs_name_existing_feature_blocks(config: dict) -> None:
    from pathlib import Path

    blocks = set(config["core"]) | set(config["extras"])
    for path in sorted(Path("configs").glob("*.yaml")):
        features = load_yaml(path).get("features")
        if isinstance(features, dict) and "hand_crafted" in features:
            unknown = sorted(set(features["hand_crafted"]) - blocks)
            assert unknown == [], f"{path.name} names unknown feature blocks {unknown}"


def test_extras_are_off_by_default(config: dict) -> None:
    assert all(not block["enabled"] for block in config["extras"].values())
    assert all(not block["enabled"] for block in config["signatures"].values())
    assert all(block["enabled"] for block in config["core"].values())
