import copy

import numpy as np
import pytest
from feature_toy import toy_record

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.view import as_recorded_reveal_rows, view_table
from sepsis.features.builder import (
    BLOCKS,
    build_features,
    feature_columns,
    with_blocks,
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_yaml(FEATURES_CONFIG)


@pytest.fixture(scope="module")
def panels() -> dict:
    return load_yaml(PANELS_CONFIG)


def _all_on(config: dict) -> dict:
    return with_blocks(config, list(BLOCKS))


# Batch equals hour by hour: tests/test_feature_leakage.py (every block, four reveal tables).


def test_output_rows_and_columns(config: dict, panels: dict) -> None:
    record = toy_record()
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    rows = build_features(view, config, panels_config=panels)
    assert len(rows) == len(record)
    assert list(rows.columns[:3]) == ["patient_id", "hour", "ICULOS"]
    model_inputs = feature_columns(view, config, panels_config=panels)
    assert model_inputs == list(rows.columns[2:])  # no keys: hour is not a model input
    for column in ("SepsisLabel", "hospital", "DBP", "SaO2"):
        assert column not in rows.columns
    for column in ("Age", "HR", "MAP", "Lactate", "shock_index", "partial_sofa", "hours_since_ABG"):
        assert column in rows.columns
    assert "news_HR" not in rows.columns  # extras off by default


def test_raw_exclusions_come_from_config(config: dict, panels: dict) -> None:
    record = toy_record()
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    cfg = copy.deepcopy(config)
    cfg["raw"]["exclude"] = []
    rows = build_features(view, cfg, panels_config=panels)
    assert {"DBP", "SaO2"} <= set(rows.columns)


def test_a_typo_in_raw_exclude_is_refused(config: dict, panels: dict) -> None:
    record = toy_record()
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    cfg = copy.deepcopy(config)
    cfg["raw"]["exclude"] = ["DBP", "SAO2"]
    with pytest.raises(ValueError, match="unknown columns .'SAO2'."):
        build_features(view, cfg, panels_config=panels)


def test_an_excluded_lab_does_not_reset_hours_since(config: dict, panels: dict) -> None:
    record = toy_record()
    record = record[record["patient_id"] == "A_1"].reset_index(drop=True)
    abg = ["pH", "PaCO2", "BaseExcess", "SaO2", "FiO2"]
    record[abg] = np.nan
    record.loc[2, "pH"] = 7.3
    record.loc[6, "SaO2"] = 92.0  # an SaO2-only entry, excluded (EDA finding 9)
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    since = build_features(view, config, panels_config=panels)["hours_since_ABG"]
    assert since.iloc[[3, 7, 9]].tolist() == [1.0, 5.0, 7.0]  # counts from the pH at row 2
    cfg = copy.deepcopy(config)
    cfg["raw"]["exclude"] = []
    since = build_features(view, cfg, panels_config=panels)["hours_since_ABG"]
    assert since.iloc[[3, 7, 9]].tolist() == [1.0, 1.0, 3.0]  # SaO2 counts when shown


def test_with_blocks(config: dict) -> None:
    core_only = with_blocks(config, ["shock_index", "vital_lags"])
    enabled = {
        name
        for section in ("core", "extras")
        for name, block in core_only[section].items()
        if block["enabled"]
    }
    assert enabled == {"shock_index", "vital_lags"}
    assert config["core"]["bun_cr"]["enabled"]  # the original is not changed
    with pytest.raises(ValueError, match="unknown feature blocks"):
        with_blocks(config, ["shock_index", "lactate_count"])


def test_b2_config_blocks_resolve(config: dict) -> None:
    names = load_yaml("configs/b2_lightgbm.yaml")["features"]["hand_crafted"]
    with_blocks(config, names)
    # B2 is exactly the core section (rank #1 blocks + SIRS, ADR-0001), so the two can't drift.
    core = {name for name, block in config["core"].items() if block["enabled"]}
    assert sorted(names) == sorted(core)


def test_mismatched_values_and_lab_rows_are_refused(config: dict, panels: dict) -> None:
    record = toy_record()
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    broken = type(view)(
        values=view.values, lab_rows=view.lab_rows.iloc[::-1].reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="same .patient_id, hour. rows"):
        build_features(broken, config, panels_config=panels)


def test_overlapping_blocks_are_refused(config: dict, panels: dict) -> None:
    cfg = _all_on(config)
    cfg["extras"]["vital_window_stats"]["stats"] = ["max"]
    cfg["extras"]["vital_window_stats"]["windows_rows"] = [6]  # same columns as vital_minmax
    record = toy_record()
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    with pytest.raises(ValueError, match="duplicate feature columns"):
        build_features(view, cfg, panels_config=panels)


def test_no_value_is_inf(config: dict, panels: dict) -> None:
    record = toy_record()
    rows = build_features(
        view_table(record, as_recorded_reveal_rows(record, delay=1)),
        _all_on(config),
        panels_config=panels,
    )
    numbers = rows.drop(columns=["patient_id"]).to_numpy(dtype=float)
    assert not np.isinf(numbers).any()
