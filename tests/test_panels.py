import copy

import pytest

from sepsis.config import load_yaml
from sepsis.data.columns import LABS
from sepsis.data.panels import (
    PANEL_NAMES,
    PANELS_CONFIG,
    PREPROCESSING_CONFIG,
    lab_to_panel,
    validate_panels,
)


@pytest.fixture
def config() -> dict:
    return load_yaml(PANELS_CONFIG)


def test_committed_config_is_valid(config: dict) -> None:
    validate_panels(config)


def test_every_lab_maps_to_exactly_one_panel(config: dict) -> None:
    listed = [lab for panel in config["panels"].values() for lab in panel["labs"]]
    assert sorted(listed) == sorted(LABS)
    assert len(listed) == len(set(listed))
    assert set(lab_to_panel(config)) == set(LABS)


def test_panel_names_match_the_glossary(config: dict) -> None:
    assert set(config["panels"]) == set(PANEL_NAMES)


def test_labs_that_move_between_hospitals_have_one_fixed_panel(config: dict) -> None:
    mapping = lab_to_panel(config)
    assert mapping["HCO3"] == "CHEM"
    assert mapping["Chloride"] == "CHEM"
    assert mapping["FiO2"] == "ABG"
    assert mapping["Glucose"] == "CHEM"


def test_reveal_delay_is_at_least_one(config: dict) -> None:
    # Look-forward itself is tested with the reveal function in M2-1 / M5-1.
    assert config["reveal_delay_rows"] >= 1


def _broken(config: dict, change) -> dict:
    bad = copy.deepcopy(config)
    change(bad)
    return bad


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c["panels"]["CBC"]["labs"].remove("Hct"), "no panel"),
        (lambda c: c["panels"]["ABG"]["labs"].append("HCO3"), "more than one panel"),
        (lambda c: c["panels"]["CBC"]["labs"].append("Bands"), "unknown lab"),
        (lambda c: c["panels"].pop("TROPONIN"), "panel names"),
        (lambda c: c["panels"]["CHEM"].update(lookback_rows=0), "lookback_rows"),
        (lambda c: c["panels"]["CHEM"].update(lookback_rows=2.5), "lookback_rows"),
        (lambda c: c.update(reveal_delay_rows=0), "reveal_delay_rows"),
    ],
)
def test_invalid_configs_are_rejected(config: dict, change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_panels(_broken(config, change))


def test_b1_has_a_fixed_fill_value_for_every_lab() -> None:
    fill = load_yaml(PREPROCESSING_CONFIG)["b1"]["lab_fill"]
    assert set(fill) == set(LABS)
    assert all(isinstance(entry["value"], int | float) for entry in fill.values())
