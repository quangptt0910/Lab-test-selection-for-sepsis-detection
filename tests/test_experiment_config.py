import copy

import pytest

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.eval.experiment_config import (
    ExperimentConfigError,
    check_model_type,
    selected_blocks,
    validate_experiment_config,
)

FEATURES = load_yaml(FEATURES_CONFIG)
VALID = {
    "id": "toy-001",
    "hypothesis": "the runner works",
    "seed": 0,
    "split": {"setups": ["A_to_B", "B_to_A", "mix"]},
    "features": {"hand_crafted": ["shock_index", "sirs"], "signatures": None, "lab_masking_p": 0.3},
    "target": "utility_diff",
    "model": {"type": "dummy", "params": {"column": "ICULOS"}},
    "note": "",
}


def _with(path: str, value) -> dict:
    cfg = copy.deepcopy(VALID)
    *parents, last = path.split(".")
    node = cfg
    for key in parents:
        node = node[key]
    if value is DELETE:
        del node[last]
    else:
        node[last] = value
    return cfg


DELETE = object()


def test_valid_config_passes() -> None:
    assert validate_experiment_config(VALID, FEATURES) == VALID
    check_model_type(VALID)


def test_the_b2_config_follows_the_schema() -> None:
    cfg = load_yaml("configs/b2_lightgbm.yaml")
    validate_experiment_config(cfg, FEATURES)
    check_model_type(cfg)  # lightgbm_regressor, registered in M3-2


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("seed", DELETE, "lacks keys"),
        ("impute", "ffill", "unknown keys"),
        ("features.impute", "ffill_nan_before_first", "unknown keys"),
        ("split.folds", 5, "unknown keys"),
        ("model.threshold", 0.5, "unknown keys"),
        ("threshold_grid", {"n_points": 5}, "unknown keys"),
        ("id", "B2_001", "id"),
        ("id", "", "id"),
        ("hypothesis", " ", "hypothesis"),
        ("note", None, "note"),
        ("seed", "0", "seed"),
        ("seed", True, "seed"),
        ("split.setups", [], "non-empty"),
        ("split.setups", ["A_to_B", "A_to_B"], "distinct"),
        ("split.setups", ["A_to_C"], "distinct setups"),
        ("features.hand_crafted", ["lactate_count"], "unknown feature blocks"),
        ("features.hand_crafted", ["sirs", "sirs"], "distinct"),
        ("features.signatures", {"depth": 2}, "signatures"),
        ("features.signatures", [], "non-empty"),
        ("features.signatures", ["sig_nope"], "unknown signature blocks"),
        ("features.signatures", ["sig_vitals", "sig_vitals"], "distinct"),
        ("features.signatures", ["sirs"], "unknown signature blocks"),
        ("features.hand_crafted", ["sirs", "sig_vitals"], "features.signatures"),
        ("features.lab_masking_p", 1.5, "lab_masking_p"),
        ("features.lab_masking_p", "0.3", "lab_masking_p"),
        ("target", "SepsisLabel", "target"),
        ("model.type", 3, "model.type"),
        ("model.params", [], "model.params"),
        ("split", ["mix"], "mapping"),
    ],
)
def test_schema_rejections(path: str, value, message: str) -> None:
    with pytest.raises(ExperimentConfigError, match=message):
        validate_experiment_config(_with(path, value), FEATURES)


def test_unknown_model_type_is_refused() -> None:
    with pytest.raises(ExperimentConfigError, match="not registered"):
        check_model_type(_with("model.type", "xgboost"))


def test_a_b3_config_passes_and_selects_both_lists() -> None:
    cfg = _with("features.signatures", ["sig_vitals", "sig_partial_sofa"])
    assert validate_experiment_config(cfg, FEATURES) == cfg
    assert selected_blocks(cfg) == ("shock_index", "sirs", "sig_vitals", "sig_partial_sofa")


def test_no_signatures_selects_the_hand_crafted_blocks() -> None:
    assert selected_blocks(VALID) == ("shock_index", "sirs")
