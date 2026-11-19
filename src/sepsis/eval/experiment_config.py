"""Check an experiment config before anything runs (ADR-0008).

The check is strict: a missing key, a wrong type or an unknown key is an error, so a typo or a
stale key fails at load time instead of silently doing nothing. The threshold grid is not an
experiment setting: it comes from `configs/eval.yaml`, the same for every experiment.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from sepsis.eval.threshold import SETUPS
from sepsis.features.builder import with_blocks
from sepsis.risk.base import MODEL_TARGETS, MODELS

# Experiment `target` -> the target-table column a model is trained on.
TARGET_COLUMN = {"utility_diff": "utility_diff", "sepsis_label": "SepsisLabel"}
TARGETS = tuple(TARGET_COLUMN)
TOP_KEYS = {"id", "hypothesis", "seed", "split", "features", "target", "model", "note"}
SPLIT_KEYS = {"setups"}
FEATURE_KEYS = {"hand_crafted", "signatures", "lab_masking_p"}
MODEL_KEYS = {"type", "params"}
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class ExperimentConfigError(ValueError):
    """Raised when an experiment config breaks the schema."""


def _keys(section: Any, expected: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(section, Mapping):
        raise ExperimentConfigError(f"{where} must be a mapping")
    missing = sorted(expected - set(section))
    unknown = sorted(set(section) - expected)
    if missing:
        raise ExperimentConfigError(f"{where} lacks keys {missing}")
    if unknown:
        raise ExperimentConfigError(f"{where} has unknown keys {unknown}")
    return section


def validate_experiment_config(
    cfg: Mapping[str, Any], features_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the config if it follows the schema; raise `ExperimentConfigError` otherwise.

    Block names are checked against `features_config` (configs/features.yaml). The model type is
    checked separately by `check_model_type`, because models are registered over several issues.
    """
    _keys(cfg, TOP_KEYS, "the config")
    if not isinstance(cfg["id"], str) or not ID_PATTERN.match(cfg["id"]):
        raise ExperimentConfigError("id must be lowercase letters, digits and dashes")
    if not isinstance(cfg["hypothesis"], str) or not cfg["hypothesis"].strip():
        raise ExperimentConfigError("hypothesis must be a non-empty string")
    if not isinstance(cfg["note"], str):
        raise ExperimentConfigError('note must be a string ("" if none)')
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int):
        raise ExperimentConfigError("seed must be an integer")

    setups = _keys(cfg["split"], SPLIT_KEYS, "split")["setups"]
    if not isinstance(setups, list) or not setups:
        raise ExperimentConfigError(f"split.setups must be a non-empty list of {SETUPS}")
    if len(set(setups)) != len(setups) or not set(setups) <= set(SETUPS):
        raise ExperimentConfigError(f"split.setups must list distinct setups from {SETUPS}")

    features = _keys(cfg["features"], FEATURE_KEYS, "features")
    blocks = features["hand_crafted"]
    if not isinstance(blocks, list) or len(set(blocks)) != len(blocks):
        raise ExperimentConfigError("features.hand_crafted must be a list of distinct block names")
    try:
        with_blocks(features_config, blocks)
    except ValueError as error:
        raise ExperimentConfigError(f"features.hand_crafted: {error}") from error
    if features["signatures"] is not None:
        raise ExperimentConfigError("features.signatures must be null until signatures exist (M4)")
    p = features["lab_masking_p"]
    if isinstance(p, bool) or not isinstance(p, int | float) or not 0 <= p <= 1:
        raise ExperimentConfigError("features.lab_masking_p must be a number between 0 and 1")

    if cfg["target"] not in TARGETS:
        raise ExperimentConfigError(f"target must be one of {TARGETS}")

    model = _keys(cfg["model"], MODEL_KEYS, "model")
    if not isinstance(model["type"], str):
        raise ExperimentConfigError("model.type must be a string")
    if not isinstance(model["params"], Mapping):
        raise ExperimentConfigError("model.params must be a mapping")
    return dict(cfg)


def check_model_type(cfg: Mapping[str, Any], models: Mapping[str, Any] = MODELS) -> None:
    """Raise `ExperimentConfigError` if `model.type` is not registered or can't take `target`."""
    model_type = cfg["model"]["type"]
    if model_type not in models:
        raise ExperimentConfigError(
            f"model.type {model_type!r} is not registered; known: {sorted(models)}"
        )
    targets = MODEL_TARGETS.get(model_type)
    if targets is not None and cfg["target"] not in targets:
        raise ExperimentConfigError(
            f"{model_type} needs target in {targets}, not {cfg['target']!r}"
        )
