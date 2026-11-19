from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from runner_toy import records

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.eval.tables import (
    KEYS,
    Tables,
    TableSettings,
    build_tables,
    cache_keys,
    check_tables,
    load_or_build,
)
from sepsis.features.builder import FORBIDDEN_COLUMNS
from sepsis.labels.utility import LABELS_CONFIG, UtilityConstants

BLOCKS = ("shock_index", "partial_sofa", "sirs", "hours_since_panel")


def settings(**changes) -> TableSettings:
    base = TableSettings(
        blocks=BLOCKS,
        lab_masking_p=0.3,
        seed=0,
        cleaning=load_yaml(Path("configs/cleaning.yaml")),
        panels=load_yaml(PANELS_CONFIG),
        features=load_yaml(FEATURES_CONFIG),
        labels=UtilityConstants.from_config(load_yaml(LABELS_CONFIG)),
    )
    return replace(base, **changes)


@pytest.fixture(scope="module")
def hours() -> pd.DataFrame:
    toy = records()
    return toy[toy["patient_id"].str.contains("_d")]  # dev patients only


@pytest.fixture(scope="module")
def tables(hours: pd.DataFrame) -> Tables:
    return build_tables(hours.sample(frac=1, random_state=1), settings())


def test_tables_are_aligned_and_hold_no_label_columns(tables: Tables, hours) -> None:
    expected = hours[KEYS].sort_values(KEYS).reset_index(drop=True)
    for table in (tables.train_X, tables.eval_X, tables.targets):
        pd.testing.assert_frame_equal(table[KEYS], expected)
    assert not set(tables.feature_columns) & set(FORBIDDEN_COLUMNS)
    assert not set(tables.feature_columns) & set(KEYS)
    assert {"utility_diff", "weight", "SepsisLabel"} <= set(tables.targets.columns)
    np.testing.assert_array_equal(
        tables.targets["SepsisLabel"], hours.sort_values(KEYS)["SepsisLabel"]
    )


def test_training_table_is_masked_and_evaluation_table_is_not(hours) -> None:
    full = build_tables(hours, settings(lab_masking_p=1.0))
    assert full.train_X["Lactate"].isna().all()  # every panel draw hidden in training
    assert full.eval_X["Lactate"].notna().any()  # the evaluation view keeps them


def test_a_patients_rows_do_not_depend_on_the_other_patients(tables: Tables, hours) -> None:
    pid = "B_d21"
    alone = build_tables(hours[hours["patient_id"] == pid], settings())
    for name in ("train_X", "eval_X", "targets"):
        whole = getattr(tables, name)
        part = whole[whole["patient_id"] == pid].reset_index(drop=True)
        pd.testing.assert_frame_equal(getattr(alone, name), part)


def test_misaligned_tables_are_refused(tables: Tables) -> None:
    shuffled = tables.eval_X.sample(frac=1, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="eval_X rows differ"):
        check_tables(replace(tables, eval_X=shuffled))
    with pytest.raises(ValueError, match="label or key"):
        check_tables(replace(tables, feature_columns=[*tables.feature_columns, "SepsisLabel"]))


# --- Cache --------------------------------------------------------------------------------------


def test_cache_keys_change_with_every_input() -> None:
    ids, base = ["A_d00", "B_d00"], cache_keys(settings(), ["A_d00", "B_d00"], "fp", source="s")
    assert cache_keys(settings(seed=1), ids, "fp", source="s") == (
        cache_keys(settings(seed=1), ids, "fp", source="s")[0],
        base[1],  # the evaluation table does not depend on the seed ...
    )
    assert cache_keys(settings(seed=1), ids, "fp", source="s")[0] != base[0]  # ... training does
    assert cache_keys(settings(lab_masking_p=0.5), ids, "fp", source="s")[0] != base[0]
    changed = [
        cache_keys(settings(blocks=("sirs",)), ids, "fp", source="s"),
        cache_keys(settings(), ["A_d00"], "fp", source="s"),
        cache_keys(settings(), ids, "other", source="s"),
        cache_keys(settings(), ids, "fp", source="other"),
        cache_keys(settings(panels={**settings().panels, "reveal_delay_rows": 2}), ids, "fp", "s"),
    ]
    for keys in changed:
        assert keys[0] != base[0] and keys[1] != base[1]


def test_cache_hit_returns_equal_tables(tmp_path: Path, hours, tables: Tables) -> None:
    first, info = load_or_build(hours, settings(), split_fingerprint="fp", cache_dir=tmp_path)
    assert info["hit"] is False
    again, info = load_or_build(hours, settings(), split_fingerprint="fp", cache_dir=tmp_path)
    assert info["hit"] is True
    for name in ("train_X", "eval_X", "targets"):
        pd.testing.assert_frame_equal(getattr(again, name), getattr(tables, name))
    assert again.feature_columns == tables.feature_columns


def test_an_unfinished_cache_is_never_read(tmp_path: Path, hours) -> None:
    ids = sorted(hours["patient_id"].unique())
    train_key, eval_key = cache_keys(settings(), ids, "fp")
    for key in (train_key, eval_key):  # a crash left only .tmp folders
        (tmp_path / f"{key}.tmp").mkdir()
        (tmp_path / f"{key}.tmp" / "meta.json").write_text("{}")
    _, info = load_or_build(hours, settings(), split_fingerprint="fp", cache_dir=tmp_path)
    assert info["hit"] is False
    assert (tmp_path / train_key / "train_X.parquet").exists()
    assert not (tmp_path / f"{train_key}.tmp").exists()
