"""Leakage suite: runs on every registered feature block automatically.

A block added to `sepsis.features.builder.BLOCKS` (and to configs/features.yaml) is tested here
without writing new tests: no future rows, batch equals hour by hour for any reveal table,
unrevealed labs never matter, no forbidden columns. The toy record holds every column the view
can show (tests/feature_toy.py).
"""

import numpy as np
import pandas as pd
import pytest
from feature_toy import toy_record
from test_features_forbidden import ALLOWED_NAMES, FORBIDDEN_NAME

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.data.columns import DEMOGRAPHICS, LABS, VITALS
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.view import as_recorded_reveal_rows, view_at, view_table
from sepsis.features.builder import (
    BLOCKS,
    FORBIDDEN_COLUMNS,
    build_features,
    features_at,
    with_blocks,
)

CONFIG = load_yaml(FEATURES_CONFIG)
PANELS = load_yaml(PANELS_CONFIG)
RECORD_COLUMNS = ["ICULOS", "SepsisLabel", *DEMOGRAPHICS, *VITALS, *LABS]
EVERY_BLOCK = list(BLOCKS)


def _features(
    record: pd.DataFrame, blocks: str | list[str], reveal: pd.DataFrame | None = None
) -> pd.DataFrame:
    if reveal is None:
        reveal = as_recorded_reveal_rows(record, delay=1)
    names = [blocks] if isinstance(blocks, str) else blocks
    return build_features(
        view_table(record, reveal), with_blocks(CONFIG, names), panels_config=PANELS
    )


def _reveals(record: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Reveal-row tables of the kinds replay and training produce."""
    reveal = as_recorded_reveal_rows(record, delay=1)
    if kind == "late":  # requested some rows after recording
        reveal[LABS] = reveal[LABS] + 3
    elif kind == "masked":  # panel-draw masking: about half the draws never revealed
        hide = np.random.default_rng(0).random(reveal[LABS].shape) < 0.5
        reveal[LABS] = reveal[LABS].mask(hide)
    elif kind == "out of order":  # an older value revealed after a newer one
        first = reveal.groupby("patient_id")[LABS].transform("first")
        reveal[LABS] = reveal[LABS].where(reveal[LABS] != first, reveal[LABS] + 10)
    return reveal


def test_every_config_block_is_registered() -> None:
    in_config = set(CONFIG["core"]) | set(CONFIG["extras"]) | set(CONFIG["signatures"])
    assert in_config == set(BLOCKS)


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("t", [0, 5, 9, 17, 28])
def test_no_future(block: str, t: int) -> None:
    """Every record column of every patient changes after t: rows up to t stay identical."""
    record = toy_record()
    changed = record.copy()
    later = (changed["hour"] > t).to_numpy()
    rng = np.random.default_rng(t)
    for column in RECORD_COLUMNS:
        if column == "SepsisLabel":
            changed.loc[later, column] = 1 - changed.loc[later, column]
        elif column == "ICULOS":  # integer column: jump the clock
            changed.loc[later, column] = changed.loc[later, column] + 100
        else:  # new values, and values appear or vanish after t
            new = rng.uniform(1, 200, later.sum())
            changed.loc[later, column] = np.where(rng.random(later.sum()) < 0.5, new, np.nan)
    before, after = _features(record, block), _features(changed, block)
    keep = ~later
    pd.testing.assert_frame_equal(before[keep], after[keep])


KINDS = ["as recorded", "late", "masked", "out of order"]


def _batch_equals_hour_by_hour(kind: str, every_row: bool) -> None:
    record = toy_record()
    reveal = _reveals(record, kind)
    cfg = with_blocks(CONFIG, EVERY_BLOCK)
    batch = build_features(view_table(record, reveal), cfg, panels_config=PANELS)
    for pid, rows in record.groupby("patient_id"):
        one_reveal = reveal[reveal["patient_id"] == pid]
        n = len(rows)
        # B_2's labs start late (a backward fill shows there), so it is checked at every row.
        hours = range(n) if every_row or pid == "B_2" else sorted({*range(0, n, 4), n - 1})
        expected = batch[batch["patient_id"] == pid].set_index("hour", drop=False).loc[list(hours)]
        hourly = pd.concat(
            [features_at(view_at(rows, one_reveal, t), cfg, panels_config=PANELS) for t in hours],
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(hourly, expected.reset_index(drop=True))


@pytest.mark.parametrize("kind", KINDS)
def test_batch_equals_hour_by_hour(kind: str) -> None:
    """Every block at once: the training rows equal what replay sees at that hour."""
    _batch_equals_hour_by_hour(kind, every_row=False)


@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_batch_equals_hour_by_hour_every_row(kind: str) -> None:
    _batch_equals_hour_by_hour(kind, every_row=True)


@pytest.mark.parametrize("panel", PANELS["panels"])
def test_an_unrevealed_panel_never_matters(panel: str) -> None:
    """Every block at once: labs never revealed can hold any values without changing a feature.

    One panel at a time, the others revealed as recorded; together the cases cover all 26 labs.
    """
    labs = PANELS["panels"][panel]["labs"]
    record = toy_record()
    reveal = as_recorded_reveal_rows(record, delay=1)
    reveal[labs] = np.nan  # never revealed
    changed = record.copy()
    rng = np.random.default_rng(0)
    changed[labs] = rng.uniform(0.1, 400, (len(changed), len(labs)))  # any values at all
    pd.testing.assert_frame_equal(
        _features(record, EVERY_BLOCK, reveal), _features(changed, EVERY_BLOCK, reveal)
    )


@pytest.mark.parametrize("block", BLOCKS)
def test_no_forbidden_columns(block: str) -> None:
    columns = _features(toy_record(), block).columns
    assert not set(columns) & set(FORBIDDEN_COLUMNS)
    flagged = [c for c in columns if c not in ALLOWED_NAMES and FORBIDDEN_NAME.search(c)]
    assert flagged == []
    excluded = CONFIG["raw"]["exclude"]  # no block brings an excluded column back
    assert [c for c in columns if c in excluded or c.split("_")[0] in excluded] == []
