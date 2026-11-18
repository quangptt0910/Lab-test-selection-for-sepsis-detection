"""Feature builder (M2-3): the view in, model-ready feature rows out.

One row per patient-hour: the keys, `ICULOS`, the raw view columns (minus `raw.exclude` in
`configs/features.yaml`) and the columns of every enabled feature block. Everything comes from the
view (`sepsis.data.view`), so a row at hour t can use only rows <= t and revealed labs.

Two paths give the same rows:
- batch, `build_features(view_table)`: every row of every patient at once (training);
- hour by hour, `features_at(view_at)`: one patient's row t (replay).

A new feature block is added to `BLOCKS` and to `configs/features.yaml`; the leakage suite
(`tests/test_feature_leakage.py`) then tests it automatically.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from sepsis.data.columns import LABS, VITALS
from sepsis.data.view import KEYS, STATIC, View, ViewTable
from sepsis.features import clinical, labs, vitals

Block = Callable[
    [pd.DataFrame, pd.DataFrame, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    pd.DataFrame,
]

# Never a feature: labels, the hospital, and every target-table column (LABEL_COLUMNS).
FORBIDDEN_COLUMNS = ("SepsisLabel", "hospital", "dt", "u1", "u0", "utility_diff", "weight")
SECTIONS = ("core", "extras")


def _components(values, lab_rows, block, t, panels) -> pd.DataFrame:  # noqa: ANN001
    return pd.concat(
        [
            clinical.sofa_components(values, t["sofa"]),
            clinical.news_components(values, t["news2"]),
            clinical.qsofa_partial(values, t["qsofa"]),
        ],
        axis=1,
    )


# Each block: (values, lab_rows, block settings, thresholds, panels config) -> feature columns.
BLOCKS: dict[str, Block] = {
    "shock_index": lambda v, lr, b, t, p: clinical.shock_index(v),
    "bun_cr": lambda v, lr, b, t, p: clinical.bun_cr(v),
    "partial_sofa": lambda v, lr, b, t, p: clinical.partial_sofa(v, t["sofa"]),
    "partial_sofa_delta": lambda v, lr, b, t, p: clinical.partial_sofa_delta(
        v, t["sofa"], window_rows=int(b["window_rows"])
    ),
    "sirs": lambda v, lr, b, t, p: clinical.sirs_count(v, t["sirs"]),
    "vital_minmax": lambda v, lr, b, t, p: vitals.rolling_minmax(v, b["vitals"], b["windows_rows"]),
    "hours_since_panel": lambda v, lr, b, t, p: labs.hours_since_panel(lr, p),
    "component_scores": _components,
    "vital_window_stats": lambda v, lr, b, t, p: vitals.window_stats(
        v, b["vitals"], b["windows_rows"], b["stats"]
    ),
    "vital_lags": lambda v, lr, b, t, p: vitals.lags(v, b["vitals"], b["lags"]),
    "vital_diffs": lambda v, lr, b, t, p: vitals.diffs(v, b["vitals"]),
}


def _blocks(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Every block in the config, by name, with its settings."""
    blocks = {name: b for section in SECTIONS for name, b in config[section].items()}
    unregistered = sorted(set(blocks) - set(BLOCKS))
    if unregistered:
        raise ValueError(f"feature blocks without a builder: {unregistered}")
    return blocks


def with_blocks(config: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """A copy of the config with exactly the named blocks enabled (an experiment's selection)."""
    unknown = sorted(set(names) - set(_blocks(config)))
    if unknown:
        raise ValueError(f"unknown feature blocks {unknown}")
    out = copy.deepcopy(dict(config))
    for section in SECTIONS:
        for name, block in out[section].items():
            block["enabled"] = name in names
    return out


def build_features(
    view: ViewTable, config: Mapping[str, Any], *, panels_config: Mapping[str, Any]
) -> pd.DataFrame:
    """Feature rows for every row of the view, in the view's order."""
    values, lab_rows = view.values, view.lab_rows
    if not values[KEYS].equals(lab_rows[KEYS]) or not values.index.equals(lab_rows.index):
        raise ValueError("values and lab_rows must have the same (patient_id, hour) rows")
    exclude = set(config.get("raw", {}).get("exclude", []))
    unknown = sorted(exclude - set(STATIC + VITALS + LABS))
    if unknown:
        raise ValueError(f"raw.exclude names unknown columns {unknown}")
    raw = [c for c in values.columns if c not in exclude]
    parts = [values[raw]]
    # An excluded lab is not shown, so it also never resets its panel's hours_since (finding 9).
    lab_rows = lab_rows.drop(columns=[c for c in lab_rows.columns if c in exclude])
    for name, block in _blocks(config).items():
        if block.get("enabled", False):
            parts.append(BLOCKS[name](values, lab_rows, block, config["thresholds"], panels_config))
    rows = pd.concat(parts, axis=1)
    duplicated = sorted(set(rows.columns[rows.columns.duplicated()]))
    if duplicated:
        raise ValueError(f"duplicate feature columns {duplicated}")
    forbidden = sorted(set(rows.columns) & set(FORBIDDEN_COLUMNS))
    if forbidden:
        raise ValueError(f"forbidden columns in the feature rows {forbidden}")
    return rows


def features_at(
    view: View, config: Mapping[str, Any], *, panels_config: Mapping[str, Any]
) -> pd.DataFrame:
    """One patient's feature row at row t, from the view at t (rows <= t only). Replay path."""
    table = ViewTable(values=view.values, lab_rows=view.lab_rows)
    rows = build_features(table, config, panels_config=panels_config)
    return rows[rows["hour"] == view.t].reset_index(drop=True)


def feature_columns(
    view: ViewTable, config: Mapping[str, Any], *, panels_config: Mapping[str, Any]
) -> list[str]:
    """The model-input columns: the feature-row columns without the keys `patient_id, hour`.

    `hour` is the row number, kept only to join; `ICULOS` is the time feature (ADR-0001).
    """
    first = ViewTable(values=view.values.iloc[:1], lab_rows=view.lab_rows.iloc[:1])
    columns = build_features(first, config, panels_config=panels_config).columns
    return [c for c in columns if c not in KEYS]
