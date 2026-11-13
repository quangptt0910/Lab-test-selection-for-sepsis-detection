"""Cleaning rules for patient-hour tables (M0-3). Rules and limits come from configs/cleaning.yaml.

Every function is pure and returns a new table. Guarantees, so cleaning can never leak or break
scoring:
- Each value is judged only by itself or by other columns of the same hour, never by other hours.
- Limits are fixed constants from config, never fitted on data.
- Rows are never dropped or reordered: the official scorer matches predictions to labels by row.
- The id columns, ICULOS and SepsisLabel are never changed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

PROTECTED = frozenset({"patient_id", "hospital", "hour", "ICULOS", "SepsisLabel"})

CONVERSIONS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "fahrenheit_to_celsius": lambda x: (x - 32.0) * 5.0 / 9.0,
    "percent_to_fraction": lambda x: x / 100.0,
}


def _check_columns(columns: Iterable[str], where: str) -> None:
    touched = sorted(set(columns) & PROTECTED)
    if touched:
        raise ValueError(f"{where} may not change protected columns: {touched}")


def convert_units(df: pd.DataFrame, suspects: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Convert values inside a unit-suspect window, for rules with `action: convert`."""
    _check_columns(suspects, "unit_suspects")
    out = df.copy()
    for column, rule in suspects.items():
        if rule.get("action", "keep") != "convert" or column not in out.columns:
            continue
        mask = out[column].between(rule["low"], rule["high"])
        out.loc[mask, column] = CONVERSIONS[rule["conversion"]](out.loc[mask, column])
    return out


def apply_ranges(df: pd.DataFrame, ranges: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Apply each column's plausible range: `nan` blanks values outside it, `clip` moves them to
    the nearest limit, `keep` leaves them. Limits are inclusive; NaN stays NaN."""
    _check_columns(ranges, "ranges")
    out = df.copy()
    for column, rule in ranges.items():
        action = rule.get("action", "nan")
        if action not in ("nan", "clip", "keep"):
            raise ValueError(f"{column}: unknown action {action!r} (use nan, clip or keep)")
        if column not in out.columns or action == "keep":
            continue
        values = out[column]
        if action == "clip":
            out[column] = values.clip(rule["low"], rule["high"])
        else:
            outside = (values < rule["low"]) | (values > rule["high"])
            out[column] = values.mask(outside, np.nan)
    return out


def apply_bp_consistency(df: pd.DataFrame, rule: Mapping[str, Any]) -> pd.DataFrame:
    """Blank all listed pressures in hours where DBP >= SBP (both present): an impossible reading
    where we can't tell which value is wrong."""
    columns = list(rule["columns"])
    _check_columns(columns, "bp_consistency")
    if rule.get("action", "nan") != "nan":
        raise ValueError("bp_consistency supports only action: nan")
    out = df.copy()
    present = [c for c in columns if c in out.columns]
    out.loc[bp_inconsistent_hours(df), present] = np.nan
    return out


def bp_inconsistent_hours(df: pd.DataFrame) -> pd.Series:
    """True for hours where DBP >= SBP, both present (all False if either column is absent)."""
    if not {"SBP", "DBP"} <= set(df.columns):
        return pd.Series(False, index=df.index)
    return df["SBP"].notna() & df["DBP"].notna() & (df["DBP"] >= df["SBP"])


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Remove columns that are not used at all (missing ones are ignored)."""
    columns = list(columns)
    _check_columns(columns, "drop_columns")
    return df.drop(columns=[c for c in columns if c in df.columns])


def clean(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Apply all rules from a cleaning config: drop columns, convert units, apply plausible
    ranges and the same-hour blood-pressure check. Row count, order and index are kept.

    The blood-pressure check looks at the values before the range rule, so a flush artefact
    (e.g. SBP = MAP = DBP = 290) loses all three pressures even though DBP alone is out of range.
    """
    out = drop_columns(df, config.get("drop_columns", {}))
    out = convert_units(out, config.get("unit_suspects", {}))
    bp_bad = bp_inconsistent_hours(out)
    out = apply_ranges(out, config.get("ranges", {}))
    if "bp_consistency" in config:
        rule = config["bp_consistency"]
        out = apply_bp_consistency(out, rule)  # validates the rule
        out.loc[bp_bad, [c for c in rule["columns"] if c in out.columns]] = np.nan
    return out
