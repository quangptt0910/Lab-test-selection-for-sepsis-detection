"""The view: what feature and action code may see at row t (M2-1, ADR-0005, ADR-0006).

This module is the only gate between a patient record and the feature/action code. At row t the
view holds:
- vitals and static columns (`ICULOS`, demographics) of rows <= t, forward-filled within the
  patient, NaN before the first value;
- for each lab, among its values whose **reveal row** is <= t, the one recorded latest (NaN if
  none), plus the row it was recorded at (`lab_rows`, for `hours_since_<panel>`).

`SepsisLabel`, `hospital` and every other column never enter the view. Nothing is filled backwards,
and nothing crosses patients.

Reveals are given as a **reveal-row table**: the record's `patient_id, hour` rows, one column per
lab, holding the row from which that recorded value is visible (NaN: never revealed). The same
format serves training (as recorded: recorded row + `reveal_delay_rows`, optionally with random
panel masking) and replay (request row + `reveal_delay_rows`, M5-1). A value can only be revealed
after the row it was recorded.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sepsis.data.clean import clean
from sepsis.data.columns import DEMOGRAPHICS, LABS, VITALS, present
from sepsis.data.panels import PANEL_NAMES, validate_panels

KEYS = ["patient_id", "hour"]
STATIC = ["ICULOS", *DEMOGRAPHICS]


@dataclass(frozen=True)
class ViewTable:
    """The view at every row: `values` (keys, static, vitals, labs) and `lab_rows` (keys, and for
    each lab the row its shown value was recorded at). Both sorted by patient and hour."""

    values: pd.DataFrame
    lab_rows: pd.DataFrame


@dataclass(frozen=True)
class View:
    """One patient's view at row t. Built from rows <= t only, so later rows are unreachable."""

    patient_id: str
    t: int
    values: pd.DataFrame
    lab_rows: pd.DataFrame

    def latest(self) -> pd.DataFrame:
        """The one row of `values` at row t."""
        return self.values[self.values["hour"] == self.t]


def _ordered(table: pd.DataFrame, what: str) -> pd.DataFrame:
    """Sorted by patient and hour; raise unless each patient has rows 0..n-1 exactly once."""
    if not pd.api.types.is_integer_dtype(table["hour"]):
        raise ValueError(f"hour must be an integer row number in the {what}")
    if table.duplicated(KEYS).any():
        raise ValueError(f"duplicate (patient_id, hour) rows in the {what}")
    out = table.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if not (out.groupby("patient_id").cumcount() == out["hour"]).all():
        raise ValueError(f"each patient in the {what} needs its rows 0..n-1")
    return out


def _delay(panels_config: Mapping[str, Any]) -> int:
    validate_panels(dict(panels_config))
    return int(panels_config["reveal_delay_rows"])


# --- Reveal-row tables --------------------------------------------------------------------------


def as_recorded_reveal_rows(hours: pd.DataFrame, delay: int) -> pd.DataFrame:
    """Every recorded lab value revealed `delay` rows after the row it was recorded."""
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        raise ValueError(f"delay must be an integer >= 1, got {delay!r}")
    record = _ordered(hours, "record")
    out = record[KEYS].copy()
    for lab in present(LABS, record.columns):
        out[lab] = (record["hour"] + delay).where(record[lab].notna()).astype(float)
    return out


def mask_panel_draws(
    reveal_rows: pd.DataFrame, panels_config: Mapping[str, Any], p: float, seed: int
) -> pd.DataFrame:
    """Hide each panel draw (one patient, one row, one panel) with probability `p`, seeded.

    A hidden draw hides every lab of that panel at that row. Masking only hides; it never reveals
    or moves a value. For training only (ADR-0002); evaluation uses `evaluation_reveal_rows`.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"lab_masking_p must be between 0 and 1, got {p!r}")
    validate_panels(dict(panels_config))
    out = _ordered(reveal_rows, "reveal-row table").copy()
    # One random stream per patient (seed + a stable hash of the id), so a patient gets the same
    # mask whichever other patients are in the table (e.g. in every CV fold).
    draws = np.empty((len(out), len(PANEL_NAMES)))
    for pid, index in out.groupby("patient_id", sort=False).indices.items():
        rng = np.random.default_rng([seed, zlib.crc32(str(pid).encode())])
        draws[index] = rng.random((len(index), len(PANEL_NAMES)))
    for j, name in enumerate(PANEL_NAMES):
        columns = present(panels_config["panels"][name]["labs"], out.columns)
        if not columns:
            continue
        hide = out[columns].notna().any(axis=1).to_numpy() & (draws[:, j] < p)
        out.loc[hide, columns] = np.nan
    return out


def training_reveal_rows(
    hours: pd.DataFrame, panels_config: Mapping[str, Any], *, lab_masking_p: float, seed: int
) -> pd.DataFrame:
    """As recorded with the panel delay, then random panel-draw masking (training only)."""
    reveal = as_recorded_reveal_rows(hours, _delay(panels_config))
    return mask_panel_draws(reveal, panels_config, lab_masking_p, seed)


def evaluation_reveal_rows(hours: pd.DataFrame, panels_config: Mapping[str, Any]) -> pd.DataFrame:
    """As recorded with the panel delay. Never masked: there is no masking argument."""
    return as_recorded_reveal_rows(hours, _delay(panels_config))


def validate_reveal_rows(
    hours: pd.DataFrame, reveal_rows: pd.DataFrame, panels_config: Mapping[str, Any]
) -> None:
    """Raise ValueError unless every reveal follows ADR-0006 for its panel.

    For a value recorded at row r with reveal row v (request row v - delay):
    - v is finite and whole, and v >= r + delay (requested at or after the row it was recorded);
    - r is inside the panel's look-back window ending at the request row:
      r >= (v - delay) - lookback_rows + 1.
    """
    delay = _delay(panels_config)
    record = _ordered(hours, "record")
    reveal = _ordered(reveal_rows, "reveal-row table")
    if not reveal[KEYS].equals(record[KEYS]):
        raise ValueError("the reveal-row table must have the same (patient_id, hour) rows")
    recorded = record["hour"].to_numpy()
    for name, panel in panels_config["panels"].items():
        for lab in present(panel["labs"], reveal.columns):
            v = reveal[lab].to_numpy(dtype=float)
            has = ~np.isnan(v)
            if not np.isfinite(v[has]).all() or (v[has] != np.floor(v[has])).any():
                raise ValueError(f"reveal rows of {lab} must be finite whole row numbers")
            if (v[has] < recorded[has] + delay).any():
                raise ValueError(f"a {lab} value is revealed less than {delay} row(s) after it")
            request = v[has] - delay
            if (recorded[has] < request - panel["lookback_rows"] + 1).any():
                raise ValueError(
                    f"a {lab} value is older than the {name} look-back window "
                    f"({panel['lookback_rows']} rows) of its request"
                )


# --- The view ---------------------------------------------------------------------------------


def _lab_view(record: pd.DataFrame, reveal: np.ndarray, lab: str) -> tuple[np.ndarray, np.ndarray]:
    """For one lab: the shown value and its recorded row at every record row (NaN if none)."""
    n = len(record)
    has = ~np.isnan(reveal) & record[lab].notna().to_numpy()
    recorded = record["hour"].to_numpy()
    if not np.isfinite(reveal[has]).all():
        raise ValueError(f"reveal rows of {lab} must be finite (NaN means never revealed)")
    if (reveal[has] != np.floor(reveal[has])).any():
        raise ValueError(f"reveal rows of {lab} must be whole row numbers")
    if (reveal[has] < recorded[has] + 1).any():
        raise ValueError(f"a {lab} value can only be revealed after the row it was recorded")
    values, rows = np.full(n, np.nan), np.full(n, np.nan)
    if not has.any():
        return values, rows

    events = pd.DataFrame(
        {
            "patient_id": record["patient_id"].to_numpy()[has],
            "reveal": reveal[has].astype(np.int64),
            "rec_row": recorded[has].astype(np.int64),
            "value": record[lab].to_numpy(dtype=float)[has],
        }
    ).sort_values(["patient_id", "reveal", "rec_row"], kind="stable")
    # In reveal order, keep the value recorded latest so far: an older value revealed later never
    # replaces a newer one.
    best_row = events.groupby("patient_id")["rec_row"].cummax()
    events["best_value"] = (
        events["value"].where(events["rec_row"] == best_row).groupby(events["patient_id"]).ffill()
    )
    events["best_row"] = best_row.astype(float)
    events = events.drop_duplicates(["patient_id", "reveal"], keep="last")

    left = pd.DataFrame(
        {"patient_id": record["patient_id"].to_numpy(), "hour": recorded.astype(np.int64)}
    )
    left["_pos"] = np.arange(n)
    merged = pd.merge_asof(
        left.sort_values("hour", kind="stable"),
        events[["patient_id", "reveal", "best_value", "best_row"]].sort_values("reveal"),
        left_on="hour",
        right_on="reveal",
        by="patient_id",
        direction="backward",
    )
    pos = merged["_pos"].to_numpy()
    values[pos] = merged["best_value"].to_numpy(dtype=float)
    rows[pos] = merged["best_row"].to_numpy(dtype=float)
    return values, rows


def view_table(hours: pd.DataFrame, reveal_rows: pd.DataFrame) -> ViewTable:
    """The view at every row of every patient, from a (cleaned) record and its reveal rows.

    Row s holds exactly what `view_at(record, reveal_rows, s).latest()` holds.
    """
    record = _ordered(hours, "record")
    reveal = _ordered(reveal_rows, "reveal-row table")
    if not reveal[KEYS].equals(record[KEYS]):
        raise ValueError("the reveal-row table must have the same (patient_id, hour) rows")
    labs = present(LABS, record.columns)
    extra = sorted(set(reveal.columns) - set(KEYS) - set(labs))
    missing = [lab for lab in labs if lab not in reveal.columns]
    if extra or missing:
        raise ValueError(
            f"reveal-row columns must be the record's labs (extra {extra}, missing {missing})"
        )

    shown = present(STATIC, record.columns) + present(VITALS, record.columns)
    values = record[KEYS].copy()
    values[shown] = record.groupby("patient_id", sort=False)[shown].ffill()
    lab_rows = record[KEYS].copy()
    for lab in labs:
        values[lab], lab_rows[lab] = _lab_view(record, reveal[lab].to_numpy(dtype=float), lab)
    return ViewTable(values=values, lab_rows=lab_rows)


def view_at(record: pd.DataFrame, reveal_rows: pd.DataFrame, t: int) -> View:
    """One patient's view at row t, built from rows <= t and reveals visible by t only.

    `record` must already be cleaned, as in training; `build_view_at` cleans and checks the
    reveal rows first.
    """
    ids = record["patient_id"].unique()
    if len(ids) != 1 or set(reveal_rows["patient_id"].unique()) != set(ids):
        raise ValueError("view_at needs the record and reveal rows of one patient")
    if not 0 <= t < len(record):
        raise ValueError(f"t must be a row of the record (0..{len(record) - 1}), got {t}")
    upto = record[record["hour"] <= t]
    reveal = reveal_rows[reveal_rows["hour"] <= t].copy()
    labs = [c for c in reveal.columns if c not in KEYS]
    reveal[labs] = reveal[labs].where(reveal[labs] <= t)
    table = view_table(upto, reveal)
    return View(patient_id=str(ids[0]), t=t, values=table.values, lab_rows=table.lab_rows)


# --- Entry points: clean, reveal, view ----------------------------------------------------------


def build_training_view(
    hours: pd.DataFrame,
    cleaning_config: Mapping[str, Any],
    panels_config: Mapping[str, Any],
    *,
    lab_masking_p: float,
    seed: int,
) -> ViewTable:
    """Clean (M0-3), reveal as recorded with random panel-draw masking, build the view."""
    cleaned = clean(hours, cleaning_config)
    reveal = training_reveal_rows(cleaned, panels_config, lab_masking_p=lab_masking_p, seed=seed)
    validate_reveal_rows(cleaned, reveal, panels_config)
    return view_table(cleaned, reveal)


def build_evaluation_view(
    hours: pd.DataFrame, cleaning_config: Mapping[str, Any], panels_config: Mapping[str, Any]
) -> ViewTable:
    """Clean (M0-3), reveal as recorded without masking, build the view."""
    cleaned = clean(hours, cleaning_config)
    reveal = evaluation_reveal_rows(cleaned, panels_config)
    validate_reveal_rows(cleaned, reveal, panels_config)
    return view_table(cleaned, reveal)


def build_view_at(
    record: pd.DataFrame,
    reveal_rows: pd.DataFrame,
    t: int,
    cleaning_config: Mapping[str, Any],
    panels_config: Mapping[str, Any],
) -> View:
    """Replay entry point: clean the patient's record (M0-3, as in training), check the reveal rows
    against ADR-0006, and return the view at row t."""
    cleaned = clean(record, cleaning_config)
    validate_reveal_rows(cleaned, reveal_rows, panels_config)
    return view_at(cleaned, reveal_rows, t)
