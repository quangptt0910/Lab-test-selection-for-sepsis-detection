"""Value-of-information targets (M6-1, ADR-0010): what one panel request at a decision row is worth.

For patient i, decision row t and panel P, one fitted risk model with its frozen threshold scores
two views of the same record, both as recorded and unmasked except for P:
- *without*: P's values in its look-back window (t - L_P, t] and every P value after t hidden;
- *with*: the same, plus the window's values revealed at t + `reveal_delay_rows`.

    gain_h = sum over s = t+1 .. min(t+h, n-1) of (U1 - U0)_s * (alarm_with,s - alarm_without,s)

The targets read the whole record, the future and U1 - U0: they are labels, never features, and no
policy may import this module (`tests/test_import_guard.py`).

Each decision row becomes a copy of the record up to t + horizon (`make_copies`), with its own id,
so one `view_table` + `build_features` call per panel and view covers every decision row at once,
and copies never see each other's reveals. Pure functions; I/O is in `sepsis.labels.build_voi`.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sepsis.data.columns import present
from sepsis.data.panels import PANEL_NAMES, validate_panels
from sepsis.data.view import KEYS, as_recorded_reveal_rows, validate_reveal_rows, view_table
from sepsis.features.builder import build_features

CONFIG_KEYS = {
    "source_run",
    "horizons",
    "decision_every_rows",
    "seed",
    "chunk_patients",
    "refit_check",
    "costs",
    "provenance",
}
REFIT_CHECK_KEYS = ("atol", "max_share_off", "max_abs_diff")
DECISION_KEYS = ["patient_id", "t"]
TARGET_STATS = ["dscore_t1", "score_without_t1", "alarm_without_t1"]


# --- Config ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VoiConfig:
    """`configs/voi.yaml`, checked."""

    source_run: str
    horizons: tuple[int, ...]
    decision_every_rows: int
    seed: int
    chunk_patients: int
    refit_check: dict[str, float]  # atol, max_share_off, max_abs_diff
    costs: dict[str, float]
    provenance: str

    @property
    def max_horizon(self) -> int:
        return self.horizons[-1]


def _positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return value


def load_voi_config(raw: Mapping[str, Any]) -> VoiConfig:
    """Check the VoI config: exactly its keys, a positive cost for each of the 7 panels, and
    increasing positive horizons."""
    if not isinstance(raw, Mapping) or set(raw) != CONFIG_KEYS:
        raise ValueError(f"the VoI config needs exactly the keys {sorted(CONFIG_KEYS)}")
    costs = raw["costs"]
    if not isinstance(costs, Mapping) or set(costs) != set(PANEL_NAMES):
        raise ValueError(f"costs must name exactly the panels {list(PANEL_NAMES)}")
    for name, cost in costs.items():
        if isinstance(cost, bool) or not isinstance(cost, int | float) or not cost > 0:
            raise ValueError(f"the cost of {name} must be a positive number, got {cost!r}")
    horizons = tuple(_positive_int(h, "every horizon") for h in raw["horizons"])
    if not horizons or list(horizons) != sorted(set(horizons)):
        raise ValueError(f"horizons must be distinct and increasing, got {list(horizons)}")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    check = raw["refit_check"]
    if not isinstance(check, Mapping) or set(check) != set(REFIT_CHECK_KEYS):
        raise ValueError(f"refit_check needs exactly the keys {list(REFIT_CHECK_KEYS)}")
    for name in REFIT_CHECK_KEYS:
        value = check[name]
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise ValueError(f"refit_check.{name} must be a positive number, got {value!r}")
    if check["max_share_off"] >= 1:
        raise ValueError("refit_check.max_share_off is a share of rows, below 1")
    return VoiConfig(
        source_run=str(raw["source_run"]),
        horizons=horizons,
        decision_every_rows=_positive_int(raw["decision_every_rows"], "decision_every_rows"),
        seed=seed,
        chunk_patients=_positive_int(raw["chunk_patients"], "chunk_patients"),
        refit_check={name: float(check[name]) for name in REFIT_CHECK_KEYS},
        costs={name: float(costs[name]) for name in PANEL_NAMES},
        provenance=str(raw["provenance"]),
    )


# --- Decision rows and availability ---------------------------------------------------------------


def decision_rows(keys: pd.DataFrame, every: int, seed: int) -> pd.DataFrame:
    """`patient_id, t`: rows o, o + every, ... <= n - 2 per patient (t + 1 must exist).

    The offset o in [0, every) comes from `rng([seed, crc32(patient_id)])`, so a patient's rows do
    not depend on which other patients are in the table (the masking scheme, ADR-0006).
    """
    _positive_int(every, "every")
    sizes = keys.groupby("patient_id", sort=True)["hour"].size()
    parts = []
    for pid, n in sizes.items():
        offset = int(np.random.default_rng([seed, zlib.crc32(str(pid).encode())]).integers(every))
        ts = np.arange(offset, int(n) - 1, every)
        parts.append(pd.DataFrame({"patient_id": pid, "t": ts}))
    if not parts:
        return pd.DataFrame({"patient_id": pd.Series(dtype=object), "t": pd.Series(dtype=int)})
    return pd.concat(parts, ignore_index=True)[DECISION_KEYS]


def _ordered(record: pd.DataFrame) -> pd.DataFrame:
    out = record.sort_values(KEYS, kind="stable").reset_index(drop=True)
    if not (out.groupby("patient_id").cumcount() == out["hour"]).all():
        raise ValueError("each patient needs its whole record, rows 0..n-1")
    return out


def availability(
    record: pd.DataFrame, panels_config: Mapping[str, Any], decisions: pd.DataFrame
) -> pd.DataFrame:
    """`patient_id, t, panel, available`: 1 iff the record has a value of the panel in its
    look-back window (t - L, t], so a request at t would reveal something (ADR-0010 §4)."""
    validate_panels(dict(panels_config))
    record = _ordered(record)
    start = record.groupby("patient_id", sort=False).indices
    first = np.array([start[pid][0] for pid in decisions["patient_id"]], dtype=int)
    t = decisions["t"].to_numpy(dtype=int)
    parts = []
    for name in PANEL_NAMES:
        panel = panels_config["panels"][name]
        labs = present(panel["labs"], record.columns)
        has = record[labs].notna().any(axis=1) if labs else pd.Series(False, index=record.index)
        # Values of the panel in rows 0 .. r of the patient, at record position first + r.
        count = has.groupby(record["patient_id"], sort=False).cumsum().to_numpy()
        lo = t - panel["lookback_rows"]
        before = np.where(lo >= 0, count[first + np.maximum(lo, 0)], 0)
        parts.append(
            decisions[DECISION_KEYS].assign(
                panel=name, available=(count[first + t] - before > 0).astype(int)
            )
        )
    return pd.concat(parts, ignore_index=True)


# --- The two views --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Copies:
    """One copy of the record per decision row, rows 0 .. min(t + horizon, n - 1).

    `record.patient_id` is the copy's own id; `t` and `source` give its decision row and real
    patient for every row of `record`.
    """

    record: pd.DataFrame
    t: np.ndarray
    source: np.ndarray


def make_copies(record: pd.DataFrame, decisions: pd.DataFrame, horizon: int) -> Copies:
    """The decision rows' copies of the (cleaned) record, sorted by copy id and hour."""
    record = _ordered(record)
    merged = decisions[DECISION_KEYS].merge(record, on="patient_id", how="inner")
    merged = merged[merged["hour"] <= merged["t"] + horizon]
    merged = merged.assign(
        _copy=merged["patient_id"].astype(str) + "|" + merged["t"].map("{:06d}".format)
    )
    merged = merged.sort_values(["_copy", "hour"], kind="stable").reset_index(drop=True)
    t, source = merged["t"].to_numpy(dtype=int), merged["patient_id"].to_numpy()
    out = merged.drop(columns=["t", "patient_id"]).rename(columns={"_copy": "patient_id"})
    return Copies(
        record=out[[*KEYS, *[c for c in out.columns if c not in KEYS]]], t=t, source=source
    )


def counterfactual_reveal_rows(
    copies: Copies, panels_config: Mapping[str, Any], panel: str, *, with_request: bool
) -> pd.DataFrame:
    """The reveal-row table of every copy: as recorded, except that the panel's values in
    (t - L, t] and after t are hidden, and with `with_request` the window's values are revealed
    at t + delay instead (ADR-0010 §3)."""
    validate_panels(dict(panels_config))
    delay = int(panels_config["reveal_delay_rows"])
    reveal = as_recorded_reveal_rows(copies.record, delay)
    lookback = panels_config["panels"][panel]["lookback_rows"]
    row, t = copies.record["hour"].to_numpy(), copies.t
    window = (row > t - lookback) & (row <= t)
    for lab in present(panels_config["panels"][panel]["labs"], reveal.columns):
        recorded = reveal[lab].notna().to_numpy()
        values = reveal[lab].to_numpy(dtype=float).copy()
        values[window | (row > t)] = np.nan
        if with_request:
            hit = window & recorded
            values[hit] = t[hit] + delay
        reveal[lab] = values
    return reveal


@dataclass(frozen=True)
class CounterfactualRows:
    """Feature rows at s = t+1 .. t+horizon of every decision row, in both views (same order)."""

    keys: pd.DataFrame  # patient_id (the real one), t, hour (= s)
    without_X: pd.DataFrame
    with_X: pd.DataFrame


def counterfactual_features(
    record: pd.DataFrame,
    decisions: pd.DataFrame,
    panel: str,
    *,
    horizon: int,
    panels_config: Mapping[str, Any],
    features_config: Mapping[str, Any],
) -> CounterfactualRows:
    """Build both views of `panel` for every decision row and their feature rows after t."""
    copies = make_copies(record, decisions, horizon)
    after = copies.record["hour"].to_numpy() > copies.t
    keys = pd.DataFrame({"patient_id": copies.source[after], "t": copies.t[after]}).assign(
        hour=copies.record["hour"].to_numpy()[after]
    )
    built = {}
    for with_request in (False, True):
        reveal = counterfactual_reveal_rows(copies, panels_config, panel, with_request=with_request)
        validate_reveal_rows(copies.record, reveal, panels_config)  # ADR-0006 rules, every copy
        view = view_table(copies.record, reveal)
        rows = build_features(view, features_config, panels_config=panels_config)
        built[with_request] = rows[after].drop(columns="patient_id").reset_index(drop=True)
    return CounterfactualRows(keys=keys, without_X=built[False], with_X=built[True])


# --- The gain -------------------------------------------------------------------------------------


def gains(
    keys: pd.DataFrame,
    score_with: np.ndarray,
    score_without: np.ndarray,
    utility_diff: np.ndarray,
    *,
    threshold: float,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """`patient_id, t, gain_h<h>..., dscore_t1, score_without_t1, alarm_without_t1`, one row per
    decision row, from the scores of its rows s = t+1 .. in both views.

    Alarm = score > threshold, strictly (ADR-0007)."""
    score_with = np.asarray(score_with, dtype=float)
    score_without = np.asarray(score_without, dtype=float)
    utility_diff = np.asarray(utility_diff, dtype=float)
    if not len(keys) == len(score_with) == len(score_without) == len(utility_diff):
        raise ValueError("keys, scores and utility_diff must have one entry per row")
    if np.isnan(utility_diff).any():
        raise ValueError("utility_diff is missing for some rows")
    ahead = keys["hour"].to_numpy() - keys["t"].to_numpy()
    if (ahead < 1).any():
        raise ValueError("gains takes the rows after the decision row only")
    flip = (score_with > threshold).astype(int) - (score_without > threshold).astype(int)
    table = keys[DECISION_KEYS].assign(_ahead=ahead, _value=flip * utility_diff)
    group = table.groupby(DECISION_KEYS, sort=True)
    out = group.size().rename("_n").reset_index()[DECISION_KEYS]
    for h in horizons:
        summed = (
            table["_value"]
            .where(table["_ahead"] <= h, 0.0)
            .groupby([table["patient_id"], table["t"]], sort=True)
            .sum()
        )
        out[f"gain_h{h}"] = summed.to_numpy()
    first = ahead == 1
    t1 = keys.loc[first, DECISION_KEYS].assign(
        dscore_t1=np.abs(score_with - score_without)[first],
        score_without_t1=score_without[first],
        alarm_without_t1=(score_without[first] > threshold).astype(int),
    )
    return out.merge(t1, on=DECISION_KEYS, how="left", validate="one_to_one")


def panel_targets(
    available: pd.DataFrame, found: pd.DataFrame, panel: str, horizons: Sequence[int]
) -> pd.DataFrame:
    """The target rows of one panel: every decision row with `available`, and the gains and
    scores where available (NaN elsewhere: a request there reveals nothing, ADR-0010 §4)."""
    mine = available[available["panel"] == panel].reset_index(drop=True)
    columns = [*(f"gain_h{h}" for h in horizons), *TARGET_STATS]
    out = mine.merge(found[[*DECISION_KEYS, *columns]], on=DECISION_KEYS, how="left")
    unavailable = out["available"].to_numpy() == 0
    out[columns] = out[columns].astype(float)
    out.loc[unavailable, columns] = np.nan
    missing = out.loc[~unavailable, columns].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{panel}: {int(missing.sum())} available decision rows have no gain")
    return out[[*DECISION_KEYS, "panel", "available", *columns]]
