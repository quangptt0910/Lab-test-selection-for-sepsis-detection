"""Replay reveals (M5-1, ADR-0006): what a panel request reveals.

A request for panel P at row t reveals every value of P's labs recorded in P's look-back window
(rows t - lookback + 1 ... t, never after t). Each joins the view at its reveal row,
t + `reveal_delay_rows`. A value already revealed keeps its earlier reveal row. Nothing is ever
written into the record: a request that finds no value is logged as unavailable, never imputed.

`RevealState` holds one record and its reveal rows for all patients in a replay, and applies
one hour's requests for every patient at once.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from sepsis.data.columns import LABS, present
from sepsis.data.panels import PANEL_NAMES, validate_panels
from sepsis.data.view import KEYS

REQUEST_COLUMNS = ["patient_id", "hour", "panel", "status", "n_values"]
REVEALED, NOTHING_NEW, UNAVAILABLE = "revealed", "nothing_new", "unavailable"
STATUSES = (REVEALED, NOTHING_NEW, UNAVAILABLE)


class RevealState:
    """One record (all patients, sorted by patient and hour) and the reveal rows so far."""

    def __init__(self, record: pd.DataFrame, panels_config: Mapping[str, Any]) -> None:
        validate_panels(dict(panels_config))
        self.record = record
        keys = record[KEYS]
        if keys.duplicated().any() or not keys.equals(keys.sort_values(KEYS, kind="stable")):
            raise ValueError("the record must be sorted by (patient_id, hour), one row each")
        if not (record.groupby("patient_id", sort=False).cumcount() == record["hour"]).all():
            raise ValueError("each patient needs rows 0..n-1")
        self.labs = present(LABS, record.columns)
        self.delay = int(panels_config["reveal_delay_rows"])
        self.panels = {
            name: (
                [self.labs.index(lab) for lab in spec["labs"] if lab in self.labs],
                int(spec["lookback_rows"]),
            )
            for name, spec in panels_config["panels"].items()
        }
        self._values = record[self.labs].to_numpy(dtype=float)
        self._reveal = np.full(self._values.shape, np.nan)
        pid = record["patient_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
        lengths = np.diff(np.r_[starts, len(pid)])
        self._start = dict(zip(pid[starts], starts, strict=True))
        self._length = dict(zip(pid[starts], lengths, strict=True))
        self._last_t = -1

    def request(self, t: int, requests: Mapping[str, Iterable[str]]) -> pd.DataFrame:
        """Apply the requests made at row t (`patient_id -> panels`); return their log rows."""
        t = int(t)
        if t < self._last_t:
            raise ValueError("requests must come hour by hour, never for an earlier row")
        self._last_t = t
        by_panel: dict[str, list[str]] = {}
        for pid in sorted(requests):
            panels = set(requests[pid])
            unknown = sorted(panels - set(PANEL_NAMES))
            if unknown:
                raise ValueError(f"unknown panels {unknown}")
            if panels and (pid not in self._start or not 0 <= t < self._length[pid]):
                raise ValueError(f"{pid} is not in the ICU at row {t}")
            for panel in panels:
                by_panel.setdefault(panel, []).append(pid)
        logs = [self._reveal_panel(t, panel, pids) for panel, pids in sorted(by_panel.items())]
        if not logs:
            return pd.DataFrame(columns=REQUEST_COLUMNS)
        log = pd.concat(logs, ignore_index=True).sort_values(["patient_id", "panel"])
        return log.reset_index(drop=True)

    def _reveal_panel(self, t: int, panel: str, pids: list[str]) -> pd.DataFrame:
        """One panel requested at row t by several patients, all at once."""
        columns, lookback = self.panels[panel]
        hours = np.arange(max(0, t - lookback + 1), t + 1)  # the window: rows <= t only
        starts = np.array([self._start[pid] for pid in pids])
        rows = starts[:, None] + hours[None, :]  # patients x window rows
        cols = np.asarray(columns, dtype=int)
        recorded = ~np.isnan(self._values[rows[:, :, None], cols[None, None, :]])
        new = recorded & np.isnan(self._reveal[rows[:, :, None], cols[None, None, :]])
        where = np.nonzero(new)
        self._reveal[rows[where[0], where[1]], cols[where[2]]] = t + self.delay
        n_new = new.sum(axis=(1, 2))
        status = np.where(
            n_new > 0, REVEALED, np.where(recorded.any(axis=(1, 2)), NOTHING_NEW, UNAVAILABLE)
        )
        return pd.DataFrame(
            {
                "patient_id": pids,
                "hour": t,
                "panel": panel,
                "status": status,
                "n_values": n_new.astype(int),
            }
        )

    def reveal_rows(self, rows: np.ndarray | None = None) -> pd.DataFrame:
        """The reveal-row table so far: the record's keys, one column per lab (NaN: hidden).

        `rows` (a boolean mask of record rows) limits it to those rows.
        """
        keys = self.record[KEYS] if rows is None else self.record.loc[rows, KEYS]
        reveal = self._reveal if rows is None else self._reveal[rows]
        table = keys.reset_index(drop=True)
        return pd.concat([table, pd.DataFrame(reveal, columns=self.labs)], axis=1)
