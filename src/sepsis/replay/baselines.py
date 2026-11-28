"""AS_RECORDED (M5-2, ADR-0009): the clinicians' ordering, replayed.

It requests panel P at row t iff the record has a value of one of P's labs at row t. That is the
one policy that reads the record, by design, so it lives here and not in `sepsis.action` (whose
import guard allows only the view). It sets `needs_record`; before the first hour the engine calls
`bind(draws)` once with the cleaned record's lab columns under the per-replay patient codes. It
never sees `SepsisLabel`, the hospital or the real ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from sepsis.action.policy import Step
from sepsis.data.view import KEYS


class AsRecordedPolicy:
    """Requests what was drawn: panel P at row t iff a lab of P has a value at row t."""

    name = "AS_RECORDED"
    needs_record = True

    def __init__(self, panels_config: Mapping[str, Any]) -> None:
        self.panel_labs = {
            name: list(spec["labs"]) for name, spec in panels_config["panels"].items()
        }
        self._drawn: dict[tuple[str, int], frozenset[str]] | None = None

    def bind(self, draws: pd.DataFrame) -> None:
        """`draws`: `patient_id, hour` and the lab columns of the cleaned record."""
        drawn: dict[tuple[str, int], set[str]] = {}
        for panel, labs in self.panel_labs.items():
            present = [lab for lab in labs if lab in draws.columns]
            if not present:
                continue
            rows = draws.loc[draws[present].notna().any(axis=1), KEYS]
            for pid, hour in zip(rows["patient_id"], rows["hour"], strict=True):
                drawn.setdefault((pid, int(hour)), set()).add(panel)
        self._drawn = {key: frozenset(panels) for key, panels in drawn.items()}

    def decide(self, step: Step) -> dict[str, frozenset[str]]:
        if self._drawn is None:
            raise RuntimeError("AS_RECORDED needs bind(draws) before the first hour")
        out = {}
        for pid in step.scores.index:
            panels = self._drawn.get((pid, step.t))
            if panels:
                out[pid] = panels
        return out
