"""The policy interface (M5-1, ADR-0009): what a policy sees each hour and what it answers.

A policy never sees the record. Each hour t the replay engine hands it a `Step`: the view at t of
the patients still in the ICU (rows <= t, revealed labs only), their feature rows at t, and the
risk model's scores and alarms. It answers with the panels to request per patient; a patient left
out, or given an empty set, waits. This package may import only the view modules (import guard).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from sepsis.data.view import ViewTable


@dataclass(frozen=True)
class Step:
    """Everything a policy may see at row t."""

    t: int
    view: ViewTable  # rows <= t of the patients in the ICU at t; labs revealed by t only
    features: pd.DataFrame  # their feature rows at t (patient_id, hour, model inputs)
    scores: pd.Series  # risk score per patient_id
    alarms: pd.Series  # 0/1 per patient_id: score > the frozen threshold


class Policy(Protocol):
    """Decides which panels to request for each patient at row t (empty or missing: WAIT)."""

    name: str

    def decide(self, step: Step) -> Mapping[str, frozenset[str]]: ...
