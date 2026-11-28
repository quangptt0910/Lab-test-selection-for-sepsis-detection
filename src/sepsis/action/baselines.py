"""Reference policies that need nothing but the Step (M5-2, ADR-0009).

- NONE requests nothing, ever: the floor, B2 on vitals only.
- ALL requests every panel every hour: the ceiling, which reveals exactly as recorded and so
  reproduces B2's batch result.

AS_RECORDED needs the record, so it lives outside this package (`sepsis.replay.baselines`).
"""

from __future__ import annotations

from sepsis.action.policy import Step
from sepsis.data.panels import PANEL_NAMES


class NonePolicy:
    """Never requests a panel: every patient waits every hour."""

    name = "NONE"

    def decide(self, step: Step) -> dict[str, frozenset[str]]:
        return {}


class AllPolicy:
    """Requests all panels for every patient in the ICU, every hour."""

    name = "ALL"

    def decide(self, step: Step) -> dict[str, frozenset[str]]:
        return {pid: frozenset(PANEL_NAMES) for pid in step.scores.index}
