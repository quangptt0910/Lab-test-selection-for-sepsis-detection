"""Lab panels and reveal rules from `configs/panels.yaml` (ADR-0006)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sepsis.data.columns import LABS

PANELS_CONFIG = Path("configs/panels.yaml")
PREPROCESSING_CONFIG = Path("configs/preprocessing.yaml")
PANEL_NAMES = ("LACTATE", "ABG", "CHEM", "CBC", "LFT", "COAG", "TROPONIN")


def validate_panels(config: dict[str, Any], labs: Sequence[str] = LABS) -> None:
    """Raise ValueError unless every lab is in exactly one panel and the reveal rules are valid."""
    panels = config["panels"]
    if set(panels) != set(PANEL_NAMES):
        raise ValueError(f"panel names {sorted(panels)} differ from {sorted(PANEL_NAMES)}")

    seen: dict[str, str] = {}
    for name, panel in panels.items():
        for lab in panel["labs"]:
            if lab not in labs:
                raise ValueError(f"unknown lab {lab!r} in panel {name}")
            if lab in seen:
                raise ValueError(f"{lab} is in more than one panel ({seen[lab]}, {name})")
            seen[lab] = name
        lookback = panel["lookback_rows"]
        if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
            raise ValueError(
                f"lookback_rows of {name} must be a positive integer, got {lookback!r}"
            )

    missing = [lab for lab in labs if lab not in seen]
    if missing:
        raise ValueError(f"labs in no panel: {missing}")

    delay = config["reveal_delay_rows"]
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
        raise ValueError(f"reveal_delay_rows must be an integer >= 1, got {delay!r}")


def lab_to_panel(config: dict[str, Any]) -> dict[str, str]:
    """Map each lab column to the name of its panel."""
    return {lab: name for name, panel in config["panels"].items() for lab in panel["labs"]}
