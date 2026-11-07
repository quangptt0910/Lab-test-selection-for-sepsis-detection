"""Read the YAML files in `configs/`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DATA_CONFIG = Path("configs/data.yaml")
PROTOCOL_CONFIG = Path("configs/protocol.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    """Return the YAML file at `path` as a dict (empty if the file is empty)."""
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
