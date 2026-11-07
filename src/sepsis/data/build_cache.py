"""Build the parquet cache of the raw patient files, one file per hospital.

Run with ``uv run python -m sepsis.data.build_cache`` (both hospitals, a few minutes each).
Paths come from ``configs/data.yaml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sepsis.config import DATA_CONFIG, load_yaml
from sepsis.data.load import HOSPITALS, cache_path, read_raw_hospital


def save_cache(table: pd.DataFrame, name: str, cache_dir: Path) -> Path:
    """Write one hospital's table to the cache and return its path."""
    path = cache_path(cache_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DATA_CONFIG)
    parser.add_argument("--hospital", choices=HOSPITALS, default=None)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    raw_dir, cache_dir = Path(config["raw_dir"]), Path(config["cache_dir"])
    for name in [args.hospital] if args.hospital else HOSPITALS:
        table = read_raw_hospital(name, raw_dir)  # ALL patients: the cache holds everything
        path = save_cache(table, name, cache_dir)
        print(
            f"hospital {name}: {table['patient_id'].nunique()} patients, "
            f"{len(table)} patient-hours -> {path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
