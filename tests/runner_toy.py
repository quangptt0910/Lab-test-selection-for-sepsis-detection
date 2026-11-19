"""A tiny project for the experiment-runner tests: split file, cache and configs in tmp_path.

Two dev patients per hospital in each of the 5 stored folds, three septic per hospital (folds 0,
2, 4), plus two locked test patients per hospital. Records come from `feature_toy._patient`, so
every view column exists. No real patient data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from feature_toy import _patient

from sepsis.data.build_cache import save_cache
from sepsis.data.splits import fingerprint, write_split

GRID = {"quantile_min": 0.0, "quantile_max": 1.0, "n_points": 5}
LOG_HEADER = Path("experiments/LOG.md").read_text(encoding="utf-8")


def _record(pid: str, n: int, seed: int, septic: bool) -> pd.DataFrame:
    record = _patient(pid, n, seed, first_lab_row=0)
    label = [0] * n
    if septic:
        label = [0] * (n - 8) + [1] * 8
    return record.assign(SepsisLabel=label)


def patients() -> pd.DataFrame:
    """`patient_id, hospital, septic, split, fold` of the toy project."""
    rows = []
    for hospital in "AB":
        for fold in range(5):
            for i in range(2):
                septic = int(i == 0 and fold in (0, 2, 4))
                rows.append((f"{hospital}_d{fold}{i}", hospital, septic, "dev", fold))
        rows += [(f"{hospital}_t{i}", hospital, int(i == 0), "test", None) for i in range(2)]
    table = pd.DataFrame(rows, columns=["patient_id", "hospital", "septic", "split", "fold"])
    table["fold"] = table["fold"].astype("Int64")
    return table


def records(table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Patient-hours of the toy patients (whole records, test patients included)."""
    table = patients() if table is None else table
    parts = [
        _record(row.patient_id, 16 + 3 * (k % 4), seed=k, septic=bool(row.septic))
        for k, row in enumerate(table.itertuples())
    ]
    return pd.concat(parts, ignore_index=True)


def project(tmp_path: Path, data_dir: Path | None = None) -> dict:
    """Write the cache, split file, data/protocol configs and an empty LOG under `tmp_path`.

    `data_dir` (default `tmp_path`) holds the cache and split file; tests share one to build the
    toy feature tables once. The LOG and run folders are always under `tmp_path`.
    """
    table = patients()
    hours = records(table)
    data_dir = tmp_path if data_dir is None else data_dir
    cache_dir = data_dir / "cache"
    split_path = data_dir / "splits" / "split.csv"
    if not split_path.exists():
        for hospital in "AB":
            save_cache(hours[hours["hospital"] == hospital], hospital, cache_dir)
        write_split(table, split_path)
    data_config = tmp_path / "data.yaml"
    data_config.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(tmp_path / "raw"),
                "cache_dir": str(cache_dir),
                "split_path": str(split_path),
            }
        )
    )
    protocol_config = tmp_path / "protocol.yaml"
    protocol_config.write_text(yaml.safe_dump({"split": {"fingerprint": fingerprint(table)}}))
    log = tmp_path / "experiments" / "LOG.md"
    log.parent.mkdir(parents=True)
    log.write_text(LOG_HEADER, encoding="utf-8")
    return {
        "root": tmp_path,
        "data_config": data_config,
        "protocol_config": protocol_config,
        "split": table,
        "hours": hours,
    }


def experiment(tmp_path: Path, **changes) -> Path:
    """Write a dummy-model experiment config to tmp_path and return its path."""
    cfg = {
        "id": "toy-001",
        "hypothesis": "the runner works end to end",
        "seed": 0,
        "split": {"setups": ["A_to_B", "B_to_A", "mix"]},
        "features": {
            "hand_crafted": ["shock_index", "partial_sofa", "sirs", "hours_since_panel"],
            "signatures": None,
            "lab_masking_p": 0.3,
        },
        "target": "utility_diff",
        "model": {"type": "dummy", "params": {"column": "partial_sofa"}},
        "note": "",
    }
    for key, value in changes.items():
        cfg[key] = value
    path = tmp_path / f"{cfg['id']}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path
