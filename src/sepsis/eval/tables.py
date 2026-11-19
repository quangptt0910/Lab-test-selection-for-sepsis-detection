"""Build the model tables of an experiment once, and cache them (ADR-0008).

- `train_X`: feature rows from the training view (as recorded, with panel-draw masking). Models are
  fitted on these.
- `eval_X`: feature rows from the evaluation view (as recorded, no masking). Out-of-fold and
  eval-side predictions are made on these.
- `targets`: `dt, u1, u0, utility_diff, weight` (whole record, label side) and `SepsisLabel`.

All three have the same `patient_id, hour` rows in the same order, so rows are aligned by position,
never merged. Features are pure per patient (masking is seeded per patient, ADR-0006), so building
them once for all dev patients and selecting fold rows later gives the same rows as building them
per fold.

The cache is keyed by everything that shapes the tables: the configs, the chosen blocks, masking,
seed, split fingerprint, the patient set, and the source code that builds them. The evaluation
table and targets do not depend on masking or the seed, so they have their own key.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import sepsis
from sepsis.data.view import build_evaluation_view, build_training_view
from sepsis.features.builder import FORBIDDEN_COLUMNS, build_features, feature_columns, with_blocks
from sepsis.labels.utility import UtilityConstants, utility_targets

KEYS = ["patient_id", "hour"]
TARGET_COLUMNS = ["dt", "u1", "u0", "utility_diff", "weight", "SepsisLabel"]
PACKAGE = Path(sepsis.__file__).resolve().parent
# Source that shapes the tables; any change gives a new cache key.
SOURCE_PATHS = ("data", "features", "labels", "eval/tables.py", "config.py")


@dataclass(frozen=True)
class Tables:
    """The model tables of one experiment, aligned row by row (`patient_id, hour`)."""

    train_X: pd.DataFrame
    eval_X: pd.DataFrame
    targets: pd.DataFrame
    feature_columns: list[str]


@dataclass(frozen=True)
class TableSettings:
    """Everything besides the patient-hours that shapes the tables."""

    blocks: tuple[str, ...]
    lab_masking_p: float
    seed: int
    cleaning: Mapping[str, Any]
    panels: Mapping[str, Any]
    features: Mapping[str, Any]
    labels: UtilityConstants


def build_tables(hours: pd.DataFrame, settings: TableSettings) -> Tables:
    """Build both feature tables and the targets for every patient in `hours` (whole records)."""
    hours = hours.sort_values(KEYS, kind="stable").reset_index(drop=True)
    features = with_blocks(settings.features, list(settings.blocks))
    eval_view = build_evaluation_view(hours, settings.cleaning, settings.panels)
    train_view = build_training_view(
        hours,
        settings.cleaning,
        settings.panels,
        lab_masking_p=settings.lab_masking_p,
        seed=settings.seed,
    )
    columns = feature_columns(eval_view, features, panels_config=settings.panels)
    eval_X = build_features(eval_view, features, panels_config=settings.panels)
    train_X = build_features(train_view, features, panels_config=settings.panels)
    targets = utility_targets(hours[[*KEYS, "SepsisLabel"]], settings.labels)
    _check_keys(hours, targets, "targets")
    targets = targets.assign(SepsisLabel=hours["SepsisLabel"].to_numpy())
    tables = Tables(
        train_X=train_X.reset_index(drop=True),
        eval_X=eval_X.reset_index(drop=True),
        targets=targets[[*KEYS, *TARGET_COLUMNS]],
        feature_columns=columns,
    )
    check_tables(tables)
    return tables


def _check_keys(expected: pd.DataFrame, table: pd.DataFrame, what: str) -> None:
    if not expected[KEYS].reset_index(drop=True).equals(table[KEYS].reset_index(drop=True)):
        raise ValueError(f"{what} rows differ from the patient-hour rows (patient_id, hour)")


def check_tables(tables: Tables) -> None:
    """Raise unless the tables are aligned row by row and no label column is a feature."""
    _check_keys(tables.targets, tables.train_X, "train_X")
    _check_keys(tables.targets, tables.eval_X, "eval_X")
    if list(tables.train_X.columns) != list(tables.eval_X.columns):
        raise ValueError("train_X and eval_X must have the same columns")
    forbidden = sorted(set(tables.feature_columns) & set(FORBIDDEN_COLUMNS))
    if forbidden or set(KEYS) & set(tables.feature_columns):
        raise ValueError(f"label or key columns among the features: {forbidden or KEYS}")
    missing = sorted(set(tables.feature_columns) - set(tables.eval_X.columns))
    if missing:
        raise ValueError(f"feature columns missing from the tables: {missing}")


# --- Cache --------------------------------------------------------------------------------------


def source_digest(package: Path = PACKAGE) -> str:
    """SHA-256 of the source that builds the tables (relative path + bytes, LF line endings)."""
    digest = hashlib.sha256()
    for entry in SOURCE_PATHS:
        path = package / entry
        files = sorted(path.rglob("*.py")) if path.is_dir() else [path]
        for file in files:
            digest.update(file.relative_to(package).as_posix().encode())
            digest.update(file.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def cache_keys(
    settings: TableSettings,
    patient_ids: Sequence[str],
    split_fingerprint: str,
    source: str | None = None,
) -> tuple[str, str]:
    """`(training-table key, evaluation-table key)`. Only the first depends on masking and seed."""
    shared = {
        "blocks": sorted(settings.blocks),
        "cleaning": settings.cleaning,
        "panels": settings.panels,
        "features": settings.features,
        "labels": asdict(settings.labels),
        "split": split_fingerprint,
        "patients": hashlib.sha256("\n".join(sorted(patient_ids)).encode()).hexdigest(),
        "source": source_digest() if source is None else source,
    }
    train = {**shared, "lab_masking_p": settings.lab_masking_p, "seed": settings.seed}
    return "train-" + _hash(train), "eval-" + _hash(shared)


def _write(directory: Path, frames: Mapping[str, pd.DataFrame], extra: Mapping[str, Any]) -> None:
    """Write to `<directory>.tmp`, then rename: a half-written cache is never read."""
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_parquet(tmp / f"{name}.parquet", index=False)
    (tmp / "meta.json").write_text(json.dumps(extra, indent=1), encoding="utf-8")
    if directory.exists():
        shutil.rmtree(directory)
    tmp.rename(directory)


def read_cached(cache_dir: Path, info: Mapping[str, Any]) -> Tables:
    """The cached tables under `info["train_key"]` and `info["eval_key"]`, checked."""
    train_dir, eval_dir = Path(cache_dir) / info["train_key"], Path(cache_dir) / info["eval_key"]
    meta = json.loads((eval_dir / "meta.json").read_text(encoding="utf-8"))
    tables = Tables(
        train_X=pd.read_parquet(train_dir / "train_X.parquet"),
        eval_X=pd.read_parquet(eval_dir / "eval_X.parquet"),
        targets=pd.read_parquet(eval_dir / "targets.parquet"),
        feature_columns=list(meta["feature_columns"]),
    )
    check_tables(tables)
    return tables


def load_or_build(
    hours: pd.DataFrame,
    settings: TableSettings,
    *,
    split_fingerprint: str,
    cache_dir: Path | None,
) -> tuple[Tables, dict[str, Any]]:
    """The tables from `cache_dir` when both keys hit, else built (and cached if `cache_dir`).

    Returns the tables and `{"train_key", "eval_key", "hit"}` for the run metadata.
    """
    ids = sorted(hours["patient_id"].unique())
    train_key, eval_key = cache_keys(settings, ids, split_fingerprint)
    info: dict[str, Any] = {"train_key": train_key, "eval_key": eval_key, "hit": False}
    if cache_dir is not None:
        train_dir, eval_dir = Path(cache_dir) / train_key, Path(cache_dir) / eval_key
        if (train_dir / "meta.json").exists() and (eval_dir / "meta.json").exists():
            tables = read_cached(cache_dir, info)
            expected = hours[KEYS].sort_values(KEYS, kind="stable").reset_index(drop=True)
            _check_keys(expected, tables.targets, "cached tables")
            return tables, {**info, "hit": True}
    tables = build_tables(hours, settings)
    if cache_dir is not None:
        _write(Path(cache_dir) / train_key, {"train_X": tables.train_X}, {"key": train_key})
        _write(
            Path(cache_dir) / eval_key,
            {"eval_X": tables.eval_X, "targets": tables.targets},
            {"key": eval_key, "feature_columns": tables.feature_columns},
        )
    return tables, info
