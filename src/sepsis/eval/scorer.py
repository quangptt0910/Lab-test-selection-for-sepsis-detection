"""Score predictions with the official PhysioNet/CinC 2019 scorer, unmodified (ADR-0007).

The scorer in `third_party/` is imported as it is, after its hash is checked against
`configs/eval.yaml`. This module only writes the folders it reads and names the values it returns.

Tables:
- labels: patient-hour rows with `patient_id, hour, SepsisLabel` (and `hospital` for
  `score_by_hospital`), from `sepsis.data.splits.load_patient_hours`.
- predictions: `patient_id, hour, score, alarm`. `score` is the raw model output on any scale;
  `alarm` is 0/1 and comes from `apply_threshold`. Scoring takes alarms, not a threshold, because
  the mix setup has one threshold per outer fold.
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.special import expit

from sepsis.config import EVAL_CONFIG, load_yaml

KEYS = ["patient_id", "hour"]
LABEL_HEADER = "SepsisLabel"
PREDICTION_HEADER = "PredictedProbability|PredictedLabel"


class ScorerModifiedError(RuntimeError):
    """Raised when the official scorer file differs from the hash in configs/eval.yaml."""


@dataclass(frozen=True)
class Scores:
    """The five values of the official scorer, by name. `utility` is the normalized utility."""

    auroc: float
    auprc: float
    accuracy: float
    f_measure: float
    utility: float


def scorer_sha256(path: Path) -> str:
    """SHA-256 of the scorer file with line endings normalised to LF (git may check out CRLF)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_official_scorer(path: Path, expected_sha256: str) -> ModuleType:
    """Import the official scorer file as a module, refusing a file that was changed."""
    actual = scorer_sha256(path)
    if actual != expected_sha256:
        raise ScorerModifiedError(
            f"{path} does not match scorer.sha256 in configs/eval.yaml; "
            "utility must come from the unmodified official scorer"
        )
    spec = importlib.util.spec_from_file_location("official_evaluate_sepsis_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def official_scorer() -> ModuleType:
    """The project's verified scorer (path and hash from configs/eval.yaml)."""
    config = load_yaml(EVAL_CONFIG)["scorer"]
    return load_official_scorer(Path(config["path"]), str(config["sha256"]))


def to_probability(score: np.ndarray | pd.Series) -> np.ndarray:
    """Map raw scores into (0, 1) with the logistic function, for `PredictedProbability`.

    The scorer's AUROC is wrong for values above 1, and U1 - U0 predictions reach about 2. The
    logistic function is fixed and increasing, so the ranking is kept and nothing is fitted on the
    scored patients. Alarms never use this value: they compare the raw score with the threshold.
    """
    return expit(np.asarray(score, dtype=float))


def apply_threshold(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add `alarm`: 1 where the raw score exceeds the threshold (strictly), else 0."""
    return predictions.assign(alarm=(predictions["score"] > threshold).astype(int))


def align(hours: pd.DataFrame, predictions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Join `columns` of the predictions onto the label rows, ordered by patient and hour.

    Raises `ValueError` unless there is exactly one prediction row for every label row, every
    patient has its complete record (hours 0..n-1: the scorer finds onset by row position), labels
    are 0 or 1 and scores are finite.
    """
    missing = [c for c in KEYS + columns if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions lack columns {missing}")
    for name, table in (("labels", hours), ("predictions", predictions)):
        if not pd.api.types.is_integer_dtype(table["hour"]):
            raise ValueError(f"hour must be an integer row number in the {name}")
    if not hours["SepsisLabel"].isin([0, 1]).all():
        raise ValueError("SepsisLabel must be 0 or 1")
    if predictions.duplicated(KEYS).any() or hours.duplicated(KEYS).any():
        raise ValueError("duplicate (patient_id, hour) rows")
    merged = hours.merge(predictions[KEYS + columns], on=KEYS, how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError(
            "predictions and labels must cover the same (patient_id, hour) rows; "
            f"{(merged['_merge'] == 'left_only').sum()} label rows have no prediction and "
            f"{(merged['_merge'] == 'right_only').sum()} predictions have no label row"
        )
    if not np.isfinite(merged["score"].to_numpy(dtype=float)).all():
        raise ValueError("scores must be finite (no NaN or inf)")
    merged = merged.drop(columns="_merge").sort_values(KEYS, kind="stable").reset_index(drop=True)
    if not (merged.groupby("patient_id").cumcount() == merged["hour"]).all():
        raise ValueError(
            "each patient needs its complete record, hours 0..n-1: "
            "the scorer finds onset by row position"
        )
    return merged


def _new_dir(directory: Path) -> Path:
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"{directory} is not empty; the scorer reads every .psv file in it")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_patient_files(directory: Path, table: pd.DataFrame, header: str, columns: list[str]):
    directory = _new_dir(directory)
    ordered = table.sort_values(KEYS, kind="stable")
    for patient_id, rows in ordered.groupby("patient_id", sort=True):
        cells = zip(*(rows[c].tolist() for c in columns), strict=True)
        lines = ["|".join(map(repr, values)) for values in cells]
        text = "\n".join([header, *lines]) + "\n"
        (directory / f"{patient_id}.psv").write_text(text, encoding="utf-8")


def write_label_dir(hours: pd.DataFrame, directory: Path) -> None:
    """Write one `<patient_id>.psv` label file per patient, rows in `hour` order."""
    labels = hours[KEYS].assign(label=hours["SepsisLabel"].astype(int).tolist())
    _write_patient_files(directory, labels, LABEL_HEADER, ["label"])


def write_prediction_dir(predictions: pd.DataFrame, directory: Path) -> None:
    """Write one `<patient_id>.psv` prediction file per patient, rows in `hour` order.

    File names equal the label files' names: the scorer pairs the two folders by sorted name.
    """
    table = predictions[KEYS].assign(
        probability=to_probability(predictions["score"]).tolist(),
        alarm=predictions["alarm"].astype(int).tolist(),
    )
    _write_patient_files(directory, table, PREDICTION_HEADER, ["probability", "alarm"])


def score_dirs(label_dir: Path, prediction_dir: Path) -> Scores:
    """Run the official scorer on a label folder and a prediction folder."""
    values = official_scorer().evaluate_sepsis_score(str(label_dir), str(prediction_dir))
    return Scores(*(float(v) for v in values))


def score(hours: pd.DataFrame, predictions: pd.DataFrame, work_dir: Path | None = None) -> Scores:
    """Score the predictions of a patient set with the official scorer.

    The folders are written to `work_dir/labels` and `work_dir/predictions`, or to a temporary
    directory. Normalized utility is undefined without a septic patient, so that is an error.
    """
    merged = align(hours, predictions, ["score", "alarm"])
    if not merged["alarm"].isin([0, 1]).all():
        raise ValueError("alarm must be 0 or 1")
    if not (merged["SepsisLabel"] == 1).any():
        raise ValueError("no septic patient in the set: normalized utility is undefined")
    if work_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return _score_in(merged, Path(tmp))
    return _score_in(merged, Path(work_dir))


def _score_in(merged: pd.DataFrame, work_dir: Path) -> Scores:
    write_label_dir(merged, work_dir / "labels")
    write_prediction_dir(merged, work_dir / "predictions")
    return score_dirs(work_dir / "labels", work_dir / "predictions")


def score_by_hospital(
    hours: pd.DataFrame, predictions: pd.DataFrame, work_dir: Path | None = None
) -> dict[str, Scores]:
    """Score each hospital's patients on their own (mix -> A and mix -> B).

    Each hospital gets one scorer call over all its patients, so its utility is normalized over
    that hospital only. With `work_dir`, the folders go to `work_dir/<hospital>/`.
    """
    if hours["hospital"].isna().any():
        raise ValueError("every label row needs a hospital")
    if (hours.groupby("patient_id")["hospital"].nunique() > 1).any():
        raise ValueError("a patient belongs to one hospital only")
    stray = set(predictions["patient_id"]) - set(hours["patient_id"])
    if stray:
        raise ValueError(f"{len(stray)} predicted patients have no label rows")
    result = {}
    for hospital, rows in hours.groupby("hospital", sort=True):
        own = predictions[predictions["patient_id"].isin(set(rows["patient_id"]))]
        sub_dir = None if work_dir is None else Path(work_dir) / str(hospital)
        result[str(hospital)] = score(rows, own, sub_dir)
    return result
