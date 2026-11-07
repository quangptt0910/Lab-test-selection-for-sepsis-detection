"""The fixed patient-level split shared by every setup (ADR-0004).

One row per patient: `patient_id, hospital, septic, split (dev|test), fold`, where fold is
0..n-1 on dev and empty on test. Dev/test and the folds are both stratified by hospital x septic,
so each hospital gets its own 80/20 split stratified by septic. Test rows are locked: only the
final evaluation may read them (`final_eval=True`).

Use `load_project_split()` and `load_patient_hours()`: together they give dev patients only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from sepsis.config import DATA_CONFIG, PROTOCOL_CONFIG, load_yaml
from sepsis.data.load import HOSPITALS, read_cache

SPLIT_COLUMNS = ["patient_id", "hospital", "septic", "split", "fold"]
# Helpers drop `septic`: it is computed from the whole record and must not reach features.
FOLD_COLUMNS = ["patient_id", "hospital", "fold"]
PARTS = ("dev", "test", "all")


class TestSplitLockedError(RuntimeError):
    """Raised when test patients are requested outside the final evaluation."""

    __test__ = False  # not a pytest test class


class SplitFingerprintError(RuntimeError):
    """Raised when a split file is unverified or does not match configs/protocol.yaml."""


def _strata(patients: pd.DataFrame, stratify_by: Sequence[str]) -> pd.Series:
    return patients[list(stratify_by)].astype(str).agg("|".join, axis=1)


def make_split(
    patients: pd.DataFrame,
    test_fraction: float,
    n_folds: int,
    seed: int,
    stratify_by: Sequence[str],
) -> pd.DataFrame:
    """Split patients into dev and test, and assign stratified CV folds to dev patients.

    `patients` needs `patient_id`, `hospital` and `septic`. The result does not depend on the
    input row order.
    """
    patients = (
        patients[["patient_id", "hospital", "septic"]]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    if not patients["patient_id"].is_unique:
        raise ValueError("patient_id must be unique")

    dev_idx, test_idx = train_test_split(
        patients.index,
        test_size=test_fraction,
        stratify=_strata(patients, stratify_by),
        random_state=seed,
    )
    split = patients.assign(split="dev", fold=pd.array([pd.NA] * len(patients), dtype="Int64"))
    split.loc[test_idx, "split"] = "test"

    dev = split.loc[sorted(dev_idx)]
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val) in enumerate(folds.split(dev, _strata(dev, stratify_by))):
        split.loc[dev.index[val], "fold"] = k
    return split[SPLIT_COLUMNS]


def fingerprint(split: pd.DataFrame) -> str:
    """SHA-256 of the sorted `(patient_id, split, fold)` rows."""
    rows = split.sort_values("patient_id")
    lines = (
        f"{pid},{part},{'' if pd.isna(fold) else int(fold)}"
        for pid, part, fold in zip(rows["patient_id"], rows["split"], rows["fold"], strict=True)
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def write_split(split: pd.DataFrame, path: Path) -> None:
    """Write the split file. Refuses to overwrite: the split is created once."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; the split is created once (ADR-0004)")
    path.parent.mkdir(parents=True, exist_ok=True)
    split[SPLIT_COLUMNS].to_csv(path, index=False)


def load_split(
    path: Path,
    expected_fingerprint: str | None,
    *,
    part: str = "dev",
    final_eval: bool = False,
    allow_unverified: bool = False,
) -> pd.DataFrame:
    """Read the split file and return its dev rows (default), test rows or all rows.

    Test rows (`part="test"` or `"all"`) are locked unless `final_eval=True`, which only the
    final evaluation may pass. The file must match `expected_fingerprint`; a missing fingerprint
    is an error unless `allow_unverified=True` (only for `make_split` and tests).
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    if part != "dev" and not final_eval:
        raise TestSplitLockedError(
            "test patients are locked until the final evaluation (ADR-0004); "
            "only the final evaluation may pass final_eval=True"
        )
    if expected_fingerprint is None and not allow_unverified:
        raise SplitFingerprintError(
            "no split fingerprint: set split.fingerprint in configs/protocol.yaml to the value "
            "printed by `python -m sepsis.data.make_split`"
        )
    split = pd.read_csv(path, dtype={"patient_id": str, "hospital": str, "split": str})
    split["fold"] = split["fold"].astype("Int64")
    if expected_fingerprint is not None and fingerprint(split) != expected_fingerprint:
        raise SplitFingerprintError(
            f"{path} does not match split.fingerprint in configs/protocol.yaml; "
            "it was created differently from the project's split"
        )
    if part != "all":
        split = split[split["split"] == part]
    return split.reset_index(drop=True)


def load_project_split(*, part: str = "dev", final_eval: bool = False) -> pd.DataFrame:
    """The project's split, verified against `split.fingerprint` (paths from configs/)."""
    path = Path(load_yaml(DATA_CONFIG)["split_path"])
    expected = load_yaml(PROTOCOL_CONFIG)["split"].get("fingerprint")
    return load_split(path, expected, part=part, final_eval=final_eval)


def rows_for(df: pd.DataFrame, patient_ids: Iterable[str]) -> pd.DataFrame:
    """Return the patient-hour rows of the given patients."""
    return df[df["patient_id"].isin(set(patient_ids))].reset_index(drop=True)


def load_patient_hours(
    hospitals: Sequence[str],
    split: pd.DataFrame,
    cache_dir: Path,
    *,
    final_eval: bool = False,
) -> pd.DataFrame:
    """Patient-hour rows of the given hospitals, restricted to the patients in `split`.

    Pass the split from `load_project_split()`, which holds dev patients only. A split that
    contains test patients is refused unless `final_eval=True`.
    """
    if (split["split"] != "dev").any() and not final_eval:
        raise TestSplitLockedError(
            "the split passed in contains test patients; only the final evaluation may load them"
        )
    return rows_for(read_cache(hospitals, cache_dir), split["patient_id"])


def _dev(split: pd.DataFrame) -> pd.DataFrame:
    return split.loc[split["split"] == "dev", FOLD_COLUMNS].reset_index(drop=True)


def mix_folds(split: pd.DataFrame) -> pd.DataFrame:
    """Dev patients of both hospitals with their folds (mix setup). Never contains test patients."""
    return _dev(split)


def cross_hospital(split: pd.DataFrame, train: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return `(training side, eval side)` for A -> B (`train="A"`) or B -> A (`train="B"`).

    The training side is the training hospital's dev patients with their folds; the eval side is
    the other hospital's dev patients. Neither contains test patients.
    """
    if train not in HOSPITALS:
        raise ValueError(f"train must be one of {HOSPITALS}")
    dev = _dev(split)
    train_side = dev[dev["hospital"] == train].reset_index(drop=True)
    eval_side = dev[dev["hospital"] != train].drop(columns="fold").reset_index(drop=True)
    return train_side, eval_side


def summarize_split(split: pd.DataFrame) -> pd.DataFrame:
    """Patients and septic share per hospital, part and fold."""
    keyed = split.assign(fold=split["fold"].astype("string").fillna("-"))
    return (
        keyed.groupby(["hospital", "split", "fold"])["septic"]
        .agg(patients="size", septic_share="mean")
        .reset_index()
    )
