"""Read PhysioNet/CinC 2019 patient files into patient-hour tables.

Each hospital is read and cached separately, and can be read alone or together with the other.
Rows keep each file's order: the official scorer matches predictions to labels by row position,
so tables are never re-sorted by ICULOS.

WARNING: `read_raw_hospital`, `read_raw_all` and `read_cache` return ALL patients, test patients
included. Only `build_cache`, `make_split` and the final evaluation may call them. Everything else
gets patient-hours through `sepsis.data.splits.load_patient_hours`, which checks the requested
patients against the verified split and reads only those (`read_cache_patients`).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import pandas as pd

HOSPITALS = ("A", "B")
HOSPITAL_DIRS = {
    "A": Path("training_setA") / "training_setA",
    "B": Path("training_setB") / "training_setB",
}
ID_COLUMNS = ["patient_id", "hospital", "hour"]


def read_patient_file(path: Path) -> pd.DataFrame:
    """Read one `.psv` patient file."""
    return pd.read_csv(path, sep="|", na_values="NaN")


def iter_patients(directory: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield `(file stem, table)` for each patient file, in sorted file order."""
    for path in sorted(Path(directory).glob("*.psv")):
        yield path.stem, read_patient_file(path)


def build_hospital_table(
    hospital: str, patients: Iterable[tuple[str, pd.DataFrame]]
) -> pd.DataFrame:
    """Stack patient tables, adding `patient_id`, `hospital` and `hour` (0-based row number)."""
    frames = []
    for stem, df in patients:
        ids = pd.DataFrame(
            {
                "patient_id": f"{hospital}_{stem}",
                "hospital": hospital,
                "hour": range(len(df)),
            }
        )
        frames.append(pd.concat([ids, df.reset_index(drop=True)], axis=1))
    if not frames:
        raise ValueError(f"no patient files for hospital {hospital}")
    return pd.concat(frames, ignore_index=True)


def _check_hospital(name: str) -> None:
    if name not in HOSPITAL_DIRS:
        raise ValueError(f"unknown hospital {name!r}; expected one of {HOSPITALS}")


def read_raw_hospital(name: str, raw_dir: Path) -> pd.DataFrame:
    """Read every raw patient file of one hospital (ALL patients, test included)."""
    _check_hospital(name)
    directory = Path(raw_dir) / HOSPITAL_DIRS[name]
    if not directory.is_dir():
        raise FileNotFoundError(f"missing directory: {directory}")
    return build_hospital_table(name, iter_patients(directory))


def read_raw_all(raw_dir: Path) -> pd.DataFrame:
    """Read both hospitals from the raw files (ALL patients, test included)."""
    return pd.concat([read_raw_hospital(name, raw_dir) for name in HOSPITALS], ignore_index=True)


def cache_path(cache_dir: Path, name: str) -> Path:
    return Path(cache_dir) / f"hospital_{name}.parquet"


def _existing_cache(cache_dir: Path, name: str) -> Path:
    _check_hospital(name)
    path = cache_path(cache_dir, name)
    if not path.exists():
        raise FileNotFoundError(
            f"missing cache {path}; build it with: uv run python -m sepsis.data.build_cache"
        )
    return path


def read_cache(hospitals: Sequence[str], cache_dir: Path) -> pd.DataFrame:
    """Read the parquet cache of the given hospitals (ALL patients, test included)."""
    frames = [pd.read_parquet(_existing_cache(cache_dir, name)) for name in hospitals]
    return pd.concat(frames, ignore_index=True)


def read_cache_patients(
    hospitals: Sequence[str], cache_dir: Path, patient_ids: Iterable[str]
) -> pd.DataFrame:
    """Read only the given patients from the cache; other patients are never loaded.

    It does not check who may be read: use `sepsis.data.splits.load_patient_hours`, which does.
    """
    ids = sorted(set(patient_ids))
    frames = [
        pd.read_parquet(_existing_cache(cache_dir, name), filters=[("patient_id", "in", ids)])
        for name in hospitals
    ]
    return pd.concat(frames, ignore_index=True)
