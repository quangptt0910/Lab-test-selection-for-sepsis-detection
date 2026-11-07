from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sepsis.data.build_cache import save_cache
from sepsis.data.inspect import load_patients, summarize
from sepsis.data.load import (
    HOSPITAL_DIRS,
    iter_patients,
    read_cache,
    read_raw_all,
    read_raw_hospital,
)

HEADER = "HR|ICULOS|SepsisLabel"


def _write(raw_dir: Path, hospital: str, name: str, rows: list[str]) -> None:
    directory = raw_dir / HOSPITAL_DIRS[hospital]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("\n".join([HEADER, *rows]) + "\n")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    # Same file name in both hospitals, to prove ids never collide.
    _write(tmp_path, "A", "p000001.psv", ["80|1|0", "NaN|2|0", "95|3|1"])
    _write(tmp_path, "A", "p000002.psv", ["70|1|0", "72|2|0"])
    # ICULOS out of order: the loader must keep the file's row order, not sort.
    _write(tmp_path, "B", "p000001.psv", ["60|5|0", "61|4|0", "62|6|0"])
    return tmp_path


def test_iter_patients_yields_file_stem_and_table(raw_dir: Path) -> None:
    items = list(iter_patients(raw_dir / HOSPITAL_DIRS["A"]))
    assert [pid for pid, _ in items] == ["p000001", "p000002"]
    assert len(items[0][1]) == 3


def test_load_patients_and_summarize_still_work(raw_dir: Path) -> None:
    s = summarize(load_patients(raw_dir / HOSPITAL_DIRS["A"]))
    assert s["n_patients"] == 2
    assert s["n_rows"] == 5


def test_read_raw_hospital_adds_ids_hospital_and_hour(raw_dir: Path) -> None:
    df = read_raw_hospital("A", raw_dir)
    assert list(df.columns[:3]) == ["patient_id", "hospital", "hour"]
    assert df["patient_id"].tolist() == ["A_p000001"] * 3 + ["A_p000002"] * 2
    assert (df["hospital"] == "A").all()
    assert df["hour"].tolist() == [0, 1, 2, 0, 1]
    assert np.isnan(df.loc[1, "HR"])


def test_read_raw_hospital_keeps_file_row_order(raw_dir: Path) -> None:
    df = read_raw_hospital("B", raw_dir)
    assert df["ICULOS"].tolist() == [5, 4, 6]
    assert df["HR"].tolist() == [60, 61, 62]
    assert df["hour"].tolist() == [0, 1, 2]


def test_read_raw_all_ids_never_collide(raw_dir: Path) -> None:
    df = read_raw_all(raw_dir)
    assert set(df["patient_id"]) == {"A_p000001", "A_p000002", "B_p000001"}
    assert df.groupby("patient_id")["hospital"].nunique().max() == 1


def test_read_raw_hospital_rejects_unknown_or_missing(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_raw_hospital("C", tmp_path)
    with pytest.raises(FileNotFoundError):
        read_raw_hospital("A", tmp_path)


def test_parquet_cache_round_trip(raw_dir: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    tables = {name: read_raw_hospital(name, raw_dir) for name in ("A", "B")}
    for name, table in tables.items():
        save_cache(table, name, cache_dir)
    pd.testing.assert_frame_equal(read_cache(["A"], cache_dir), tables["A"])
    both = read_cache(["A", "B"], cache_dir)
    pd.testing.assert_frame_equal(both, pd.concat(tables.values(), ignore_index=True))


def test_read_cache_missing_names_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_cache"):
        read_cache(["A"], tmp_path)
