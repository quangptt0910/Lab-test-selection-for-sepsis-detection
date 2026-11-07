from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sepsis.data import make_split as make_split_cli
from sepsis.data.build_cache import save_cache
from sepsis.data.splits import (
    SplitFingerprintError,
    TestSplitLockedError,
    cross_hospital,
    fingerprint,
    load_patient_hours,
    load_split,
    make_split,
    mix_folds,
    rows_for,
    write_split,
)

SETTINGS = {"test_fraction": 0.2, "n_folds": 5, "seed": 0, "stratify_by": ["hospital", "septic"]}


def _patients(n_a: int = 2000, n_b: int = 2000, share_a: float = 0.09, share_b: float = 0.06):
    rows = []
    for hospital, n, share in (("A", n_a, share_a), ("B", n_b, share_b)):
        n_septic = round(n * share)
        for i in range(n):
            rows.append((f"{hospital}_p{i:06d}", hospital, int(i < n_septic)))
    return pd.DataFrame(rows, columns=["patient_id", "hospital", "septic"])


@pytest.fixture(scope="module")
def split() -> pd.DataFrame:
    return make_split(_patients(), **SETTINGS)


def test_split_columns_and_one_row_per_patient(split: pd.DataFrame) -> None:
    assert list(split.columns) == ["patient_id", "hospital", "septic", "split", "fold"]
    assert split["patient_id"].is_unique
    assert len(split) == 4000


def test_dev_test_fraction_per_hospital(split: pd.DataFrame) -> None:
    for _, group in split.groupby("hospital"):
        assert (group["split"] == "test").mean() == pytest.approx(0.2, abs=0.005)


def test_folds_only_on_dev_and_each_dev_patient_in_one_fold(split: pd.DataFrame) -> None:
    dev, test = split[split["split"] == "dev"], split[split["split"] == "test"]
    assert test["fold"].isna().all()
    assert dev["fold"].notna().all()
    assert sorted(dev["fold"].unique()) == [0, 1, 2, 3, 4]


def test_septic_share_within_one_point(split: pd.DataFrame) -> None:
    for _, group in split.groupby("hospital"):
        overall = group["septic"].mean()
        test = group[group["split"] == "test"]
        assert abs(test["septic"].mean() - overall) <= 0.01
        for _, fold in group[group["split"] == "dev"].groupby("fold"):
            assert abs(fold["septic"].mean() - overall) <= 0.01


def test_same_seed_same_split_and_fingerprint() -> None:
    a = make_split(_patients(), **SETTINGS)
    b = make_split(
        _patients().sample(frac=1, random_state=3), **SETTINGS
    )  # input order must not matter
    pd.testing.assert_frame_equal(a, b)
    assert fingerprint(a) == fingerprint(b)
    c = make_split(_patients(), **{**SETTINGS, "seed": 1})
    assert fingerprint(c) != fingerprint(a)


def test_write_split_refuses_overwrite(split: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "splits" / "split.csv"
    write_split(split, path)
    with pytest.raises(FileExistsError):
        write_split(split, path)


def test_load_split_round_trip_returns_dev_only(split: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    write_split(split, path)
    dev = load_split(path, fingerprint(split))
    assert (dev["split"] == "dev").all()
    expected = split[split["split"] == "dev"].reset_index(drop=True)
    pd.testing.assert_frame_equal(dev, expected)


def test_test_rows_are_locked_without_final_eval(split: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    write_split(split, path)
    with pytest.raises(TestSplitLockedError):
        load_split(path, fingerprint(split), part="test")
    with pytest.raises(TestSplitLockedError):
        load_split(path, fingerprint(split), part="all")
    test = load_split(path, fingerprint(split), part="test", final_eval=True)
    assert (test["split"] == "test").all() and len(test) == 800


def test_fingerprint_mismatch_is_refused(split: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    write_split(split, path)
    with pytest.raises(SplitFingerprintError):
        load_split(path, "0" * 64)


def test_mix_folds_has_no_test_patients(split: pd.DataFrame) -> None:
    mix = mix_folds(split)
    test_ids = set(split.loc[split["split"] == "test", "patient_id"])
    assert not test_ids & set(mix["patient_id"])
    assert set(mix["hospital"]) == {"A", "B"}
    assert mix["fold"].notna().all()


@pytest.mark.parametrize(("train", "other"), [("A", "B"), ("B", "A")])
def test_cross_hospital_sides(split: pd.DataFrame, train: str, other: str) -> None:
    train_side, eval_side = cross_hospital(split, train=train)
    test_ids = set(split.loc[split["split"] == "test", "patient_id"])
    assert (train_side["hospital"] == train).all() and (eval_side["hospital"] == other).all()
    assert not test_ids & (set(train_side["patient_id"]) | set(eval_side["patient_id"]))
    assert train_side["fold"].notna().all()
    assert "fold" not in eval_side.columns


def test_cross_hospital_rejects_unknown_hospital(split: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        cross_hospital(split, train="C")


def test_rows_for_selects_patient_hours() -> None:
    df = pd.DataFrame({"patient_id": ["A_p1", "A_p1", "B_p2"], "HR": [1.0, 2.0, np.nan]})
    out = rows_for(df, ["A_p1"])
    assert out["HR"].tolist() == [1.0, 2.0]


def test_missing_fingerprint_is_refused_unless_allowed(split: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "split.csv"
    write_split(split, path)
    with pytest.raises(SplitFingerprintError):
        load_split(path, None)
    assert len(load_split(path, None, allow_unverified=True)) == 3200


def test_helpers_never_return_septic(split: pd.DataFrame) -> None:
    train_side, eval_side = cross_hospital(split, train="A")
    for frame in (mix_folds(split), train_side, eval_side):
        assert "septic" not in frame.columns


def _cache(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    """A tiny two-hospital cache: 10 patients each, 2 septic, 3 hours per patient."""
    cache_dir = tmp_path / "cache"
    for hospital in ("A", "B"):
        rows = [
            (f"{hospital}_p{i:06d}", hospital, h, 80.0, int(i < 2 and h == 2))
            for i in range(10)
            for h in range(3)
        ]
        table = pd.DataFrame(rows, columns=["patient_id", "hospital", "hour", "HR", "SepsisLabel"])
        save_cache(table, hospital, cache_dir)
    split = make_split(
        pd.DataFrame(
            {
                "patient_id": [f"{h}_p{i:06d}" for h in "AB" for i in range(10)],
                "hospital": [h for h in "AB" for _ in range(10)],
                "septic": [int(i < 2) for _ in "AB" for i in range(10)],
            }
        ),
        test_fraction=0.2,
        n_folds=2,
        seed=0,
        stratify_by=["hospital", "septic"],
    )
    return cache_dir, split


def test_load_patient_hours_returns_dev_rows_only(tmp_path: Path) -> None:
    cache_dir, split = _cache(tmp_path)
    dev = split[split["split"] == "dev"]
    hours = load_patient_hours(["A", "B"], dev, cache_dir)
    assert set(hours["patient_id"]) == set(dev["patient_id"])
    assert len(hours) == 3 * len(dev)
    a_only = load_patient_hours(["A"], dev, cache_dir)
    assert (a_only["hospital"] == "A").all()


def test_load_patient_hours_refuses_test_patients(tmp_path: Path) -> None:
    cache_dir, split = _cache(tmp_path)
    with pytest.raises(TestSplitLockedError):
        load_patient_hours(["A", "B"], split, cache_dir)
    assert len(load_patient_hours(["A", "B"], split, cache_dir, final_eval=True)) == 60


def _configs(tmp_path: Path, cache_dir: Path, fingerprint_value: str | None) -> list[str]:
    data = tmp_path / "data.yaml"
    data.write_text(
        f"raw_dir: unused\ncache_dir: {cache_dir.as_posix()}\n"
        + (f"split_path: {(tmp_path / 'splits' / 'split.csv').as_posix()}\n")
    )
    protocol = tmp_path / "protocol.yaml"
    fp = "null" if fingerprint_value is None else fingerprint_value
    protocol.write_text(
        "split:\n  seed: 0\n  test_fraction: 0.2\n  n_folds: 2\n"
        f"  stratify_by: [hospital, septic]\n  fingerprint: {fp}\n"
    )
    return ["--data-config", str(data), "--protocol", str(protocol)]


def test_make_split_cli_creates_once(tmp_path: Path) -> None:
    cache_dir, _ = _cache(tmp_path)
    args = _configs(tmp_path, cache_dir, None)
    assert make_split_cli.main(args) == 0
    written = load_split(
        tmp_path / "splits" / "split.csv", None, part="all", final_eval=True, allow_unverified=True
    )
    assert len(written) == 20 and set(written["split"]) == {"dev", "test"}
    assert make_split_cli.main(args) == 1  # the file exists: never re-split


def test_make_split_cli_rebuilds_when_fingerprint_matches(tmp_path: Path) -> None:
    cache_dir, expected = _cache(tmp_path)
    args = _configs(tmp_path, cache_dir, fingerprint(expected))
    assert make_split_cli.main(args) == 0
    path = tmp_path / "splits" / "split.csv"
    written = load_split(path, fingerprint(expected), part="all", final_eval=True)
    pd.testing.assert_frame_equal(written, expected)


def test_make_split_cli_writes_nothing_when_rebuild_differs(tmp_path: Path) -> None:
    cache_dir, _ = _cache(tmp_path)
    assert make_split_cli.main(_configs(tmp_path, cache_dir, "f" * 64)) == 1
    assert not (tmp_path / "splits" / "split.csv").exists()


def test_make_split_cli_verifies_an_existing_file(tmp_path: Path) -> None:
    cache_dir, expected = _cache(tmp_path)
    path = tmp_path / "splits" / "split.csv"
    write_split(expected, path)
    before = path.read_bytes()
    assert make_split_cli.main(_configs(tmp_path, cache_dir, fingerprint(expected))) == 0
    assert make_split_cli.main(_configs(tmp_path, cache_dir, "f" * 64)) == 1
    assert path.read_bytes() == before  # never rewritten
