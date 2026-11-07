"""Checks on the real PhysioNet files. Marked slow: they need the data and take minutes."""

from pathlib import Path

import pytest

from sepsis.config import load_yaml
from sepsis.data.load import HOSPITAL_DIRS, read_raw_hospital

EXPECTED_PATIENTS = {"A": 20_336, "B": 20_000}


@pytest.mark.slow
def test_real_patient_counts() -> None:
    raw_dir = Path(load_yaml(Path("configs/data.yaml"))["raw_dir"])
    if not all((raw_dir / d).is_dir() for d in HOSPITAL_DIRS.values()):
        pytest.skip("real data not found")
    counts = {
        name: read_raw_hospital(name, raw_dir)["patient_id"].nunique() for name in HOSPITAL_DIRS
    }
    assert counts == EXPECTED_PATIENTS
    assert sum(counts.values()) == 40_336
