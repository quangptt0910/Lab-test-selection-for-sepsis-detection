"""Feature rows on the real view of dev patients. Slow: needs the data cache."""

import time

import numpy as np
import pytest
from test_features_forbidden import all_features

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.quality import CLEANING_CONFIG
from sepsis.data.view import build_evaluation_view
from sepsis.features.builder import build_features


def _dev_hours(hospitals: list[str], per_hospital: int | None = None):  # noqa: ANN202
    from sepsis.data.splits import (
        IncompleteCacheError,
        SplitFingerprintError,
        load_patient_hours,
        load_project_split,
    )

    try:
        split = load_project_split()  # dev patients only (ADR-0004)
        split = split[split["hospital"].isin(hospitals)]
        if per_hospital is not None:
            split = split.groupby("hospital").sample(per_hospital, random_state=0)
        return load_patient_hours(hospitals, split["patient_id"].tolist())
    except (FileNotFoundError, IncompleteCacheError, SplitFingerprintError):
        pytest.skip("real data cache or split file not found")


@pytest.mark.slow
def test_features_on_real_dev_patients() -> None:
    hours = _dev_hours(["A", "B"], per_hospital=200)
    view = build_evaluation_view(hours, load_yaml(CLEANING_CONFIG), load_yaml(PANELS_CONFIG))
    features = all_features(view, load_yaml(FEATURES_CONFIG)).drop(columns=["patient_id"])

    assert len(features) == len(hours)
    assert not np.isinf(features.to_numpy(dtype=float)).any()
    assert features["partial_sofa"].between(0, 13).all()  # 4 + 4 + 1 + 4
    assert features["sirs_count"].between(0, 4).all()
    assert features["qsofa_partial"].between(0, 2).all()
    since = features.filter(like="hours_since_").to_numpy(dtype=float)
    shown = since[~np.isnan(since)]
    assert shown.size and (shown >= 1).all()  # a lab is never shown at the row it was recorded
    assert features["partial_sofa_delta_24"].min() >= 0


@pytest.mark.slow
def test_time_all_dev_patients() -> None:
    """Acceptance (M2-3): time to build feature rows for all dev patients. Run with -s."""
    hours = _dev_hours(["A", "B"])
    panels = load_yaml(PANELS_CONFIG)
    start = time.perf_counter()
    view = build_evaluation_view(hours, load_yaml(CLEANING_CONFIG), panels)
    view_s = time.perf_counter() - start
    for name, run in (
        ("core", lambda: build_features(view, load_yaml(FEATURES_CONFIG), panels_config=panels)),
        ("all blocks", lambda: all_features(view, load_yaml(FEATURES_CONFIG))),
    ):
        start = time.perf_counter()
        rows = run()
        features_s = time.perf_counter() - start
        print(
            f"\n{name}: {rows['patient_id'].nunique()} patients, {len(rows)} rows, "
            f"{rows.shape[1]} columns; view {view_s:.1f} s, features {features_s:.1f} s"
        )
        assert len(rows) == len(hours)
