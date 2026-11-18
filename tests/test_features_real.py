"""Every feature (all flags on) on the real view of dev patients. Slow: needs the data cache."""

import numpy as np
import pytest
from test_features_forbidden import _all_features

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.quality import CLEANING_CONFIG
from sepsis.data.view import build_evaluation_view


@pytest.mark.slow
def test_features_on_real_dev_patients() -> None:
    from sepsis.data.splits import (
        IncompleteCacheError,
        SplitFingerprintError,
        load_patient_hours,
        load_project_split,
    )

    try:
        split = load_project_split()  # dev patients only (ADR-0004)
        ids = split.groupby("hospital").sample(200, random_state=0)["patient_id"].tolist()
        hours = load_patient_hours(["A", "B"], ids)
    except (FileNotFoundError, IncompleteCacheError, SplitFingerprintError):
        pytest.skip("real data cache or split file not found")
    view = build_evaluation_view(hours, load_yaml(CLEANING_CONFIG), load_yaml(PANELS_CONFIG))
    features = _all_features(view.values, view.lab_rows, load_yaml(FEATURES_CONFIG))

    assert len(features) == len(hours)
    assert not np.isinf(features.to_numpy(dtype=float)).any()
    assert features["partial_sofa"].between(0, 13).all()  # 4 + 4 + 1 + 4
    assert features["sirs_count"].between(0, 4).all()
    assert features["qsofa_partial"].between(0, 2).all()
    since = features.filter(like="hours_since_").to_numpy(dtype=float)
    shown = since[~np.isnan(since)]
    assert shown.size and (shown >= 1).all()  # a lab is never shown at the row it was recorded
    assert features["partial_sofa_delta_24"].min() >= 0
