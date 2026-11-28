"""Replay on real dev patients (M5-1). Slow: needs the data cache; dev patients only."""

import numpy as np
import pandas as pd
import pytest
from test_replay_engine import AllPolicy, NonePolicy

from sepsis.config import CLEANING_CONFIG, FEATURES_CONFIG, load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.view import KEYS, build_evaluation_view
from sepsis.features.builder import build_features, feature_columns
from sepsis.replay.engine import replay


class Score:
    """A fixed score from a lab value, a lab's age and a vital, so reveals matter."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            X["partial_sofa"].fillna(0.0)
            + 0.2 * X["Lactate"].fillna(0.0)
            - 0.01 * X["hours_since_CBC"].fillna(48.0)
            + 0.01 * X["HR"].fillna(0.0)
        ).to_numpy(dtype=float)


@pytest.mark.slow
def test_all_reproduces_the_batch_evaluation_view_on_dev_patients() -> None:
    from sepsis.data.splits import (
        IncompleteCacheError,
        SplitFingerprintError,
        load_patient_hours,
        load_project_split,
    )

    try:
        split = load_project_split()  # dev patients only (ADR-0004)
        ids = split.groupby("hospital").sample(100, random_state=1)["patient_id"].tolist()
        hours = load_patient_hours(["A", "B"], ids)
    except (FileNotFoundError, IncompleteCacheError, SplitFingerprintError):
        pytest.skip("real data cache or split file not found")
    hours = hours.sort_values(KEYS).reset_index(drop=True)
    cleaning, panels, features = (
        load_yaml(CLEANING_CONFIG),
        load_yaml(PANELS_CONFIG),
        load_yaml(FEATURES_CONFIG),
    )
    view = build_evaluation_view(hours, cleaning, panels)
    columns = feature_columns(view, features, panels_config=panels)
    batch = Score().predict(build_features(view, features, panels_config=panels)[columns])

    def run(policy):  # noqa: ANN001, ANN202
        return replay(
            hours,
            model=Score(),
            threshold=2.0,
            policy=policy,
            features_config=features,
            feature_columns=columns,
            panels_config=panels,
            cleaning_config=cleaning,
        )

    every = run(AllPolicy())
    np.testing.assert_allclose(every.predictions["score"], batch, rtol=0, atol=1e-12)
    nothing = run(NonePolicy())
    assert nothing.requests.empty
    assert not np.allclose(nothing.predictions["score"], batch)  # hidden labs do change scores
