import numpy as np
import pandas as pd
import pytest

from sepsis.risk.base import MODELS, DummyModel, RiskModel, make_model


def test_dummy_scores_one_column_and_fills_nan_from_training_rows() -> None:
    train = pd.DataFrame({"x": [3.0, np.nan, 5.0], "y": [9.0, 9.0, 9.0]})
    other = pd.DataFrame({"x": [np.nan, 1.0, 7.0], "y": [0.0, 0.0, 0.0]})
    model = make_model("dummy", {"column": "x"}, seed=0)
    assert model.fit(train, np.zeros(3), np.ones(3)) is model
    # NaN becomes the training minimum (3), not the minimum of the rows being scored (1).
    np.testing.assert_array_equal(model.predict(other), [3.0, 1.0, 7.0])


def test_dummy_must_be_fitted_first() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        DummyModel("x").predict(pd.DataFrame({"x": [1.0]}))


def test_registry_builds_fresh_models_and_refuses_unknown_types() -> None:
    a = make_model("dummy", {"column": "x"}, seed=0)
    b = make_model("dummy", {"column": "x"}, seed=0)
    assert a is not b
    assert isinstance(a, RiskModel)
    assert "dummy" in MODELS
    with pytest.raises(ValueError, match="unknown model type"):
        make_model("lightgbm_classifier", {}, seed=0)
    with pytest.raises(ValueError, match="column"):
        make_model("dummy", {}, seed=0)
