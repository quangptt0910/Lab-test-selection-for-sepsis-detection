"""B0, B1 and B2 baselines (M3-2) on small hand-built feature tables. No real patient data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from sepsis.config import FEATURES_CONFIG, load_yaml
from sepsis.eval.experiment_config import (
    ExperimentConfigError,
    check_model_type,
    validate_experiment_config,
)
from sepsis.risk.base import MODEL_TARGETS, MODELS, RiskModel, make_model
from sepsis.risk.gbm import REFUSED_PARAMS, LightGBMRegressor
from sepsis.risk.logistic import LogisticBaseline
from sepsis.risk.rules import RuleScore

NAN = np.nan
PREPROCESSING = load_yaml(Path("configs/preprocessing.yaml"))["b1"]
LAB_FILL = {"Lactate": 1.0, "BUN": 14.0, "Creatinine": 0.9}
LOG_LABS = ["Lactate", "BUN", "Creatinine"]


def table(seed: int, n: int = 40, shift: float = 0.0) -> pd.DataFrame:
    """A feature table with labs, a lab ratio, a vital, a hours_since column and two scores."""
    rng = np.random.default_rng(seed)

    def gappy(values: np.ndarray, p: float = 0.3) -> np.ndarray:
        return np.where(rng.random(n) < p, NAN, values)

    return pd.DataFrame(
        {
            "Lactate": gappy(rng.uniform(0.5, 6, n) + shift),
            "BUN": gappy(rng.uniform(5, 60, n) + shift),
            "Creatinine": gappy(rng.uniform(0.4, 4, n) + shift),
            "bun_cr": gappy(rng.uniform(5, 40, n) + shift),
            "HR": gappy(rng.uniform(50, 140, n) + shift),
            "Unit1": gappy(rng.integers(0, 2, n).astype(float)),
            "hours_since_LACTATE": gappy(rng.integers(0, 30, n).astype(float) + shift),
            "sirs_count": rng.integers(0, 5, n).astype(float),
            "partial_sofa": rng.integers(0, 8, n).astype(float),
        }
    )


def labels(n: int = 40, seed: int = 0) -> np.ndarray:
    y = np.random.default_rng(seed).integers(0, 2, n).astype(float)
    y[:2] = [0.0, 1.0]  # both classes present
    return y


def logistic(**changes) -> LogisticBaseline:
    kwargs = {"lab_fill": LAB_FILL, "log_labs": LOG_LABS, "C": 1.0, "max_iter": 1000, "seed": 0}
    return LogisticBaseline(**{**kwargs, **changes})


SMALL_GBM = {"n_estimators": 5, "min_child_samples": 2, "colsample_bytree": 0.5, "num_leaves": 4}


# --- B0 -----------------------------------------------------------------------------------------


def test_rule_score_is_the_weighted_sum() -> None:
    X = table(0)
    model = RuleScore({"sirs_count": 1.0, "partial_sofa": 2.0})
    before = dict(vars(model))
    assert model.fit(X, labels(), np.ones(len(X))) is model
    assert vars(model) == before  # nothing is fitted
    np.testing.assert_array_equal(model.predict(X), X["sirs_count"] + 2 * X["partial_sofa"])


def test_rule_score_refuses_missing_columns_nan_and_empty_weights() -> None:
    X = table(0)
    with pytest.raises(ValueError, match="no weights"):
        RuleScore({})
    with pytest.raises(ValueError, match="missing"):
        RuleScore({"qsofa_partial": 1.0}).fit(X, labels(), np.ones(len(X)))
    with pytest.raises(ValueError, match="NaN"):
        RuleScore({"Lactate": 1.0}).fit(X, labels(), np.ones(len(X))).predict(X)


# --- B1: fitted statistics come from the training rows only (acceptance) -------------------------


def test_logistic_fills_and_scaler_come_from_the_training_rows_only() -> None:
    X_train, X_eval = table(0), table(1, shift=100.0)  # eval values far from training values
    model = logistic().fit(X_train, labels(), np.ones(len(X_train)))
    expected_fill = {
        "Lactate": 1.0,  # labs: fixed normals, even though X_train has measured values
        "BUN": 14.0,
        "Creatinine": 0.9,
        "bun_cr": 14.0 / 0.9,  # ratio of the normals
        "HR": np.nanmedian(X_train["HR"]),
        "Unit1": np.nanmedian(X_train["Unit1"]),
        "hours_since_LACTATE": np.nanmax(X_train["hours_since_LACTATE"]),
        "sirs_count": np.nanmedian(X_train["sirs_count"]),
        "partial_sofa": np.nanmedian(X_train["partial_sofa"]),
    }
    assert model.fill_ == pytest.approx(expected_fill)

    filled = X_train.fillna(expected_fill)
    filled[LOG_LABS] = np.log1p(filled[LOG_LABS])
    np.testing.assert_allclose(model.scaler_.mean_, filled.mean().to_numpy())
    np.testing.assert_allclose(model.scaler_.scale_, filled.std(ddof=0).to_numpy())

    # Predicting other data never refits, and its NaN get the training statistics.
    before = (dict(model.fill_), model.scaler_.mean_.copy(), model.coef_.copy())
    scores = model.predict(X_eval)
    assert model.fill_ == before[0]
    np.testing.assert_array_equal(model.scaler_.mean_, before[1])
    np.testing.assert_array_equal(model.coef_, before[2])
    eval_filled = X_eval.fillna(expected_fill)
    eval_filled[LOG_LABS] = np.log1p(eval_filled[LOG_LABS])
    expected = model.classifier_.decision_function(model.scaler_.transform(eval_filled.to_numpy()))
    np.testing.assert_allclose(scores, expected)


def test_logistic_fallbacks_for_columns_with_no_training_value() -> None:
    X = table(0).assign(hours_since_LACTATE=NAN, Unit1=NAN, ICULOS=np.arange(40.0) + 3)
    model = logistic().fit(X, labels(), np.ones(len(X)))
    assert model.fill_["hours_since_LACTATE"] == 42.0  # never revealed: the longest stay
    assert model.fill_["Unit1"] == 0.0
    assert np.isfinite(model.predict(table(1).assign(ICULOS=np.arange(40.0)))).all()


def test_logistic_returns_log_odds() -> None:
    X = table(0)
    model = logistic().fit(X, labels(), np.ones(len(X)))
    scores = model.predict(X)
    np.testing.assert_allclose(expit(scores), model.predict_proba(X))


def test_logistic_ignores_rows_with_weight_zero() -> None:
    X, y = table(0), labels()
    w = np.ones(len(X))
    w[10:20] = 0.0
    flipped = y.copy()
    flipped[10:20] = 1.0 - flipped[10:20]
    a = logistic().fit(X, y, w)
    b = logistic().fit(X, flipped, w)
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-6)


def test_logistic_refusals() -> None:
    X = table(0)
    with pytest.raises(ValueError, match="0/1"):
        logistic().fit(X, np.linspace(-1, 1, len(X)), np.ones(len(X)))
    model = logistic().fit(X, labels(), np.ones(len(X)))
    with pytest.raises(ValueError, match="columns"):
        model.predict(X.drop(columns="HR"))
    with pytest.raises(RuntimeError, match="fit"):
        logistic().predict(X)


def test_logistic_refuses_lab_derived_columns_without_a_fixed_fill() -> None:
    X = table(0).assign(sofa_renal=NAN)
    with pytest.raises(ValueError, match="sofa_renal"):
        logistic().fit(X, labels(), np.ones(len(X)))


def test_logistic_is_deterministic() -> None:  # lbfgs uses no randomness, whatever the seed
    X = table(0)
    a = logistic(seed=0).fit(X, labels(), np.ones(len(X))).predict(table(1))
    b = logistic(seed=0).fit(X, labels(), np.ones(len(X))).predict(table(1))
    np.testing.assert_array_equal(a, b)


# --- B2 -----------------------------------------------------------------------------------------


def test_lightgbm_takes_nan_and_ignores_rows_with_weight_zero() -> None:
    X = table(0, n=200)
    y = np.random.default_rng(3).normal(size=len(X))
    w = np.ones(len(X))
    w[:50] = 0.0
    flipped = y.copy()
    flipped[:50] = -100 * flipped[:50]  # weight-0 rows: change y, never drop them (bins)
    a = LightGBMRegressor(SMALL_GBM, seed=0).fit(X, y, w).predict(X)
    b = LightGBMRegressor(SMALL_GBM, seed=0).fit(X, flipped, w).predict(X)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)


def test_lightgbm_is_reproducible_with_a_seed_and_the_seed_matters() -> None:
    X = table(0, n=200)
    y = np.random.default_rng(3).normal(size=len(X))
    w = np.ones(len(X))

    def scores(seed: int) -> np.ndarray:
        return LightGBMRegressor(SMALL_GBM, seed=seed).fit(X, y, w).predict(X)

    np.testing.assert_array_equal(scores(0), scores(0))
    assert not np.array_equal(scores(0), scores(1))


def test_lightgbm_refuses_every_alias_of_its_seed_and_determinism_params() -> None:
    from lightgbm.basic import _ConfigAliases

    names = ["seed", "bagging_seed", "feature_fraction_seed", "data_random_seed", "extra_seed"]
    names += ["drop_seed", "objective_seed", "deterministic", "verbosity", "force_row_wise"]
    names += ["force_col_wise"]
    aliases = set().union(*(_ConfigAliases.get(name) for name in names))
    assert aliases <= REFUSED_PARAMS


@pytest.mark.parametrize("key", sorted(REFUSED_PARAMS))
def test_lightgbm_refuses_seed_and_determinism_params(key: str) -> None:
    with pytest.raises(ValueError, match="set by the runner"):
        LightGBMRegressor({**SMALL_GBM, key: 1}, seed=0)


# --- Registry and configs -------------------------------------------------------------------------

PARAMS = {
    "rule_score": {"weights": {"sirs_count": 1, "partial_sofa": 1}},
    "logistic": {"C": 1.0, "max_iter": 100},
    "lightgbm_regressor": SMALL_GBM,
}


@pytest.mark.parametrize("model_type", sorted(PARAMS))
def test_registry_builds_every_baseline(model_type: str) -> None:
    model = make_model(model_type, PARAMS[model_type], seed=0, preprocessing=PREPROCESSING)
    assert isinstance(model, RiskModel)
    assert model_type in MODELS


def test_logistic_factory_needs_the_preprocessing_config() -> None:
    with pytest.raises(ValueError, match="preprocessing"):
        make_model("logistic", PARAMS["logistic"], seed=0)
    with pytest.raises(ValueError, match="unknown"):
        make_model("logistic", {"C": 1.0, "penalty": "l1"}, seed=0, preprocessing=PREPROCESSING)
    for key, value in (("vital_fill", "mean"), ("scaler", "robust")):
        with pytest.raises(ValueError, match=key):
            make_model(
                "logistic", PARAMS["logistic"], 0, preprocessing={**PREPROCESSING, key: value}
            )
    model = make_model("logistic", PARAMS["logistic"], seed=0, preprocessing=PREPROCESSING)
    assert model.lab_fill["Lactate"] == PREPROCESSING["lab_fill"]["Lactate"]["value"]


def test_logistic_factory_needs_its_params_from_the_config() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        make_model("logistic", {"C": 1.0}, seed=0, preprocessing=PREPROCESSING)


def test_rule_score_factory_needs_weights() -> None:
    with pytest.raises(ValueError, match="weights"):
        make_model("rule_score", {}, seed=0)


CONFIGS = ["configs/b0_rules.yaml", "configs/b1_logistic.yaml", "configs/b2_lightgbm.yaml"]


@pytest.mark.parametrize("path", CONFIGS)
def test_baseline_configs_are_valid_and_share_the_feature_blocks(path: str) -> None:
    features = load_yaml(FEATURES_CONFIG)
    cfg = validate_experiment_config(load_yaml(Path(path)), features)
    check_model_type(cfg)
    b2 = load_yaml(Path("configs/b2_lightgbm.yaml"))
    assert cfg["features"] == b2["features"]  # one cached feature table for all three
    assert cfg["seed"] == b2["seed"] and cfg["split"] == b2["split"]


def test_b1_is_trained_on_sepsis_label_only() -> None:
    cfg = load_yaml(Path("configs/b1_logistic.yaml"))
    assert cfg["target"] == "sepsis_label"
    assert MODEL_TARGETS["logistic"] == ("sepsis_label",)
    with pytest.raises(ExperimentConfigError, match="target"):
        check_model_type({**cfg, "target": "utility_diff"})


# --- Real data ------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["logistic", "lightgbm_regressor"])
def test_baselines_fit_one_real_fold(model_type: str) -> None:
    import time

    from sepsis.config import DATA_CONFIG

    data = load_yaml(DATA_CONFIG)
    if not Path(data["split_path"]).exists() or not Path(data["cache_dir"]).exists():
        pytest.skip("real data cache or split file not found")
    from dataclasses import replace

    from test_tables import settings

    from sepsis.data.splits import load_patient_hours, load_project_split
    from sepsis.eval.tables import build_tables

    split = load_project_split()
    a_dev = split[(split["hospital"] == "A") & (split["split"] == "dev")]
    hours = load_patient_hours(["A"], a_dev["patient_id"])
    b2 = load_yaml(Path("configs/b2_lightgbm.yaml"))
    tables = build_tables(hours, replace(settings(), blocks=tuple(b2["features"]["hand_crafted"])))
    fold = tables.targets["patient_id"].map(a_dev.set_index("patient_id")["fold"])
    train, held_out = (fold != 0).to_numpy(), (fold == 0).to_numpy()
    target = "SepsisLabel" if model_type == "logistic" else "utility_diff"
    params = PARAMS["logistic"] if model_type == "logistic" else b2["model"]["params"]
    model = make_model(model_type, params, seed=0, preprocessing=PREPROCESSING)
    start = time.perf_counter()
    model.fit(
        tables.train_X.loc[train, tables.feature_columns],
        tables.targets.loc[train, target].to_numpy(dtype=float),
        tables.targets.loc[train, "weight"].to_numpy(dtype=float),
    )
    seconds = time.perf_counter() - start
    scores = model.predict(tables.eval_X.loc[held_out, tables.feature_columns])
    print(f"{model_type}: fit on {train.sum()} rows in {seconds:.1f} s")
    assert np.isfinite(scores).all() and len(scores) == held_out.sum()
