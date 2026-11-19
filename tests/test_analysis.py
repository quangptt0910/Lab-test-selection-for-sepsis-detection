"""M3-3 analysis modules: refit, importance and the ordering diagnostic. Toy data only."""

import numpy as np
import pandas as pd
import pytest
from runner_toy import experiment
from test_runner import _main, shared, toy  # noqa: F401  (pytest fixtures)

import sepsis.eval.refit as refit_module
import sepsis.eval.run_experiment as cli
from sepsis.config import CLEANING_CONFIG, FEATURES_CONFIG, load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.eval.importance import shap_importance, suspicion_checks
from sepsis.eval.ordering_diagnostic import PREFIXES, ordering_columns, with_ordering
from sepsis.eval.refit import check_against_run, load_run, refit_cross_hospital
from sepsis.eval.runner import plan_setup
from sepsis.features.builder import BLOCKS

CLEANING = load_yaml(CLEANING_CONFIG)
PANELS = load_yaml(PANELS_CONFIG)
SMALL_B2 = {"n_estimators": 5, "min_child_samples": 2, "num_leaves": 4}


def _record(lactate: list[float]) -> pd.DataFrame:
    n = len(lactate)
    return pd.DataFrame(
        {"patient_id": "A_1", "hospital": "A", "hour": range(n), "ICULOS": range(1, n + 1)}
    ).assign(Lactate=lactate, HR=80.0, SepsisLabel=0)


# --- Ordering diagnostic --------------------------------------------------------------------------


def test_ordering_columns_show_only_what_the_view_can_show() -> None:
    nan = np.nan
    cols = ordering_columns(_record([2.0, nan, 3.0, nan, nan]), CLEANING, PANELS)
    # Recorded at rows 0 and 2, revealed one row later (reveal delay 1).
    assert cols["revealed_now_Lactate"].tolist() == [0, 1, 0, 1, 0]
    assert cols["n_revealed_Lactate"].tolist() == [0, 1, 1, 2, 2]


def test_ordering_counts_restart_for_each_patient() -> None:
    nan = np.nan
    first = _record([2.0, 3.0, nan])
    second = _record([nan, 4.0, nan, nan]).assign(patient_id="A_2")
    shuffled = pd.concat([second, first]).sample(frac=1, random_state=0)
    cols = ordering_columns(shuffled, CLEANING, PANELS).set_index(["patient_id", "hour"])
    assert cols.loc["A_1", "n_revealed_Lactate"].tolist() == [0, 1, 2]
    assert cols.loc["A_2", "n_revealed_Lactate"].tolist() == [0, 0, 1, 1]
    assert cols.loc["A_2", "revealed_now_Lactate"].tolist() == [0, 0, 1, 0]


def test_ordering_columns_never_use_later_rows() -> None:
    nan = np.nan
    base = ordering_columns(_record([2.0, nan, 3.0, nan, nan]), CLEANING, PANELS)
    later = ordering_columns(_record([2.0, nan, 3.0, 9.0, 4.0]), CLEANING, PANELS)
    pd.testing.assert_frame_equal(base.iloc[:4], later.iloc[:4])


def test_ordering_columns_are_in_no_feature_block_or_config() -> None:
    names = set(BLOCKS) | set(load_yaml(FEATURES_CONFIG).get("core", {}))
    names |= set(load_yaml(FEATURES_CONFIG).get("extras", {}))
    text = open(FEATURES_CONFIG, encoding="utf-8").read()
    for prefix in PREFIXES:
        assert prefix not in text
        assert not any(prefix in name for name in names)


# --- Refit and importance on a toy B2 run ---------------------------------------------------------


@pytest.fixture
def b2_run(toy: dict):  # noqa: F811
    config = experiment(
        toy["root"],
        id="toy-b2",
        split={"setups": ["A_to_B", "B_to_A"]},
        model={"type": "lightgbm_regressor", "params": SMALL_B2},
    )
    assert _main(toy, config) == 0
    run_dir = toy["root"] / "experiments" / "runs" / "toy-b2"
    cfg, preprocessing = load_run(run_dir)
    dev = cli.load_dev_tables(cfg)
    return run_dir, cfg, preprocessing, dev.tables, dev.split


def test_refit_reproduces_the_runs_scored_predictions(b2_run) -> None:
    run_dir, cfg, preprocessing, tables, split = b2_run
    for setup in ("A_to_B", "B_to_A"):
        refitted = refit_cross_hospital(cfg, preprocessing, tables, split, setup)
        assert set(refitted.keys["patient_id"]) == set(plan_setup(split, setup)[0].scored)
        check_against_run(refitted, run_dir)  # raises if the scores differ


def test_refit_fits_on_the_planned_training_patients_only(b2_run, monkeypatch) -> None:
    _, cfg, preprocessing, tables, split = b2_run
    fitted_rows = []
    real = refit_module.model_factory

    def spy_factory(*args, **kwargs):
        new_model = real(*args, **kwargs)

        def spy():
            model = new_model()
            fit = model.fit

            def recording_fit(X, y, sample_weight):
                fitted_rows.append(X.index)
                return fit(X, y, sample_weight)

            model.fit = recording_fit
            return model

        return spy

    monkeypatch.setattr(refit_module, "model_factory", spy_factory)
    for setup in ("A_to_B", "B_to_A"):
        fitted_rows.clear()
        refit_cross_hospital(cfg, preprocessing, tables, split, setup)
        (rows,) = fitted_rows
        fitted = set(tables.targets.loc[rows, "patient_id"])
        assert fitted == set(plan_setup(split, setup)[0].refit)


def test_refit_check_refuses_a_different_model(b2_run) -> None:
    run_dir, cfg, preprocessing, tables, split = b2_run
    path = run_dir / "A_to_B" / "predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[predictions["kind"] == "scored", "score"] += 1.0
    predictions.to_parquet(path, index=False)
    refitted = refit_cross_hospital(cfg, preprocessing, tables, split, "A_to_B")
    with pytest.raises(ValueError, match="reproduce"):
        check_against_run(refitted, run_dir)


def test_shap_importance_adds_up_to_the_prediction(b2_run) -> None:
    _, cfg, preprocessing, tables, split = b2_run
    refitted = refit_cross_hospital(cfg, preprocessing, tables, split, "A_to_B")
    table = shap_importance(refitted)
    assert list(table.columns) == ["feature", "gain", "mean_abs_shap", "direction", "rank"]
    assert sorted(table["feature"]) == sorted(tables.feature_columns)
    contrib = refitted.model.booster.predict(refitted.X_eval, pred_contrib=True)
    assert contrib.shape[1] == len(tables.feature_columns) + 1  # plus the bias
    np.testing.assert_allclose(contrib.sum(axis=1), refitted.scores, rtol=1e-6, atol=1e-9)
    notes = suspicion_checks({"A_to_B": table, "B_to_A": table})
    assert any("rank of hospital/stay columns" in n for n in notes)


def test_importance_refuses_a_model_that_is_not_lightgbm(toy: dict) -> None:  # noqa: F811
    config = experiment(toy["root"], id="toy-dummy", split={"setups": ["A_to_B"]})
    _main(toy, config)
    cfg, preprocessing = load_run(toy["root"] / "experiments" / "runs" / "toy-dummy")
    dev = cli.load_dev_tables(cfg)
    refitted = refit_cross_hospital(cfg, preprocessing, dev.tables, dev.split, "A_to_B")
    with pytest.raises(ValueError, match="lightgbm"):
        shap_importance(refitted)


def test_with_ordering_adds_the_same_columns_to_both_views(b2_run) -> None:
    _, _, _, tables, _ = b2_run
    columns = tables.targets[["patient_id", "hour"]].assign(n_revealed_Lactate=1.0)
    added = with_ordering(tables, columns)
    assert added.feature_columns[-1] == "n_revealed_Lactate"
    assert added.train_X["n_revealed_Lactate"].eq(1.0).all()
    assert added.eval_X["n_revealed_Lactate"].eq(1.0).all()
    with pytest.raises(ValueError, match="aligned"):
        with_ordering(tables, columns.iloc[::-1])
