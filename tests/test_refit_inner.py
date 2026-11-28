"""Inner fold refits (M6-1, ADR-0010 §6): the out-of-fold models a run tuned its threshold on,
refit and checked against the run's stored `oof` predictions. Toy data only."""

import numpy as np
import pandas as pd
import pytest
from runner_toy import experiment
from test_runner import _main, shared, toy  # noqa: F401  (pytest fixtures)

import sepsis.eval.run_experiment as cli
from sepsis.eval.refit import check_oof_against_run, load_run, refit_inner
from sepsis.eval.runner import plan_setup


@pytest.fixture
def run(toy: dict):  # noqa: F811
    config = experiment(toy["root"], id="toy-inner")  # dummy model, all three setups
    assert _main(toy, config) == 0
    run_dir = toy["root"] / "experiments" / "runs" / "toy-inner"
    cfg, preprocessing = load_run(run_dir)
    return run_dir, cfg, preprocessing, cli.load_dev_tables(cfg)


@pytest.mark.parametrize("setup", ["A_to_B", "B_to_A", "mix"])
def test_every_inner_refit_reproduces_the_runs_out_of_fold_scores(run, setup: str) -> None:
    run_dir, cfg, preprocessing, dev = run
    stored = pd.read_parquet(run_dir / setup / "predictions.parquet")
    n_oof = 0
    for plan in plan_setup(dev.split, setup):
        for fold, fit_ids, oof_ids in plan.inner:
            refit = refit_inner(cfg, preprocessing, dev.tables, plan, fold)
            assert (refit.outer_fold, refit.inner_fold) == (plan.outer_fold, fold)
            assert set(refit.keys["patient_id"]) == set(oof_ids)
            assert not set(fit_ids) & set(oof_ids)
            check_oof_against_run(refit, run_dir)  # raises unless it is the run's model
            n_oof += len(refit.keys)
    assert n_oof == int((stored["kind"] == "oof").sum())  # every stored oof row is reproduced


def test_an_inner_refit_that_differs_from_the_run_is_refused(run) -> None:
    run_dir, cfg, preprocessing, dev = run
    path = run_dir / "A_to_B" / "predictions.parquet"
    stored = pd.read_parquet(path)
    plan = plan_setup(dev.split, "A_to_B")[0]
    fold = plan.inner[0][0]
    hit = (stored["kind"] == "oof") & (stored["inner_fold"] == fold)
    stored.loc[hit[hit].index[0], "score"] += 1.0
    stored.to_parquet(path, index=False)
    refit = refit_inner(cfg, preprocessing, dev.tables, plan, fold)
    with pytest.raises(ValueError, match="does not reproduce"):
        check_oof_against_run(refit, run_dir)
    other = plan.inner[1][0]  # the other folds are untouched
    check_oof_against_run(refit_inner(cfg, preprocessing, dev.tables, plan, other), run_dir)
    assert np.isfinite(refit.scores).all()
