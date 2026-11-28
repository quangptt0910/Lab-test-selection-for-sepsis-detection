"""Inner fold refits (M6-1, ADR-0010 §6): the out-of-fold models a run tuned its threshold on,
refit and checked against the run's stored `oof` predictions. Toy data only."""

import json

import numpy as np
import pandas as pd
import pytest
from runner_toy import experiment
from test_runner import _main, shared, toy  # noqa: F401  (pytest fixtures)

import sepsis.eval.run_experiment as cli
from sepsis.eval.refit import RefitTolerance, check_oof_against_run, load_run, refit_inner
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


# --- The tie tolerance (ADR-0010 §6): float ties across machines, never a different decision ----

TOLERANCE = RefitTolerance(atol=1e-9, max_share_off=0.05, max_abs_diff=0.01)


def _nudged(run, delta_of) -> tuple:
    """Refit A_to_B's first inner model after changing stored oof rows of that fold.

    `delta_of(scores, threshold)` gives the new stored scores of the fold's rows.
    """
    run_dir, cfg, preprocessing, dev = run
    path = run_dir / "A_to_B" / "predictions.parquet"
    stored = pd.read_parquet(path)
    plan = plan_setup(dev.split, "A_to_B")[0]
    fold = plan.inner[0][0]
    threshold = float(
        json.loads((run_dir / "A_to_B" / "thresholds.json").read_text())[0]["threshold"]
    )
    hit = (stored["kind"] == "oof") & (stored["inner_fold"] == fold)
    stored.loc[hit, "score"] = delta_of(stored.loc[hit, "score"].to_numpy(), threshold)
    stored.to_parquet(path, index=False)
    return refit_inner(cfg, preprocessing, dev.tables, plan, fold), run_dir, threshold


def _far_from(scores: np.ndarray, threshold: float, n: int) -> np.ndarray:
    """Positions of the first n scores at least 0.01 away from the threshold."""
    return np.flatnonzero(np.abs(scores - threshold) > 0.01)[:n]


def test_a_tie_level_difference_without_alarm_change_passes_the_tolerance(run) -> None:
    def nudge(scores, threshold):
        out = scores.copy()
        out[_far_from(scores, threshold, 1)] += 1e-4
        return out

    refit, run_dir, threshold = _nudged(run, nudge)
    with pytest.raises(ValueError, match="does not reproduce"):
        check_oof_against_run(refit, run_dir)  # exact by default
    report = check_oof_against_run(refit, run_dir, threshold=threshold, tolerance=TOLERANCE)
    assert report["rows_off"] == 1 and report["max_abs_diff"] == pytest.approx(1e-4)
    assert report["rows"] == len(refit.keys)


def test_an_alarm_change_is_refused_even_within_the_tolerance(run) -> None:
    def flip(scores, threshold):
        out = scores.copy()
        i = int(np.argmin(np.abs(scores - threshold)))  # the score nearest the threshold
        out[i] = threshold + 1e-6 if scores[i] <= threshold else threshold - 1e-6
        return out

    refit, run_dir, threshold = _nudged(run, flip)
    loose = RefitTolerance(atol=1e-9, max_share_off=0.05, max_abs_diff=1e6)
    with pytest.raises(ValueError, match="alarm"):
        check_oof_against_run(refit, run_dir, threshold=threshold, tolerance=loose)


def test_too_many_or_too_large_differences_are_refused(run) -> None:
    def many(scores, threshold):
        out = scores.copy()
        out[_far_from(scores, threshold, len(scores))] += 1e-4  # far more than 5% of the rows
        return out

    refit, run_dir, threshold = _nudged(run, many)
    with pytest.raises(ValueError, match="rows differ"):
        check_oof_against_run(refit, run_dir, threshold=threshold, tolerance=TOLERANCE)


def test_a_difference_above_the_largest_allowed_is_refused(run) -> None:
    def big(scores, threshold):
        out = scores.copy()
        i = np.flatnonzero(np.abs(scores - threshold) > 1.0)[:1]  # stays on its side
        out[i] += 0.5 * np.sign(scores[i] - threshold)
        return out

    refit, run_dir, threshold = _nudged(run, big)
    with pytest.raises(ValueError, match="max"):
        check_oof_against_run(refit, run_dir, threshold=threshold, tolerance=TOLERANCE)
