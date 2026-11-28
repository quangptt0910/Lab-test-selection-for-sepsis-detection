"""Replay models (M5-1): each scored patient gets a model that never trained on them, with the
setup's frozen threshold from the run folder (ADR-0008: no saved models, replay refits).
Toy data only."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from runner_toy import experiment
from test_runner import _main, shared, toy  # noqa: F401  (pytest fixtures)

import sepsis.eval.run_experiment as cli
from sepsis.action.baselines import AllPolicy
from sepsis.eval.refit import load_run
from sepsis.eval.runner import plan_setup
from sepsis.replay.models import by_hospital, replay_models, replay_setup, score_replay

SETUPS = ("A_to_B", "B_to_A", "mix")


@pytest.fixture
def run(toy: dict):  # noqa: F811
    config = experiment(toy["root"], id="toy-replay")  # dummy model, all three setups
    assert _main(toy, config) == 0
    run_dir = toy["root"] / "experiments" / "runs" / "toy-replay"
    cfg, _ = load_run(run_dir)
    return run_dir, cli.load_dev_tables(cfg)


@pytest.mark.parametrize("setup", SETUPS)
def test_every_scored_patient_has_a_model_that_never_saw_them(run, setup: str) -> None:
    run_dir, dev = run
    models = replay_models(run_dir, setup, dev=dev)
    plans = plan_setup(dev.split, setup)
    assert len(models) == len(plans)
    frozen = json.loads((run_dir / setup / "thresholds.json").read_text())
    for model, plan, entry in zip(models, plans, frozen, strict=True):
        assert set(model.trained_ids) == set(plan.refit)
        assert set(model.scored_ids) == set(plan.scored)
        assert not set(model.trained_ids) & set(model.scored_ids)
        fold = None if entry["outer_fold"] is None else int(entry["outer_fold"])
        assert model.outer_fold == plan.outer_fold == fold
        assert model.threshold == entry["threshold"]  # frozen, never re-tuned
    scored = [pid for m in models for pid in m.scored_ids]
    assert len(scored) == len(set(scored))  # each patient replayed once


def test_a_refit_that_differs_from_the_run_is_refused(run) -> None:
    run_dir, dev = run
    path = run_dir / "mix" / "predictions.parquet"
    predictions = pd.read_parquet(path)
    predictions.loc[predictions["kind"] == "scored", "score"] += 1.0
    predictions.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="does not reproduce"):
        replay_models(run_dir, "mix", dev=dev)


@pytest.mark.parametrize("setup", ["A_to_B", "B_to_A", pytest.param("mix", marks=pytest.mark.slow)])
def test_replaying_all_reproduces_the_runs_scored_predictions(run, setup: str) -> None:
    run_dir, dev = run
    result = replay_setup(run_dir, setup, AllPolicy(), dev=dev)
    stored = pd.read_parquet(run_dir / setup / "predictions.parquet")
    stored = stored[stored["kind"] == "scored"].sort_values(["patient_id", "hour"])
    mine = result.predictions.sort_values(["patient_id", "hour"])
    assert (
        mine[["patient_id", "hour"]].values.tolist()
        == stored[["patient_id", "hour"]].values.tolist()
    )
    np.testing.assert_allclose(mine["score"], stored["score"], rtol=0, atol=1e-9)
    assert mine["alarm"].tolist() == stored["alarm"].astype(int).tolist()


@pytest.mark.slow
def test_mix_results_split_by_hospital(run) -> None:
    run_dir, dev = run
    result = replay_setup(run_dir, "mix", AllPolicy(), dev=dev)
    parts = by_hospital(result, dev.split)
    assert set(parts) == {"A", "B"}
    for name, part in parts.items():
        assert part.predictions["patient_id"].str.startswith(f"{name}_").all()
        assert part.requests["patient_id"].str.startswith(f"{name}_").all()
    assert sum(len(p.predictions) for p in parts.values()) == len(result.predictions)
    assert sum(len(p.requests) for p in parts.values()) == len(result.requests)


@pytest.mark.parametrize("setup", ["A_to_B", pytest.param("mix", marks=pytest.mark.slow)])
def test_scoring_the_all_replay_reproduces_the_runs_scores(run, setup: str) -> None:
    run_dir, dev = run
    models = replay_models(run_dir, setup, dev=dev)
    result = replay_setup(run_dir, setup, AllPolicy(), dev=dev, models=models)
    scores = score_replay(result, setup, dev)
    stored = json.loads((run_dir / "scores.json").read_text())
    keys = ["mix_A", "mix_B"] if setup == "mix" else [setup]
    assert sorted(scores) == keys
    for key in keys:
        for name, value in stored[key].items():
            assert scores[key][name] == pytest.approx(value, rel=0, abs=1e-12), (key, name)
        assert scores[key]["utility"] == stored[key]["utility"]  # same alarms, same utility
        assert scores[key]["labs_per_patient_day"] == pytest.approx(7 * 24)
        assert 0 < scores[key]["unavailable_rate"] < 1
        assert 0 <= scores[key]["nothing_new_rate"] < 1


def test_replay_setup_uses_the_models_it_is_given(run) -> None:
    run_dir, dev = run
    models = replay_models(run_dir, "A_to_B", dev=dev)

    class Constant:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), 42.0)

    swapped = [dataclasses.replace(m, model=Constant()) for m in models]
    result = replay_setup(run_dir, "A_to_B", AllPolicy(), dev=dev, models=swapped)
    assert result.predictions["score"].eq(42.0).all()
