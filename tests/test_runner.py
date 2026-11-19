import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from runner_toy import GRID, RED_FLAGS, experiment, patients, project, records
from test_tables import settings

import sepsis.data.load as load_module
import sepsis.data.splits as splits_module
import sepsis.eval.run_experiment as cli
import sepsis.eval.runner as runner_module
from sepsis.config import load_yaml
from sepsis.eval.run_experiment import RunRefusedError, logged_ids, main
from sepsis.eval.runner import (
    PARTIAL_NOTE,
    SetupResult,
    log_row,
    plan_setup,
    red_flags,
    run_setup,
)
from sepsis.eval.scorer import Scores
from sepsis.eval.tables import KEYS, Tables, build_tables
from sepsis.eval.threshold import training_side_patients
from sepsis.risk.base import MODELS, DummyModel, make_model

SPLIT = patients()
REAL_APPEND = cli.append_log_row
DEV = SPLIT[SPLIT["split"] == "dev"]
TEST_IDS = set(SPLIT.loc[SPLIT["split"] == "test", "patient_id"])
CLEAN = lambda: ("abc1234", False)  # noqa: E731  (an injected git check: clean tree)


@pytest.fixture(scope="module")
def tables() -> Tables:
    hours = records()
    return build_tables(hours[hours["patient_id"].isin(DEV["patient_id"])], settings())


@pytest.fixture(scope="module")
def labels(tables: Tables) -> pd.DataFrame:
    hospital = tables.targets["patient_id"].str[0]
    return tables.targets[[*KEYS, "SepsisLabel"]].assign(hospital=hospital)


class Spy(DummyModel):
    """Records which rows each model instance was fitted on and predicted."""

    log: list = []

    def fit(self, X, y, sample_weight):
        self.fit_index, self.fit_X, self.y, self.w = X.index, X, y, sample_weight
        Spy.log.append(self)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        self.predict_index, self.predict_X = X.index, X
        return super().predict(X)


def _run(tables, labels, setup, target="utility_diff", column="partial_sofa"):
    Spy.log = []
    result = run_setup(
        tables,
        labels,
        SPLIT,
        setup,
        new_model=lambda: Spy(column),
        target=target,
        grid=GRID,
    )
    return result, list(Spy.log)


# --- Fold plans and fit sides ---------------------------------------------------------------------


def test_fold_plans_follow_the_stored_folds() -> None:
    a_to_b = plan_setup(SPLIT, "A_to_B")
    assert len(a_to_b) == 1 and len(a_to_b[0].inner) == 5
    assert set(a_to_b[0].refit) == set(DEV.loc[DEV["hospital"] == "A", "patient_id"])
    assert set(a_to_b[0].scored) == set(DEV.loc[DEV["hospital"] == "B", "patient_id"])
    mix = plan_setup(SPLIT, "mix")
    assert [p.outer_fold for p in mix] == [0, 1, 2, 3, 4]
    for plan in mix:
        assert len(plan.inner) == 4
        assert set(plan.scored) == set(DEV.loc[DEV["fold"] == plan.outer_fold, "patient_id"])
        assert not set(plan.refit) & set(plan.scored)


@pytest.mark.parametrize("setup", ["A_to_B", "B_to_A", "mix"])
def test_models_fit_only_on_the_training_side_and_never_predict_their_own_patients(
    tables, labels, setup
) -> None:
    _, models = _run(tables, labels, setup)
    pid = tables.targets["patient_id"]
    # Models are made in plan order: each plan's inner folds, then its refit.
    expected = []
    for plan in plan_setup(SPLIT, setup):
        side = training_side_patients(SPLIT, setup, plan.outer_fold)
        expected += [(set(fit), set(oof), side, set(plan.scored)) for _, fit, oof in plan.inner]
        expected.append((set(plan.refit), set(plan.scored), side, set()))
    assert len(models) == len(expected)
    for model, (fit_ids, predict_ids, side, scored) in zip(models, expected, strict=True):
        fitted = set(pid[model.fit_index])
        predicted = set(pid[model.predict_index])
        assert fitted == fit_ids and predicted == predict_ids
        # Each model stays inside its own plan's training side: never the evaluated hospital,
        # never the held-out mix fold k (inner models neither fit nor predict fold k).
        assert fitted <= side
        assert not (fitted | predicted) & scored
        assert not fitted & TEST_IDS and not predicted & TEST_IDS
        assert not fitted & predicted  # out-of-fold and scored rows come from other patients
        # The dummy's fitted statistic comes from its training rows only.
        assert model.fill_ == np.nanmin(tables.train_X.loc[model.fit_index, "partial_sofa"])


@pytest.mark.parametrize("setup", ["A_to_B", "B_to_A"])
def test_a_fitted_statistic_comes_from_the_training_hospital(tables, labels, setup) -> None:
    # Toy lactate: every B value is above every A value, and both have NaNs.
    _, models = _run(tables, labels, setup, column="Lactate")
    hospital = tables.targets["patient_id"].str[0]
    lactate = tables.train_X["Lactate"]
    train, other = setup[0], setup[-1]
    train_min = np.nanmin(lactate[hospital == train])
    other_min = np.nanmin(tables.eval_X["Lactate"][hospital == other])
    assert train_min != other_min
    refit = models[-1]
    assert refit.fill_ == train_min
    # The scored rows with no lactate get the training-side fill, not the evaluated hospital's.
    missing = tables.eval_X.loc[refit.predict_index, "Lactate"].isna().to_numpy()
    assert missing.any()
    np.testing.assert_array_equal(refit.predict(refit.predict_X)[missing], train_min)


@pytest.mark.parametrize("setup", ["A_to_B", "mix"])
def test_thresholds_are_tuned_on_out_of_fold_rows_only(tables, labels, setup, monkeypatch) -> None:
    calls = []
    real = runner_module.tune_threshold

    def spy(hours, oof, split, setup_, *, outer_fold=None, grid):
        calls.append((outer_fold, set(hours["patient_id"]), oof.copy()))
        return real(hours, oof, split, setup_, outer_fold=outer_fold, grid=grid)

    monkeypatch.setattr(runner_module, "tune_threshold", spy)
    result, models = _run(tables, labels, setup)
    plans = plan_setup(SPLIT, setup)
    assert len(calls) == len(plans)
    pid = tables.targets["patient_id"]
    per_plan = len(plans[0].inner) + 1
    for i, (plan, (outer_fold, label_ids, oof)) in enumerate(zip(plans, calls, strict=True)):
        oof_ids = set().union(*(set(ids) for _, _, ids in plan.inner))
        assert outer_fold == plan.outer_fold
        assert label_ids == oof_ids == set(oof["patient_id"])
        assert not oof_ids & set(plan.scored)
        # The scores are exactly the inner models' out-of-fold predictions, never the refit's.
        inner = models[i * per_plan : (i + 1) * per_plan - 1]
        expected = pd.concat(
            [
                tables.targets.loc[m.predict_index, KEYS].assign(score=m.predict(m.predict_X))
                for m in inner
            ]
        )
        pd.testing.assert_frame_equal(
            oof[[*KEYS, "score"]].sort_values(KEYS).reset_index(drop=True),
            expected.sort_values(KEYS).reset_index(drop=True),
            check_dtype=False,
        )
        assert set(pid[models[(i + 1) * per_plan - 1].predict_index]) == set(plan.scored)


PREPROCESSING = load_yaml(Path("configs/preprocessing.yaml"))["b1"]
BASELINES = {
    "rule_score": ({"weights": {"sirs_count": 1, "partial_sofa": 1}}, "utility_diff"),
    "logistic": ({"C": 1.0, "max_iter": 200}, "sepsis_label"),
    "lightgbm_regressor": ({"n_estimators": 5, "min_child_samples": 2}, "utility_diff"),
}


def _run_baseline(tables, labels, setup, model_type):
    params, target = BASELINES[model_type]
    built = []

    def new_model():
        built.append(make_model(model_type, params, seed=0, preprocessing=PREPROCESSING))
        return built[-1]

    result = run_setup(tables, labels, SPLIT, setup, new_model=new_model, target=target, grid=GRID)
    return result, built


@pytest.mark.parametrize("model_type", sorted(BASELINES))
def test_every_baseline_runs_under_the_runner(tables, labels, model_type) -> None:
    result, _ = _run_baseline(tables, labels, "mix", model_type)
    assert set(result.scores) == {"A", "B"}
    assert np.isfinite(result.predictions["score"]).all()


def test_logistic_scaler_sees_only_the_training_hospital(tables, labels) -> None:
    _, built = _run_baseline(tables, labels, "A_to_B", "logistic")
    refit = built[-1]  # fitted on all of A-dev, scores B-dev
    column = refit.columns_.index("Lactate")
    rows = (tables.targets["patient_id"].str[0] == "A").to_numpy()
    normal = PREPROCESSING["lab_fill"]["Lactate"]["value"]
    lactate = tables.train_X.loc[rows, "Lactate"].fillna(normal)
    # Toy B lactate lies above every A value, so any B row in the fit would move this mean.
    assert refit.scaler_.mean_[column] == pytest.approx(np.log1p(lactate).mean())
    assert refit.fill_["HR"] == np.nanmedian(tables.train_X.loc[rows, "HR"])


def test_fits_use_the_training_view_and_predictions_the_evaluation_view(tables, labels) -> None:
    _, models = _run(tables, labels, "B_to_A")
    columns = tables.feature_columns
    for model in models:
        pd.testing.assert_frame_equal(model.fit_X, tables.train_X.loc[model.fit_index, columns])
        pd.testing.assert_frame_equal(
            model.predict_X, tables.eval_X.loc[model.predict_index, columns]
        )


@pytest.mark.parametrize(
    ("target", "column"), [("utility_diff", "utility_diff"), ("sepsis_label", "SepsisLabel")]
)
def test_targets_and_weights_come_from_the_target_table(tables, labels, target, column) -> None:
    _, models = _run(tables, labels, "A_to_B", target=target)
    for model in models:
        rows = tables.targets.loc[model.fit_index]
        np.testing.assert_array_equal(model.y, rows[column].to_numpy(dtype=float))
        np.testing.assert_array_equal(model.w, rows["weight"].to_numpy(dtype=float))


def test_mix_tunes_one_threshold_per_outer_fold_and_scores_each_hospital(tables, labels) -> None:
    result, _ = _run(tables, labels, "mix")
    assert [t["outer_fold"] for t in result.thresholds] == [0, 1, 2, 3, 4]
    assert set(result.scores) == {"A", "B"}
    scored = result.predictions[result.predictions["kind"] == "scored"]
    assert set(scored["patient_id"]) == set(DEV["patient_id"])  # each dev patient scored once
    assert not scored.duplicated(KEYS).any()


# --- Summaries ----------------------------------------------------------------------------------


def _result(setup: str, utility: float, auroc: float = 0.8) -> SetupResult:
    s = Scores(auroc=auroc, auprc=0.1, accuracy=0.9, f_measure=0.2, utility=utility)
    scores = {"A": s, "B": s} if setup == "mix" else {"all": s}
    return SetupResult(setup, scores, [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


CFG = {"id": "x-1", "note": "a|b"}


def test_log_row_for_a_full_run() -> None:
    results = {s: _result(s, 0.3) for s in ("A_to_B", "B_to_A", "mix")}
    row = log_row(CFG, "configs/x.yaml", results, sha="abc", date="2026-11-19", flags=[])
    assert row == (
        "| x-1 | 2026-11-19 | abc | configs/x.yaml | dev | 0.300 | 0.300 | 0.300 | 0.300 "
        "| 0.800 | — | a\\|b |"
    )


def test_log_row_for_a_partial_run_with_red_flags() -> None:
    results = {"mix": _result("mix", 0.2), "B_to_A": _result("B_to_A", 0.4, auroc=0.97)}
    flags = red_flags(results, RED_FLAGS)
    row = log_row({"id": "x-2", "note": ""}, "c.yaml", results, sha="s", date="d", flags=flags)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[5:9] == ["—", "0.400", "0.200", "0.200"]
    assert cells[9] == "0.857 (3 of 4)"
    assert PARTIAL_NOTE in cells[11]
    assert "B→A AUROC 0.970 > 0.95" in cells[11]
    assert "B→A utility beats mix→A" in cells[11]


def test_red_flag_when_the_two_directions_differ() -> None:
    results = {"A_to_B": _result("A_to_B", 0.05), "B_to_A": _result("B_to_A", 0.3)}
    assert any("differ by more than 0.15" in f for f in red_flags(results, RED_FLAGS))


# --- The CLI end to end ---------------------------------------------------------------------------


def _main(p: dict, config: Path, *extra: str, git=CLEAN) -> int:
    # In-process (--jobs 1) unless a test asks: worker processes don't see monkeypatches.
    jobs = [] if "--jobs" in extra else ["--jobs", "1"]
    return main(
        [str(config), *jobs, *extra],
        root=p["root"],
        git_check=git,
        grid=GRID,
    )


@pytest.fixture(scope="module")
def shared(tmp_path_factory) -> Path:
    """The toy cache, split file and feature cache, built once for all CLI tests."""
    return tmp_path_factory.mktemp("toy_data")


@pytest.fixture
def toy(tmp_path: Path, shared: Path, monkeypatch) -> dict:
    p = project(tmp_path, data_dir=shared)
    requested = []

    def forbidden(*args, **kwargs):
        raise AssertionError("the runner read every patient (test patients included)")

    real = splits_module.read_cache_patients

    def spy(hospitals, cache_dir, ids):
        ids = list(ids)
        requested.extend(ids)
        return real(hospitals, cache_dir, ids)

    for name in ("read_cache", "read_raw_hospital", "read_raw_all"):
        monkeypatch.setattr(load_module, name, forbidden)
    monkeypatch.setattr(splits_module, "read_cache_patients", spy)
    # Point the CLI at the toy project: the real loaders, with the toy configs as their defaults.
    configs = {"_data_config": p["data_config"], "_protocol_config": p["protocol_config"]}
    monkeypatch.setattr(cli, "DATA_CONFIG", p["data_config"])
    monkeypatch.setattr(cli, "PROTOCOL_CONFIG", p["protocol_config"])
    monkeypatch.setattr(
        cli, "load_project_split", partial(splits_module.load_project_split, **configs)
    )
    monkeypatch.setattr(
        cli, "load_patient_hours", partial(splits_module.load_patient_hours, **configs)
    )
    p["requested"] = requested
    return p


def test_end_to_end_run_writes_the_run_folder_and_one_log_row(toy: dict) -> None:
    config = experiment(toy["root"])
    assert _main(toy, config) == 0
    run = toy["root"] / "experiments" / "runs" / "toy-001"
    for name in [
        "config.yaml",
        "meta.json",
        "scores.json",
        "feature_columns.json",
        "resolved/features.yaml",
        "resolved/eval_grid.yaml",
        *(
            f"{s}/{f}"
            for s in ("A_to_B", "B_to_A", "mix")
            for f in ("thresholds.json", "threshold_curve.csv", "folds.csv", "predictions.parquet")
        ),
    ]:
        assert (run / name).exists(), name
    rows = [
        r
        for r in (toy["root"] / "experiments" / "LOG.md").read_text(encoding="utf-8").splitlines()
        if "toy-001" in r
    ]
    assert len(rows) == 1
    cells = [c.strip() for c in rows[0].strip("|").split("|")]
    assert all(c not in ("", "—") for c in cells[5:9])  # all four numbers
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    assert set(scores) == {"A_to_B", "B_to_A", "mix_A", "mix_B"}
    # Test patients were never requested from the cache (ADR-0004).
    assert toy["requested"] and not set(toy["requested"]) & TEST_IDS


def test_partial_run_writes_dashes_and_the_partial_note(toy: dict) -> None:
    config = experiment(toy["root"], split={"setups": ["B_to_A"]})
    _main(toy, config)
    row = (toy["root"] / "experiments" / "LOG.md").read_text(encoding="utf-8").splitlines()[-1]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[5] == "—" and cells[7] == "—" and cells[8] == "—"
    assert PARTIAL_NOTE in cells[11]


def test_duplicate_ids_and_finished_runs_are_refused(toy: dict) -> None:
    config = experiment(toy["root"], split={"setups": ["A_to_B"]})
    _main(toy, config)
    with pytest.raises(RunRefusedError, match="already in"):
        _main(toy, config)
    # Exact match on the id cell: toy-0011 is a different id.
    other = experiment(toy["root"], id="toy-0011", split={"setups": ["A_to_B"]})
    _main(toy, other)
    assert {"toy-001", "toy-0011"} <= logged_ids(toy["root"] / "experiments" / "LOG.md")


def test_a_dirty_tree_is_refused_unless_allowed(toy: dict) -> None:
    config = experiment(toy["root"], split={"setups": ["A_to_B"]})
    with pytest.raises(RunRefusedError, match="dirty"):
        _main(toy, config, git=lambda: ("abc1234", True))
    _main(toy, config, "--allow-dirty", git=lambda: ("abc1234", True))
    row = (toy["root"] / "experiments" / "LOG.md").read_text(encoding="utf-8").splitlines()[-1]
    assert "abc1234-dirty" in row


def test_a_crashed_run_writes_no_log_row_and_can_be_rerun(toy: dict, monkeypatch) -> None:
    config = experiment(toy["root"], split={"setups": ["A_to_B", "mix"]})
    real = runner_module.run_setup

    def crash_on_mix(*args, **kwargs):
        if args[3] == "mix":
            raise RuntimeError("crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "run_setup", crash_on_mix)
    with pytest.raises(RuntimeError, match="crash"):
        _main(toy, config)
    assert "toy-001" not in logged_ids(toy["root"] / "experiments" / "LOG.md")
    assert not (toy["root"] / "experiments" / "runs" / "toy-001").exists()
    monkeypatch.setattr(cli, "run_setup", real)
    assert _main(toy, config) == 0
    assert "toy-001" in logged_ids(toy["root"] / "experiments" / "LOG.md")


def test_a_run_that_fails_after_writing_its_folder_can_be_rerun(toy: dict, monkeypatch) -> None:
    config = experiment(toy["root"], split={"setups": ["A_to_B"]})
    runs = toy["root"] / "experiments" / "runs"
    log = toy["root"] / "experiments" / "LOG.md"

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "append_log_row", fail)
    with pytest.raises(OSError, match="disk full"):
        _main(toy, config)
    # The finished folder went back to .partial, so the id is neither logged nor finished.
    assert "toy-001" not in logged_ids(log)
    assert not (runs / "toy-001").exists()
    assert (runs / "toy-001.partial" / "scores.json").exists()
    (runs / "toy-001.partial" / "stale.txt").write_text("from the failed run")

    monkeypatch.setattr(cli, "append_log_row", REAL_APPEND)
    assert _main(toy, config) == 0
    assert "toy-001" in logged_ids(log)
    assert not (runs / "toy-001.partial").exists()
    assert not (runs / "toy-001" / "stale.txt").exists()  # the stale .partial was replaced


def test_a_logistic_run_records_its_preprocessing(toy: dict) -> None:
    config = experiment(
        toy["root"],
        id="toy-b1",
        split={"setups": ["A_to_B"]},
        target="sepsis_label",
        model={"type": "logistic", "params": {"C": 1.0, "max_iter": 200}},
    )
    assert _main(toy, config) == 0
    run = toy["root"] / "experiments" / "runs" / "toy-b1"
    recorded = load_yaml(run / "resolved" / "preprocessing.yaml")
    assert recorded == PREPROCESSING


def test_an_unregistered_model_is_refused_before_any_work(toy: dict, monkeypatch) -> None:
    config = experiment(toy["root"], model={"type": "xgboost", "params": {}})
    with pytest.raises(ValueError, match="not registered"):
        _main(toy, config)
    assert toy["requested"] == []  # nothing was loaded
    assert "dummy" in MODELS


# --- Real data ------------------------------------------------------------------------------------


@pytest.mark.slow
def test_dummy_b_to_a_on_real_dev_data(tmp_path: Path) -> None:
    from sepsis.config import DATA_CONFIG, load_yaml

    data = load_yaml(DATA_CONFIG)
    if not Path(data["split_path"]).exists() or not Path(data["cache_dir"]).exists():
        pytest.skip("real data cache or split file not found")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "LOG.md").write_text(
        Path("experiments/LOG.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = experiment(tmp_path, id="smoke-001", split={"setups": ["B_to_A"]})
    assert (
        main(
            [str(config), "--no-cache"],
            root=tmp_path,
            git_check=CLEAN,
        )
        == 0
    )
    run = tmp_path / "experiments" / "runs" / "smoke-001"
    scores = json.loads((run / "scores.json").read_text(encoding="utf-8"))
    assert 0.0 < scores["B_to_A"]["auroc"] < 1.0


def test_parallel_setups_give_the_same_run_as_serial(toy: dict) -> None:
    serial = experiment(toy["root"], id="toy-serial")
    parallel = experiment(toy["root"], id="toy-parallel")
    assert _main(toy, serial) == 0
    assert _main(toy, parallel, "--jobs", "3") == 0
    runs = toy["root"] / "experiments" / "runs"
    one, other = (
        json.loads((runs / i / "scores.json").read_text()) for i in ("toy-serial", "toy-parallel")
    )
    assert one == other
    for setup in ("A_to_B", "B_to_A", "mix"):
        a = pd.read_parquet(runs / "toy-serial" / setup / "predictions.parquet")
        b = pd.read_parquet(runs / "toy-parallel" / setup / "predictions.parquet")
        pd.testing.assert_frame_equal(a, b)
        assert (runs / "toy-serial" / setup / "thresholds.json").read_text() == (
            runs / "toy-parallel" / setup / "thresholds.json"
        ).read_text()
    meta = json.loads((runs / "toy-parallel" / "meta.json").read_text())
    assert meta["setup_jobs"] == 3
