"""The VoI-target CLI (M6-1, ADR-0010): refusals, out-of-fold scorers, table shape and the cache.

Runs on the toy project of the runner tests (no real patient data).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from runner_toy import experiment
from test_runner import CLEAN, _main, shared, toy  # noqa: F401  (pytest fixtures)

import sepsis.labels.build_voi as cli
from sepsis.data.panels import PANEL_NAMES
from sepsis.data.view import KEYS
from sepsis.eval.run_experiment import RunRefusedError
from sepsis.labels.voi import decision_rows

VOI = yaml.safe_load(Path("configs/voi.yaml").read_text(encoding="utf-8"))
GAINS = ["gain_h1", "gain_h6", "gain_h12"]
COLUMNS = [
    "setup",
    "outer_fold",
    "inner_fold",
    "patient_id",
    "t",
    "panel",
    "available",
    *GAINS,
    "dscore_t1",
    "score_without_t1",
    "alarm_without_t1",
]


def _config(toy: dict, source: str, **changes) -> Path:  # noqa: F811
    path = toy["root"] / "voi.yaml"
    path.write_text(yaml.safe_dump({**VOI, "source_run": source, "chunk_patients": 7, **changes}))
    return path


def _run(toy: dict, config: Path, *extra: str, git=CLEAN) -> int:  # noqa: F811
    return cli.main([str(config), *extra], root=toy["root"], git_check=git)


def _source(toy: dict) -> str:  # noqa: F811
    assert _main(toy, experiment(toy["root"], id="toy-src")) == 0  # dummy model, three setups
    return "toy-src"


def _written(capsys) -> Path:
    out = capsys.readouterr().out
    return Path(re.search(r"\[voi\] (?:wrote|cached:) (.+)", out).group(1).strip())


# --- Refusals -------------------------------------------------------------------------------------


def test_refuses_a_missing_source_run(toy) -> None:  # noqa: F811
    with pytest.raises(RunRefusedError, match="no-such-run"):
        _run(toy, _config(toy, "no-such-run"))


def test_refuses_a_source_run_that_is_not_in_the_log(toy) -> None:  # noqa: F811
    (toy["root"] / "experiments" / "runs" / "unlogged").mkdir(parents=True)
    with pytest.raises(RunRefusedError, match="unlogged"):
        _run(toy, _config(toy, "unlogged"))


def test_refuses_a_dirty_tree_unless_allowed(toy) -> None:  # noqa: F811
    source = _source(toy)
    with pytest.raises(RunRefusedError, match="dirty"):
        _run(toy, _config(toy, source), git=lambda: ("abc1234", True))


def test_a_refit_that_differs_from_the_source_run_writes_nothing(toy) -> None:  # noqa: F811
    source = _source(toy)
    path = toy["root"] / "experiments" / "runs" / source / "B_to_A" / "predictions.parquet"
    stored = pd.read_parquet(path)
    stored.loc[stored["kind"] == "oof", "score"] += 1.0
    stored.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="does not reproduce"):
        _run(toy, _config(toy, source, seed=11))
    assert not list(Path(toy["root"]).rglob("voi-*"))


# --- A full build ---------------------------------------------------------------------------------


@pytest.fixture
def built(toy, capsys):  # noqa: F811
    source = _source(toy)
    capsys.readouterr()
    assert _run(toy, _config(toy, source)) == 0
    out_dir = _written(capsys)
    tables = {s: pd.read_parquet(out_dir / f"{s}.parquet") for s in ("A_to_B", "B_to_A", "mix")}
    return toy, source, out_dir, tables


def test_one_table_per_setup_with_the_agreed_columns(built) -> None:
    _, _, out_dir, tables = built
    assert (out_dir / "meta.json").exists() and not out_dir.with_name(
        out_dir.name + ".partial"
    ).exists()
    for setup, table in tables.items():
        assert list(table.columns) == COLUMNS
        assert set(table["setup"]) == {setup}
        assert set(table["panel"]) == set(PANEL_NAMES)
        assert set(table["available"]) <= {0, 1}
        gone = table["available"] == 0
        assert table.loc[gone, GAINS].isna().all().all()
        assert table.loc[~gone, GAINS].notna().all().all()


def test_training_side_only_and_scored_by_the_model_that_held_each_patient_out(built) -> None:
    project, _, _, tables = built
    split = project["split"].set_index("patient_id")
    for setup, hospital in (("A_to_B", "A"), ("B_to_A", "B")):
        table = tables[setup]
        assert set(split.loc[table["patient_id"].unique(), "hospital"]) == {hospital}
        assert set(split.loc[table["patient_id"].unique(), "split"]) == {"dev"}
        assert table["outer_fold"].isna().all()
        assert (
            table["inner_fold"].to_numpy() == split.loc[table["patient_id"], "fold"].to_numpy()
        ).all()
    mix = tables["mix"]
    fold = split.loc[mix["patient_id"], "fold"].to_numpy()
    assert (mix["inner_fold"].to_numpy() == fold).all()  # held out by the inner model
    assert (mix["outer_fold"].to_numpy() != fold).all()  # in the outer fold's training side
    per = mix.groupby(["patient_id", "t", "panel"])["outer_fold"].nunique()
    assert (per == 4).all()  # one row per outer fold that trains on the patient


def test_decision_rows_follow_the_config(built) -> None:
    project, _, _, tables = built
    table = tables["A_to_B"]
    hours = project["hours"]
    keys = hours[hours["patient_id"].isin(table["patient_id"].unique())][KEYS]
    expected = decision_rows(keys, every=VOI["decision_every_rows"], seed=VOI["seed"])
    mine = table[table["panel"] == "CBC"][["patient_id", "t"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        mine, expected.sort_values(["patient_id", "t"]).reset_index(drop=True)
    )


def test_some_panel_changes_a_score_on_the_toy_data(built) -> None:
    _, _, _, tables = built  # the toy model scores partial_sofa: CBC, CHEM and LFT feed it
    hit = tables["A_to_B"][tables["A_to_B"]["available"] == 1]
    assert (hit["dscore_t1"] > 0).any()
    assert np.isfinite(hit["score_without_t1"]).all()


def test_a_second_build_reuses_the_targets(built, monkeypatch, capsys) -> None:
    project, source, out_dir, _ = built

    def forbidden(*args, **kwargs):
        raise AssertionError("refit although the targets are cached")

    monkeypatch.setattr(cli, "refit_inner", forbidden)
    assert _run(project, _config(project, source)) == 0
    assert _written(capsys) == out_dir


def test_changed_source_run_outputs_are_not_served_from_the_cache(built, monkeypatch) -> None:
    project, source, _, _ = built
    path = project["root"] / "experiments" / "runs" / source / "A_to_B" / "thresholds.json"
    path.write_text(path.read_text(encoding="utf-8").replace("{", '{"note": "x", ', 1))

    def recomputed(*args, **kwargs):
        raise AssertionError("recomputed")

    monkeypatch.setattr(cli, "refit_inner", recomputed)
    with pytest.raises(AssertionError, match="recomputed"):
        _run(project, _config(project, source))


def test_meta_records_the_refit_check_of_every_inner_model(built) -> None:
    _, _, out_dir, _ = built
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    checks = meta["refit_check"]
    assert len(checks) == meta["inner_models"] == 5 + 5 + 5 * 4  # A_to_B, B_to_A, mix
    assert all(c["rows_off"] == 0 and c["rows"] > 0 for c in checks)  # same machine: exact
    assert meta["config"]["refit_check"] == VOI["refit_check"]
