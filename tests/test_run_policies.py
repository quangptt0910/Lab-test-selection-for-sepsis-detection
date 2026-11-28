"""The replay-baselines CLI (M5-2): refusals, run folders, LOG rows and the hard M5 check.

Runs on the toy project of the runner tests (no real patient data).
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from runner_toy import experiment
from test_runner import CLEAN, _main, shared, toy  # noqa: F401  (pytest fixtures)

import sepsis.replay.run_policies as cli
from sepsis.action.policy import Step
from sepsis.eval.run_experiment import RunRefusedError, logged_ids
from sepsis.eval.runner import PARTIAL_NOTE
from sepsis.replay.baselines import AsRecordedPolicy

IDS = {"NONE": "r-none-t", "ALL": "r-all-t", "AS_RECORDED": "r-asrec-t"}
SETUP_KEYS = {"A_to_B": ["A_to_B"], "B_to_A": ["B_to_A"], "mix": ["mix_A", "mix_B"]}


def _source(toy: dict, setups: list[str]) -> str:  # noqa: F811
    config = experiment(toy["root"], id="toy-src", split={"setups": setups})
    assert _main(toy, config) == 0
    return "toy-src"


def _config(toy: dict, source: str, policies: dict | None = None, note: str = "") -> Path:  # noqa: F811
    path = toy["root"] / "replay.yaml"
    body = {"source_run": source, "policies": IDS if policies is None else policies, "note": note}
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _run(toy: dict, config: Path, *extra: str, git=CLEAN) -> int:  # noqa: F811
    return cli.main([str(config), *extra], root=toy["root"], git_check=git)


def _runs(toy: dict) -> Path:  # noqa: F811
    return toy["root"] / "experiments" / "runs"


def _log(toy: dict) -> Path:  # noqa: F811
    return toy["root"] / "experiments" / "LOG.md"


# --- Refusals, before any computation ------------------------------------------------------------


@pytest.fixture
def fake_source(toy: dict) -> str:  # noqa: F811
    (_runs(toy) / "toy-src").mkdir(parents=True)
    log = _log(toy)
    log.write_text(log.read_text(encoding="utf-8") + "| toy-src | x |\n", encoding="utf-8")
    return "toy-src"


def _no_compute(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("computed before refusing")

    monkeypatch.setattr(cli, "load_run", forbidden)
    monkeypatch.setattr(cli.run_experiment, "load_dev_tables", forbidden)


def test_refuses_an_id_already_in_the_log(toy, fake_source, monkeypatch) -> None:  # noqa: F811
    _no_compute(monkeypatch)
    log = _log(toy)
    log.write_text(log.read_text(encoding="utf-8") + "| r-all-t | x |\n", encoding="utf-8")
    with pytest.raises(RunRefusedError, match="r-all-t"):
        _run(toy, _config(toy, fake_source))


def test_refuses_an_existing_run_folder(toy, fake_source, monkeypatch) -> None:  # noqa: F811
    _no_compute(monkeypatch)
    (_runs(toy) / "r-none-t").mkdir()
    with pytest.raises(RunRefusedError, match="r-none-t"):
        _run(toy, _config(toy, fake_source))


def test_refuses_a_dirty_tree_unless_allowed(toy, fake_source, monkeypatch) -> None:  # noqa: F811
    _no_compute(monkeypatch)
    with pytest.raises(RunRefusedError, match="dirty"):
        _run(toy, _config(toy, fake_source), git=lambda: ("abc1234", True))
    with pytest.raises(AssertionError, match="computed"):  # allowed: it gets past the refusals
        _run(toy, _config(toy, fake_source), "--allow-dirty", git=lambda: ("abc1234", True))


def test_refuses_a_missing_source_run(toy, monkeypatch) -> None:  # noqa: F811
    _no_compute(monkeypatch)
    with pytest.raises(RunRefusedError, match="no-such-run"):
        _run(toy, _config(toy, "no-such-run"))


def test_refuses_a_source_run_that_is_not_in_the_log(toy, monkeypatch) -> None:  # noqa: F811
    _no_compute(monkeypatch)
    (_runs(toy) / "unlogged").mkdir(parents=True)
    with pytest.raises(RunRefusedError, match="unlogged"):
        _run(toy, _config(toy, "unlogged"))


def test_refuses_an_unknown_policy(toy, fake_source, monkeypatch) -> None:  # noqa: F811
    _no_compute(monkeypatch)
    with pytest.raises(RunRefusedError, match="RANDOM"):
        _run(toy, _config(toy, fake_source, {**IDS, "RANDOM": "r-rand-t"}))


# --- A full run -----------------------------------------------------------------------------------


def test_writes_one_run_folder_and_one_log_row_per_policy(toy) -> None:  # noqa: F811
    source = _source(toy, ["A_to_B", "B_to_A", "mix"])
    assert _run(toy, _config(toy, source, note="toy baselines")) == 0
    source_scores = json.loads((_runs(toy) / source / "scores.json").read_text())
    header = next(
        line
        for line in _log(toy).read_text(encoding="utf-8").splitlines()
        if line.startswith("| id")
    )
    rows = {
        line.split("|")[1].strip(): line
        for line in _log(toy).read_text(encoding="utf-8").splitlines()
        if line.startswith("| r-")
    }
    assert set(rows) == set(IDS.values()) <= logged_ids(_log(toy))
    for policy, run_id in IDS.items():
        folder = _runs(toy) / run_id
        assert not folder.with_name(run_id + ".partial").exists()
        assert (folder / "config.yaml").exists()
        for setup in SETUP_KEYS:
            predictions = pd.read_parquet(folder / setup / "predictions.parquet")
            assert list(predictions.columns) == ["patient_id", "hour", "score", "alarm"]
            requests = pd.read_parquet(folder / setup / "requests.parquet")
            assert not requests["patient_id"].astype(str).str.startswith("p0").any()  # real ids
        scores = json.loads((folder / "scores.json").read_text())
        assert sorted(scores) == ["A_to_B", "B_to_A", "mix_A", "mix_B"]
        for value in scores.values():
            assert {"utility", "auroc", "labs_per_patient_day", "unavailable_rate"} <= set(value)
            assert "nothing_new_rate" in value
        meta = json.loads((folder / "meta.json").read_text())
        assert meta["id"] == run_id and meta["policy"] == policy and meta["source_run"] == source
        assert {"sha", "dirty", "date", "command", "scorer_sha256", "versions"} <= set(meta)
        row = rows[run_id]
        assert row.count("|") == header.count("|")
        assert "source toy-src" in row and "toy baselines" in row
    alls = json.loads((_runs(toy) / "r-all-t" / "scores.json").read_text())
    nones = json.loads((_runs(toy) / "r-none-t" / "scores.json").read_text())
    asrec = json.loads((_runs(toy) / "r-asrec-t" / "scores.json").read_text())
    for key in alls:
        assert alls[key]["utility"] == source_scores[key]["utility"]
        assert alls[key]["labs_per_patient_day"] == pytest.approx(7 * 24)
        assert nones[key]["labs_per_patient_day"] == 0.0
        assert nones[key]["unavailable_rate"] is None and nones[key]["nothing_new_rate"] is None
        assert asrec[key]["utility"] == alls[key]["utility"]
        assert 0 < asrec[key]["labs_per_patient_day"] < alls[key]["labs_per_patient_day"]
        assert asrec[key]["unavailable_rate"] == 0.0
    assert "| 168.0 |" in rows["r-all-t"] and "| 0.0 |" in rows["r-none-t"]


# --- The hard M5 check ----------------------------------------------------------------------------


def _assert_refused(toy: dict, code: int) -> None:  # noqa: F811
    assert code != 0
    assert not set(IDS.values()) & logged_ids(_log(toy))
    for run_id in IDS.values():
        assert not (_runs(toy) / run_id).exists()
        assert (_runs(toy) / f"{run_id}.partial").exists()


def test_a_none_policy_that_requests_fails_the_m5_check(toy, monkeypatch, capsys) -> None:  # noqa: F811
    source = _source(toy, ["A_to_B"])

    class Leaky:
        name = "NONE"

        def decide(self, step: Step) -> dict[str, frozenset[str]]:
            first = step.scores.index[0]
            return {first: frozenset({"LACTATE"})} if step.t == 0 else {}

    monkeypatch.setitem(cli.POLICIES, "NONE", lambda panels: Leaky())
    _assert_refused(toy, _run(toy, _config(toy, source)))
    assert "NONE" in capsys.readouterr().err


def test_all_differing_from_the_source_run_fails_the_m5_check(toy, capsys) -> None:  # noqa: F811
    source = _source(toy, ["A_to_B"])
    path = _runs(toy) / source / "A_to_B" / "predictions.parquet"
    stored = pd.read_parquet(path)
    first = stored.index[stored["kind"] == "scored"][0]
    stored.loc[first, "alarm"] = 1 - stored.loc[first, "alarm"]
    stored.to_parquet(path, index=False)
    _assert_refused(toy, _run(toy, _config(toy, source)))
    assert "ALL" in capsys.readouterr().err


def test_as_recorded_requests_that_find_nothing_fail_the_m5_check(toy, monkeypatch, capsys) -> None:  # noqa: F811
    source = _source(toy, ["A_to_B"])

    class Greedy(AsRecordedPolicy):
        """Predicts like ALL (CBC reveals only values up to t) but over-requests."""

        def decide(self, step: Step) -> dict[str, frozenset[str]]:
            out = super().decide(step)
            return {pid: out.get(pid, frozenset()) | {"CBC"} for pid in step.scores.index}

    monkeypatch.setitem(cli.POLICIES, "AS_RECORDED", Greedy)
    _assert_refused(toy, _run(toy, _config(toy, source)))
    assert "AS_RECORDED" in capsys.readouterr().err


def test_a_partial_source_run_is_marked_in_the_log_row() -> None:
    value = {"utility": 0.3, "auroc": 0.8, "labs_per_patient_day": 10.0, "unavailable_rate": 0.0}
    row = cli.replay_log_row(
        "r-x",
        date="d",
        sha="s",
        config_path="c.yaml",
        scores={"A_to_B": value},
        note="",
        source_run="src",
    )
    assert PARTIAL_NOTE in row and "0.800 (1 of 4)" in row
