"""The keep decision from run folders (`sepsis.eval.compare`). Toy run folders only."""

import json
from pathlib import Path

import pytest
import yaml

from sepsis.eval.compare import (
    CompareError,
    Run,
    cross_run_flags,
    group_stats,
    keep_decision,
    main,
)

KEYS = ("A_to_B", "B_to_A", "mix_A", "mix_B")
# A resolved features.yaml: one block on, one off.
FEATURES = {
    "raw": {"exclude": ["DBP"]},
    "core": {"sirs": {"enabled": True}},
    "extras": {"vital_lags": {"enabled": False, "lags": [1]}},
}


def write_run(
    runs: Path,
    run_id: str,
    utilities: tuple[float, float, float, float],
    *,
    seed: int = 0,
    sha: str = "abc1234",
    oof: tuple[float, float, float] = (0.40, 0.38, 0.39),
    auroc: float = 0.80,
    config: dict | None = None,
    setups: tuple[str, ...] = KEYS,
    features: dict | None = None,
    columns: tuple[str, ...] = ("ICULOS", "HR", "sirs_count"),
    meta: dict | None = None,
) -> None:
    """A run folder as `run_experiment` writes it: scores, thresholds, meta, config, the resolved
    configs and the feature columns."""
    folder = runs / run_id
    folder.mkdir(parents=True)
    scores = {
        key: {"auroc": auroc, "auprc": 0.1, "accuracy": 0.9, "f_measure": 0.1, "utility": u}
        for key, u in zip(KEYS, utilities, strict=True)
        if key in setups
    }
    (folder / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    for setup, value in zip(("A_to_B", "B_to_A"), oof[:2], strict=True):
        (folder / setup).mkdir()
        rows = [{"outer_fold": None, "threshold": 0.01, "tuning_utility": value}]
        (folder / setup / "thresholds.json").write_text(json.dumps(rows), encoding="utf-8")
    (folder / "mix").mkdir()
    folds = [{"outer_fold": str(k), "threshold": 0.01, "tuning_utility": oof[2]} for k in range(5)]
    (folder / "mix" / "thresholds.json").write_text(json.dumps(folds), encoding="utf-8")
    record = {"id": run_id, "sha": sha, "dirty": False, "seed": seed, "scorer_sha256": "26b8"}
    (folder / "meta.json").write_text(json.dumps({**record, **(meta or {})}), encoding="utf-8")
    (folder / "feature_columns.json").write_text(json.dumps(list(columns)), encoding="utf-8")
    (folder / "resolved").mkdir()
    resolved = {"features.yaml": features or FEATURES, "labels.yaml": {"u_fp": -0.05}}
    for name, value in resolved.items():
        (folder / "resolved" / name).write_text(yaml.safe_dump(value), encoding="utf-8")
    cfg = {
        "id": run_id,
        "hypothesis": "h",
        "seed": seed,
        "note": "",
        "features": {"hand_crafted": ["sirs"], "signatures": None, "lab_masking_p": 0.3},
        "model": {"type": "lightgbm_regressor", "params": {"n_estimators": 100}},
        **(config or {}),
    }
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def compare(runs: Path, reference: list[str], candidate: list[str]) -> int:
    return main(["--runs", str(runs), "--reference", *reference, "--candidate", *candidate])


def test_a_partial_run_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "r-1", (0.3, 0.3, 0.4, 0.36))
    write_run(tmp_path, "c-1", (0.3, 0.3, 0.4, 0.36), setups=("A_to_B", "B_to_A"))
    with pytest.raises(CompareError, match="partial"):
        compare(tmp_path, ["r-1"], ["c-1"])


def test_two_runs_of_a_group_with_the_same_seed_are_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "r-1", (0.3, 0.3, 0.4, 0.36), seed=0)
    write_run(tmp_path, "r-2", (0.3, 0.3, 0.4, 0.36), seed=0)
    write_run(tmp_path, "c-1", (0.3, 0.3, 0.4, 0.36))
    with pytest.raises(CompareError, match="seed"):
        compare(tmp_path, ["r-1", "r-2"], ["c-1"])


@pytest.mark.parametrize(
    "change",
    [
        {
            "features": {
                "hand_crafted": ["sirs"],
                "signatures": ["sig_vitals"],
                "lab_masking_p": 0.3,
            }
        },
        {"model": {"type": "lightgbm_regressor", "params": {"n_estimators": 500}}},
    ],
)
def test_a_seed_repeat_that_changed_more_than_the_seed_is_refused(tmp_path: Path, change) -> None:
    write_run(tmp_path, "r-1", (0.3, 0.3, 0.4, 0.36), seed=0)
    write_run(tmp_path, "c-1", (0.3, 0.3, 0.4, 0.36), seed=0)
    write_run(tmp_path, "c-2", (0.3, 0.3, 0.4, 0.36), seed=1, config=change)
    with pytest.raises(CompareError, match="c-2"):
        compare(tmp_path, ["r-1"], ["c-1", "c-2"])


def test_seed_repeats_may_differ_in_id_seed_hypothesis_and_note(tmp_path: Path) -> None:
    write_run(tmp_path, "r-1", (0.3, 0.3, 0.4, 0.36), seed=0)
    write_run(
        tmp_path, "r-2", (0.3, 0.3, 0.4, 0.36), seed=1, config={"hypothesis": "x", "note": "y"}
    )
    write_run(tmp_path, "c-1", (0.3, 0.3, 0.4, 0.36))
    assert compare(tmp_path, ["r-1", "r-2"], ["c-1"]) == 0


def make_run(
    run_id: str, utilities, *, seed: int = 0, oof=(0.40, 0.38, 0.39), auroc: float = 0.8
) -> Run:
    scores = {
        key: {"auroc": auroc, "auprc": 0.1, "accuracy": 0.9, "f_measure": 0.1, "utility": u}
        for key, u in zip(KEYS, utilities, strict=True)
    }
    oof_by_setup = dict(zip(("A→B", "B→A", "mix"), oof, strict=True))
    return Run(run_id, seed, "abc1234", scores, {}, oof_by_setup, {}, "26b8", False)


def test_group_stats_are_means_and_max_minus_min_spreads_of_unrounded_values() -> None:
    runs = [
        make_run("r-1", (0.30, 0.31, 0.42345, 0.36345), seed=0, oof=(0.40, 0.38, 0.39)),
        make_run("r-2", (0.31, 0.28, 0.42, 0.37), seed=1, oof=(0.42, 0.37, 0.41)),
        make_run("r-3", (0.29, 0.30, 0.41, 0.36), seed=2, oof=(0.41, 0.39, 0.40)),
    ]
    stats = group_stats(runs)
    # Per-run mix mean from unrounded values: (0.42345 + 0.36345) / 2 = 0.39345 (rounding to
    # three places first would give 0.393).
    assert stats.per_run["r-1"]["mix mean"] == pytest.approx(0.39345)
    assert stats.mean["A→B"] == pytest.approx(0.30)
    assert stats.spread["A→B"] == pytest.approx(0.02)
    assert stats.spread["B→A"] == pytest.approx(0.03)
    assert stats.mean["mix mean"] == pytest.approx((0.39345 + 0.395 + 0.385) / 3)
    assert stats.spread["mix mean"] == pytest.approx(0.395 - 0.385)
    assert stats.oof_mean == pytest.approx({"A→B": 0.41, "B→A": 0.38, "mix": 0.40})
    assert stats.oof_spread == pytest.approx({"A→B": 0.02, "B→A": 0.02, "mix": 0.02})
    assert stats.n == 3


def side(*rows: tuple[float, float, float, float]):
    """The group stats of seed repeats, one row of (A→B, B→A, mix→A, mix→B) per seed."""
    return group_stats([make_run(f"x-{k}", row, seed=k) for k, row in enumerate(rows)])


# Reference: mix means 0.390, 0.392, 0.388 (mean 0.390, spread 0.004); A→B 0.30, B→A 0.30.
REFERENCE = ((0.30, 0.30, 0.42, 0.36), (0.30, 0.30, 0.424, 0.36), (0.30, 0.30, 0.416, 0.36))


def test_a_mix_gain_above_the_larger_seed_spread_is_kept() -> None:
    # Candidate mix means 0.400, 0.402, 0.398: gain 0.010 > margin 0.004.
    cand = side((0.30, 0.30, 0.44, 0.36), (0.30, 0.30, 0.444, 0.36), (0.30, 0.30, 0.436, 0.36))
    decision = keep_decision(side(*REFERENCE), cand, tolerance=0.01, min_seeds=3)
    assert decision.verdict == "keep"
    assert decision.gain == pytest.approx(0.010)
    assert decision.margin == pytest.approx(0.004)


def test_a_mix_gain_within_the_seed_spread_is_dropped() -> None:
    # Candidate mix means 0.3925, 0.394, 0.391 (mean 0.3925, gain 0.0025 < margin 0.004).
    cand = side((0.30, 0.30, 0.425, 0.36), (0.30, 0.30, 0.428, 0.36), (0.30, 0.30, 0.422, 0.36))
    decision = keep_decision(side(*REFERENCE), cand, tolerance=0.01, min_seeds=3)
    assert decision.verdict == "drop"
    assert decision.gain == pytest.approx(0.0025)


def test_a_mix_gain_is_dropped_when_b_to_a_falls_by_more_than_the_tolerance() -> None:
    # Gain 0.010 as in the keep test, but B→A falls from 0.30 to 0.285.
    cand = side((0.30, 0.285, 0.44, 0.36), (0.30, 0.285, 0.444, 0.36), (0.30, 0.285, 0.436, 0.36))
    decision = keep_decision(side(*REFERENCE), cand, tolerance=0.01, min_seeds=3)
    assert decision.verdict == "drop"
    assert any("B→A" in reason for reason in decision.reasons)


@pytest.mark.parametrize("n_candidate", [1, 2])
def test_fewer_seeds_than_the_minimum_is_undecided_however_large_the_gain(n_candidate) -> None:
    cand = side(*[(0.30, 0.30, 0.50, 0.40)] * n_candidate)  # gain of about 0.06
    decision = keep_decision(side(*REFERENCE), cand, tolerance=0.01, min_seeds=3)
    assert decision.verdict == "undecided"
    assert any("3 seeds" in reason for reason in decision.reasons)


LIMITS = {"max_auroc": 0.95, "max_direction_gap": 0.15, "max_utility_jump": 0.05}


def test_no_flag_for_runs_within_the_limits() -> None:
    ref = [make_run("r-1", (0.30, 0.31, 0.42, 0.36), seed=0)]
    cand = [make_run("c-1", (0.32, 0.30, 0.43, 0.37), seed=0)]
    assert cross_run_flags(ref, cand, LIMITS) == []


def test_each_run_gets_the_runner_red_flags() -> None:
    cand = [make_run("c-1", (0.40, 0.31, 0.42, 0.36), seed=0, auroc=0.97)]
    flags = cross_run_flags([], cand, LIMITS)
    assert "c-1: red flag: A→B AUROC 0.970 > 0.95" in flags
    assert "c-1: red flag: A→B utility beats mix→B (mix trains on that hospital)" in flags


def test_a_utility_jump_from_the_reference_run_with_the_same_seed_is_flagged() -> None:
    ref = [
        make_run("r-1", (0.30, 0.30, 0.42, 0.36), seed=0),
        make_run("r-2", (0.30, 0.25, 0.42, 0.36), seed=1),
    ]
    # c-2 (seed 1) is 0.06 above r-2 on B→A; c-1 (seed 0) is within 0.05 of r-1.
    cand = [
        make_run("c-1", (0.30, 0.34, 0.42, 0.36), seed=0),
        make_run("c-2", (0.30, 0.31, 0.42, 0.36), seed=1),
    ]
    flags = cross_run_flags(ref, cand, LIMITS)
    assert flags == ["c-2: red flag: B→A utility +0.060 vs r-2 (same seed) moves more than 0.05"]


def _three(runs: Path, prefix: str, rows, sha: str = "abc1234") -> list[str]:
    for k, row in enumerate(rows):
        write_run(runs, f"{prefix}-{k}", row, seed=k, sha=sha)
    return [f"{prefix}-{k}" for k in range(len(rows))]


def test_the_cli_prints_the_tables_and_a_drop_decision(tmp_path: Path, capsys) -> None:
    ref = _three(tmp_path, "r", REFERENCE)
    cand = _three(
        tmp_path,
        "c",
        ((0.30, 0.30, 0.425, 0.36), (0.30, 0.30, 0.428, 0.36), (0.30, 0.30, 0.422, 0.36)),
    )
    assert compare(tmp_path, ref, cand) == 0
    out = capsys.readouterr().out
    assert "| r-0 | 0 | 0.3000 | 0.3000 | 0.4200 | 0.3600 | 0.3900 |" in out
    assert "| change | +0.0000 | +0.0000 | +0.0050 | +0.0000 | +0.0025 |" in out
    assert "**Decision: drop**" in out
    assert "mix mean +0.0025 <= margin 0.0040" in out
    assert "OOF" in out


def test_the_cli_says_undecided_for_a_single_seed(tmp_path: Path, capsys) -> None:
    ref = _three(tmp_path, "r", REFERENCE)
    cand = _three(tmp_path, "c", ((0.30, 0.30, 0.50, 0.40),))
    assert compare(tmp_path, ref, cand) == 0
    out = capsys.readouterr().out
    assert "**Decision: undecided**" in out
    assert "keep" not in out.split("**Decision:")[1].split("\n")[0]


def test_the_cli_warns_when_a_side_spans_several_commits(tmp_path: Path, capsys) -> None:
    ref = _three(tmp_path, "r", REFERENCE[:2])
    write_run(tmp_path, "r-2", REFERENCE[2], seed=2, sha="def5678")
    cand = _three(tmp_path, "c", REFERENCE)
    compare(tmp_path, [*ref, "r-2"], cand)
    assert (
        "warning: reference runs span commits: abc1234 (r-0, r-1), def5678 (r-2)"
        in capsys.readouterr().out
    )


def _with(**changes) -> dict:
    out = {k: dict(v) for k, v in FEATURES.items()}
    for section, blocks in changes.items():
        out[section] = {**out.get(section, {}), **blocks}
    return out


def test_seed_repeats_may_differ_in_blocks_that_are_switched_off(tmp_path: Path) -> None:
    # A later commit added a block that is off: the runs still compute the same features.
    write_run(tmp_path, "r-1", REFERENCE[0], seed=0)
    added = _with(signatures={"sig_vitals": {"enabled": False, "depth": 3}})
    write_run(tmp_path, "r-2", REFERENCE[1], seed=1, features=added)
    write_run(tmp_path, "c-1", REFERENCE[0])
    assert compare(tmp_path, ["r-1", "r-2"], ["c-1"]) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"features": _with(core={"sirs": {"enabled": False}})},  # an enabled block switched off
        {"features": _with(raw={"exclude": ["DBP", "SaO2"]})},
        {"columns": ("ICULOS", "HR")},
    ],
)
def test_a_seed_repeat_with_different_effective_features_is_refused(tmp_path, change) -> None:
    write_run(tmp_path, "r-1", REFERENCE[0], seed=0)
    write_run(tmp_path, "r-2", REFERENCE[1], seed=1, **change)
    write_run(tmp_path, "c-1", REFERENCE[0])
    with pytest.raises(CompareError, match="r-2"):
        compare(tmp_path, ["r-1", "r-2"], ["c-1"])


def test_runs_scored_by_different_scorer_files_are_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "r-1", REFERENCE[0])
    write_run(tmp_path, "c-1", REFERENCE[0], meta={"scorer_sha256": "ffff"})
    with pytest.raises(CompareError, match="scorer"):
        compare(tmp_path, ["r-1"], ["c-1"])


def test_a_partial_folder_or_a_folder_of_another_run_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "r-1", REFERENCE[0])
    write_run(tmp_path, "c-1.partial", REFERENCE[0], meta={"id": "c-1"})
    write_run(tmp_path, "c-2", REFERENCE[0], meta={"id": "c-9"})
    with pytest.raises(CompareError, match="partial"):
        compare(tmp_path, ["r-1"], ["c-1.partial"])
    with pytest.raises(CompareError, match="c-9"):
        compare(tmp_path, ["r-1"], ["c-2"])


def test_no_seed_spread_means_no_margin_so_no_verdict() -> None:
    # Seeds that change nothing (e.g. no masking, a seedless model): a margin of 0 would keep
    # any gain at all, so the rule can't decide.
    flat = side(*[(0.30, 0.30, 0.42, 0.36)] * 3)
    better = side(*[(0.30, 0.30, 0.43, 0.36)] * 3)
    decision = keep_decision(flat, better, tolerance=0.01, min_seeds=3)
    assert decision.verdict == "undecided"
    assert any("spread is 0" in reason for reason in decision.reasons)


def test_the_cli_warns_about_uncommitted_code_and_different_seed_sets(tmp_path, capsys) -> None:
    write_run(tmp_path, "r-0", REFERENCE[0], seed=0)
    write_run(tmp_path, "c-0", REFERENCE[0], seed=5, meta={"dirty": True})
    compare(tmp_path, ["r-0"], ["c-0"])
    out = capsys.readouterr().out
    assert "warning: c-0 ran on uncommitted code (dirty tree)" in out
    assert "warning: the sides use different seeds ([0] vs [5])" in out


def test_a_run_id_given_twice_is_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "r-1", REFERENCE[0])
    write_run(tmp_path, "c-1", REFERENCE[0])
    with pytest.raises(CompareError, match="r-1"):
        compare(tmp_path, ["r-1"], ["r-1"])
    with pytest.raises(CompareError, match="c-1"):
        compare(tmp_path, ["r-1"], ["c-1", "c-1"])


def test_a_mix_gain_is_dropped_when_a_to_b_falls_by_more_than_the_tolerance() -> None:
    # Gain 0.010 as in the keep test, but A→B falls from 0.30 to 0.289.
    cand = side((0.289, 0.30, 0.44, 0.36), (0.289, 0.30, 0.444, 0.36), (0.289, 0.30, 0.436, 0.36))
    decision = keep_decision(side(*REFERENCE), cand, tolerance=0.01, min_seeds=3)
    assert decision.verdict == "drop"
    assert any(reason.startswith("A→B -0.0110") for reason in decision.reasons)


def test_the_commit_warning_names_the_runs_of_each_commit(tmp_path: Path, capsys) -> None:
    write_run(tmp_path, "r-0", REFERENCE[0], seed=0, sha="abc1234")
    write_run(tmp_path, "r-1", REFERENCE[1], seed=1, sha="def5678")
    write_run(tmp_path, "c-0", REFERENCE[0], seed=0)
    write_run(tmp_path, "c-1", REFERENCE[1], seed=1)
    compare(tmp_path, ["r-0", "r-1"], ["c-0", "c-1"])
    out = capsys.readouterr().out
    assert "warning: reference runs span commits: abc1234 (r-0), def5678 (r-1)" in out
