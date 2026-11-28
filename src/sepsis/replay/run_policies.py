"""Replay the baseline policies under every setup of a finished run and log them (M5-2).

    uv run python -m sepsis.replay.run_policies configs/replay_b2.yaml [--allow-dirty]

The config names a finished source run (`experiments/runs/<source_run>/`) and a new run id per
policy. Each setup's models are refitted once (`replay_models`, checked against the source run) and
shared by all policies. Every replay is scored with the official scorer and written to
`experiments/runs/<id>.partial/`.

Then the M5 check, which is hard: ALL must reproduce the source run's scored predictions and
utilities exactly, NONE must request nothing, and AS_RECORDED must predict exactly like ALL. If
anything fails, the CLI exits non-zero, the folders stay `.partial` and no LOG row is written.
Otherwise the folders are renamed and one LOG row per policy is appended (ADR-0008, ADR-0009).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sepsis.action.baselines import AllPolicy, NonePolicy
from sepsis.action.policy import Policy
from sepsis.config import EVAL_CONFIG, load_yaml
from sepsis.eval import run_experiment
from sepsis.eval.refit import load_run
from sepsis.eval.run_experiment import (
    LOG_PATH,
    RUNS_DIR,
    GitCheck,
    RunRefusedError,
    append_log_row,
    git_state,
    logged_ids,
)
from sepsis.eval.runner import PARTIAL_NOTE, RESULT_NAMES, _cell
from sepsis.eval.scorer import scorer_sha256
from sepsis.eval.tables import KEYS
from sepsis.replay.baselines import AsRecordedPolicy
from sepsis.replay.engine import ReplayResult
from sepsis.replay.models import replay_models, replay_setup, score_replay
from sepsis.replay.reveal import REVEALED

# Policy name -> factory taking the panels config.
POLICIES: dict[str, Callable[[Mapping[str, Any]], Policy]] = {
    "NONE": lambda panels: NonePolicy(),
    "ALL": lambda panels: AllPolicy(),
    "AS_RECORDED": lambda panels: AsRecordedPolicy(panels),
}
CONFIG_KEYS = {"source_run", "policies", "note"}
SCORE_KEYS = ("A_to_B", "B_to_A", "mix_A", "mix_B")  # the LOG's four results, in order
ATOL = 1e-9


def load_replay_config(path: Path) -> dict[str, Any]:
    """The replay config, checked: `source_run`, `policies` (name -> new id) and `note`."""
    cfg = load_yaml(path)
    if not isinstance(cfg, dict) or set(cfg) != CONFIG_KEYS:
        raise ValueError(f"{path}: the replay config needs exactly the keys {sorted(CONFIG_KEYS)}")
    policies = cfg["policies"]
    if not isinstance(policies, dict) or not policies:
        raise ValueError(f"{path}: `policies` maps policy names to new run ids")
    ids = [str(i) for i in policies.values()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: every policy needs its own run id")
    cfg["note"] = cfg["note"] or ""
    return cfg


def _same_predictions(mine: pd.DataFrame, reference: pd.DataFrame) -> str | None:
    """None if both tables have the same rows, scores (within ATOL) and alarms; else why not."""
    a = mine.sort_values(KEYS).reset_index(drop=True)
    b = reference.sort_values(KEYS).reset_index(drop=True)
    if not a[KEYS].astype(str).equals(b[KEYS].astype(str)):
        return "other (patient_id, hour) rows"
    if not np.allclose(a["score"].to_numpy(float), b["score"].to_numpy(float), rtol=0, atol=ATOL):
        return "other scores"
    if not np.array_equal(a["alarm"].to_numpy(int), b["alarm"].to_numpy(int)):
        return "other alarms"
    return None


def m5_check(
    results: Mapping[str, Mapping[str, ReplayResult]],
    scores: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_dir: Path,
) -> list[str]:
    """The M5 check (ADR-0009 amendment); an empty list means it passed.

    `results[policy][setup]` and `scores[policy][key]` come from this CLI.
    """
    failures = []
    source_scores = json.loads((source_dir / "scores.json").read_text(encoding="utf-8"))
    if "ALL" in results:
        for setup, result in results["ALL"].items():
            stored = pd.read_parquet(source_dir / setup / "predictions.parquet")
            stored = stored[stored["kind"] == "scored"]
            why = _same_predictions(result.predictions, stored)
            if why:
                failures.append(f"ALL {setup}: {why} than the source run's scored predictions")
        for key, value in scores["ALL"].items():
            expected = source_scores[key]["utility"]
            if value["utility"] != expected:
                failures.append(f"ALL {key}: utility {value['utility']} != source {expected}")
    if "NONE" in results:
        for setup, result in results["NONE"].items():
            if len(result.requests):
                failures.append(f"NONE {setup}: {len(result.requests)} requests, expected 0")
    if "AS_RECORDED" in results:
        for setup, result in results["AS_RECORDED"].items():
            why = _same_predictions(result.predictions, results["ALL"][setup].predictions)
            if why:
                failures.append(f"AS_RECORDED {setup}: {why} than ALL")
            wasted = int((result.requests["status"] != REVEALED).sum())
            if wasted:
                failures.append(
                    f"AS_RECORDED {setup}: {wasted} requests revealed nothing (expected 0)"
                )
    return failures


def _fmt(value: float | None, spec: str) -> str:
    return "—" if value is None else format(value, spec)


def replay_log_row(
    run_id: str,
    *,
    date: str,
    sha: str,
    config_path: str,
    scores: Mapping[str, Mapping[str, Any]],
    note: str,
    source_run: str,
) -> str:
    """One `experiments/LOG.md` row for one policy's replay (the header in that file)."""
    found = [scores.get(key) for key in SCORE_KEYS]
    utilities = [_fmt(None if v is None else v["utility"], ".3f") for v in found]
    run = [v for v in found if v is not None]
    auroc = f"{np.mean([v['auroc'] for v in run]):.3f}"
    partial = len(run) < len(found)
    if partial:
        auroc += f" ({len(run)} of {len(found)})"
    burden = f"{np.mean([v['labs_per_patient_day'] for v in run]):.1f}"
    per_setup = ", ".join(
        f"{name} {_fmt(None if v is None else v['labs_per_patient_day'], '.1f')}"
        for name, v in zip(RESULT_NAMES, found, strict=True)
    )
    unavailable = ", ".join(
        f"{name} {_fmt(None if v is None else v['unavailable_rate'], '.3f')}"
        for name, v in zip(RESULT_NAMES, found, strict=True)
    )
    parts = [note] if note else []
    parts += [PARTIAL_NOTE] if partial else []
    parts += [
        f"source {source_run}",
        f"labs/pt-day {per_setup}",
        f"unavailable rate {unavailable}",
    ]
    cells = [run_id, date, sha, config_path, "dev", *utilities, auroc, burden, "; ".join(parts)]
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")


def main(
    argv: list[str] | None = None, *, root: Path = Path("."), git_check: GitCheck = git_state
) -> int:
    """Replay, score, check and log. The keyword arguments exist for tests only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree (-dirty)")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    root = Path(root)

    # 1. Refuse runs that must not start, before any computation.
    cfg = load_replay_config(args.config)
    unknown = sorted(set(cfg["policies"]) - set(POLICIES))
    if unknown:
        raise RunRefusedError(f"unknown policies {unknown}; known: {sorted(POLICIES)}")
    if "AS_RECORDED" in cfg["policies"] and "ALL" not in cfg["policies"]:
        raise RunRefusedError("the M5 check compares AS_RECORDED with ALL; run both")
    log_path = root / LOG_PATH
    ids = {name: str(run_id) for name, run_id in cfg["policies"].items()}
    logged = sorted(set(ids.values()) & logged_ids(log_path))
    if logged:
        raise RunRefusedError(f"ids {logged} are already in {log_path}; never reuse an id")
    for run_id in ids.values():
        if (root / RUNS_DIR / run_id).exists():
            raise RunRefusedError(f"{root / RUNS_DIR / run_id} exists; never overwritten")
    source_dir = root / RUNS_DIR / str(cfg["source_run"])
    if not source_dir.is_dir():
        raise RunRefusedError(f"source run {source_dir} not found")
    if str(cfg["source_run"]) not in logged_ids(log_path):
        raise RunRefusedError(f"source run {cfg['source_run']!r} is not in {log_path}")
    sha, dirty = git_check()
    if dirty and not args.allow_dirty:
        raise RunRefusedError("the git tree is dirty; commit first or pass --allow-dirty")
    sha = f"{sha}-dirty" if dirty else sha

    # 2. Dev patients and the source run's models, refitted once per setup.
    source_cfg, _ = load_run(source_dir)
    dev = run_experiment.load_dev_tables(source_cfg)
    panels = dev.settings.panels
    results: dict[str, dict[str, ReplayResult]] = {name: {} for name in ids}
    seconds: dict[str, float] = dict.fromkeys(ids, 0.0)
    for setup in source_cfg["split"]["setups"]:
        print(f"[replay] {setup}: refitting the source run's models ...", flush=True)
        models = replay_models(source_dir, setup, dev=dev)
        for name in ids:
            print(f"[replay] {setup}: {name} ...", flush=True)
            t0 = time.perf_counter()
            results[name][setup] = replay_setup(
                source_dir, setup, POLICIES[name](panels), dev=dev, models=models
            )
            seconds[name] += time.perf_counter() - t0

    # 3. Score and write each policy's run folder as <id>.partial.
    scores = {
        name: {
            key: value
            for s, r in by_setup.items()
            for key, value in score_replay(r, s, dev).items()
        }
        for name, by_setup in results.items()
    }
    eval_config = load_yaml(EVAL_CONFIG)
    today = date.today().isoformat()
    partials = {}
    for name, run_id in ids.items():
        partial = root / RUNS_DIR / f"{run_id}.partial"
        if partial.exists():
            shutil.rmtree(partial)  # left by a crashed or refused run of the same id
        partial.mkdir(parents=True)
        (partial / "config.yaml").write_text(
            Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
        )
        for setup, result in results[name].items():
            (partial / setup).mkdir()
            result.predictions.to_parquet(partial / setup / "predictions.parquet", index=False)
            result.requests.to_parquet(partial / setup / "requests.parquet", index=False)
        _write_json(partial / "scores.json", scores[name])
        _write_json(
            partial / "meta.json",
            {
                "id": run_id,
                "policy": name,
                "source_run": cfg["source_run"],
                "sha": sha,
                "dirty": dirty,
                "date": today,
                "command": " ".join([Path(sys.argv[0]).name, *(argv or sys.argv[1:])]),
                "scorer_sha256": scorer_sha256(Path(eval_config["scorer"]["path"])),
                "versions": run_experiment._versions(),
                "replay_seconds": round(seconds[name], 1),
                "wall_seconds": round(time.perf_counter() - started, 1),
            },
        )
        partials[name] = partial
    print(json.dumps(scores, indent=1))

    # 4. The hard M5 check: on failure, no LOG row and the folders stay .partial.
    failures = m5_check(results, scores, source_dir)
    if failures:
        for failure in failures:
            print(f"M5 check failed: {failure}", file=sys.stderr)
        return 1

    # 5. Finished folders, then one LOG row each.
    rows = {
        name: replay_log_row(
            run_id,
            date=today,
            sha=sha,
            config_path=Path(args.config).as_posix(),
            scores=scores[name],
            note=cfg["note"],
            source_run=str(cfg["source_run"]),
        )
        for name, run_id in ids.items()
    }
    for name, run_id in ids.items():
        partials[name].rename(root / RUNS_DIR / run_id)
    pending = dict(ids)
    try:
        for name in ids:
            append_log_row(log_path, rows[name])
            pending.pop(name)
            print(rows[name])
    except BaseException:
        for run_id in pending.values():  # an id is in the LOG only when its folder is complete
            (root / RUNS_DIR / run_id).rename(root / RUNS_DIR / f"{run_id}.partial")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
