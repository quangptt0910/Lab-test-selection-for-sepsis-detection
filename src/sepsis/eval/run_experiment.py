"""Run one experiment config under the setups of ADR-0004 and log it (ADR-0008).

    uv run python -m sepsis.eval.run_experiment configs/<name>.yaml [--allow-dirty] [--no-cache]

All file I/O of an experiment lives here: read and check the config, load dev patients, build or
load the cached model tables, run each setup (`sepsis.eval.runner`), write the run folder
`experiments/runs/<id>/`, and append one row to `experiments/LOG.md`. A-test and B-test are never
loaded: patients come only from `load_patient_hours` without `final_eval`.

The run folder is written as `<id>.partial`, renamed to `<id>`, and only then is the LOG row
appended (if that fails, the folder goes back to `<id>.partial`). So an id is in the LOG only when
its run folder is complete, and a crashed run can be rerun under the same id.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sepsis.config import (
    CLEANING_CONFIG,
    DATA_CONFIG,
    EVAL_CONFIG,
    FEATURES_CONFIG,
    PROTOCOL_CONFIG,
    load_yaml,
)
from sepsis.data.panels import PANELS_CONFIG, PREPROCESSING_CONFIG
from sepsis.data.splits import load_patient_hours, load_project_split
from sepsis.eval.experiment_config import (
    check_model_type,
    model_factory,
    validate_experiment_config,
)
from sepsis.eval.runner import SetupResult, log_row, red_flags, run_setup
from sepsis.eval.scorer import scorer_sha256
from sepsis.eval.tables import KEYS, Tables, TableSettings, load_or_build, read_cached
from sepsis.features.builder import with_blocks
from sepsis.labels.utility import LABELS_CONFIG, UtilityConstants

LOG_PATH = Path("experiments/LOG.md")
RUNS_DIR = Path("experiments/runs")
# The LOG is appended by every run, so it does not make the tree dirty for the next run.
DIRTY_IGNORED = {"experiments/LOG.md"}

GitCheck = Callable[[], tuple[str, bool]]


class RunRefusedError(RuntimeError):
    """Raised when a run must not start (duplicate id, dirty tree, finished run folder)."""


def git_state() -> tuple[str, bool]:
    """`(short SHA, dirty)`. Dirty = tracked files changed, except `experiments/LOG.md`."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    changed = {line[3:].strip() for line in status if line.strip()}
    return sha, bool(changed - DIRTY_IGNORED)


def logged_ids(log_path: Path) -> set[str]:
    """Experiment ids already in the LOG (the first cell of every table row)."""
    ids = set()
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells and cells[0] not in ("id", "") and set(cells[0]) != {"-"}:
            ids.add(cells[0])
    return ids


@dataclass(frozen=True)
class DevData:
    """The dev patients of one experiment: patient-hours, split and model tables, aligned."""

    tables: Tables
    hours: pd.DataFrame  # every dev patient-hour, sorted by KEYS (the tables' row order)
    split: pd.DataFrame
    cache: dict[str, Any]  # train_key, eval_key, hit
    cache_dir: Path | None
    settings: TableSettings

    @property
    def labels(self) -> pd.DataFrame:
        """`patient_id, hour, hospital, SepsisLabel`, aligned with the tables."""
        return self.hours[[*KEYS, "hospital", "SepsisLabel"]]


def load_dev_tables(
    cfg: Mapping[str, Any], *, no_cache: bool = False, lab_masking_p: float | None = None
) -> DevData:
    """Dev patients only (ADR-0004), then the model tables of `cfg` (cached unless `no_cache`).

    `lab_masking_p` overrides the config's value (the ordering diagnostic's full-information
    setting).
    """
    split = load_project_split()
    hours = load_patient_hours(["A", "B"], split["patient_id"])
    hours = hours.sort_values(KEYS, kind="stable").reset_index(drop=True)
    masking = cfg["features"]["lab_masking_p"] if lab_masking_p is None else lab_masking_p
    settings = TableSettings(
        blocks=tuple(cfg["features"]["hand_crafted"]),
        lab_masking_p=float(masking),
        seed=int(cfg["seed"]),
        cleaning=load_yaml(CLEANING_CONFIG),
        panels=load_yaml(PANELS_CONFIG),
        features=load_yaml(FEATURES_CONFIG),
        labels=UtilityConstants.from_config(load_yaml(LABELS_CONFIG)),
    )
    cache_dir = None if no_cache else Path(load_yaml(DATA_CONFIG)["cache_dir"]) / "features"
    fingerprint = str(load_yaml(PROTOCOL_CONFIG)["split"]["fingerprint"])
    tables, cache = load_or_build(
        hours, settings, split_fingerprint=fingerprint, cache_dir=cache_dir
    )
    return DevData(tables, hours, split, cache, cache_dir, settings)


@dataclass(frozen=True)
class SetupJob:
    """One setup for a worker process: everything but the tables, which it reads from the cache
    by key (the same files the parent just checked), so no table is copied between processes."""

    setup: str
    cfg: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    labels: pd.DataFrame
    split: pd.DataFrame
    cache_dir: Path
    cache: Mapping[str, Any]
    grid: Mapping[str, float]
    score_jobs: int


def run_setup_job(job: SetupJob) -> SetupResult:
    """Worker entry point: the same `run_setup` call the serial path makes."""
    print(f"[{job.cfg['id']}] {job.setup} ...", flush=True)
    return run_setup(
        read_cached(job.cache_dir, job.cache),
        job.labels,
        job.split,
        job.setup,
        new_model=model_factory(job.cfg, job.preprocessing),
        target=job.cfg["target"],
        grid=job.grid,
        score_jobs=job.score_jobs,
    )


def append_log_row(log_path: Path, row: str) -> None:
    """Append one row to the LOG; past rows are never edited."""
    text = Path(log_path).read_text(encoding="utf-8")
    newline = "" if text.endswith("\n") else "\n"
    Path(log_path).write_text(text + newline + row + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")


def _versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "lightgbm", "scikit-learn", "pyarrow")
    return {"python": platform.python_version(), **{p: version(p) for p in packages}}


def main(
    argv: list[str] | None = None,
    *,
    root: Path = Path("."),
    git_check: GitCheck = git_state,
    grid: Mapping[str, float] | None = None,
) -> int:
    """Run the experiment. The keyword arguments exist for tests only; never pass them elsewhere.

    Data always comes from the project's configs (`DATA_CONFIG`, `PROTOCOL_CONFIG`) through the
    split loaders' defaults; no caller can point a run at another split.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree (-dirty)")
    parser.add_argument("--no-cache", action="store_true", help="rebuild the model tables")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="setups run in parallel processes (default: parallel.setup_jobs in configs/eval.yaml;"
        " 1 runs them in this process)",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    root = Path(root)

    # 1. Check the config and refuse runs that must not start, before any computation.
    features_config = load_yaml(FEATURES_CONFIG)
    cfg = validate_experiment_config(load_yaml(args.config), features_config)
    check_model_type(cfg)
    log_path, run_dir = root / LOG_PATH, root / RUNS_DIR / cfg["id"]
    if cfg["id"] in logged_ids(log_path):
        raise RunRefusedError(f"id {cfg['id']!r} is already in {log_path}; never reuse an id")
    if run_dir.exists():
        raise RunRefusedError(f"{run_dir} exists; a finished run is never overwritten")
    sha, dirty = git_check()
    if dirty and not args.allow_dirty:
        raise RunRefusedError("the git tree is dirty; commit first or pass --allow-dirty")
    sha = f"{sha}-dirty" if dirty else sha

    # 2. Dev patients only (ADR-0004): the split and patient-hours never include test patients.
    # 3. Model tables, built once for all dev patients (cached unless --no-cache).
    dev = load_dev_tables(cfg, no_cache=args.no_cache)
    tables, settings = dev.tables, dev.settings

    # 4. Run each setup.
    eval_config = load_yaml(EVAL_CONFIG)
    limits = load_yaml(PROTOCOL_CONFIG)["red_flags"]
    preprocessing = load_yaml(PREPROCESSING_CONFIG)["b1"]  # B1's fills and log labs (ADR-0005)
    grid = dict(eval_config["threshold"]["grid"] if grid is None else grid)
    parallel = eval_config["parallel"]
    jobs = int(parallel["setup_jobs"] if args.jobs is None else args.jobs)
    setups = list(cfg["split"]["setups"])
    results = {}
    if jobs > 1 and len(setups) > 1 and dev.cache_dir is not None:
        # Speed only: each worker makes the serial path's run_setup call on the same cached
        # tables, so the results are the same (tests/test_runner.py checks this).
        common = dict(
            cfg=cfg,
            preprocessing=preprocessing,
            labels=dev.labels,
            split=dev.split,
            cache_dir=dev.cache_dir,
            cache=dev.cache,
            grid=grid,
            score_jobs=int(parallel["score_jobs"]),
        )
        with ProcessPoolExecutor(max_workers=min(jobs, len(setups))) as pool:
            futures = {s: pool.submit(run_setup_job, SetupJob(setup=s, **common)) for s in setups}
            results = {s: future.result() for s, future in futures.items()}
    else:
        for setup in setups:
            print(f"[{cfg['id']}] {setup} ...", flush=True)
            results[setup] = run_setup(
                tables,
                dev.labels,
                dev.split,
                setup,
                new_model=model_factory(cfg, preprocessing),
                target=cfg["target"],
                grid=grid,
                score_jobs=int(parallel["score_jobs"]),
            )

    # 5. Run folder, written as <id>.partial.
    partial = run_dir.with_name(run_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)  # left by a crashed run of the same id
    partial.mkdir(parents=True)
    (partial / "config.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8"
    )
    resolved = {
        "features.yaml": with_blocks(features_config, list(settings.blocks)),
        "panels.yaml": settings.panels,
        "cleaning.yaml": settings.cleaning,
        "labels.yaml": asdict(settings.labels),
        "eval_grid.yaml": grid,
        "red_flags.yaml": limits,
        "preprocessing.yaml": preprocessing,
    }
    (partial / "resolved").mkdir()
    for name, value in resolved.items():
        (partial / "resolved" / name).write_text(yaml.safe_dump(value), encoding="utf-8")
    _write_json(partial / "feature_columns.json", tables.feature_columns)
    scores = {}
    for setup, result in results.items():
        folder = partial / setup
        folder.mkdir()
        _write_json(folder / "thresholds.json", result.thresholds)
        result.curves.to_csv(folder / "threshold_curve.csv", index=False)
        result.folds.to_csv(folder / "folds.csv", index=False)
        result.predictions.to_parquet(folder / "predictions.parquet", index=False)
        for name, value in result.scores.items():
            key = setup if name == "all" else f"{setup}_{name}"
            scores[key] = asdict(value)
    _write_json(partial / "scores.json", scores)
    _write_json(
        partial / "meta.json",
        {
            "id": cfg["id"],
            "sha": sha,
            "dirty": dirty,
            "date": date.today().isoformat(),
            "command": " ".join([Path(sys.argv[0]).name, *(argv or sys.argv[1:])]),
            "config": str(args.config),
            "seed": cfg["seed"],
            "cache": dev.cache,
            "setup_jobs": jobs,
            "scorer_sha256": scorer_sha256(Path(eval_config["scorer"]["path"])),
            "versions": _versions(),
            "setup_seconds": {s: r.seconds for s, r in results.items()},
            "wall_seconds": round(time.perf_counter() - started, 1),
        },
    )

    # 6. The finished run folder, then the LOG row.
    flags = red_flags(results, limits)
    row = log_row(
        cfg,
        Path(args.config).as_posix(),
        results,
        sha=sha,
        date=date.today().isoformat(),
        flags=flags,
    )
    partial.rename(run_dir)
    try:
        append_log_row(log_path, row)
    except BaseException:
        run_dir.rename(partial)  # an id is in the LOG only when its run folder is complete
        raise

    print(json.dumps(scores, indent=1))
    for flag in flags:
        print(flag)
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
