"""Build the VoI training targets of a finished run (M6-1, ADR-0010).

    uv run python -m sepsis.labels.build_voi configs/voi.yaml [--allow-dirty]

Every setup of the source run is built on its training side only. Each training-side patient is
scored by the run's inner (out-of-fold) model that held it out, refit here and checked against the
run's stored out-of-fold predictions, with the setup's frozen threshold (mix: per outer fold, so
4 rows per patient). Features of the counterfactual views are built once per chunk of patients
and shared by every model.

Writes `<cache_dir>/voi/<key>/{A_to_B,B_to_A,mix}.parquet` and `meta.json` (via `<key>.partial`),
and prints a summary per setup and panel. A dataset, not a scored experiment: no LOG row. An
existing key is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sepsis
from sepsis.config import load_yaml
from sepsis.data.clean import clean
from sepsis.data.panels import PANEL_NAMES
from sepsis.eval import run_experiment
from sepsis.eval.refit import RefitTolerance, check_oof_against_run, load_run, refit_inner
from sepsis.eval.run_experiment import (
    LOG_PATH,
    RUNS_DIR,
    GitCheck,
    RunRefusedError,
    git_state,
    logged_ids,
)
from sepsis.eval.runner import plan_setup
from sepsis.eval.tables import KEYS
from sepsis.features.builder import with_blocks
from sepsis.labels.voi import (
    DECISION_KEYS,
    TARGET_STATS,
    availability,
    counterfactual_features,
    decision_rows,
    gains,
    load_voi_config,
    panel_targets,
)
from sepsis.risk.base import RiskModel

# Code that shapes the targets beyond the feature tables (whose key already covers data/,
# features/ and labels/).
KEY_SOURCES = ("eval/refit.py", "eval/runner.py", "eval/experiment_config.py", "risk")
# The source run's outputs the targets depend on: its frozen thresholds and the out-of-fold
# scores the refits are checked against.
RUN_OUTPUTS = ("thresholds.json", "predictions.parquet")
PACKAGE = Path(sepsis.__file__).resolve().parent


@dataclass(frozen=True)
class Scorer:
    """One inner model of the source run, its frozen threshold and the patients it held out."""

    setup: str
    outer_fold: int | None
    inner_fold: int
    model: RiskModel
    threshold: float
    ids: frozenset[str]


def _fold(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[call-overload]


def voi_scorers(
    run_dir: Path,
    cfg: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None,
    dev: run_experiment.DevData,
    tolerance: RefitTolerance,
) -> tuple[list[Scorer], list[dict[str, Any]]]:
    """Every inner model of every setup of the run, refit and checked against its stored
    out-of-fold predictions (within `tolerance`, alarms identical), with the frozen threshold of
    its plan. Also returns the check's report per model."""
    scorers, reports = [], []
    for setup in cfg["split"]["setups"]:
        entries = json.loads((run_dir / setup / "thresholds.json").read_text(encoding="utf-8"))
        frozen = {_fold(e["outer_fold"]): float(e["threshold"]) for e in entries}
        for plan in plan_setup(dev.split, setup):
            for fold, _fit_ids, oof_ids in plan.inner:
                refit = refit_inner(cfg, preprocessing, dev.tables, plan, fold)
                threshold = frozen[_fold(plan.outer_fold)]
                report = check_oof_against_run(  # raises unless it is the run's model
                    refit, run_dir, threshold=threshold, tolerance=tolerance
                )
                reports.append(
                    {"setup": setup, "outer_fold": _fold(plan.outer_fold), "inner_fold": int(fold)}
                    | report
                )
                scorers.append(
                    Scorer(
                        setup=setup,
                        outer_fold=_fold(plan.outer_fold),
                        inner_fold=int(fold),
                        model=refit.model,
                        threshold=threshold,
                        ids=frozenset(oof_ids),
                    )
                )
    return scorers, reports


def voi_key(
    raw: Mapping[str, Any], source_cfg: Mapping[str, Any], eval_key: str, run_dir: Path
) -> str:
    """Hash of the VoI config, the source run's config and outputs (thresholds, out-of-fold
    scores), its feature tables, and the refit and model code."""
    code = hashlib.sha256()
    for entry in KEY_SOURCES:
        path = PACKAGE / entry
        for file in sorted(path.rglob("*.py")) if path.is_dir() else [path]:
            code.update(file.relative_to(PACKAGE).as_posix().encode())
            code.update(file.read_bytes().replace(b"\r\n", b"\n"))
    outputs = hashlib.sha256()
    for setup in source_cfg["split"]["setups"]:
        for name in RUN_OUTPUTS:
            outputs.update(f"{setup}/{name}".encode())
            outputs.update((Path(run_dir) / setup / name).read_bytes())
    payload = {
        "voi": dict(raw),
        "source": dict(source_cfg),
        "outputs": outputs.hexdigest(),
        "tables": eval_key,
        "code": code.hexdigest(),
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return "voi-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _no_gains(horizons: Sequence[int]) -> pd.DataFrame:
    columns = [*DECISION_KEYS, *(f"gain_h{h}" for h in horizons), *TARGET_STATS]
    return pd.DataFrame(
        {c: pd.Series(dtype=object if c == "patient_id" else float) for c in columns}
    )


def _tag(targets: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    return targets.assign(
        setup=scorer.setup,
        outer_fold=pd.array([scorer.outer_fold] * len(targets), dtype="Int64"),
        inner_fold=scorer.inner_fold,
    )


def build_targets(
    record: pd.DataFrame,
    utility: pd.DataFrame,
    scorers: Sequence[Scorer],
    *,
    feature_columns: Sequence[str],
    features_config: Mapping[str, Any],
    panels_config: Mapping[str, Any],
    every: int,
    seed: int,
    horizons: Sequence[int],
) -> dict[str, pd.DataFrame]:
    """The target table of every setup for the patients in `record` (cleaned, whole records).

    `utility` holds `patient_id, hour, utility_diff` for the same rows.
    """
    ids = sorted(set().union(*(s.ids for s in scorers)) & set(record["patient_id"]))
    record = record[record["patient_id"].isin(ids)]
    decisions = decision_rows(record[KEYS], every, seed)
    available = availability(record, panels_config, decisions)
    ud = utility.set_index(KEYS)["utility_diff"]
    parts: dict[str, list[pd.DataFrame]] = {s.setup: [] for s in scorers}
    for panel in PANEL_NAMES:
        mine = available[(available["panel"] == panel) & (available["available"] == 1)]
        if mine.empty:  # nothing to reveal in this chunk: every row unavailable, no gain
            for scorer in scorers:
                ours = available[available["patient_id"].isin(scorer.ids)]
                if not ours.empty:
                    targets = panel_targets(ours, _no_gains(horizons), panel, horizons)
                    parts[scorer.setup].append(_tag(targets, scorer))
            continue
        rows = counterfactual_features(
            record,
            mine[DECISION_KEYS],
            panel,
            horizon=max(horizons),
            panels_config=panels_config,
            features_config=features_config,
        )
        index = pd.MultiIndex.from_frame(rows.keys[["patient_id", "hour"]])
        values = ud.reindex(index).to_numpy(dtype=float)
        X_with = rows.with_X[list(feature_columns)]
        X_without = rows.without_X[list(feature_columns)]
        for scorer in scorers:
            held_out = rows.keys["patient_id"].isin(scorer.ids).to_numpy()
            ours = available[available["patient_id"].isin(scorer.ids)]
            if ours.empty:
                continue
            found = gains(
                rows.keys[held_out].reset_index(drop=True),
                scorer.model.predict(X_with[held_out]) if held_out.any() else np.empty(0),
                scorer.model.predict(X_without[held_out]) if held_out.any() else np.empty(0),
                values[held_out],
                threshold=scorer.threshold,
                horizons=horizons,
            )
            parts[scorer.setup].append(_tag(panel_targets(ours, found, panel, horizons), scorer))
    out = {}
    for setup, frames in parts.items():
        if not frames:  # no decision row of this setup's patients in this chunk
            continue
        table = pd.concat(frames, ignore_index=True)
        front = ["setup", "outer_fold", "inner_fold"]
        table = table[[*front, *(c for c in table.columns if c not in front)]]
        out[setup] = table.sort_values(
            ["outer_fold", "patient_id", "t", "panel"], kind="stable"
        ).reset_index(drop=True)
    return out


def summary(tables: Mapping[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Per setup and panel: decision rows, available share, share of available rows with a
    non-zero gain, mean gain and mean |Δscore| at t+1 where available."""
    gain = f"gain_h{horizon}"
    rows = []
    for setup, table in tables.items():
        for panel, part in table.groupby("panel", sort=False):
            hit = part[part["available"] == 1]
            rows.append(
                {
                    "setup": setup,
                    "panel": panel,
                    "rows": len(part),
                    "available": round(float(part["available"].mean()), 4),
                    f"{gain}!=0": round(float((hit[gain] != 0).mean()), 4) if len(hit) else None,
                    f"mean {gain}": round(float(hit[gain].mean()), 5) if len(hit) else None,
                    "mean dscore_t1": round(float(hit["dscore_t1"].mean()), 5)
                    if len(hit)
                    else None,
                }
            )
    return pd.DataFrame(rows)


def main(
    argv: list[str] | None = None, *, root: Path = Path("."), git_check: GitCheck = git_state
) -> int:
    """Refit, build, write and summarise. The keyword arguments exist for tests only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("--allow-dirty", action="store_true", help="run on a dirty tree (-dirty)")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    root = Path(root)

    # 1. Refuse before any computation.
    raw = load_yaml(args.config)
    cfg = load_voi_config(raw)
    log_path = root / LOG_PATH
    source_dir = root / RUNS_DIR / cfg.source_run
    if not source_dir.is_dir():
        raise RunRefusedError(f"source run {source_dir} not found")
    if cfg.source_run not in logged_ids(log_path):
        raise RunRefusedError(f"source run {cfg.source_run!r} is not in {log_path}")
    sha, dirty = git_check()
    if dirty and not args.allow_dirty:
        raise RunRefusedError("the git tree is dirty; commit first or pass --allow-dirty")

    # 2. Dev patients; reuse finished targets.
    source_cfg, preprocessing = load_run(source_dir)
    dev = run_experiment.load_dev_tables(source_cfg)
    if dev.cache_dir is None:
        raise RunRefusedError("the VoI targets are written next to the feature cache")
    key = voi_key(raw, source_cfg, dev.cache["eval_key"], source_dir)
    out_dir = Path(dev.cache_dir).parent / "voi" / key
    if (out_dir / "meta.json").exists():
        print(f"[voi] cached: {out_dir}", flush=True)
        tables = {
            s: pd.read_parquet(out_dir / f"{s}.parquet") for s in source_cfg["split"]["setups"]
        }
        print(summary(tables, cfg.max_horizon).to_string(index=False))
        return 0

    # 3. The run's inner models, checked.
    print("[voi] refitting the source run's inner models ...", flush=True)
    tolerance = RefitTolerance(**cfg.refit_check)
    scorers, checks = voi_scorers(source_dir, source_cfg, preprocessing, dev, tolerance)
    print(f"[voi] {len(scorers)} inner models reproduce the run's out-of-fold scores", flush=True)
    for check in checks:
        if check["rows_off"]:
            print(
                f"[voi]   within the tie tolerance: {check['setup']} outer {check['outer_fold']} "
                f"inner {check['inner_fold']}: {check['rows_off']} of {check['rows']} rows, "
                f"max |diff| {check['max_abs_diff']:.3g}, no alarm changed",
                flush=True,
            )

    # 4. Targets, chunk by chunk.
    settings = dev.settings
    record = clean(dev.hours, settings.cleaning)
    utility = dev.tables.targets[[*KEYS, "utility_diff"]]
    ids = sorted(set().union(*(s.ids for s in scorers)))
    chunks = [ids[i : i + cfg.chunk_patients] for i in range(0, len(ids), cfg.chunk_patients)]
    parts: dict[str, list[pd.DataFrame]] = {}
    for n, chunk in enumerate(chunks, start=1):
        t0 = time.perf_counter()
        keep = set(chunk)
        found = build_targets(
            record[record["patient_id"].isin(keep)],
            utility[utility["patient_id"].isin(keep)],
            [s for s in scorers if s.ids & keep],
            feature_columns=dev.tables.feature_columns,
            features_config=with_blocks(settings.features, list(settings.blocks)),
            panels_config=settings.panels,
            every=cfg.decision_every_rows,
            seed=cfg.seed,
            horizons=cfg.horizons,
        )
        for setup, table in found.items():
            parts.setdefault(setup, []).append(table)
        print(
            f"[voi] chunk {n}/{len(chunks)}: {len(chunk)} patients, "
            f"{time.perf_counter() - t0:.0f} s",
            flush=True,
        )
    tables = {
        setup: pd.concat(frames, ignore_index=True)
        .sort_values(["outer_fold", "patient_id", "t", "panel"], kind="stable")
        .reset_index(drop=True)
        for setup, frames in parts.items()
    }

    # 5. Write via .partial, then rename.
    partial = out_dir.with_name(key + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for setup, table in tables.items():
        table.to_parquet(partial / f"{setup}.parquet", index=False)
    meta = {
        "key": key,
        "config": dict(raw),
        "source_run": cfg.source_run,
        "feature_tables": dev.cache["eval_key"],
        "sha": f"{sha}-dirty" if dirty else sha,
        "dirty": dirty,
        "date": date.today().isoformat(),
        "command": " ".join([Path(sys.argv[0]).name, *(argv or sys.argv[1:])]),
        "versions": run_experiment._versions(),
        "inner_models": len(scorers),
        "refit_check": checks,
        "rows": {setup: len(table) for setup, table in tables.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (partial / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    partial.rename(out_dir)
    print(f"[voi] wrote {out_dir}", flush=True)
    print(summary(tables, cfg.max_horizon).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
