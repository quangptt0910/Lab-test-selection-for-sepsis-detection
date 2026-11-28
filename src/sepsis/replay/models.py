"""Models for replay (M5-1): each scored patient gets a model that never trained on them.

No models are saved (ADR-0008), so replay refits each fold plan of a finished run with the run's
config, seed and training rows, and checks the refit reproduces the run's scored predictions.
The threshold is the run's frozen one (`<run>/<setup>/thresholds.json`), never re-tuned.
- A_to_B / B_to_A: one model, fitted on the training hospital's dev patients, scores the other.
- mix: one model per outer fold, fitted on the other four folds, scores its own fold.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from sepsis.action.policy import Policy
from sepsis.eval import run_experiment
from sepsis.eval.refit import check_against_run, load_run, refit_plan
from sepsis.eval.runner import plan_setup
from sepsis.eval.scorer import score, score_by_hospital
from sepsis.features.builder import with_blocks
from sepsis.replay.engine import ReplayResult, labs_per_patient_day, replay, request_shares
from sepsis.risk.base import RiskModel


@dataclass(frozen=True)
class ReplayModel:
    """One fitted model, its frozen threshold and the patients it may score."""

    setup: str
    outer_fold: int | None  # mix only
    model: RiskModel
    threshold: float
    trained_ids: tuple[str, ...]
    scored_ids: tuple[str, ...]


def _fold(value: object) -> int | None:
    """An outer fold as an int (the run writes numpy ints through json as strings), or None."""
    return None if value is None else int(value)  # type: ignore[call-overload]


def _dev(run_dir: Path, dev: run_experiment.DevData | None) -> run_experiment.DevData:
    if dev is not None:
        return dev
    cfg, _ = load_run(run_dir)
    return run_experiment.load_dev_tables(cfg)


def replay_models(
    run_dir: Path, setup: str, *, dev: run_experiment.DevData | None = None
) -> list[ReplayModel]:
    """The refit model of every fold plan of `setup`, checked against the run, with the run's
    frozen threshold."""
    run_dir = Path(run_dir)
    cfg, preprocessing = load_run(run_dir)
    dev = _dev(run_dir, dev)
    entries = json.loads((run_dir / setup / "thresholds.json").read_text(encoding="utf-8"))
    frozen = {_fold(e["outer_fold"]): float(e["threshold"]) for e in entries}
    models = []
    for plan in plan_setup(dev.split, setup):
        fold = _fold(plan.outer_fold)
        refit = refit_plan(cfg, preprocessing, dev.tables, plan)
        check_against_run(refit, run_dir)  # raises unless it is the run's model
        if fold not in frozen:
            raise ValueError(f"{setup}: no frozen threshold for outer fold {fold}")
        models.append(
            ReplayModel(
                setup=setup,
                outer_fold=fold,
                model=refit.model,
                threshold=frozen[fold],
                trained_ids=plan.refit,
                scored_ids=plan.scored,
            )
        )
    return models


def replay_setup(
    run_dir: Path,
    setup: str,
    policy: Policy,
    *,
    dev: run_experiment.DevData | None = None,
    models: Sequence[ReplayModel] | None = None,
) -> ReplayResult:
    """Replay every scored patient of `setup` under `policy`, each with its own fold's model.

    `models` (from `replay_models`) skips the refit, so several policies share one set of models.
    """
    run_dir = Path(run_dir)
    dev = _dev(run_dir, dev)
    settings = dev.settings
    features = with_blocks(settings.features, list(settings.blocks))
    if models is None:
        models = replay_models(run_dir, setup, dev=dev)
    parts = []
    for item in models:
        hours = dev.hours[dev.hours["patient_id"].isin(item.scored_ids)]
        parts.append(
            replay(
                hours,
                model=item.model,
                threshold=item.threshold,
                policy=policy,
                features_config=features,
                feature_columns=dev.tables.feature_columns,
                panels_config=settings.panels,
                cleaning_config=settings.cleaning,
            )
        )
    return ReplayResult(
        predictions=pd.concat([p.predictions for p in parts], ignore_index=True),
        requests=pd.concat([p.requests for p in parts], ignore_index=True),
        reveal_rows=pd.concat([p.reveal_rows for p in parts], ignore_index=True),
    )


def by_hospital(result: ReplayResult, split: pd.DataFrame) -> dict[str, ReplayResult]:
    """One result per hospital, from the split file. Mix pools both hospitals, but mix → A and
    mix → B are scored and reported apart, never merged (ADR-0004)."""
    hospital = split.set_index("patient_id")["hospital"]
    out = {}
    for name in sorted(set(hospital[result.predictions["patient_id"].unique()])):

        def mine(table: pd.DataFrame, name: str = name) -> pd.DataFrame:
            keep = table["patient_id"].map(hospital).eq(name)
            return table[keep.to_numpy()].reset_index(drop=True)

        out[name] = ReplayResult(
            mine(result.predictions), mine(result.requests), mine(result.reveal_rows)
        )
    return out


def _metrics(scores: object, part: ReplayResult) -> dict[str, float | None]:
    return {
        **asdict(scores),  # type: ignore[call-overload]
        "labs_per_patient_day": labs_per_patient_day(part.requests, part.predictions),
        **request_shares(part.requests),
    }


def score_replay(
    result: ReplayResult, setup: str, dev: run_experiment.DevData
) -> dict[str, dict[str, float | None]]:
    """Official scores and test burden of one setup's replay, under the runner's `scores.json`
    keys: `A_to_B` / `B_to_A`, or `mix_A` and `mix_B` (scored apart, ADR-0004)."""
    patients = set(result.predictions["patient_id"])
    labels = dev.labels[dev.labels["patient_id"].isin(patients)]
    if setup != "mix":
        return {setup: _metrics(score(labels, result.predictions), result)}
    scores = score_by_hospital(labels, result.predictions)
    parts = by_hospital(result, dev.split)
    return {f"mix_{h}": _metrics(scores[h], parts[h]) for h in sorted(scores)}
