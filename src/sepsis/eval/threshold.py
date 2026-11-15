"""Tune the alarm threshold on the training side of a setup (ADR-0004, ADR-0007).

The threshold is tuned on out-of-fold predictions of the training side only, then frozen. Whether
it transfers to the other hospital is part of what A -> B and B -> A measure, so `tune_threshold`
refuses any patient outside the training side: the evaluated hospital, the held-out fold of mix,
and every test patient.

Utility comes from the official scorer's own `compute_prediction_utility`; nothing is reimplemented.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import ModuleType

import numpy as np
import pandas as pd

from sepsis.data.splits import cross_hospital, mix_folds
from sepsis.eval.scorer import align, official_scorer

SETUPS = ("A_to_B", "B_to_A", "mix")


class ThresholdLeakError(RuntimeError):
    """Raised when threshold tuning receives a patient outside the setup's training side."""


@dataclass(frozen=True)
class ThresholdResult:
    """The chosen threshold, its normalized utility, and the whole `threshold, utility` curve."""

    threshold: float
    utility: float
    curve: pd.DataFrame


def threshold_grid(
    scores: np.ndarray | pd.Series, quantile_min: float, quantile_max: float, n_points: int
) -> np.ndarray:
    """Candidate thresholds: observed scores at evenly spaced quantiles, sorted and unique.

    Only observed scores are candidates, because a threshold between two neighbouring scores gives
    the same alarms as the lower one.
    """
    if not 0 <= quantile_min <= quantile_max <= 1:
        raise ValueError("need 0 <= quantile_min <= quantile_max <= 1")
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    quantiles = np.linspace(quantile_min, quantile_max, int(n_points))
    return np.unique(np.quantile(np.asarray(scores, dtype=float), quantiles, method="lower"))


def utility_curve(
    labels: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    grid: np.ndarray,
    scorer: ModuleType,
) -> pd.DataFrame:
    """Normalized utility of `score > threshold` for each threshold in `grid`.

    `labels` and `scores` hold one array per patient, rows in hour order. Normalization follows
    the official scorer: (observed - inaction) / (best - inaction) over all given patients.
    """
    parts = [_patient_utilities(scorer, np.asarray(y, dtype=np.int8).tobytes()) for y in labels]
    span = sum(best for _, best, _ in parts) - sum(inaction for inaction, _, _ in parts)
    if span == 0:
        raise ValueError("no septic patient in the set: normalized utility is undefined")

    # Observed - inaction is the summed gain of the alarmed hours, so one sort serves every
    # threshold: sum the gains of the rows whose score exceeds it.
    all_scores = np.concatenate(scores)
    order = np.argsort(all_scores, kind="stable")
    sorted_scores = all_scores[order]
    below = np.concatenate([[0.0], np.cumsum(np.concatenate([g for _, _, g in parts])[order])])
    not_alarmed = np.searchsorted(sorted_scores, grid, side="right")
    gained = below[-1] - below[not_alarmed]
    return pd.DataFrame({"threshold": np.asarray(grid, dtype=float), "utility": gained / span})


@cache
def _patient_utilities(scorer: ModuleType, label_bytes: bytes) -> tuple[float, float, np.ndarray]:
    """One label record's inaction utility, best utility and per-hour gain of an alarm.

    Every number is a call to the scorer's `compute_prediction_utility`. The gain of hour t is the
    utility of alarming at t alone minus the inaction utility: the scorer sums per-hour terms that
    depend only on that hour's prediction, so a patient's utility is inaction + the gains of its
    alarmed hours. `test_tuned_utility_equals_the_official_score_at_that_threshold` guards this.
    Cached by label record, since many patients share one (every non-septic stay of a length).
    """
    utility = scorer.compute_prediction_utility
    defaults = inspect.signature(utility).parameters
    dt_early, dt_optimal, dt_late = (
        int(defaults[name].default) for name in ("dt_early", "dt_optimal", "dt_late")
    )
    y = np.frombuffer(label_bytes, dtype=np.int8)
    n = len(y)
    # The scorer's best predictions (evaluate_sepsis_score): alarm from t_sepsis + dt_early to
    # t_sepsis + dt_late.
    best_predictions = np.zeros(n)
    if np.any(y):
        t_sepsis = int(np.argmax(y)) - dt_optimal
        best_predictions[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, n)] = 1
    inaction = float(utility(y, np.zeros(n), check_errors=False))
    best = float(utility(y, best_predictions, check_errors=False))
    gains = np.zeros(n)
    for t in range(n):
        one_alarm = np.zeros(n)
        one_alarm[t] = 1
        gains[t] = utility(y, one_alarm, check_errors=False) - inaction
    gains.setflags(write=False)
    return inaction, best, gains


def best_threshold(curve: pd.DataFrame) -> tuple[float, float]:
    """Return `(threshold, utility)` with the highest utility; ties go to the highest threshold."""
    top = curve[curve["utility"] == curve["utility"].max()]
    row = top.loc[top["threshold"].idxmax()]
    return float(row["threshold"]), float(row["utility"])


def training_side_patients(
    split: pd.DataFrame, setup: str, outer_fold: int | None = None
) -> set[str]:
    """Patients a setup may tune its threshold on, taken from the split file.

    `A_to_B`: A-dev. `B_to_A`: B-dev. `mix`: dev patients of both hospitals outside `outer_fold`.
    Test patients are never included, whatever rows `split` holds.
    """
    if setup not in SETUPS:
        raise ValueError(f"setup must be one of {SETUPS}")
    if not split["patient_id"].is_unique:
        raise ValueError("the split table lists a patient more than once")
    if split.loc[split["split"] == "dev", "fold"].isna().any():
        raise ValueError("the split table has a dev patient without a fold")
    if setup != "mix":
        if outer_fold is not None:
            raise ValueError("outer_fold is only for the mix setup")
        train_side, _ = cross_hospital(split, train=setup[0])
        return set(train_side["patient_id"])
    folds = mix_folds(split)
    if outer_fold is None or outer_fold not in set(folds["fold"]):
        raise ValueError(
            "the mix setup needs outer_fold, the stored fold being scored; "
            f"got {outer_fold!r}, the split has folds {sorted(set(folds['fold']))}"
        )
    return set(folds.loc[folds["fold"] != outer_fold, "patient_id"])


def tune_threshold(
    hours: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    split: pd.DataFrame,
    setup: str,
    *,
    outer_fold: int | None = None,
    grid: Mapping[str, float],
) -> ThresholdResult:
    """Pick the alarm threshold with the best official utility on the training side.

    `hours` are the label rows and `oof_predictions` the out-of-fold `patient_id, hour, score` of
    the same patients. `split` is the table from `load_project_split()`. `grid` is
    `threshold.grid` from configs/eval.yaml. For mix, call it once per outer fold with the
    inner-CV predictions of that fold's training patients.

    Raises `ThresholdLeakError` if any patient is outside the setup's training side.
    """
    allowed = training_side_patients(split, setup, outer_fold)
    given = set(oof_predictions["patient_id"]) | set(hours["patient_id"])
    if not given:
        raise ValueError("no predictions to tune on")
    leaked = sorted(given - allowed)
    if leaked:
        raise ThresholdLeakError(
            f"{len(leaked)} patients are outside the training side of {setup}"
            f"{'' if outer_fold is None else f' (outer fold {outer_fold})'}, e.g. {leaked[:3]}; "
            "the threshold is tuned on training-side out-of-fold predictions only"
        )

    merged = align(hours, oof_predictions, ["score"])
    groups = [rows for _, rows in merged.groupby("patient_id", sort=True)]
    labels = [rows["SepsisLabel"].to_numpy() for rows in groups]
    scores = [rows["score"].to_numpy(dtype=float) for rows in groups]
    candidates = threshold_grid(
        merged["score"], grid["quantile_min"], grid["quantile_max"], grid["n_points"]
    )
    curve = utility_curve(labels, scores, candidates, official_scorer())
    threshold, utility = best_threshold(curve)
    return ThresholdResult(threshold=threshold, utility=utility, curve=curve)
