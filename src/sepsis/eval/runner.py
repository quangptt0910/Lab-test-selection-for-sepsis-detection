"""Run one model config under one setup of ADR-0004 and score it (ADR-0008).

No config or data files are read here; the CLI (`sepsis.eval.run_experiment`) does that. The one
exception is the official scorer, which reads label and prediction folders written to a temporary
directory.

Per setup:
- **A_to_B / B_to_A**: 5-fold CV over the stored folds of the training hospital's dev part gives
  out-of-fold scores. The threshold is tuned on them, then a model refit on that whole dev part
  scores the other hospital's dev part with the frozen threshold.
- **mix**: for each stored outer fold k, an inner CV over the other 4 folds gives out-of-fold
  scores to tune fold k's threshold. A model refit on those 4 folds scores fold k. The held-out
  folds are pooled and scored per hospital (mix -> A, mix -> B).

Every model is fitted on `train_X` rows (training view, masked) of training-side patients only,
and predicts `eval_X` rows (evaluation view) of patients it was not fitted on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sepsis.data.splits import cross_hospital, mix_folds
from sepsis.eval.experiment_config import TARGET_COLUMN
from sepsis.eval.scorer import Scores, apply_threshold, score, score_by_hospital
from sepsis.eval.tables import KEYS, Tables
from sepsis.eval.threshold import SETUPS, tune_threshold
from sepsis.risk.base import RiskModel

# LOG columns, in order: (setup, hospital scored). None: the setup's single score.
RESULTS = (("A_to_B", None), ("B_to_A", None), ("mix", "A"), ("mix", "B"))
RESULT_NAMES = ("A→B", "B→A", "mix→A", "mix→B")
PARTIAL_NOTE = "partial run: not comparable"


@dataclass(frozen=True)
class FoldPlan:
    """One threshold and one scored set: inner CV for out-of-fold scores, then one refit."""

    setup: str
    outer_fold: int | None  # mix only
    inner: tuple[tuple[int, tuple[str, ...], tuple[str, ...]], ...]  # (fold, fit ids, oof ids)
    refit: tuple[str, ...]
    scored: tuple[str, ...]


@dataclass
class SetupResult:
    """Everything one setup produced; the CLI writes it to `experiments/runs/<id>/<setup>/`."""

    setup: str
    scores: dict[str, Scores]  # "all" for A_to_B / B_to_A; "A" and "B" for mix
    thresholds: list[dict[str, Any]]  # one per plan: outer_fold, threshold, tuning_utility
    curves: pd.DataFrame  # threshold -> tuning utility, per plan
    predictions: pd.DataFrame  # KEYS, kind (oof|scored), outer_fold, inner_fold, score, alarm
    folds: pd.DataFrame  # one row per model: patients, rows, fit and predict seconds
    seconds: float = 0.0


def _ids(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(table["patient_id"]))


def plan_setup(split: pd.DataFrame, setup: str) -> list[FoldPlan]:
    """The fold plans of a setup, from the stored folds of the split file (dev patients only)."""
    if setup not in SETUPS:
        raise ValueError(f"setup must be one of {SETUPS}")
    if setup == "mix":
        folds = mix_folds(split)
        outer = sorted(set(folds["fold"]))
        return [
            _plan(setup, folds[folds["fold"] != k], scored=_ids(folds[folds["fold"] == k]), k=k)
            for k in outer
        ]
    train_side, eval_side = cross_hospital(split, train=setup[0])
    return [_plan(setup, train_side, scored=_ids(eval_side), k=None)]


def _plan(setup: str, training: pd.DataFrame, scored: tuple[str, ...], k: int | None) -> FoldPlan:
    inner = tuple(
        (int(f), _ids(training[training["fold"] != f]), _ids(training[training["fold"] == f]))
        for f in sorted(set(training["fold"]))
    )
    if len(inner) < 2:
        raise ValueError(f"{setup}: the training side needs at least 2 stored folds")
    return FoldPlan(setup, k, inner, refit=_ids(training), scored=scored)


@dataclass(frozen=True)
class PatientIndex:
    """Integer patient codes of the table rows, built once per table and reused for every fold.

    np.isin on string ids took ~6 min per call on all dev rows; on integer codes it is instant.
    """

    codes: np.ndarray
    patients: pd.Index

    @classmethod
    def of(cls, tables: Tables) -> PatientIndex:
        codes, patients = pd.factorize(tables.targets["patient_id"])
        return cls(codes, pd.Index(patients))

    def rows(self, ids: Sequence[str]) -> np.ndarray:
        """Boolean mask of the table rows of `ids`; raises if a patient has no rows."""
        wanted = self.patients.get_indexer(list(ids))
        if (wanted < 0).any():
            raise ValueError("a planned patient has no rows in the tables")
        return np.isin(self.codes, wanted)


def _check_aligned(tables: Tables, labels: pd.DataFrame) -> None:
    if not labels[KEYS].reset_index(drop=True).equals(tables.targets[KEYS]):
        raise ValueError("labels must have the same (patient_id, hour) rows as the tables")


def run_setup(
    tables: Tables,
    labels: pd.DataFrame,
    split: pd.DataFrame,
    setup: str,
    *,
    new_model: Callable[[], RiskModel],
    target: str,
    grid: Mapping[str, float],
    score_jobs: int = 1,
) -> SetupResult:
    """Fit, tune and score one setup.

    `labels` are the patient-hour rows (`patient_id, hour, hospital, SepsisLabel`) aligned with
    the tables. `new_model()` returns a fresh, unfitted model. `grid` is `threshold.grid` from
    configs/eval.yaml. `score_jobs` > 1 scores mix's hospitals in parallel processes.
    """
    _check_aligned(tables, labels)
    start = time.perf_counter()
    y = tables.targets[TARGET_COLUMN[target]].to_numpy(dtype=float)
    weight = tables.targets["weight"].to_numpy(dtype=float)
    X_train = tables.train_X[tables.feature_columns]
    X_eval = tables.eval_X[tables.feature_columns]

    rows = PatientIndex.of(tables).rows

    def fit_predict(
        fit_ids: Sequence[str], predict_ids: Sequence[str], fold_row: dict[str, Any]
    ) -> pd.DataFrame:
        fit_rows, predict_rows = rows(fit_ids), rows(predict_ids)
        t0 = time.perf_counter()
        model = new_model().fit(X_train[fit_rows], y[fit_rows], weight[fit_rows])
        t1 = time.perf_counter()
        scores = np.asarray(model.predict(X_eval[predict_rows]), dtype=float)
        fold_row.update(
            fit_patients=len(fit_ids),
            fit_rows=int(fit_rows.sum()),
            predict_patients=len(predict_ids),
            predict_rows=int(predict_rows.sum()),
            fit_seconds=round(t1 - t0, 3),
            predict_seconds=round(time.perf_counter() - t1, 3),
        )
        return tables.targets.loc[predict_rows, KEYS].assign(score=scores).reset_index(drop=True)

    thresholds, curves, predictions, fold_rows, scored_parts = [], [], [], [], []
    for plan in plan_setup(split, setup):
        oof_parts = []
        for fold, fit_ids, oof_ids in plan.inner:
            row = {"outer_fold": plan.outer_fold, "inner_fold": fold, "kind": "oof"}
            oof_parts.append(fit_predict(fit_ids, oof_ids, row).assign(inner_fold=fold))
            fold_rows.append(row)
        oof = pd.concat(oof_parts, ignore_index=True)
        tuned = tune_threshold(
            labels[rows(_ids(oof.drop_duplicates("patient_id")))],
            oof[[*KEYS, "score"]],
            split,
            setup,
            outer_fold=plan.outer_fold,
            grid=grid,
        )
        thresholds.append(
            {
                "outer_fold": plan.outer_fold,
                "threshold": tuned.threshold,
                "tuning_utility": tuned.utility,
            }
        )
        curves.append(tuned.curve.assign(outer_fold=plan.outer_fold))
        row = {"outer_fold": plan.outer_fold, "inner_fold": None, "kind": "scored"}
        scored = apply_threshold(fit_predict(plan.refit, plan.scored, row), tuned.threshold)
        fold_rows.append(row)
        scored_parts.append(scored)
        predictions.append(
            apply_threshold(oof, tuned.threshold).assign(kind="oof", outer_fold=plan.outer_fold)
        )
        predictions.append(scored.assign(kind="scored", outer_fold=plan.outer_fold, inner_fold=-1))

    pooled = pd.concat(scored_parts, ignore_index=True)
    scored_labels = labels[rows(_ids(pooled.drop_duplicates("patient_id")))]
    if setup == "mix":
        result_scores = score_by_hospital(scored_labels, pooled, jobs=score_jobs)
    else:
        result_scores = {"all": score(scored_labels, pooled)}
    columns = [*KEYS, "kind", "outer_fold", "inner_fold", "score", "alarm"]
    table = pd.concat(predictions, ignore_index=True)[columns]
    return SetupResult(
        setup=setup,
        scores=result_scores,
        thresholds=thresholds,
        curves=pd.concat(curves, ignore_index=True),
        predictions=table.sort_values([*KEYS, "kind", "outer_fold"], kind="stable").reset_index(
            drop=True
        ),
        folds=pd.DataFrame(fold_rows),
        seconds=round(time.perf_counter() - start, 1),
    )


# --- Summaries ----------------------------------------------------------------------------------


def four_scores(results: Mapping[str, SetupResult]) -> dict[str, Scores | None]:
    """The four LOG results by name (A→B, B→A, mix→A, mix→B); None for setups not run."""
    out: dict[str, Scores | None] = {}
    for name, (setup, hospital) in zip(RESULT_NAMES, RESULTS, strict=True):
        result = results.get(setup)
        out[name] = (
            None if result is None else result.scores["all" if hospital is None else hospital]
        )
    return out


def red_flags(results: Mapping[str, SetupResult], limits: Mapping[str, float]) -> list[str]:
    """The `run-experiment` red flags that need no earlier run (printed and written to the note).

    `limits` is the `red_flags` section of configs/protocol.yaml.
    """
    return flags_from_scores(four_scores(results), limits)


def flags_from_scores(s: Mapping[str, Scores | None], limits: Mapping[str, float]) -> list[str]:
    """`red_flags` from the four results by LOG name (`four_scores`); also used by
    `sepsis.eval.compare` on finished run folders."""
    max_auroc, max_gap = limits["max_auroc"], limits["max_direction_gap"]
    flags = [
        f"red flag: {name} AUROC {v.auroc:.3f} > {max_auroc}"
        for name, v in s.items()
        if v and v.auroc > max_auroc
    ]
    if s["A→B"] and s["B→A"] and abs(s["A→B"].utility - s["B→A"].utility) > max_gap:
        flags.append(f"red flag: A→B and B→A utility differ by more than {max_gap}")
    for cross, mix in (("A→B", "mix→B"), ("B→A", "mix→A")):
        if s[cross] and s[mix] and s[cross].utility > s[mix].utility:
            flags.append(f"red flag: {cross} utility beats {mix} (mix trains on that hospital)")
    return flags


def _cell(text: str) -> str:
    return " ".join(str(text).replace("|", "\\|").split())


def log_row(
    cfg: Mapping[str, Any],
    config_path: str,
    results: Mapping[str, SetupResult],
    *,
    sha: str,
    date: str,
    flags: Sequence[str],
) -> str:
    """One `experiments/LOG.md` row (header in that file); `flags` from `red_flags`."""
    s = four_scores(results)
    run = [v for v in s.values() if v is not None]
    utilities = [("—" if v is None else f"{v.utility:.3f}") for v in s.values()]
    auroc = f"{np.mean([v.auroc for v in run]):.3f}"
    partial = len(run) < len(s)
    if partial:
        auroc += f" ({len(run)} of {len(s)})"
    note = [cfg["note"]] if cfg["note"] else []
    note += [PARTIAL_NOTE] if partial else []
    note += list(flags)
    cells = [cfg["id"], date, sha, config_path, "dev", *utilities, auroc, "—", "; ".join(note)]
    return "| " + " | ".join(_cell(c) for c in cells) + " |"
