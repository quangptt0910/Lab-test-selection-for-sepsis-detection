"""The replay engine (M5-1, ADR-0009): view -> risk -> alarm -> policy -> reveal, hour by hour.

All patients step through row t together: the view at t of every patient still in the ICU is built
in one batch from their rows <= t and the labs revealed by t, then their feature rows at t are
scored, the policy answers, and its requests are revealed from t + delay (ADR-0006). This gives
the same rows as `view_at` + `features_at` per patient (tested), at a fraction of the cost.

The record is cleaned once up front: cleaning judges each hour on its own, so it equals cleaning
the rows <= t. Only the engine holds the record; the policy sees `Step`s.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sepsis.action.policy import Policy, Step
from sepsis.data.clean import clean
from sepsis.data.view import KEYS, validate_reveal_rows, view_table
from sepsis.features.builder import build_features
from sepsis.replay.reveal import REQUEST_COLUMNS, RevealState
from sepsis.risk.base import RiskModel

PREDICTION_COLUMNS = ["patient_id", "hour", "score", "alarm"]


@dataclass
class ReplayResult:
    """What one replay produced."""

    predictions: pd.DataFrame  # patient_id, hour, score, alarm: the prediction table (ADR-0007)
    requests: pd.DataFrame  # patient_id, hour (request row), panel, status, n_values
    reveal_rows: pd.DataFrame  # the final reveal-row table, checked against ADR-0006


def replay(
    hours: pd.DataFrame,
    *,
    model: RiskModel,
    threshold: float,
    policy: Policy,
    features_config: Mapping[str, Any],
    feature_columns: Sequence[str],
    panels_config: Mapping[str, Any],
    cleaning_config: Mapping[str, Any],
) -> ReplayResult:
    """Replay every patient in `hours` (whole records) under `policy`, with a fitted model and its
    frozen threshold. Labs start hidden; only the policy's requests reveal them."""
    record = clean(hours, cleaning_config).sort_values(KEYS, kind="stable").reset_index(drop=True)
    # Patient ids name the hospital (A_..., B_...), so the policy sees per-replay codes instead.
    # Codes follow the sorted ids, so every table keeps its row order.
    codes, real_ids = pd.factorize(record["patient_id"], sort=True)
    real_of = {f"p{i:06d}": pid for i, pid in enumerate(real_ids)}
    record = record.assign(patient_id=[f"p{i:06d}" for i in codes])
    state = RevealState(record, panels_config)
    columns = list(feature_columns)
    hour = record["hour"].to_numpy()
    length = record.groupby("patient_id", sort=False)["hour"].transform("size").to_numpy()
    predictions, logs = [], []
    for t in range(int(length.max()) if len(record) else 0):
        rows = (hour <= t) & (length > t)  # rows <= t of the patients in the ICU at t
        reveal = state.reveal_rows(rows)
        labs = state.labs
        reveal[labs] = reveal[labs].where(reveal[labs] <= t)  # visible by t only
        view = view_table(record.loc[rows].reset_index(drop=True), reveal)
        features = build_features(view, features_config, panels_config=panels_config)
        at_t = (features["hour"] == t).to_numpy()
        current = features.loc[at_t].reset_index(drop=True)
        scores = np.asarray(model.predict(current[columns]), dtype=float)
        alarms = (scores > threshold).astype(int)
        ids = pd.Index(current["patient_id"], name="patient_id")
        step = Step(
            t=t,
            view=view,
            features=current,
            scores=pd.Series(scores, index=ids, name="score"),
            alarms=pd.Series(alarms, index=ids, name="alarm"),
        )
        predictions.append(current[KEYS].assign(score=scores, alarm=alarms))
        answer = {pid: frozenset(p) for pid, p in policy.decide(step).items() if p}
        strangers = sorted(set(answer) - set(ids))
        if strangers:
            raise ValueError(f"{strangers[:3]} not in the ICU at row {t}")
        if answer:
            logs.append(state.request(t, answer))
    table = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    table = table.sort_values(KEYS, kind="stable").reset_index(drop=True)[PREDICTION_COLUMNS]
    requests = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=REQUEST_COLUMNS)
    reveal_rows = state.reveal_rows()
    validate_reveal_rows(record, reveal_rows, panels_config)
    for frame in (table, requests, reveal_rows):  # back to the real ids for scoring
        frame["patient_id"] = frame["patient_id"].map(real_of)
    return ReplayResult(table, requests, reveal_rows)


def labs_per_patient_day(requests: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Panel requests per 24 patient-hours, unavailable requests included (ADR-0009).

    Pass one hospital's rows at a time: mix → A and mix → B are reported apart (ADR-0004).
    """
    patient_days = len(predictions) / 24
    return float(len(requests) / patient_days) if patient_days else 0.0
