"""Replay engine (M5-1): view -> risk -> alarm -> policy -> reveal, all patients in step.

Checked against a slow reference written here, independently of the engine: one patient at a
time, `view_at` + `features_at`, and the ADR-0006 reveal rule as plain loops. Toy records only.
"""

import numpy as np
import pandas as pd
import pytest
from feature_toy import toy_record

from sepsis.action.baselines import AllPolicy
from sepsis.action.baselines import NonePolicy as BaseNone
from sepsis.action.policy import Step
from sepsis.config import CLEANING_CONFIG, FEATURES_CONFIG, load_yaml
from sepsis.data.clean import clean
from sepsis.data.columns import DEMOGRAPHICS, LABS, VITALS
from sepsis.data.panels import PANEL_NAMES, PANELS_CONFIG
from sepsis.data.view import (
    KEYS,
    STATIC,
    build_evaluation_view,
    evaluation_reveal_rows,
    view_at,
)
from sepsis.features.builder import build_features, feature_columns, features_at
from sepsis.replay.engine import ReplayResult, labs_per_patient_day, replay

CLEANING = load_yaml(CLEANING_CONFIG)
PANELS = load_yaml(PANELS_CONFIG)
FEATURES = load_yaml(FEATURES_CONFIG)  # core blocks on
THRESHOLD = 1.5


class ToyModel:
    """A fixed score that uses a vital, a lab value and a lab's age, so reveals change it."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        score = (
            X["partial_sofa"].fillna(0.0)
            + 0.2 * X["Lactate"].fillna(0.0)
            + 0.05 * X["hours_since_CHEM"].fillna(30.0)
            + 0.01 * X["HR"].fillna(0.0)
        )
        return score.to_numpy(dtype=float)


def _columns(hours: pd.DataFrame) -> list[str]:
    view = build_evaluation_view(hours, CLEANING, PANELS)
    return feature_columns(view, FEATURES, panels_config=PANELS)


def _choose(row: pd.Series, score: float) -> frozenset[str]:
    """A policy rule that depends only on the patient's own feature row at t (never the id):
    pseudo-random from the row's hour and vitals, plus one score rule."""
    seed = [int(row["hour"]), *(int(np.nan_to_num(row[v]) * 10) for v in ("HR", "Resp"))]
    rng = np.random.default_rng(seed)
    wanted = {p for p in PANEL_NAMES if rng.random() < 0.15}
    if score > THRESHOLD:
        wanted.add("LACTATE")
    return frozenset(wanted)


class RulePolicy:
    name = "rule"

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def decide(self, step: Step) -> dict[str, frozenset[str]]:
        self.steps.append(step)
        rows = step.features.set_index("patient_id")
        return {pid: _choose(rows.loc[pid], float(s)) for pid, s in step.scores.items()}


class NonePolicy(BaseNone):
    """NONE, recording the steps it saw."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def decide(self, step: Step) -> dict[str, frozenset[str]]:
        self.steps.append(step)
        return super().decide(step)


def _replay(hours: pd.DataFrame, policy, model=None, threshold: float = THRESHOLD) -> ReplayResult:
    return replay(
        hours,
        model=model or ToyModel(),
        threshold=threshold,
        policy=policy,
        features_config=FEATURES,
        feature_columns=_columns(hours),
        panels_config=PANELS,
        cleaning_config=CLEANING,
    )


def _slow_replay(hours: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The reference: one patient-hour at a time, reveals as plain loops (ADR-0006).

    Returns every feature row at t, the scores and the request log with status and n_values.
    """
    cleaned = clean(hours, CLEANING)
    labs = [lab for lab in LABS if lab in cleaned.columns]
    columns = _columns(hours)
    delay = PANELS["reveal_delay_rows"]
    rows, scores, requests = [], [], []
    for pid, record in cleaned.groupby("patient_id", sort=False):
        record = record.reset_index(drop=True)
        reveal = record[KEYS].copy()
        reveal[labs] = np.nan
        for t in range(len(record)):
            row = features_at(view_at(record, reveal, t), FEATURES, panels_config=PANELS)
            score = float(ToyModel().predict(row[columns])[0])
            rows.append(row)
            scores.append((pid, t, score))
            for panel in sorted(_choose(row.iloc[0], score)):
                spec = PANELS["panels"][panel]
                n_new, seen = 0, False
                for lab in (lab for lab in spec["labs"] if lab in labs):
                    for r in range(max(0, t - spec["lookback_rows"] + 1), t + 1):
                        if pd.notna(record.at[r, lab]):
                            seen = True
                            if pd.isna(reveal.at[r, lab]):
                                reveal.at[r, lab] = t + delay
                                n_new += 1
                status = "revealed" if n_new else ("nothing_new" if seen else "unavailable")
                requests.append((pid, t, panel, status, n_new))
    return (
        pd.concat(rows, ignore_index=True),
        pd.DataFrame(scores, columns=["patient_id", "hour", "score"]),
        pd.DataFrame(requests, columns=["patient_id", "hour", "panel", "status", "n_values"]),
    )


def _sorted(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return table.sort_values(by).reset_index(drop=True)


def test_all_patients_in_step_equal_one_patient_at_a_time() -> None:
    hours = toy_record()
    policy = RulePolicy()
    result = _replay(hours, policy)
    slow_rows, slow_scores, slow_requests = _slow_replay(hours)
    np.testing.assert_allclose(
        _sorted(result.predictions, KEYS)["score"], _sorted(slow_scores, KEYS)["score"]
    )
    # Every feature column the policy saw, not only the ones the model uses. Step ids are
    # per-replay codes in the same order as the real ids, so rows line up after sorting.
    seen = pd.concat([step.features for step in policy.steps], ignore_index=True)
    seen = seen.sort_values(KEYS).drop(columns="patient_id").reset_index(drop=True)
    slow = slow_rows.sort_values(KEYS).drop(columns="patient_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(seen, slow, check_dtype=False)
    by = [*KEYS, "panel"]
    pd.testing.assert_frame_equal(
        _sorted(result.requests, by), _sorted(slow_requests, by), check_dtype=False
    )
    assert len(result.requests) > 20  # the rule really requests
    assert {"revealed", "unavailable"} <= set(result.requests["status"])


def test_prediction_table_and_strict_alarm() -> None:
    hours = toy_record()
    result = _replay(hours, RulePolicy())
    assert list(result.predictions.columns) == ["patient_id", "hour", "score", "alarm"]
    assert len(result.predictions) == len(hours)
    expected = (result.predictions["score"] > THRESHOLD).astype(int)
    assert result.predictions["alarm"].tolist() == expected.tolist()

    class Constant:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), 0.7)

    tie = _replay(hours, NonePolicy(), model=Constant(), threshold=0.7)
    assert tie.predictions["alarm"].eq(0).all()  # score == threshold is no alarm


def test_all_policy_reveals_exactly_as_recorded_and_scores_like_the_batch_view() -> None:
    hours = toy_record()
    result = _replay(hours, AllPolicy())
    expected = evaluation_reveal_rows(clean(hours, CLEANING), PANELS)
    pd.testing.assert_frame_equal(
        result.reveal_rows.sort_index(axis=1), expected.sort_index(axis=1), check_dtype=False
    )
    batch = build_features(
        build_evaluation_view(hours, CLEANING, PANELS), FEATURES, panels_config=PANELS
    )
    np.testing.assert_allclose(
        result.predictions["score"], ToyModel().predict(batch[_columns(hours)])
    )
    assert set(result.requests["status"]) <= {"revealed", "nothing_new", "unavailable"}


def test_none_policy_never_shows_a_lab() -> None:
    policy = NonePolicy()
    result = _replay(toy_record(), policy)
    assert result.requests.empty
    assert list(result.requests.columns) == ["patient_id", "hour", "panel", "status", "n_values"]
    labs = [lab for lab in LABS if lab in policy.steps[0].view.values.columns]
    assert all(step.view.values[labs].isna().all().all() for step in policy.steps)
    assert labs_per_patient_day(result.requests, result.predictions) == 0.0


def test_labs_per_patient_day_counts_every_request() -> None:
    result = _replay(toy_record(), AllPolicy())
    patient_days = len(result.predictions) / 24
    assert labs_per_patient_day(result.requests, result.predictions) == pytest.approx(
        len(PANEL_NAMES) * len(result.predictions) / patient_days
    )


class AskOnceAt10(NonePolicy):
    """Requests LACTATE for every patient at row 10 only."""

    def __init__(self, ask: bool = True) -> None:
        super().__init__()
        self.ask = ask

    def decide(self, step: Step) -> dict[str, frozenset[str]]:
        self.steps.append(step)
        if self.ask and step.t == 10:
            return {pid: frozenset({"LACTATE"}) for pid in step.scores.index}
        return {}


def test_a_value_requested_at_t_shows_only_from_t_plus_1() -> None:
    hours = toy_record()
    hours = hours[hours["patient_id"] == "A_1"].reset_index(drop=True)
    hours["Lactate"] = np.nan
    hours.loc[10, "Lactate"] = 3.0
    policy = AskOnceAt10()
    _replay(hours, policy)
    shown = [step.features["Lactate"].item() for step in policy.steps]
    assert all(np.isnan(v) for v in shown[:11])
    assert shown[11:] == [3.0] * (len(hours) - 11)


def test_a_request_at_t_changes_nothing_up_to_t() -> None:
    """ADR-0006 section 5: values revealed at t + 1 change no feature or view row at rows <= t."""
    asked, waited = AskOnceAt10(ask=True), AskOnceAt10(ask=False)
    first, second = _replay(toy_record(), asked), _replay(toy_record(), waited)
    for a, b in zip(asked.steps[:11], waited.steps[:11], strict=True):
        pd.testing.assert_frame_equal(a.view.values, b.view.values)
        pd.testing.assert_frame_equal(a.view.lab_rows, b.view.lab_rows)
        pd.testing.assert_frame_equal(a.features, b.features)
    keep = first.predictions["hour"] <= 10
    pd.testing.assert_frame_equal(first.predictions[keep], second.predictions[keep])
    assert not asked.steps[11].features.equals(waited.steps[11].features)  # it does show later


@pytest.mark.parametrize(
    "t", [pytest.param(0, marks=pytest.mark.slow), 7, pytest.param(16, marks=pytest.mark.slow)]
)
def test_rows_after_t_change_nothing_seen_up_to_t(t: int) -> None:
    hours = toy_record()
    changed = hours.copy()
    later = (changed["hour"] > t).to_numpy()
    rng = np.random.default_rng(t)
    for column in [*VITALS, *LABS, *DEMOGRAPHICS]:
        new = rng.uniform(1, 200, later.sum())
        changed.loc[later, column] = np.where(rng.random(later.sum()) < 0.5, new, np.nan)
    changed.loc[later, "SepsisLabel"] = 1 - changed.loc[later, "SepsisLabel"]
    before, after = RulePolicy(), RulePolicy()
    first, second = _replay(hours, before), _replay(changed, after)
    keep = first.predictions["hour"] <= t
    pd.testing.assert_frame_equal(first.predictions[keep], second.predictions[keep])
    for a, b in zip(before.steps[: t + 1], after.steps[: t + 1], strict=True):
        pd.testing.assert_frame_equal(a.view.values, b.view.values)
        pd.testing.assert_frame_equal(a.features, b.features)
    early = first.requests["hour"] <= t
    pd.testing.assert_frame_equal(
        first.requests[early].reset_index(drop=True),
        second.requests[second.requests["hour"] <= t].reset_index(drop=True),
    )


def test_a_policy_sees_only_the_view_up_to_t() -> None:
    policy = RulePolicy()
    _replay(toy_record(), policy)
    allowed = set(KEYS) | set(STATIC) | set(VITALS) | set(LABS)
    for step in policy.steps:
        assert set(step.view.values.columns) <= allowed
        assert set(step.view.lab_rows.columns) <= set(KEYS) | set(LABS)
        assert step.view.values["hour"].max() == step.t
        assert set(step.features["hour"]) == {step.t}
        assert "SepsisLabel" not in step.features.columns
        assert "hospital" not in step.features.columns
        assert set(step.scores.index) == set(step.features["patient_id"])
    # B_2 has 18 rows: only one patient is left after its stay ends.
    assert len(policy.steps[17].scores) == 2 and len(policy.steps[20].scores) == 1


def test_a_policy_never_sees_the_real_patient_ids() -> None:
    """Ids name the hospital (A_..., B_...): a policy gets per-replay codes instead."""
    policy = RulePolicy()
    result = _replay(toy_record(), policy)
    for step in policy.steps:
        for ids in (step.view.values["patient_id"], step.features["patient_id"], step.scores.index):
            assert all(str(pid).startswith("p") for pid in ids)
            assert not {"A_1", "B_2"} & set(ids)
    assert set(result.predictions["patient_id"]) == {"A_1", "B_2"}  # real ids come back
    assert set(result.requests["patient_id"]) <= {"A_1", "B_2"}


@pytest.mark.slow
def test_relabelled_ids_change_nothing() -> None:
    hours = toy_record()
    renamed = hours.assign(patient_id=hours["patient_id"].map({"A_1": "B_7", "B_2": "C_8"}))
    first, second = _replay(hours, RulePolicy()), _replay(renamed, RulePolicy())
    back = {"B_7": "A_1", "C_8": "B_2"}
    for table in (second.predictions, second.requests):
        table["patient_id"] = table["patient_id"].map(back)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.requests, second.requests)


def test_bad_policy_answers_are_refused() -> None:
    class Unknown(NonePolicy):
        def decide(self, step: Step) -> dict[str, frozenset[str]]:
            return {step.scores.index[0]: frozenset({"BLOOD_CULTURE"})}

    class Stranger(NonePolicy):
        def decide(self, step: Step) -> dict[str, frozenset[str]]:
            return {"C_9": frozenset({"LACTATE"})}

    with pytest.raises(ValueError, match="unknown panels"):
        _replay(toy_record(), Unknown())
    with pytest.raises(ValueError, match="not in the ICU"):
        _replay(toy_record(), Stranger())
