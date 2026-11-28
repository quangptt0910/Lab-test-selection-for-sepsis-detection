"""Baseline policies (M5-2): NONE, ALL and AS_RECORDED, and the `bind` record hook (ADR-0009).

With a reveal delay of 1, AS_RECORDED reveals each value at its recorded row + 1: the as-recorded
view, the same one ALL reveals. Toy records only.
"""

import importlib
import inspect
import pkgutil

import numpy as np
import pandas as pd
from feature_toy import toy_record
from test_replay_engine import PANELS, _replay

from sepsis.action.baselines import AllPolicy, NonePolicy
from sepsis.data.columns import LABS, VITALS
from sepsis.data.view import KEYS
from sepsis.replay.baselines import AsRecordedPolicy
from sepsis.replay.engine import request_shares


def test_as_recorded_predicts_exactly_like_all() -> None:
    hours = toy_record()
    every, recorded = _replay(hours, AllPolicy()), _replay(hours, AsRecordedPolicy(PANELS))
    pd.testing.assert_frame_equal(recorded.predictions, every.predictions)
    pd.testing.assert_frame_equal(
        recorded.reveal_rows.sort_values(KEYS).reset_index(drop=True),
        every.reveal_rows.sort_values(KEYS).reset_index(drop=True),
    )
    assert np.isfinite(recorded.predictions["score"]).all()


def test_as_recorded_requests_less_than_all_and_never_finds_nothing() -> None:
    hours = toy_record()
    every, recorded = _replay(hours, AllPolicy()), _replay(hours, AsRecordedPolicy(PANELS))
    assert 0 < len(recorded.requests) < len(every.requests)
    # Each request asks at a row with a value of that panel, so it always reveals something.
    assert set(recorded.requests["status"]) == {"revealed"}
    assert "unavailable" in set(every.requests["status"])


def test_as_recorded_requests_a_panel_exactly_where_one_of_its_labs_was_drawn() -> None:
    hours = toy_record()
    hours = hours[hours["patient_id"] == "A_1"].reset_index(drop=True)
    hours[LABS] = np.nan
    hours.loc[3, "Lactate"] = 2.0
    hours.loc[5, ["WBC", "Platelets"]] = [9.0, 200.0]
    hours.loc[5, "pH"] = 7.3
    result = _replay(hours, AsRecordedPolicy(PANELS))
    asked = sorted(zip(result.requests["hour"], result.requests["panel"], strict=True))
    assert asked == [(3, "LACTATE"), (5, "ABG"), (5, "CBC")]


class Spy(AsRecordedPolicy):
    def bind(self, draws: pd.DataFrame) -> None:
        self.draws = draws
        super().bind(draws)


def test_bind_gets_only_the_lab_columns_under_per_replay_codes() -> None:
    policy = Spy(PANELS)
    _replay(toy_record(), policy)
    draws = policy.draws
    assert "SepsisLabel" not in draws.columns and "hospital" not in draws.columns
    assert set(draws.columns) == {*KEYS, *(lab for lab in LABS if lab in draws.columns)}
    assert not set(VITALS) & set(draws.columns)
    assert all(str(pid).startswith("p") for pid in draws["patient_id"])
    assert not {"A_1", "B_2"} & set(draws["patient_id"])


def test_a_policy_without_needs_record_is_never_bound() -> None:
    class Wait(NonePolicy):
        def bind(self, draws: pd.DataFrame) -> None:
            raise AssertionError("bound a policy that does not need the record")

    _replay(toy_record(), Wait())


def test_no_policy_in_the_action_package_reads_the_record() -> None:
    import sepsis.action

    offenders = []
    for info in pkgutil.walk_packages(sepsis.action.__path__, "sepsis.action."):
        module = importlib.import_module(info.name)
        for name, value in vars(module).items():
            if inspect.isclass(value) and getattr(value, "needs_record", False):
                offenders.append(f"{info.name}.{name}")
    assert offenders == []


def test_request_shares_divide_by_every_request() -> None:
    log = pd.DataFrame(
        {
            "patient_id": ["A_1"] * 5,
            "hour": [0, 0, 1, 2, 3],
            "panel": ["LACTATE", "CBC", "CBC", "ABG", "LACTATE"],
            "status": ["revealed", "unavailable", "nothing_new", "unavailable", "revealed"],
            "n_values": [1, 0, 0, 0, 2],
        }
    )
    assert request_shares(log) == {"unavailable_rate": 0.4, "nothing_new_rate": 0.2}


def test_request_shares_are_none_without_requests() -> None:
    empty = pd.DataFrame(columns=["patient_id", "hour", "panel", "status", "n_values"])
    assert request_shares(empty) == {"unavailable_rate": None, "nothing_new_rate": None}
