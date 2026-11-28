"""Replay reveals (M5-1, ADR-0006): a panel request at row t reveals the panel's recorded values in
its look-back window, never a value after t, visible from t + reveal delay. Toy records only."""

import numpy as np
import pandas as pd
import pytest

from sepsis.config import load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.view import validate_reveal_rows
from sepsis.replay.reveal import REQUEST_COLUMNS, RevealState

PANELS = load_yaml(PANELS_CONFIG)
NAN = np.nan


def _record(pid: str = "A_1", n: int = 30, **labs: dict[int, float]) -> pd.DataFrame:
    record = pd.DataFrame({"patient_id": pid, "hour": range(n), "HR": 80.0})
    for lab in ("Lactate", "pH", "Creatinine", "BUN", "WBC", "Glucose"):
        column = np.full(n, NAN)
        for row, value in labs.get(lab, {}).items():
            column[row] = value
        record[lab] = column
    return record


def _state(record: pd.DataFrame) -> RevealState:
    return RevealState(record, PANELS)


def test_a_request_reveals_its_window_from_the_next_row() -> None:
    state = _state(_record(Lactate={2: 1.5, 5: 3.0, 9: 2.0}))
    log = state.request(6, {"A_1": {"LACTATE"}})
    lactate = state.reveal_rows()["Lactate"]
    assert lactate.iloc[[2, 5]].tolist() == [7.0, 7.0]  # requested at 6, visible from 7
    assert lactate.drop(index=[2, 5]).isna().all()  # row 9 is after t
    assert log.to_dict("records") == [
        {"patient_id": "A_1", "hour": 6, "panel": "LACTATE", "status": "revealed", "n_values": 2}
    ]
    assert list(log.columns) == REQUEST_COLUMNS


def test_never_a_value_after_t() -> None:
    state = _state(_record(Lactate={7: 2.0}))
    log = state.request(6, {"A_1": {"LACTATE"}})
    assert state.reveal_rows()["Lactate"].isna().all()
    assert log["status"].tolist() == ["unavailable"]
    assert log["n_values"].tolist() == [0]


@pytest.mark.parametrize(
    ("panel", "lab", "lookback"), [("LACTATE", "Lactate", 12), ("CHEM", "Creatinine", 24)]
)
def test_look_back_window_edges(panel: str, lab: str, lookback: int) -> None:
    t = 28
    oldest = t - lookback + 1  # the first row inside the window
    state = _state(_record(**{lab: {oldest - 1: 1.0, oldest: 2.0}}))
    state.request(t, {"A_1": {panel}})
    reveal = state.reveal_rows()[lab]
    assert np.isnan(reveal.iloc[oldest - 1])  # just outside: never revealed
    assert reveal.iloc[oldest] == t + 1


def test_an_unavailable_request_imputes_nothing() -> None:
    record = _record(Lactate={20: 2.0})
    state = _state(record)
    log = state.request(10, {"A_1": {"LACTATE"}})
    assert log["status"].tolist() == ["unavailable"]
    assert state.reveal_rows()["Lactate"].isna().all()
    pd.testing.assert_frame_equal(state.record, record)  # the record is never written


def test_an_already_revealed_value_keeps_its_reveal_row() -> None:
    state = _state(_record(Lactate={3: 2.0}))
    state.request(4, {"A_1": {"LACTATE"}})
    log = state.request(8, {"A_1": {"LACTATE"}})
    assert state.reveal_rows()["Lactate"].iloc[3] == 5
    assert log.to_dict("records")[0] | {"hour": 8} == {
        "patient_id": "A_1",
        "hour": 8,
        "panel": "LACTATE",
        "status": "nothing_new",
        "n_values": 0,
    }


def test_several_panels_and_patients_in_one_hour() -> None:
    record = pd.concat(
        [
            _record("A_1", 10, Lactate={1: 2.0}, Creatinine={0: 1.0}, BUN={0: 20.0}),
            _record("B_2", 6, pH={2: 7.3}),
        ],
        ignore_index=True,
    )
    state = _state(record)
    log = state.request(3, {"A_1": {"LACTATE", "CHEM"}, "B_2": {"ABG"}})
    by_key = log.set_index(["patient_id", "panel"])
    assert by_key.loc[("A_1", "CHEM"), "n_values"] == 2  # Creatinine and BUN, one draw
    assert by_key.loc[("A_1", "LACTATE"), "status"] == "revealed"
    assert by_key.loc[("B_2", "ABG"), "n_values"] == 1
    reveal = state.reveal_rows()
    assert reveal.loc[(reveal.patient_id == "B_2") & (reveal.hour == 2), "pH"].item() == 4


def test_requests_after_a_patients_last_row_are_refused() -> None:
    state = _state(_record("A_1", 5, Lactate={1: 2.0}))
    with pytest.raises(ValueError, match="not in the ICU"):
        state.request(5, {"A_1": {"LACTATE"}})


def test_unknown_panels_are_refused() -> None:
    state = _state(_record())
    with pytest.raises(ValueError, match="unknown panels"):
        state.request(3, {"A_1": {"LACTATES"}})


def test_hours_must_go_forward() -> None:
    state = _state(_record(Lactate={1: 2.0}))
    state.request(5, {"A_1": {"LACTATE"}})
    with pytest.raises(ValueError, match="hour by hour"):
        state.request(4, {"A_1": {"LACTATE"}})


def test_the_replayed_reveal_table_follows_adr_0006() -> None:
    rng = np.random.default_rng(0)
    record = _record(
        Lactate={int(r): 2.0 for r in rng.choice(30, 8)},
        Creatinine={int(r): 1.0 for r in rng.choice(30, 8)},
    )
    state = _state(record)
    for t in range(30):
        wanted = {p for p in ("LACTATE", "CHEM") if rng.random() < 0.3}
        if wanted:
            state.request(t, {"A_1": wanted})
    validate_reveal_rows(record, state.reveal_rows(), PANELS)  # raises on any broken rule
