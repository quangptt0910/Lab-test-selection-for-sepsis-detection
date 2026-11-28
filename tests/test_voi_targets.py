"""VoI targets (M6-1, ADR-0010): decision rows, availability, the with/without views and the gain.

The acceptance test uses a stub risk model (score = the Lactate value in the view) on a hand-built
patient, with every expected gain worked out by hand from the official utility. Toy records only.
"""

import numpy as np
import pandas as pd
import pytest
from feature_toy import _patient, toy_record

from sepsis.config import CLEANING_CONFIG, FEATURES_CONFIG, load_yaml
from sepsis.data.clean import clean
from sepsis.data.columns import LABS, present
from sepsis.data.panels import PANEL_NAMES, PANELS_CONFIG
from sepsis.data.view import KEYS, validate_reveal_rows, view_table
from sepsis.features.builder import build_features
from sepsis.labels.utility import LABELS_CONFIG, UtilityConstants, utility_targets
from sepsis.labels.voi import (
    availability,
    counterfactual_features,
    counterfactual_reveal_rows,
    decision_rows,
    gains,
    load_voi_config,
    make_copies,
    panel_targets,
)

CLEANING = load_yaml(CLEANING_CONFIG)
PANELS = load_yaml(PANELS_CONFIG)
FEATURES = load_yaml(FEATURES_CONFIG)
CONSTANTS = UtilityConstants.from_config(load_yaml(LABELS_CONFIG))
VOI = load_yaml("configs/voi.yaml")
HORIZONS = (1, 6, 12)
THRESHOLD = 1.5
LACTATE_L = PANELS["panels"]["LACTATE"]["lookback_rows"]  # 12


class LactateModel:
    """Stub risk model: the Lactate value in the view, 0 where none is shown."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["Lactate"].fillna(0.0).to_numpy(dtype=float)


def _hand_patient(n: int = 24, septic_from: int | None = 10, lactate: dict | None = None):
    """One patient, no Lactate except `lactate` {row: value}; SepsisLabel 1 from `septic_from`."""
    record = _patient("A_1", n, seed=1, first_lab_row=0)
    record["Lactate"] = np.nan
    for row, value in (lactate or {3: 2.0}).items():
        record.loc[row, "Lactate"] = value
    label = [0] * n if septic_from is None else [0] * septic_from + [1] * (n - septic_from)
    return clean(record.assign(SepsisLabel=label), CLEANING)


def _targets(record: pd.DataFrame, t: int | list[int], panel: str = "LACTATE") -> pd.DataFrame:
    ts = [t] if isinstance(t, int) else t
    decisions = pd.DataFrame({"patient_id": "A_1", "t": ts})
    rows = counterfactual_features(
        record,
        decisions,
        panel,
        horizon=max(HORIZONS),
        panels_config=PANELS,
        features_config=FEATURES,
    )
    utility = utility_targets(record[[*KEYS, "SepsisLabel"]], CONSTANTS)
    ud = rows.keys.merge(utility, on=KEYS, how="left")["utility_diff"].to_numpy()
    model = LactateModel()
    found = gains(
        rows.keys,
        model.predict(rows.with_X),
        model.predict(rows.without_X),
        ud,
        threshold=THRESHOLD,
        horizons=HORIZONS,
    )
    return panel_targets(availability(record, PANELS, decisions), found, panel, HORIZONS)


# --- Config ---------------------------------------------------------------------------------------


def test_the_voi_config_names_every_panel_with_a_positive_cost() -> None:
    cfg = load_voi_config(VOI)
    assert set(cfg.costs) == set(PANEL_NAMES)
    assert all(cost > 0 for cost in cfg.costs.values())
    assert cfg.horizons == HORIZONS and cfg.max_horizon == 12
    assert cfg.provenance == "project_assumption"


@pytest.mark.parametrize(
    "change",
    [
        {"costs": {**VOI["costs"], "BLOOD_CULTURE": 1}},
        {"costs": {k: v for k, v in VOI["costs"].items() if k != "LACTATE"}},
        {"costs": {**VOI["costs"], "CBC": 0}},
        {"horizons": [0, 6]},
        {"horizons": [12, 6]},
        {"decision_every_rows": 0},
        {"refit_check": {**VOI["refit_check"], "max_share_off": 1.5}},
        {"refit_check": {**VOI["refit_check"], "atol": 0}},
        {"refit_check": {k: v for k, v in VOI["refit_check"].items() if k != "max_abs_diff"}},
        {"surprise": 1},
    ],
)
def test_a_bad_voi_config_is_refused(change: dict) -> None:
    with pytest.raises(ValueError):
        load_voi_config({**VOI, **change})


def test_a_missing_key_is_refused() -> None:
    with pytest.raises(ValueError):
        load_voi_config({k: v for k, v in VOI.items() if k != "seed"})


# --- Decision rows --------------------------------------------------------------------------------


def test_decision_rows_every_12_rows_from_an_offset_below_12() -> None:
    keys = toy_record()[KEYS]
    rows = decision_rows(keys, every=12, seed=0)
    n = keys.groupby("patient_id").size()
    for pid, ts in rows.groupby("patient_id")["t"]:
        ts = ts.to_list()
        assert 0 <= ts[0] < 12
        assert np.all(np.diff(ts) == 12)
        assert ts[-1] <= n[pid] - 2 < ts[-1] + 12  # t + 1 exists; none missed at the end
    assert list(rows.columns) == ["patient_id", "t"]


def test_a_patients_decision_rows_do_not_depend_on_the_other_patients() -> None:
    keys = toy_record()[KEYS]
    alone = decision_rows(keys[keys["patient_id"] == "A_1"], every=12, seed=0)
    together = decision_rows(keys, every=12, seed=0)
    pd.testing.assert_frame_equal(
        alone.reset_index(drop=True),
        together[together["patient_id"] == "A_1"].reset_index(drop=True),
    )


def test_the_seed_moves_the_offsets() -> None:
    keys = pd.DataFrame(
        {"patient_id": np.repeat([f"A_{i}" for i in range(40)], 30), "hour": np.tile(range(30), 40)}
    )
    first = decision_rows(keys, every=12, seed=0).groupby("patient_id")["t"].min()
    other = decision_rows(keys, every=12, seed=1).groupby("patient_id")["t"].min()
    assert (first != other).any()
    assert first.nunique() > 3  # offsets spread over [0, 12)


# --- Availability ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("t", "expected"), [(3, 1), (14, 1), (15, 0), (2, 0)])
def test_available_iff_a_value_is_in_the_look_back_window(t: int, expected: int) -> None:
    record = _hand_patient(lactate={3: 2.0})  # LACTATE window at t: rows t-11 .. t
    decisions = pd.DataFrame({"patient_id": ["A_1"], "t": [t]})
    found = availability(record, PANELS, decisions)
    row = found[found["panel"] == "LACTATE"]
    assert row["available"].to_list() == [expected]
    assert set(found["panel"]) == set(PANEL_NAMES)


# --- The two views --------------------------------------------------------------------------------


def _reveal(record: pd.DataFrame, t: int, with_request: bool) -> pd.DataFrame:
    copies = make_copies(record, pd.DataFrame({"patient_id": ["A_1"], "t": [t]}), horizon=12)
    reveal = counterfactual_reveal_rows(copies, PANELS, "LACTATE", with_request=with_request)
    validate_reveal_rows(copies.record, reveal, PANELS)
    return reveal.set_index("hour")


def test_without_hides_the_window_and_after_t_and_keeps_older_values() -> None:
    record = _hand_patient(lactate={3: 2.0, 4: 3.0, 20: 4.0})
    reveal = _reveal(record, t=15, with_request=False)  # window rows 4 .. 15
    assert reveal.loc[3, "Lactate"] == 4  # before the window: as recorded (row + 1)
    assert np.isnan(reveal.loc[4, "Lactate"])  # in the window: hidden
    assert np.isnan(reveal.loc[20, "Lactate"])  # after t: hidden


def test_with_reveals_the_window_at_t_plus_1_and_still_hides_after_t() -> None:
    record = _hand_patient(lactate={3: 2.0, 4: 3.0, 20: 4.0})
    reveal = _reveal(record, t=15, with_request=True)
    assert reveal.loc[3, "Lactate"] == 4
    assert reveal.loc[4, "Lactate"] == 16  # requested at 15, visible from 16
    assert np.isnan(reveal.loc[20, "Lactate"])  # a later draw is never revealed (ADR-0010 §3)


def test_other_panels_are_as_recorded_in_both_views() -> None:
    record = _hand_patient()
    for with_request in (False, True):
        reveal = _reveal(record, t=15, with_request=with_request)
        recorded = record.set_index("hour")
        for lab in present(PANELS["panels"]["CBC"]["labs"], record.columns):
            has = recorded[lab].notna()
            assert (reveal.loc[has, lab] == recorded.index[has] + 1).all()
            assert reveal.loc[~has, lab].isna().all()


def _hand_reveal(record: pd.DataFrame, t: int, with_request: bool) -> pd.DataFrame:
    """The ADR-0010 §3 reveal rule as plain loops, independent of the module."""
    upto = record[record["hour"] <= t + 12].reset_index(drop=True)
    lactate = set(PANELS["panels"]["LACTATE"]["labs"])
    out = upto[KEYS].copy()
    for lab in present(LABS, upto.columns):
        rows = []
        for r, value in zip(upto["hour"], upto[lab], strict=True):
            if np.isnan(value):
                rows.append(np.nan)
            elif lab not in lactate or r <= t - LACTATE_L:
                rows.append(r + 1.0)
            elif r <= t and with_request:
                rows.append(t + 1.0)
            else:
                rows.append(np.nan)
        out[lab] = rows
    return out


def test_copy_features_equal_an_independent_build_of_each_decision_row() -> None:
    record = clean(toy_record(), CLEANING)
    record = record[record["patient_id"] == "A_1"].reset_index(drop=True)
    decisions = pd.DataFrame({"patient_id": ["A_1", "A_1"], "t": [5, 13]})  # overlapping copies
    rows = counterfactual_features(
        record, decisions, "LACTATE", horizon=12, panels_config=PANELS, features_config=FEATURES
    )
    for t in (5, 13):
        for with_request, built in ((False, rows.without_X), (True, rows.with_X)):
            upto = record[record["hour"] <= t + 12].reset_index(drop=True)
            view = view_table(upto, _hand_reveal(record, t, with_request))
            expected = build_features(view, FEATURES, panels_config=PANELS)
            expected = expected[expected["hour"] > t].reset_index(drop=True)
            mine = built[(rows.keys["t"] == t).to_numpy()].reset_index(drop=True)
            assert (
                rows.keys.loc[rows.keys["t"] == t, "hour"].to_list() == expected["hour"].to_list()
            )
            columns = [c for c in expected.columns if c not in KEYS]
            pd.testing.assert_frame_equal(mine[columns], expected[columns])


# --- The gain (acceptance: the target matches its definition on a hand-built patient) -------------


def test_the_gain_matches_its_definition_on_a_hand_built_septic_patient() -> None:
    # Lactate 2.0 recorded at row 3; decision row t = 3. Without: no Lactate shown, score 0, no
    # alarm. With: shown from row 4, score 2.0 > 1.5, alarm at rows 4 .. 15.
    # SepsisLabel from row 10, so t_sepsis = 16 and dt = s - 16.
    #   dt <= -6:      U1 - U0 = max(dt / 6 + 2, -0.05)            (rows 4 .. 10: 0, 1/6, ..., 1)
    #   -6 < dt <= 3:  U1 - U0 = (3 - dt) / 9 - (-2 dt - 12) / 9   (rows 11 .. 15: 10/9 .. 14/9)
    found = _targets(_hand_patient(n=24, septic_from=10), t=3)
    assert found["available"].to_list() == [1]
    row = found.iloc[0]
    assert row["gain_h1"] == pytest.approx(0.0)  # row 4: dt = -12
    assert row["gain_h6"] == pytest.approx((0 + 1 + 2 + 3 + 4 + 5) / 6)  # rows 4 .. 9
    assert row["gain_h12"] == pytest.approx(21 / 6 + (10 + 11 + 12 + 13 + 14) / 9)  # rows 4 .. 15
    assert row["dscore_t1"] == pytest.approx(2.0)
    assert row["score_without_t1"] == 0.0 and row["alarm_without_t1"] == 0


def test_an_alarm_for_a_patient_who_never_becomes_septic_costs_u_fp_per_row() -> None:
    found = _targets(_hand_patient(n=24, septic_from=None), t=3)
    row = found.iloc[0]
    assert row["gain_h1"] == pytest.approx(-0.05)
    assert row["gain_h6"] == pytest.approx(-0.30)
    assert row["gain_h12"] == pytest.approx(-0.60)


def test_the_end_of_the_stay_cuts_the_horizon() -> None:
    found = _targets(_hand_patient(n=24, septic_from=None, lactate={18: 2.0}), t=20)
    row = found.iloc[0]  # only rows 21 .. 23 exist after t = 20
    assert row["gain_h1"] == pytest.approx(-0.05)
    assert row["gain_h6"] == pytest.approx(-0.15)
    assert row["gain_h12"] == pytest.approx(-0.15)


def test_rows_the_scorer_ignores_add_nothing() -> None:
    # Septic from row 2: t_sepsis = 8, so rows with dt > 3 (s >= 12) have U1 - U0 = 0.
    found = _targets(_hand_patient(n=24, septic_from=2, lactate={3: 2.0}), t=3)
    # rows 4 .. 11: dt -4 .. 3, U1 - U0 = (15 + dt) / 9; rows 12 .. 15 add 0
    assert found.iloc[0]["gain_h12"] == pytest.approx(sum(15 + dt for dt in range(-4, 4)) / 9)


def test_a_later_draw_after_t_does_not_enter_the_with_view() -> None:
    # Lactate 0.5 at row 6 would switch the alarm off from row 7 if it were revealed.
    once = _targets(_hand_patient(n=24, septic_from=None, lactate={3: 2.0}), t=3)
    later = _targets(_hand_patient(n=24, septic_from=None, lactate={3: 2.0, 6: 0.5}), t=3)
    assert later.iloc[0]["gain_h12"] == once.iloc[0]["gain_h12"] == pytest.approx(-0.60)


def test_an_unavailable_panel_has_no_gain() -> None:
    found = _targets(_hand_patient(lactate={3: 2.0}), t=[3, 20])  # row 3 is out of t=20's window
    by_t = found.set_index("t")
    assert by_t.loc[20, "available"] == 0
    assert by_t[["gain_h1", "gain_h6", "gain_h12"]].loc[20].isna().all()
    assert by_t.loc[3, "available"] == 1 and np.isfinite(by_t.loc[3, "gain_h12"])


def test_a_reveal_that_breaks_the_reveal_rules_is_refused(monkeypatch) -> None:
    import sepsis.labels.voi as voi

    real = voi.counterfactual_reveal_rows

    def too_old(copies, panels_config, panel, *, with_request):
        # Row 3 is outside the window (rows 4 .. 15) of a request at t = 15: revealing it at 16
        # breaks the look-back rule, which `view_table` alone would not notice.
        reveal = real(copies, panels_config, panel, with_request=with_request)
        reveal.loc[reveal["hour"] == 3, "Lactate"] = 16.0
        return reveal

    monkeypatch.setattr(voi, "counterfactual_reveal_rows", too_old)
    record = _hand_patient(lactate={3: 2.0})
    decisions = pd.DataFrame({"patient_id": ["A_1"], "t": [15]})
    with pytest.raises(ValueError, match="look-back"):
        counterfactual_features(
            record, decisions, "LACTATE", horizon=12, panels_config=PANELS, features_config=FEATURES
        )
