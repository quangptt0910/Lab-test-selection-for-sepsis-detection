import numpy as np
import pandas as pd
import pytest

from sepsis.config import load_yaml
from sepsis.data.panels import PANELS_CONFIG
from sepsis.data.quality import CLEANING_CONFIG
from sepsis.data.view import (
    as_recorded_reveal_rows,
    build_evaluation_view,
    build_training_view,
    build_view_at,
    evaluation_reveal_rows,
    mask_panel_draws,
    training_reveal_rows,
    validate_reveal_rows,
    view_at,
    view_table,
)

NAN = np.nan
LABS_HERE = ["Lactate", "pH", "PaCO2", "WBC"]


@pytest.fixture(scope="module")
def panels() -> dict:
    return load_yaml(PANELS_CONFIG)


def _patient(pid: str, n: int, **columns) -> pd.DataFrame:
    base = {
        "patient_id": pid,
        "hospital": pid[0],
        "hour": range(n),
        "ICULOS": range(1, n + 1),
        "Age": 60.0,
        "SepsisLabel": 0,
        "HR": [80.0 + i for i in range(n)],
        "MAP": [NAN] * n,
        **{lab: [NAN] * n for lab in LABS_HERE},
    }
    base.update(columns)
    return pd.DataFrame(base)


def _record() -> pd.DataFrame:
    """Two patients. A_1: lactate at rows 1 and 4, blood gas at row 2, WBC at row 3."""
    a = _patient(
        "A_1",
        7,
        MAP=[NAN, 70.0, NAN, NAN, 65.0, NAN, NAN],
        Lactate=[NAN, 2.0, NAN, NAN, 4.0, NAN, NAN],
        pH=[NAN, NAN, 7.30, NAN, NAN, NAN, NAN],
        PaCO2=[NAN, NAN, 35.0, NAN, NAN, NAN, NAN],
        WBC=[NAN, NAN, NAN, 12.0, NAN, NAN, NAN],
    )
    b = _patient("B_1", 4, Lactate=[1.1, NAN, NAN, NAN], HR=[NAN, 90.0, NAN, 95.0])
    return pd.concat([a, b], ignore_index=True)


def _rows(table: pd.DataFrame, pid: str) -> pd.DataFrame:
    return table[table["patient_id"] == pid].set_index("hour")


# --- As-recorded reveal rows and the view table -------------------------------------------------


def test_as_recorded_value_appears_one_row_after_it_was_recorded() -> None:
    record = _record()
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    lactate = _rows(view.values, "A_1")["Lactate"]
    assert lactate.isna().loc[[0, 1]].all()  # recorded at 1, not visible at 1
    assert lactate.loc[[2, 3, 4]].tolist() == [2.0, 2.0, 2.0]
    assert lactate.loc[[5, 6]].tolist() == [4.0, 4.0]
    rows = _rows(view.lab_rows, "A_1")["Lactate"]
    assert rows.loc[[2, 5]].tolist() == [1, 4]


def test_unrevealed_labs_are_nan() -> None:
    record = _record()
    reveal = as_recorded_reveal_rows(record, delay=1)
    reveal["WBC"] = NAN
    view = view_table(record, reveal)
    assert view.values["WBC"].isna().all()
    assert view.lab_rows["WBC"].isna().all()


def test_vitals_and_static_columns_are_visible_at_their_own_row() -> None:
    view = view_table(_record(), as_recorded_reveal_rows(_record(), delay=1)).values
    a = _rows(view, "A_1")
    assert a["HR"].tolist() == [80.0 + i for i in range(7)]
    # Vitals have no reveal delay: MAP at row 1 is visible at row 1, then forward-filled.
    assert np.isnan(a["MAP"].iloc[0])
    assert a["MAP"].iloc[1:].tolist() == [70.0, 70.0, 70.0, 65.0, 65.0, 65.0]
    assert a["ICULOS"].tolist() == list(range(1, 8))
    assert (a["Age"] == 60.0).all()


def test_no_backfill_and_no_crossing_patients() -> None:
    record = _record()
    view = view_table(record, as_recorded_reveal_rows(record, delay=1)).values
    b = _rows(view, "B_1")
    assert np.isnan(b.loc[0, "HR"])  # B_1 has no HR at row 0: never filled backwards or from A_1
    assert b.loc[[1, 2, 3], "HR"].tolist() == [90.0, 90.0, 95.0]
    assert np.isnan(b.loc[0, "Lactate"]) and b.loc[[1, 2, 3], "Lactate"].tolist() == [1.1] * 3
    assert np.isnan(_rows(view, "A_1").loc[0, "MAP"])


def test_labels_and_hospital_are_not_in_the_view() -> None:
    view = view_table(_record(), as_recorded_reveal_rows(_record(), delay=1))
    for table in (view.values, view.lab_rows):
        assert "SepsisLabel" not in table.columns
        assert "hospital" not in table.columns


def test_one_row_per_record_row_in_record_order() -> None:
    record = _record().sample(frac=1, random_state=1)
    view = view_table(record, as_recorded_reveal_rows(record, delay=1))
    assert len(view.values) == len(record) == len(view.lab_rows)
    expected = record[["patient_id", "hour"]].sort_values(["patient_id", "hour"])
    pd.testing.assert_frame_equal(
        view.values[["patient_id", "hour"]], expected.reset_index(drop=True)
    )


# --- Replay-style reveal rows (ADR-0006 section 5) ---------------------------------------------


def test_late_reveal_never_changes_earlier_rows() -> None:
    record = _record()
    reveal = as_recorded_reveal_rows(record, delay=1)
    a = reveal["patient_id"] == "A_1"
    # Lactate recorded at row 1 is only requested at row 4, so it is revealed at row 5.
    reveal.loc[a, "Lactate"] = NAN
    reveal.loc[a & (reveal["hour"] == 1), "Lactate"] = 5
    view = view_table(record, reveal)
    lactate = _rows(view.values, "A_1")["Lactate"]
    assert lactate.loc[0:4].isna().all()
    assert lactate.loc[[5, 6]].tolist() == [2.0, 2.0]
    assert _rows(view.lab_rows, "A_1").loc[5, "Lactate"] == 1  # age counts from the recorded row


def test_latest_recorded_value_wins_when_revealed_together() -> None:
    record = _record()
    reveal = as_recorded_reveal_rows(record, delay=1)
    a = reveal["patient_id"] == "A_1"
    reveal.loc[a & reveal["hour"].isin([1, 4]), "Lactate"] = 5  # both revealed at row 5
    lactate = _rows(view_table(record, reveal).values, "A_1")["Lactate"]
    assert lactate.loc[[5, 6]].tolist() == [4.0, 4.0]


def test_an_older_value_revealed_later_does_not_replace_a_newer_one() -> None:
    record = _record()
    reveal = as_recorded_reveal_rows(record, delay=1)
    a = reveal["patient_id"] == "A_1"
    reveal.loc[a & (reveal["hour"] == 1), "Lactate"] = 6  # row-1 value revealed after row-4 value
    view = view_table(record, reveal)
    assert _rows(view.values, "A_1").loc[6, "Lactate"] == 4.0
    assert _rows(view.lab_rows, "A_1").loc[6, "Lactate"] == 4


@pytest.mark.parametrize("reveal_row", [1, 0])
def test_reveal_at_or_before_the_recorded_row_is_refused(reveal_row: int) -> None:
    record = _record()
    reveal = as_recorded_reveal_rows(record, delay=1)
    reveal.loc[(reveal["patient_id"] == "A_1") & (reveal["hour"] == 1), "Lactate"] = reveal_row
    with pytest.raises(ValueError, match="after the row it was recorded"):
        view_table(record, reveal)


def test_reveal_rows_must_match_the_record() -> None:
    record = _record()
    reveal = as_recorded_reveal_rows(record, delay=1).iloc[:-1]
    with pytest.raises(ValueError, match="same .patient_id, hour. rows"):
        view_table(record, reveal)


# --- No future ---------------------------------------------------------------------------------


@pytest.mark.parametrize("t", [0, 2, 3, 5])
def test_changing_rows_after_t_leaves_rows_up_to_t_unchanged(t: int) -> None:
    record = _record()
    changed = record.copy()
    later = (changed["patient_id"] == "A_1") & (changed["hour"] > t)
    for column in ["HR", "MAP", "SepsisLabel", "ICULOS", *LABS_HERE]:
        changed.loc[later, column] = 999.0 if column != "SepsisLabel" else 1
    before = view_table(record, as_recorded_reveal_rows(record, delay=1))
    after = view_table(changed, as_recorded_reveal_rows(changed, delay=1))
    for name in ("values", "lab_rows"):
        old, new = getattr(before, name), getattr(after, name)
        keep = (old["patient_id"] != "A_1") | (old["hour"] <= t)
        pd.testing.assert_frame_equal(old[keep], new[keep])


def _late_reveals(*pairs: tuple[int, int]) -> pd.DataFrame:
    """A_1's reveal rows with Lactate recorded at row r revealed at row v, for each (r, v)."""
    reveal = as_recorded_reveal_rows(_record(), delay=1)
    a = reveal["patient_id"] == "A_1"
    for recorded, revealed in pairs:
        reveal.loc[a & (reveal["hour"] == recorded), "Lactate"] = revealed
    return reveal


REVEAL_CASES = {
    "as recorded": (),
    "row 1 revealed at 5": ((1, 5),),
    "row 1 revealed at 6, row 4 at 5": ((1, 6), (4, 5)),
    "rows 1 and 4 revealed at 5": ((1, 5), (4, 5)),
}


@pytest.mark.parametrize("case", REVEAL_CASES)
def test_view_at_equals_the_table_row_at_every_t(case: str) -> None:
    record = _record()
    reveal = _late_reveals(*REVEAL_CASES[case])
    table = view_table(record, reveal)
    one, one_reveal = (x[x["patient_id"] == "A_1"] for x in (record, reveal))
    for t in range(len(one)):
        view = view_at(one, one_reveal, t)
        assert view.values["hour"].max() == t and view.lab_rows["hour"].max() == t
        for name in ("values", "lab_rows"):
            full = getattr(table, name)
            expected = full[(full["patient_id"] == "A_1") & (full["hour"] == t)]
            got = getattr(view, name)
            pd.testing.assert_frame_equal(
                got[got["hour"] == t].reset_index(drop=True), expected.reset_index(drop=True)
            )


@pytest.mark.parametrize("t", [0, 2, 4, 5])
def test_fixed_replay_reveals_ignore_changes_after_t(t: int) -> None:
    record = _record()
    reveal = _late_reveals((1, 5), (4, 5))  # replay-style table, kept fixed
    changed = record.copy()
    later = (changed["patient_id"] == "A_1") & (changed["hour"] > t)
    for column in ["HR", "MAP", "ICULOS", *LABS_HERE]:
        changed.loc[later & changed[column].notna(), column] = 999.0
    before, after = view_table(record, reveal), view_table(changed, reveal)
    for name in ("values", "lab_rows"):
        old, new = getattr(before, name), getattr(after, name)
        keep = (old["patient_id"] != "A_1") | (old["hour"] <= t)
        pd.testing.assert_frame_equal(old[keep], new[keep])


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_non_finite_reveal_rows_are_refused(bad: float) -> None:
    record = _record()
    reveal = _late_reveals()
    reveal.loc[(reveal["patient_id"] == "A_1") & (reveal["hour"] == 4), "Lactate"] = bad
    with pytest.raises(ValueError, match="finite"):
        view_table(record, reveal)
    with pytest.raises(ValueError, match="finite"):
        validate_reveal_rows(record, reveal, load_yaml(PANELS_CONFIG))


def test_reveal_rows_follow_the_panel_rules(panels: dict) -> None:
    record = _record()
    validate_reveal_rows(record, _late_reveals((1, 5)), panels)  # within LACTATE's 12 rows
    long = pd.concat([_patient("A_9", 20, Lactate=[2.0] + [NAN] * 19)], ignore_index=True)
    reveal = as_recorded_reveal_rows(long, delay=1)
    reveal.loc[0, "Lactate"] = 13  # request at 12: row 0 is outside rows 1..12
    with pytest.raises(ValueError, match="look-back window"):
        validate_reveal_rows(long, reveal, panels)
    reveal.loc[0, "Lactate"] = 12  # request at 11: rows 0..11, allowed
    validate_reveal_rows(long, reveal, panels)
    two_rows = dict(panels, reveal_delay_rows=2)
    with pytest.raises(ValueError, match="less than 2 row"):
        validate_reveal_rows(record, _late_reveals(), two_rows)


def test_build_view_at_cleans_and_checks(panels: dict) -> None:
    cleaning = load_yaml(CLEANING_CONFIG)
    record = _patient("A_1", 4, HR=[80.0, 400.0, 85.0, NAN], Lactate=[2.0, NAN, NAN, NAN])
    reveal = as_recorded_reveal_rows(record, delay=1)
    view = build_view_at(record, reveal, 1, cleaning, panels)
    assert view.latest()["HR"].tolist() == [80.0]  # 400 bpm blanked as in training
    assert view.latest()["Lactate"].tolist() == [2.0]
    reveal.loc[0, "Lactate"] = 0
    with pytest.raises(ValueError):
        build_view_at(record, reveal, 1, cleaning, panels)


@pytest.mark.parametrize("t", [1.5, 2.0, True, "2", -1, 7])
def test_view_at_needs_an_integer_row_of_the_record(t) -> None:
    record = _record()
    one = record[record["patient_id"] == "A_1"]  # rows 0..6
    reveal = as_recorded_reveal_rows(one, delay=1)
    with pytest.raises(ValueError, match="t must be"):
        view_at(one, reveal, t)


def test_view_at_accepts_numpy_integers() -> None:
    record = _record()
    one = record[record["patient_id"] == "A_1"]
    view = view_at(one, as_recorded_reveal_rows(one, delay=1), np.int64(3))
    assert view.t == 3 and len(view.latest()) == 1


def test_view_at_needs_one_patient() -> None:
    record = _record()
    with pytest.raises(ValueError, match="one patient"):
        view_at(record, as_recorded_reveal_rows(record, delay=1), 2)


# --- Masking -----------------------------------------------------------------------------------


def _big_record(n_patients: int = 200, n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n_patients):
        lab = {name: np.where(rng.random(n) < 0.4, rng.random(n) + 1, NAN) for name in LABS_HERE}
        lab["PaCO2"] = np.where(np.isnan(lab["pH"]), NAN, 40.0)  # blood gas drawn together
        frames.append(_patient(f"A_{i}", n, **lab))
    return pd.concat(frames, ignore_index=True)


def test_masking_hides_whole_panel_draws_only(panels: dict) -> None:
    record = _big_record()
    full = as_recorded_reveal_rows(record, delay=1)
    masked = mask_panel_draws(full, panels, p=0.3, seed=7)
    hidden = full[LABS_HERE].notna() & masked[LABS_HERE].isna()
    # pH and PaCO2 are one ABG draw: hidden together or not at all.
    both = full["pH"].notna()
    assert (hidden.loc[both, "pH"] == hidden.loc[both, "PaCO2"]).all()
    assert not (full[LABS_HERE].isna() & masked[LABS_HERE].notna()).any().any()  # never reveals
    kept = masked[LABS_HERE].notna()
    unchanged = masked[LABS_HERE] == full[LABS_HERE]
    assert unchanged[kept].fillna(True).all().all()  # kept reveal rows are not moved
    rate = hidden.sum().sum() / full[LABS_HERE].notna().sum().sum()
    assert 0.25 < rate < 0.35


def test_masking_is_seeded(panels: dict) -> None:
    full = as_recorded_reveal_rows(_big_record(50), delay=1)
    one = mask_panel_draws(full, panels, p=0.3, seed=1)
    pd.testing.assert_frame_equal(one, mask_panel_draws(full, panels, p=0.3, seed=1))
    assert not one.equals(mask_panel_draws(full, panels, p=0.3, seed=2))


def test_a_patients_mask_does_not_depend_on_other_patients(panels: dict) -> None:
    record = _big_record(30)
    full = as_recorded_reveal_rows(record, delay=1)
    together = mask_panel_draws(full, panels, p=0.3, seed=3)
    alone = mask_panel_draws(full[full["patient_id"] == "A_5"], panels, p=0.3, seed=3)
    pd.testing.assert_frame_equal(
        together[together["patient_id"] == "A_5"].reset_index(drop=True), alone
    )


def test_masking_extremes(panels: dict) -> None:
    full = as_recorded_reveal_rows(_big_record(20), delay=1)
    pd.testing.assert_frame_equal(mask_panel_draws(full, panels, p=0.0, seed=0), full)
    assert mask_panel_draws(full, panels, p=1.0, seed=0)[LABS_HERE].isna().all().all()
    with pytest.raises(ValueError, match="between 0 and 1"):
        mask_panel_draws(full, panels, p=1.5, seed=0)


def test_masking_never_touches_vitals(panels: dict) -> None:
    record = _big_record(20)
    view = view_table(record, training_reveal_rows(record, panels, lab_masking_p=1.0, seed=0))
    plain = view_table(record, as_recorded_reveal_rows(record, delay=1))
    pd.testing.assert_frame_equal(
        view.values[["patient_id", "hour", "ICULOS", "Age", "HR", "MAP"]],
        plain.values[["patient_id", "hour", "ICULOS", "Age", "HR", "MAP"]],
    )
    assert view.values[LABS_HERE].isna().all().all()


def test_evaluation_reveal_rows_are_never_masked(panels: dict) -> None:
    record = _big_record(20)
    pd.testing.assert_frame_equal(
        evaluation_reveal_rows(record, panels),
        as_recorded_reveal_rows(record, delay=panels["reveal_delay_rows"]),
    )


# --- Real dev data -----------------------------------------------------------------------------


@pytest.mark.slow
def test_real_dev_patients(panels: dict) -> None:
    from sepsis.data.splits import (
        IncompleteCacheError,
        SplitFingerprintError,
        load_patient_hours,
        load_project_split,
    )

    try:
        split = load_project_split()  # dev patients only (ADR-0004)
        ids = split.groupby("hospital").sample(200, random_state=0)["patient_id"].tolist()
        hours = load_patient_hours(["A", "B"], ids)
    except (FileNotFoundError, IncompleteCacheError, SplitFingerprintError):
        pytest.skip("real data cache or split file not found")
    view = build_evaluation_view(hours, load_yaml(CLEANING_CONFIG), panels)
    assert len(view.values) == len(hours)
    assert "SepsisLabel" not in view.values.columns
    labs = [c for c in view.lab_rows.columns if c not in ("patient_id", "hour")]
    age = view.lab_rows["hour"].to_numpy()[:, None] - view.lab_rows[labs].to_numpy()
    shown = ~np.isnan(age)
    assert (age[shown] >= 1).all()  # no lab is visible at the row it was recorded
    assert shown.any()


# --- Cleaning and the two entry points ---------------------------------------------------------


def test_build_views_clean_first(panels: dict) -> None:
    cleaning = load_yaml(CLEANING_CONFIG)
    record = _patient("A_1", 4, HR=[80.0, 400.0, 85.0, NAN], Lactate=[2.0, 99.0, NAN, NAN])
    for view in (
        build_evaluation_view(record, cleaning, panels),
        build_training_view(record, cleaning, panels, lab_masking_p=0.0, seed=0),
    ):
        a = _rows(view.values, "A_1")
        assert a["HR"].tolist() == [80.0, 80.0, 85.0, 85.0]  # 400 bpm blanked, 80 carried
        assert np.isnan(a.loc[0, "Lactate"]) and a.loc[[1, 2, 3], "Lactate"].tolist() == [2.0] * 3
