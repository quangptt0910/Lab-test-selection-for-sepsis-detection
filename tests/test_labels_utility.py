import inspect
import math

import numpy as np
import pandas as pd
import pytest
from eval_toy import hours as toy_hours

from sepsis.config import load_yaml
from sepsis.eval.scorer import official_scorer
from sepsis.labels.utility import (
    LABELS_CONFIG,
    UtilityConstants,
    patient_utility,
    utility_targets,
)


@pytest.fixture(scope="module")
def constants() -> UtilityConstants:
    return UtilityConstants.from_config(load_yaml(LABELS_CONFIG))


def _septic(first_label_row: int, n: int) -> np.ndarray:
    labels = np.zeros(n, dtype=int)
    labels[first_label_row:] = 1
    return labels


# --- The rank1-features table -------------------------------------------------------------------

# First label row 20 -> t_sepsis 26, so row = 26 + dt. 40 rows cover dt -26 .. 13.
RANK1_TABLE = {-15: -0.05, -12: 0.0, -9: 0.5, -6: 1.0, -3: 4 / 3, 0: 5 / 3, 3: 2.0}


@pytest.mark.parametrize("dt, expected", RANK1_TABLE.items())
def test_rank1_table_values(constants: UtilityConstants, dt: int, expected: float) -> None:
    table = patient_utility(_septic(20, 40), constants)
    row = table.loc[table["dt"] == dt].iloc[0]
    assert row["utility_diff"] == pytest.approx(expected)
    assert row["weight"] == 1


def test_non_septic_hours(constants: UtilityConstants) -> None:
    table = patient_utility(np.zeros(12, dtype=int), constants)
    assert table["utility_diff"].tolist() == pytest.approx([-0.05] * 12)
    assert (table["u1"] == -0.05).all() and (table["u0"] == 0).all()
    assert table["dt"].isna().all()
    assert (table["weight"] == 1).all()


def test_hours_after_dt_late_are_not_scored(constants: UtilityConstants) -> None:
    table = patient_utility(_septic(20, 40), constants)
    late = table[table["dt"] > 3]
    assert len(late) == 10
    assert (late["utility_diff"] == 0).all() and (late["u1"] == 0).all()
    assert (late["u0"] == 0).all() and (late["weight"] == 0).all()
    assert (table.loc[table["dt"] <= 3, "weight"] == 1).all()


def test_u1_and_u0_in_the_label_window(constants: UtilityConstants) -> None:
    table = patient_utility(_septic(20, 40), constants).set_index("dt")
    # dt <= -6: no alarm costs nothing; -6 < dt <= 3: a missed onset costs up to -2.
    assert table.loc[-8, "u0"] == 0
    assert table.loc[0, "u1"] == pytest.approx(1 / 3)
    assert table.loc[0, "u0"] == pytest.approx(-4 / 3)
    assert table.loc[3, "u0"] == pytest.approx(-2.0)


def test_onset_counts_rows_from_the_first_row(constants: UtilityConstants) -> None:
    # Labelled from row 0 (EDA finding 2): t_sepsis = 6, first row has dt = -6.
    table = patient_utility(np.ones(10, dtype=int), constants)
    assert table["dt"].tolist() == list(range(-6, 4))
    assert table["utility_diff"].iloc[0] == pytest.approx(1.0)


def test_iculos_is_ignored(constants: UtilityConstants) -> None:
    labels = _septic(5, 15)
    base = pd.DataFrame({"patient_id": "A_x", "hour": range(15), "SepsisLabel": labels})
    late_start = base.assign(ICULOS=range(10, 25))
    early_start = base.assign(ICULOS=range(1, 16))
    pd.testing.assert_frame_equal(
        utility_targets(late_start, constants), utility_targets(early_start, constants)
    )


# --- Cross-check with the official scorer -------------------------------------------------------


def _check_against_scorer(labels: np.ndarray, constants: UtilityConstants, seed: int) -> None:
    utility = official_scorer().compute_prediction_utility
    table = patient_utility(labels, constants)
    n = len(labels)
    assert utility(labels, np.ones(n)) == pytest.approx(table["u1"].sum())
    assert utility(labels, np.zeros(n)) == pytest.approx(table["u0"].sum())
    predictions = np.random.default_rng(seed).integers(0, 2, n)
    for t in range(n):
        alarm, quiet = predictions.copy(), predictions.copy()
        alarm[t], quiet[t] = 1, 0
        flip = utility(labels, alarm) - utility(labels, quiet)
        assert flip == pytest.approx(table["utility_diff"].iloc[t], abs=1e-12)


@pytest.mark.parametrize(
    "labels",
    [
        _septic(14, 30),  # onset mid-stay
        _septic(0, 10),  # labelled from the first row
        _septic(3, 5),  # record ends before dt_late
        _septic(25, 60),  # long tail after dt_late
        np.zeros(9, dtype=int),  # never septic
    ],
)
def test_flipping_one_hour_changes_utility_by_u1_minus_u0(
    constants: UtilityConstants, labels: np.ndarray
) -> None:
    _check_against_scorer(labels, constants, seed=len(labels))


def test_toy_patients_match_the_scorer(constants: UtilityConstants) -> None:
    targets = utility_targets(toy_hours(), constants)
    for seed, (pid, rows) in enumerate(toy_hours().groupby("patient_id")):
        labels = rows.sort_values("hour")["SepsisLabel"].to_numpy()
        _check_against_scorer(labels, constants, seed)
        ours = targets[targets["patient_id"] == pid]["utility_diff"].to_numpy()
        np.testing.assert_allclose(ours, patient_utility(labels, constants)["utility_diff"])


def test_config_matches_the_scorer_defaults(constants: UtilityConstants) -> None:
    defaults = inspect.signature(official_scorer().compute_prediction_utility).parameters
    for name in UtilityConstants.__dataclass_fields__:
        assert getattr(constants, name) == defaults[name].default, name


# --- Table form and input checks ---------------------------------------------------------------


def test_utility_targets_columns_and_order(constants: UtilityConstants) -> None:
    shuffled = toy_hours().sample(frac=1, random_state=0)
    targets = utility_targets(shuffled, constants)
    assert list(targets.columns) == [
        "patient_id",
        "hour",
        "dt",
        "u1",
        "u0",
        "utility_diff",
        "weight",
    ]
    assert len(targets) == len(shuffled)
    ordered = targets.sort_values(["patient_id", "hour"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(targets, ordered)


def _bad(change) -> pd.DataFrame:
    table = toy_hours(["A_s"])
    return change(table)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda t: t[t["hour"] != 3], "complete record"),
        (lambda t: t.assign(SepsisLabel=t["SepsisLabel"].replace(1, 2)), "0 or 1"),
        (lambda t: pd.concat([t, t.iloc[[0]]]), "duplicate"),
        (lambda t: t.assign(hour=t["hour"].astype(float)), "integer"),
    ],
)
def test_bad_input_is_refused(constants: UtilityConstants, change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        utility_targets(_bad(change), constants)


@pytest.mark.parametrize("labels", [[0, 2], [0, -1, 1], [0.5, 1]])
def test_patient_utility_refuses_labels_other_than_0_or_1(
    constants: UtilityConstants, labels: list
) -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        patient_utility(labels, constants)


def test_inconsistent_constants_are_refused() -> None:
    config = load_yaml(LABELS_CONFIG)
    config["utility"]["dt_optimal"] = -5  # SepsisLabel is shifted by 6 rows, not 5
    with pytest.raises(ValueError, match="dt_optimal"):
        UtilityConstants.from_config(config)
    config["utility"].update(dt_optimal=-6, dt_early=-6)
    with pytest.raises(ValueError, match="dt_early"):
        UtilityConstants.from_config(config)


@pytest.mark.slow
def test_real_dev_patients_match_the_scorer(constants: UtilityConstants) -> None:
    from sepsis.data.splits import (
        IncompleteCacheError,
        SplitFingerprintError,
        load_patient_hours,
        load_project_split,
    )

    try:
        split = load_project_split()  # dev patients only (ADR-0004)
        sample = pd.concat(
            [
                split[split["septic"] == s].groupby("hospital").sample(50, random_state=0)
                for s in (0, 1)
            ]
        )
        hours = load_patient_hours(["A", "B"], sample["patient_id"].tolist())
    except (FileNotFoundError, IncompleteCacheError, SplitFingerprintError):
        pytest.skip("real data cache or split file not found")
    targets = utility_targets(hours[["patient_id", "hour", "SepsisLabel"]], constants)
    assert targets["patient_id"].nunique() == 200
    for seed, (_, rows) in enumerate(targets.groupby("patient_id")):
        labels = hours.loc[hours["patient_id"] == rows["patient_id"].iloc[0]]
        labels = labels.sort_values("hour")["SepsisLabel"].to_numpy()
        _check_against_scorer(labels, constants, seed)
        # The table form returns the same values, in hour order, for this patient.
        assert rows["hour"].tolist() == list(range(len(labels)))
        np.testing.assert_allclose(
            rows["utility_diff"].to_numpy(), patient_utility(labels, constants)["utility_diff"]
        )


def test_non_septic_dt_is_nan(constants: UtilityConstants) -> None:
    targets = utility_targets(toy_hours(["A_n"]), constants)
    assert all(math.isnan(v) for v in targets["dt"])
