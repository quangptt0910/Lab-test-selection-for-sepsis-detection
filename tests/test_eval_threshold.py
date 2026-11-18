import numpy as np
import pandas as pd
import pytest
from eval_toy import hours, in_best_window, split_table

from sepsis.eval.scorer import apply_threshold, score
from sepsis.eval.threshold import (
    ThresholdLeakError,
    best_threshold,
    threshold_grid,
    training_side_patients,
    tune_threshold,
)

GRID = {"quantile_min": 0.0, "quantile_max": 1.0, "n_points": 50}
A_SIDE = ["A_s", "A_n"]


def _scores(table: pd.DataFrame) -> pd.DataFrame:
    """0.9 inside the best alarm window, 0.2 elsewhere, 0.5 on non-septic patients."""
    septic = table.groupby("patient_id")["SepsisLabel"].transform("max") == 1
    value = np.where(in_best_window(table), 0.9, np.where(septic, 0.2, 0.5))
    return table[["patient_id", "hour"]].assign(score=value)


def test_grid_holds_observed_scores_sorted_and_unique() -> None:
    scores = np.array([0.3, 0.1, 0.1, 0.7, 0.5])
    grid = threshold_grid(scores, quantile_min=0.0, quantile_max=1.0, n_points=20)
    assert grid.tolist() == [0.1, 0.3, 0.5, 0.7]
    upper = threshold_grid(scores, quantile_min=0.5, quantile_max=1.0, n_points=20)
    assert upper.min() >= 0.3


@pytest.mark.parametrize("bad", [(0.5, 0.2, 10), (-0.1, 1.0, 10), (0.0, 1.1, 10), (0.0, 1.0, 0)])
def test_grid_rejects_bad_settings(bad) -> None:
    with pytest.raises(ValueError):
        threshold_grid(np.array([0.1, 0.2]), *bad)


def test_best_threshold_takes_the_highest_of_tied_thresholds() -> None:
    curve = pd.DataFrame({"threshold": [0.1, 0.2, 0.3, 0.4], "utility": [0.5, 0.7, 0.7, 0.1]})
    assert best_threshold(curve) == (0.3, 0.7)


def test_tuning_finds_the_best_threshold_on_a_toy_set() -> None:
    table = hours(A_SIDE)
    result = tune_threshold(table, _scores(table), split_table(), "A_to_B", grid=GRID)
    # Above 0.2 the non-septic patient also alarms; above 0.5 only the best window does.
    assert result.threshold == 0.5
    assert result.utility == pytest.approx(1.0)
    assert result.curve["threshold"].tolist() == [0.2, 0.5, 0.9]
    assert result.curve["utility"].iloc[-1] == 0.0  # no score exceeds 0.9: no alarm


@pytest.mark.parametrize("patients", [A_SIDE, ["B_s", "B_n"]])
def test_tuned_utility_equals_the_official_score_at_that_threshold(patients) -> None:
    table = hours(patients)
    setup = "A_to_B" if patients == A_SIDE else "B_to_A"
    noisy = table[["patient_id", "hour"]].assign(
        score=np.random.default_rng(3).normal(size=len(table))
    )
    result = tune_threshold(table, noisy, split_table(), setup, grid=GRID)
    for threshold, utility in result.curve.itertuples(index=False):
        official = score(table, apply_threshold(noisy, threshold)).utility
        assert utility == pytest.approx(official, abs=1e-9)
    assert result.utility == result.curve["utility"].max()


def test_training_side_per_setup() -> None:
    split = split_table()
    assert training_side_patients(split, "A_to_B") == {"A_s", "A_n"}
    assert training_side_patients(split, "B_to_A") == {"B_s", "B_n"}
    assert training_side_patients(split, "mix", outer_fold=0) == {"A_n", "B_s"}
    assert training_side_patients(split, "mix", outer_fold=1) == {"A_s", "B_n"}


@pytest.mark.parametrize(
    ("setup", "outer_fold", "patients"),
    [
        ("A_to_B", None, ["A_s", "A_n", "B_n"]),  # a patient of the evaluated hospital
        ("B_to_A", None, ["B_s", "B_n", "A_n"]),
        ("mix", 0, ["A_n", "B_s", "B_n"]),  # a patient of the held-out fold
        ("A_to_B", None, ["A_s", "A_n", "A_t"]),  # a locked test patient
        ("mix", 1, ["A_s", "B_n", "A_t"]),
    ],
)
def test_tuning_refuses_patients_outside_the_training_side(setup, outer_fold, patients) -> None:
    table = hours(patients)
    with pytest.raises(ThresholdLeakError):
        tune_threshold(
            table, _scores(table), split_table(), setup, outer_fold=outer_fold, grid=GRID
        )


def test_tuning_works_on_a_mix_training_side() -> None:
    table = hours(["A_n", "B_s"])
    result = tune_threshold(table, _scores(table), split_table(), "mix", outer_fold=0, grid=GRID)
    assert result.threshold == 0.5


def test_mix_needs_an_outer_fold_and_setups_are_checked() -> None:
    table = hours(A_SIDE)
    with pytest.raises(ValueError, match="outer_fold"):
        tune_threshold(table, _scores(table), split_table(), "mix", grid=GRID)
    with pytest.raises(ValueError, match="outer_fold"):
        tune_threshold(table, _scores(table), split_table(), "mix", outer_fold=7, grid=GRID)
    with pytest.raises(ValueError, match="outer_fold"):
        tune_threshold(table, _scores(table), split_table(), "A_to_B", outer_fold=0, grid=GRID)
    with pytest.raises(ValueError, match="setup"):
        tune_threshold(table, _scores(table), split_table(), "A_to_C", grid=GRID)


def test_a_malformed_split_table_is_refused() -> None:
    table = hours(A_SIDE + ["A_t"])
    split = split_table()
    # The test patient listed a second time as a dev patient must not unlock it.
    forged = pd.concat([split, split[split["patient_id"] == "A_t"].assign(split="dev", fold=0)])
    with pytest.raises(ValueError, match="more than once"):
        tune_threshold(table, _scores(table), forged, "A_to_B", grid=GRID)
    no_fold = split.assign(fold=split["fold"].astype("float").where(split["patient_id"] != "A_n"))
    with pytest.raises(ValueError, match="without a fold"):
        training_side_patients(no_fold, "mix", outer_fold=0)


def test_tuning_refuses_bad_labels_and_scores() -> None:
    table = hours(A_SIDE)
    bad_label = table.assign(SepsisLabel=table["SepsisLabel"].replace(1, 2))
    with pytest.raises(ValueError, match="SepsisLabel"):
        tune_threshold(bad_label, _scores(table), split_table(), "A_to_B", grid=GRID)
    with pytest.raises(ValueError, match="finite"):
        tune_threshold(
            table, _scores(table).assign(score=np.inf), split_table(), "A_to_B", grid=GRID
        )


def test_tuning_refuses_empty_or_misaligned_predictions() -> None:
    table = hours(A_SIDE)
    with pytest.raises(ValueError):
        tune_threshold(table.iloc[:0], _scores(table).iloc[:0], split_table(), "A_to_B", grid=GRID)
    with pytest.raises(ValueError):
        tune_threshold(table, _scores(table).iloc[1:], split_table(), "A_to_B", grid=GRID)
    non_septic = hours(["A_n"])
    with pytest.raises(ValueError, match="septic"):
        tune_threshold(non_septic, _scores(non_septic), split_table(), "A_to_B", grid=GRID)
