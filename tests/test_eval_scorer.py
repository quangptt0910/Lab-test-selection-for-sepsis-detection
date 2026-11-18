import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from eval_toy import DEV, hours, in_best_window, predictions
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis.config import DATA_CONFIG, EVAL_CONFIG, load_yaml
from sepsis.data.load import cache_path
from sepsis.eval.scorer import (
    ScorerModifiedError,
    Scores,
    apply_threshold,
    load_official_scorer,
    official_scorer,
    score,
    score_by_hospital,
    score_dirs,
    to_probability,
    write_label_dir,
    write_prediction_dir,
)

SCORER = load_yaml(EVAL_CONFIG)["scorer"]


def _all_zero(table: pd.DataFrame) -> pd.DataFrame:
    return predictions(table, 0.0, 0)


def _best(table: pd.DataFrame) -> pd.DataFrame:
    inside = in_best_window(table)
    return predictions(table, np.where(inside, 0.9, 0.1), inside.astype(int).to_numpy())


# --- the official scorer is used unmodified ---------------------------------------------------


def test_scorer_file_matches_the_stored_hash() -> None:
    scorer = load_official_scorer(Path(SCORER["path"]), SCORER["sha256"])
    assert callable(scorer.evaluate_sepsis_score)
    assert callable(scorer.compute_prediction_utility)


def test_a_changed_scorer_is_refused(tmp_path: Path) -> None:
    changed = tmp_path / "evaluate_sepsis_score.py"
    changed.write_text(
        Path(SCORER["path"]).read_text(encoding="utf-8").replace("u_fp     = -0.05", "u_fp = 0"),
        encoding="utf-8",
    )
    with pytest.raises(ScorerModifiedError):
        load_official_scorer(changed, SCORER["sha256"])


def test_line_endings_do_not_change_the_hash(tmp_path: Path) -> None:
    crlf = tmp_path / "evaluate_sepsis_score.py"
    text = Path(SCORER["path"]).read_bytes().replace(b"\r\n", b"\n")
    crlf.write_bytes(text.replace(b"\n", b"\r\n"))
    load_official_scorer(crlf, SCORER["sha256"])


# --- label and prediction folders -------------------------------------------------------------


def test_folders_have_one_file_per_patient_with_matching_names(tmp_path: Path) -> None:
    table = hours()
    shuffled = table.sample(frac=1, random_state=0)
    preds = _best(table).sample(frac=1, random_state=1)
    write_label_dir(shuffled, tmp_path / "labels")
    write_prediction_dir(preds, tmp_path / "predictions")

    names = sorted(p.name for p in (tmp_path / "labels").iterdir())
    assert names == sorted(f"{pid}.psv" for pid in DEV)
    assert names == sorted(p.name for p in (tmp_path / "predictions").iterdir())

    labels = (tmp_path / "labels" / "A_s.psv").read_text().splitlines()
    assert labels[0] == "SepsisLabel"
    assert [int(v) for v in labels[1:]] == [0] * 14 + [1] * 16  # hour order, not input order

    lines = (tmp_path / "predictions" / "A_s.psv").read_text().splitlines()
    assert lines[0] == "PredictedProbability|PredictedLabel"
    assert [int(line.split("|")[1]) for line in lines[1:]] == [0] * 8 + [1] * 16 + [0] * 6


def test_only_the_requested_patients_are_written(tmp_path: Path) -> None:
    write_label_dir(hours(["A_s", "B_n"]), tmp_path / "labels")
    assert sorted(p.stem for p in (tmp_path / "labels").iterdir()) == ["A_s", "B_n"]


def test_writers_refuse_a_non_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "old.psv").write_text("SepsisLabel\n0\n")
    with pytest.raises(FileExistsError):
        write_label_dir(hours(), tmp_path)
    with pytest.raises(FileExistsError):
        write_prediction_dir(_all_zero(hours()), tmp_path)


# --- scoring ----------------------------------------------------------------------------------


def test_all_zero_predictions_score_zero() -> None:
    table = hours()
    assert score(table, _all_zero(table)).utility == 0.0


def test_best_predictions_score_one() -> None:
    table = hours()
    result = score(table, _best(table))
    assert isinstance(result, Scores)
    assert result.utility == pytest.approx(1.0, abs=1e-12)


def test_score_equals_running_the_scorer_as_a_script(tmp_path: Path) -> None:
    table = hours()
    preds = apply_threshold(
        predictions(table, np.random.default_rng(0).normal(size=len(table)), 0), 0.3
    )
    ours = score(table, preds, work_dir=tmp_path)
    out = tmp_path / "scores.psv"
    subprocess.run(
        [
            sys.executable,
            SCORER["path"],
            str(tmp_path / "labels"),
            str(tmp_path / "predictions"),
            str(out),
        ],
        check=True,
    )
    header, values = out.read_text().splitlines()
    assert header == "AUROC|AUPRC|Accuracy|F-measure|Utility"
    official = [float(v) for v in values.split("|")]
    assert [ours.auroc, ours.auprc, ours.accuracy, ours.f_measure, ours.utility] == official
    assert score_dirs(tmp_path / "labels", tmp_path / "predictions") == ours


def test_scores_outside_0_1_keep_their_ranking() -> None:
    table = hours()
    raw = np.random.default_rng(1).uniform(-0.5, 2.5, size=len(table))  # like U1 - U0 outputs
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the scorer warns on out-of-range or inconsistent values
        result = score(table, apply_threshold(predictions(table, raw, 0), 1.0))
    # Rank-based references on the raw scores: the transform must not change the ranking.
    labels = table["SepsisLabel"].to_numpy()
    assert result.auroc == pytest.approx(roc_auc_score(labels, raw))
    assert result.auprc == pytest.approx(average_precision_score(labels, raw))

    # Written as they are, scores above 1 collapse into one ROC point in the scorer.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unsquashed, _ = official_scorer().compute_auc(labels, raw)
    assert unsquashed != pytest.approx(result.auroc)


def test_to_probability_is_monotone_and_inside_0_1() -> None:
    raw = np.array([-3.0, -0.05, 0.0, 0.5, 1.0, 2.0, 8.0])
    prob = to_probability(raw)
    assert ((prob > 0) & (prob < 1)).all()
    assert (np.diff(prob) > 0).all()


def test_apply_threshold_is_strict() -> None:
    table = pd.DataFrame({"patient_id": "A_s", "hour": [0, 1, 2], "score": [0.4, 0.5, 0.6]})
    assert apply_threshold(table, 0.5)["alarm"].tolist() == [0, 0, 1]


@pytest.mark.parametrize(
    "broken",
    [
        lambda p: p.iloc[1:],  # a missing row
        lambda p: pd.concat([p, p.iloc[[0]]]),  # a duplicated row
        lambda p: p.assign(score=np.nan),
        lambda p: p.assign(alarm=2),
        lambda p: p.drop(columns="alarm"),
    ],
)
def test_bad_prediction_tables_are_refused(broken) -> None:
    table = hours()
    with pytest.raises(ValueError):
        score(table, broken(_all_zero(table)))


def test_incomplete_records_and_non_integer_hours_are_refused() -> None:
    table = hours()
    preds = _all_zero(table)
    # Dropping a patient's first rows from both tables would move its onset row.
    late = table["hour"] >= 10
    with pytest.raises(ValueError, match="complete record"):
        score(table[late], preds[late])
    as_text = {"hour": table["hour"].astype(str)}
    with pytest.raises(ValueError, match="integer"):
        score(table.assign(**as_text), preds.assign(**as_text))
    with pytest.raises(ValueError, match="finite"):
        score(table, preds.assign(score=np.inf))


def test_score_by_hospital_refuses_missing_hospitals_and_stray_predictions() -> None:
    table = hours()
    preds = _all_zero(table)
    unknown = table.assign(hospital=table["hospital"].where(table["patient_id"] != "B_n"))
    with pytest.raises(ValueError, match="hospital"):
        score_by_hospital(unknown, preds)
    with pytest.raises(ValueError, match="no label rows"):
        score_by_hospital(table[table["patient_id"] != "B_n"], preds)


def test_a_set_without_septic_patients_is_refused() -> None:
    table = hours(["A_n", "B_n"])
    with pytest.raises(ValueError, match="septic"):
        score(table, _all_zero(table))


def test_score_by_hospital_scores_each_hospital_alone() -> None:
    table = hours()
    # Good alarms for hospital A, none for hospital B.
    inside = in_best_window(table) & (table["hospital"] == "A")
    preds = predictions(table, np.where(inside, 0.9, 0.1), inside.astype(int).to_numpy())
    by_hospital = score_by_hospital(table, preds)

    assert sorted(by_hospital) == ["A", "B"]
    for name in ("A", "B"):
        rows = table[table["hospital"] == name]
        alone = score(rows, preds[preds["patient_id"].isin(rows["patient_id"])])
        assert by_hospital[name] == alone
    assert by_hospital["A"].utility == pytest.approx(1.0)
    assert by_hospital["B"].utility == 0.0
    pooled = score(table, preds).utility
    assert 0.0 < pooled < 1.0


# --- real data --------------------------------------------------------------------------------


@pytest.mark.slow
def test_all_zero_scores_zero_on_a_dev() -> None:
    from sepsis.data.splits import cross_hospital, load_patient_hours, load_project_split
    from sepsis.eval.threshold import tune_threshold

    data = load_yaml(DATA_CONFIG)
    if not (
        cache_path(Path(data["cache_dir"]), "A").exists() and Path(data["split_path"]).exists()
    ):
        pytest.skip("cache or split file not found")
    split = load_project_split()
    train_side, _ = cross_hospital(split, "A")
    table = load_patient_hours(["A"], train_side["patient_id"])
    assert score(table, _all_zero(table)).utility == 0.0

    # Random scores: this only measures how long one tuning takes on a full training side.
    noise = predictions(table, np.random.default_rng(0).normal(size=len(table)), 0)
    start = time.perf_counter()
    result = tune_threshold(
        table, noise[["patient_id", "hour", "score"]], split, "A_to_B",
        grid=load_yaml(EVAL_CONFIG)["threshold"]["grid"],
    )  # fmt: skip
    print(f"tuned on {table['patient_id'].nunique()} patients, {len(result.curve)} thresholds, "
          f"{time.perf_counter() - start:.1f} s")  # fmt: skip
    assert np.isfinite(result.threshold)
