import numpy as np
import pandas as pd

from sepsis.eda.run import END, START, analyse, markdown_table, merge_readme


def test_merge_readme_creates_and_keeps_findings() -> None:
    first = merge_readme(None, "tables v1", "# Title")
    assert first.startswith("# Title") and "tables v1" in first and "## Findings" in first
    edited = first.replace("## Findings\n", "## Findings\n\n- Hand-written finding.\n")
    second = merge_readme(edited, "tables v2", "# Title")
    assert "tables v2" in second and "tables v1" not in second
    assert "- Hand-written finding." in second
    assert second.count(START) == 1 and second.count(END) == 1


def test_markdown_table_rounds_and_blanks_nan() -> None:
    text = markdown_table(pd.DataFrame({"a": ["x"], "b": [1.234], "c": [np.nan]}), digits=1)
    assert text.splitlines() == ["| a | b | c |", "|---|---|---|", "| x | 1.2 |  |"]


def test_analyse_smoke_on_tiny_tables(tmp_path) -> None:
    rng = np.random.default_rng(0)
    frames = []
    for hospital in "AB":
        for i in range(6):
            n = 30
            septic = i < 2
            label = [0] * 10 + [1] * 20 if septic else [0] * n
            frames.append(
                pd.DataFrame(
                    {
                        "patient_id": f"{hospital}_p{i}",
                        "hospital": hospital,
                        "hour": range(n),
                        "ICULOS": range(1, n + 1),
                        "SepsisLabel": label,
                        "HR": rng.normal(85, 10, n),
                        "MAP": rng.normal(75, 8, n),
                        "Lactate": np.where(np.arange(n) % 6 == 0, rng.normal(2, 0.5, n), np.nan),
                        "pH": np.where(np.arange(n) % 6 == 0, rng.normal(7.4, 0.05, n), np.nan),
                        "WBC": np.where(np.arange(n) % 12 == 3, rng.normal(10, 2, n), np.nan),
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    config = {
        "septic_on_arrival_rows": 1,
        "trajectory": {
            "before": 10,
            "after": 3,
            "min_patients": 1,
            "forward_fill": True,
            "columns": ["HR", "Lactate"],
        },
        "min_shared_hours": 5,
        "min_values": 5,
        "redundancy_threshold": 0.9,
        "panel_distance_threshold": 0.6,
        "shift_threshold": 0.2,
        "figure_dpi": 50,
    }
    text = analyse(df, config, tmp_path / "figures")
    assert "## 1. Cohort" in text and "## 6. Hospital A vs B" in text
    figures = {p.name for p in (tmp_path / "figures").iterdir()}
    assert {"cohort.png", "co_measurement.png", "trajectories.png", "pca.png"} <= figures
    assert "A_p0" not in text  # aggregates only: no patient ids in the report
