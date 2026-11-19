"""Time serial vs parallel scoring on a finished run, without refitting anything.

    uv run python scripts/bench_scoring.py experiments/runs/<id>

Re-scores the run's saved scored predictions (`<setup>/predictions.parquet`) with the official
scorer, twice: one call after another (the old runner), then all calls at once in separate
processes (the runner with `parallel.setup_jobs` and `parallel.score_jobs`). Both must reproduce
the run's `scores.json` exactly, so the speed-up changes no number. Dev patients only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from sepsis.data.splits import load_patient_hours, load_project_split
from sepsis.eval.scorer import Scores, score
from sepsis.eval.tables import KEYS


def scoring_calls(run_dir: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """`{result name: (label rows, scored predictions)}`, one entry per official-scorer call."""
    split = load_project_split()
    hours = load_patient_hours(["A", "B"], split["patient_id"])
    labels = hours[[*KEYS, "hospital", "SepsisLabel"]]
    calls = {}
    for setup in ("A_to_B", "B_to_A", "mix"):
        path = run_dir / setup / "predictions.parquet"
        if not path.exists():
            continue
        predictions = pd.read_parquet(path)
        scored = predictions[predictions["kind"] == "scored"][[*KEYS, "score", "alarm"]]
        rows = labels[labels["patient_id"].isin(set(scored["patient_id"]))]
        if setup == "mix":
            for hospital, own in rows.groupby("hospital"):
                mine = scored[scored["patient_id"].isin(set(own["patient_id"]))]
                calls[f"mix_{hospital}"] = (own, mine)
        else:
            calls[setup] = (rows, scored)
    return calls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    expected = json.loads((args.run_dir / "scores.json").read_text(encoding="utf-8"))
    calls = scoring_calls(args.run_dir)

    start = time.perf_counter()
    serial = {name: score(*call) for name, call in calls.items()}
    serial_seconds = time.perf_counter() - start

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(calls)) as pool:
        futures = {name: pool.submit(score, *call) for name, call in calls.items()}
        parallel: dict[str, Scores] = {name: f.result() for name, f in futures.items()}
    parallel_seconds = time.perf_counter() - start

    same = True
    for name in calls:
        want = expected[name]["utility"]
        same &= serial[name].utility == want == parallel[name].utility
        print(
            f"{name:7s} run {want:.6f}  serial {serial[name].utility:.6f}  "
            f"parallel {parallel[name].utility:.6f}"
        )
    print(
        f"{len(calls)} scorer calls: serial {serial_seconds:.0f} s, parallel "
        f"{parallel_seconds:.0f} s ({serial_seconds / parallel_seconds:.1f}x)"
    )
    print("identical to scores.json" if same else "MISMATCH")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
