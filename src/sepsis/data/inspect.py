"""Print a descriptive summary of the dev patients of each hospital.

Run with ``uv run python -m sepsis.data.inspect`` (always ``-m``: this module's
name shadows the stdlib ``inspect`` if its folder lands on ``sys.path``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.config import DATA_CONFIG, load_yaml
from sepsis.data.load import HOSPITAL_DIRS, ID_COLUMNS, iter_patients
from sepsis.data.splits import SplitFingerprintError, load_patient_hours, load_project_split


def load_patients(hospital_dir: Path) -> Iterator[pd.DataFrame]:
    """Yield one table per patient file, one at a time."""
    for _, df in iter_patients(hospital_dir):
        yield df


def summarize(patients: Iterable[pd.DataFrame]) -> dict:
    """Count patients, hours per patient, septic share and missingness."""
    hours: list[int] = []
    septic = 0
    nan_counts: pd.Series | None = None
    for df in patients:
        hours.append(len(df))
        septic += int(df["SepsisLabel"].max() == 1)
        counts = df.isna().sum()
        nan_counts = counts if nan_counts is None else nan_counts.add(counts, fill_value=0)
    n = len(hours)
    total_rows = sum(hours)
    pct_missing = {} if nan_counts is None else (100.0 * nan_counts / total_rows).to_dict()
    return {
        "n_patients": n,
        "n_rows": total_rows,
        "hours_min": int(min(hours)) if n else 0,
        "hours_median": float(np.median(hours)) if n else 0.0,
        "hours_max": int(max(hours)) if n else 0,
        "pct_septic": 100.0 * septic / n if n else 0.0,
        "pct_missing": pct_missing,
    }


def format_report(name: str, s: dict) -> str:
    lines = [
        f"== Hospital {name} ==",
        f"patients: {s['n_patients']}  (patient-hours: {s['n_rows']})",
        f"hours per patient: min {s['hours_min']} / median {s['hours_median']:g} "
        f"/ max {s['hours_max']}",
        f"septic patients: {s['pct_septic']:.2f}%",
        "% missing per column:",
    ]
    for col, pct in sorted(s["pct_missing"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {col:<18}{pct:6.2f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hospital", choices=sorted(HOSPITAL_DIRS), default=None)
    args = parser.parse_args(argv)

    # Dev patients only (ADR-0004): needs the cache and the split.
    try:
        split = load_project_split()
        cache_dir = Path(load_yaml(DATA_CONFIG)["cache_dir"])
        names = [args.hospital] if args.hospital else sorted(HOSPITAL_DIRS)
        hours = load_patient_hours(names, split, cache_dir)
    except (FileNotFoundError, SplitFingerprintError) as error:
        print(f"{error}\nRun build_cache and make_split first (see README).", file=sys.stderr)
        return 1

    reports = []
    for name in names:
        rows = hours[hours["hospital"] == name].drop(columns=ID_COLUMNS[1:])
        patients = (group.drop(columns="patient_id") for _, group in rows.groupby("patient_id"))
        reports.append(format_report(f"{name} (dev patients)", summarize(patients)))
    print("\n\n".join(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
