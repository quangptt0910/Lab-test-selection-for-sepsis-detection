"""Data-quality checks on patient-hour tables (M0-3), and the report written from them.

Run with ``uv run python -m sepsis.data.quality``. It checks dev patients only (ADR-0004) and
writes aggregates (counts of rows and patients), never patient rows, to
``docs/eda/data_quality.md``. Limits come from ``configs/cleaning.yaml``.

The check functions are pure: each takes a table with `patient_id` and `hour` and returns counts.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from sepsis.config import load_yaml
from sepsis.data.splits import (
    IncompleteCacheError,
    SplitFingerprintError,
    load_patient_hours,
    load_project_split,
)

CLEANING_CONFIG = Path("configs/cleaning.yaml")
REPORT_PATH = Path("docs/eda/data_quality.md")
NOT_CHECKED = {"patient_id", "hospital", "hour", "ICULOS", "SepsisLabel"}


def _in_hour_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["patient_id", "hour"], kind="stable")


def range_violations(df: pd.DataFrame, ranges: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Rows and patients with a value below `low` or above `high`, per column and side.

    NaN is never a violation. Only sides with at least one violation are returned.
    """
    out = []
    for column, rule in ranges.items():
        if column not in df.columns:
            continue
        values = df[column]
        for side, mask in (("low", values < rule["low"]), ("high", values > rule["high"])):
            if mask.any():
                out.append((column, side, int(mask.sum()), df.loc[mask, "patient_id"].nunique()))
    return pd.DataFrame(out, columns=["column", "side", "rows", "patients"])


def unit_suspects(df: pd.DataFrame, suspects: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Rows and patients whose value falls in a window that suggests another unit."""
    out = []
    for column, rule in suspects.items():
        if column not in df.columns:
            continue
        mask = df[column].between(rule["low"], rule["high"])
        if mask.any():
            patients = df.loc[mask, "patient_id"].nunique()
            out.append((column, rule["meaning"], int(mask.sum()), patients))
    return pd.DataFrame(out, columns=["column", "meaning", "rows", "patients"])


def iculos_problems(df: pd.DataFrame) -> dict[str, int]:
    """Patients whose ICULOS doesn't start at 1, repeats, jumps by more than 1 or decreases.

    Steps are taken in row (`hour`) order, never in ICULOS order.
    """
    ordered = _in_hour_order(df)
    by_patient = ordered.groupby("patient_id", sort=False)["ICULOS"]
    step = by_patient.diff()
    pids = ordered["patient_id"]
    return {
        "not_starting_at_1": int((by_patient.first() != 1).sum()),
        "duplicates": pids[step == 0].nunique(),
        "gaps": pids[step > 1].nunique(),
        "decreasing": pids[step < 0].nunique(),
    }


def label_reversals(df: pd.DataFrame) -> int:
    """Patients whose SepsisLabel goes from 1 back to 0."""
    ordered = _in_hour_order(df)
    step = ordered.groupby("patient_id", sort=False)["SepsisLabel"].diff()
    return ordered.loc[step < 0, "patient_id"].nunique()


def empty_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Columns with no value at all."""
    return [c for c in columns if c in df.columns and df[c].isna().all()]


def constant_columns(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Columns with values that are all the same (empty columns excluded)."""
    return [c for c in columns if c in df.columns and df[c].nunique(dropna=True) == 1]


def bp_inconsistent(df: pd.DataFrame) -> tuple[int, int]:
    """(rows, patients) where DBP >= SBP in the same hour; rows missing either are skipped."""
    if not {"SBP", "DBP"} <= set(df.columns):
        return 0, 0
    bad = df["SBP"].notna() & df["DBP"].notna() & (df["DBP"] >= df["SBP"])
    return int(bad.sum()), df.loc[bad, "patient_id"].nunique()


def short_stays(df: pd.DataFrame, min_hours: int) -> int:
    """Patients with fewer than `min_hours` rows."""
    return int((df.groupby("patient_id").size() < min_hours).sum())


def _hospital_findings(df: pd.DataFrame, config: Mapping[str, Any]) -> list[tuple]:
    """(check, column, rows, patients) for one hospital; rows is None for patient-level checks."""
    out: list[tuple] = []
    for r in range_violations(df, config.get("ranges", {})).itertuples():
        out.append((f"out of range ({r.side})", r.column, r.rows, r.patients))
    for r in unit_suspects(df, config.get("unit_suspects", {})).itertuples():
        out.append((f"looks like {r.meaning}", r.column, r.rows, r.patients))
    if {"SBP", "DBP"} <= set(df.columns):
        rows, patients = bp_inconsistent(df)
        out.append(("DBP >= SBP", _bp_label(config), rows, patients))
    if "ICULOS" in df.columns:
        names = {
            "not_starting_at_1": "ICULOS doesn't start at 1",
            "duplicates": "ICULOS repeats",
            "gaps": "ICULOS skips hours",
            "decreasing": "ICULOS decreases",
        }
        for key, count in iculos_problems(df).items():
            out.append((names[key], "ICULOS", None, count))
    if "SepsisLabel" in df.columns:
        out.append(("label goes 1 -> 0", "", None, label_reversals(df)))
    out.append(
        (f"fewer than {config['min_hours']} hours", "", None, short_stays(df, config["min_hours"]))
    )
    columns = [c for c in df.columns if c not in NOT_CHECKED]
    out += [
        ("empty column", c, None, df["patient_id"].nunique()) for c in empty_columns(df, columns)
    ]
    out += [("constant column", c, None, None) for c in constant_columns(df, columns)]
    return out


ROW_LEVEL_CHECKS = ("out of range", "looks like", "DBP >= SBP")
CHECK_ORDER = (
    "out of range",
    "looks like",
    "DBP >= SBP",
    "ICULOS",
    "label",
    "fewer",
    "empty",
    "constant",
)


def _bp_label(config: Mapping[str, Any]) -> str:
    """The pressures the consistency rule blanks, as shown in the report's `column`."""
    return ", ".join(config.get("bp_consistency", {}).get("columns", ["SBP", "DBP"]))


def _check_rank(check: str) -> int:
    """Position of a check's group in the report, so each group's rows stay together."""
    return next(i for i, prefix in enumerate(CHECK_ORDER) if check.startswith(prefix))


def _decision(check: str, column: str, config: Mapping[str, Any]) -> tuple[str, str]:
    """(decision, reason) configured for a finding; empty strings if nothing is configured."""
    rule: Mapping[str, Any] = {}
    if check.startswith("out of range"):
        rule = config.get("ranges", {}).get(column, {})
    elif check.startswith("looks like"):
        rule = config.get("unit_suspects", {}).get(column, {})
    elif check == "DBP >= SBP":
        rule = config.get("bp_consistency", {})
    elif column and column in config.get("drop_columns", {}):
        return "drop column", config["drop_columns"][column]
    return rule.get("action", ""), rule.get("reason", "")


def quality_findings(df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """One row per (check, column), with `<hospital>_rows` and `<hospital>_patients` columns.

    `decision` and `reason` come from the config, empty where nothing is configured yet.
    """
    hospitals = sorted(df["hospital"].unique())
    found: dict[tuple[str, str], dict[str, tuple]] = {}
    for hospital in hospitals:
        for check, column, rows, patients in _hospital_findings(
            df[df["hospital"] == hospital], config
        ):
            found.setdefault((check, column), {})[hospital] = (rows, patients)

    records = []
    for (check, column), by_hospital in found.items():
        record: dict[str, Any] = {"check": check, "column": column}
        row_level = check.startswith(ROW_LEVEL_CHECKS)
        for h in hospitals:
            if h in by_hospital:
                rows, patients = by_hospital[h]
            else:  # this hospital did not report the problem: none found there
                rows = 0 if row_level else None
                patients = None if check == "constant column" else 0
            record[f"{h}_rows"], record[f"{h}_patients"] = rows, patients
        record["decision"], record["reason"] = _decision(check, column, config)
        records.append(record)

    columns = [f"{h}_{kind}" for h in hospitals for kind in ("rows", "patients")]
    out = pd.DataFrame(records, columns=["check", "column", *columns, "decision", "reason"])
    for column in columns:
        out[column] = pd.array(out[column].tolist(), dtype="Int64")
    order = out["check"].map(_check_rank)
    return out.iloc[order.argsort(kind="stable")].reset_index(drop=True)


def format_report(findings: pd.DataFrame, df: pd.DataFrame, notes: Sequence[str] = ()) -> str:
    """Markdown report: totals per hospital, the findings table and notes (aggregates only)."""
    totals = [
        f"- {h}: {g['patient_id'].nunique()} patients, {len(g)} patient-hours"
        for h, g in df.groupby("hospital")
    ]
    header = "| " + " | ".join(findings.columns) + " |"
    rule = "|" + "---|" * len(findings.columns)
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in findings.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# Data quality (dev patients)",
            "",
            "Generated by `python -m sepsis.data.quality` from `configs/cleaning.yaml`. "
            "Counts only; test patients are never read.",
            "",
            *totals,
            "",
            "Empty `rows` means the check counts patients, not rows. "
            "`decision` and `reason` come from `configs/cleaning.yaml` (empty: no cleaning).",
            "",
            header,
            rule,
            *body,
            "",
            *(["## Notes", "", *(f"- {note}" for note in notes), ""] if notes else []),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CLEANING_CONFIG)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    try:
        split = load_project_split()  # dev patients only
        df = load_patient_hours(sorted(split["hospital"].unique()), split["patient_id"])
    except (FileNotFoundError, SplitFingerprintError, IncompleteCacheError) as error:
        print(f"{error}\nRun build_cache and make_split first (see README).", file=sys.stderr)
        return 1

    config = load_yaml(args.config)
    findings = quality_findings(df, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = format_report(findings, df, notes=config.get("notes", []))
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
