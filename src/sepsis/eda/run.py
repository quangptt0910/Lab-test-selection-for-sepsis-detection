"""Run the exploratory analysis of the dev patients (M0-4) and write docs/eda/.

Run with ``uv run python -m sepsis.eda.run``. It loads dev patients only (ADR-0004), applies the
M0-3 cleaning, and writes figures to ``docs/eda/figures/`` and tables to the generated part of
``docs/eda/README.md``. The hand-written findings below the generated part are kept.
Settings come from ``configs/eda.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sepsis.config import load_yaml
from sepsis.data.clean import clean
from sepsis.data.columns import LABS, MEASUREMENTS, VITALS, present
from sepsis.data.patients import patient_table
from sepsis.data.quality import CLEANING_CONFIG
from sepsis.data.splits import (
    IncompleteCacheError,
    SplitFingerprintError,
    load_patient_hours,
    load_project_split,
)
from sepsis.eda import plots, stats

EDA_CONFIG = Path("configs/eda.yaml")
OUT_DIR = Path("docs/eda")
START = "<!-- generated:start (python -m sepsis.eda.run; do not edit by hand) -->"
END = "<!-- generated:end -->"
FINDINGS_PLACEHOLDER = (
    "## Findings\n\n"
    "Hand-written after reading the tables and figures above: finding -> what it means for the "
    "project. Kept when the analysis is re-run.\n"
)


def merge_readme(existing: str | None, generated: str, title: str) -> str:
    """Put `generated` between the markers, keeping everything outside them."""
    block = f"{START}\n{generated.strip()}\n{END}"
    if not existing or START not in existing or END not in existing:
        return f"{title}\n\n{block}\n\n{FINDINGS_PLACEHOLDER}"
    head, rest = existing.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{block}{tail}"


def markdown_table(df: pd.DataFrame, digits: int = 1) -> str:
    """A pipe table; floats rounded to `digits`, NaN shown empty."""

    def cell(v) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.{digits}f}"
        return str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def _panels_table(panels: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for hospital, mapping in panels.items():
        groups: dict[int, list[str]] = {}
        for lab, cluster in mapping.items():
            groups.setdefault(cluster, []).append(lab)
        for labs in sorted(groups.values(), key=lambda g: (-len(g), g[0])):
            rows.append({"hospital": hospital, "size": len(labs), "labs": ", ".join(labs)})
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame, config: dict, fig_dir: Path) -> str:
    """Compute every table and figure; return the generated README part (markdown)."""
    dpi = config["figure_dpi"]
    vitals, labs = present(VITALS, df.columns), present(LABS, df.columns)
    measured = present(MEASUREMENTS, df.columns)
    hospitals = sorted(df["hospital"].unique())
    by_hospital = {h: df[df["hospital"] == h] for h in hospitals}
    sections: list[str] = []

    # 1. Cohort
    cohort = stats.cohort_summary(df, config["septic_on_arrival_rows"]).reset_index()
    plots.length_and_onset(patient_table(df), fig_dir / "cohort.png", dpi)
    sections += [
        "## 1. Cohort",
        markdown_table(cohort),
        "Hours are rows. `onset_hour_median` is t_sepsis (first label row + 6) of septic "
        "patients; `septic_on_arrival_pct` is the share of septic patients whose label is 1 "
        f"within the first {config['septic_on_arrival_rows']} row(s).",
        "![cohort](figures/cohort.png)",
    ]

    # 2. Missingness and time between measurements
    miss = stats.missingness(df, measured)
    gaps = stats.measurement_gaps(df, measured)
    plots.paired_bars(
        miss, "pct_hours", "Hours with a value", "% of hours", fig_dir / "missingness.png", dpi
    )
    plots.paired_bars(
        gaps[gaps["column"].isin(labs)],
        "median_hours",
        "Median hours between two values of a lab",
        "hours",
        fig_dir / "lab_gaps.png",
        dpi,
    )
    wide = miss.merge(gaps, on=["hospital", "column"]).pivot(
        index="column", columns="hospital", values=["pct_hours", "pct_patients", "median_hours"]
    )
    wide.columns = [f"{v}_{h}" for v, h in wide.columns]
    wide = wide.reindex(measured).reset_index()
    sections += [
        "## 2. Missingness and time between measurements",
        "`pct_hours`: % of hours with a value. `pct_patients`: % of patients with at least one "
        "value. `median_hours`: median rows between two values of the same patient. "
        "Describes how often labs were ordered; never a feature (ADR-0002).",
        markdown_table(wide),
        "![missingness](figures/missingness.png)",
        "![lab gaps](figures/lab_gaps.png)",
    ]

    # 3. Co-measurement and candidate panels
    matrices = {h: stats.co_measurement(g, labs) for h, g in by_hospital.items()}
    panels = {
        h: stats.candidate_panels(m, config["panel_distance_threshold"])
        for h, m in matrices.items()
    }
    order = sorted(labs, key=lambda lab: (panels[hospitals[0]].get(lab, 99), lab))
    plots.co_measurement_heatmaps(matrices, order, fig_dir / "co_measurement.png", dpi)
    sections += [
        "## 3. Labs measured together (candidate panels)",
        "Labs clustered on 1 - (how often they share an hour), average linkage, distance "
        f"threshold {config['panel_distance_threshold']}. Input for `configs/panels.yaml` (M0-6).",
        markdown_table(_panels_table(panels)),
        "![co-measurement](figures/co_measurement.png)",
    ]

    # 4. Septic vs non-septic
    traj_cfg = config["trajectory"]
    traj_columns = present(traj_cfg["columns"], df.columns)
    contrast = stats.septic_vs_nonseptic(df, measured)
    traj = stats.onset_trajectories(
        df,
        traj_columns,
        traj_cfg["before"],
        traj_cfg["after"],
        traj_cfg["min_patients"],
        forward_fill=traj_cfg["forward_fill"],
    )
    plots.trajectories(traj, contrast, traj_columns, fig_dir / "trajectories.png", dpi)
    plots.smd_bars(
        contrast,
        "Septic (label-1 hours) vs never-septic hours",
        fig_dir / "septic_vs_nonseptic.png",
        dpi,
    )
    enough = contrast[contrast[["n_septic", "n_nonseptic"]].min(axis=1) >= config["min_values"]]
    top = enough.assign(abs_smd=enough["smd"].abs()).sort_values("abs_smd", ascending=False)
    top = top.groupby("hospital").head(10).drop(columns="abs_smd")
    sections += [
        "## 4. Septic vs non-septic",
        "Label-1 hours of septic patients against all hours of patients who never become septic. "
        f"Top 10 per hospital by |standardized mean difference|, among variables with at least "
        f"{config['min_values']} values on each side:",
        markdown_table(top, digits=2),
        "![septic vs non-septic](figures/septic_vs_nonseptic.png)",
        "Onset-aligned medians with IQR band, each patient's last value carried forward "
        f"(earlier hours only) ({traj_cfg['before']} h before to "
        f"{traj_cfg['after']} h after t_sepsis, rows; points with fewer than "
        f"{traj_cfg['min_patients']} patients hidden). Dashed: median of never-septic patients "
        "over hours where the value was measured (not carried forward), so for labs it "
        "reflects the patients who got that lab ordered.",
        "![trajectories](figures/trajectories.png)",
    ]

    # 5. Correlation and redundancy
    redundancy = []
    for h, g in by_hospital.items():
        corr = stats.correlations(g, measured, config["min_shared_hours"])
        plots.correlation_heatmap(
            corr,
            f"Hospital {h}: Spearman correlation (hours)",
            fig_dir / f"correlation_{h}.png",
            dpi,
        )
        pairs = stats.redundant_pairs(corr, config["redundancy_threshold"])
        redundancy.append(pairs.assign(hospital=h)[["hospital", "a", "b", "r"]])
    points, explained = stats.patient_pca(df, vitals + labs)
    plots.pca_density(points, explained, fig_dir / "pca.png", dpi)
    sections += [
        "## 5. Correlation and redundant variables",
        f"Pairs with |Spearman r| >= {config['redundancy_threshold']}, among pairs measured "
        f"together in at least {config['min_shared_hours']} hours:",
        markdown_table(pd.concat(redundancy, ignore_index=True), digits=2),
        *(f"![correlation {h}](figures/correlation_{h}.png)" for h in hospitals),
        "PCA of per-patient medians, as a density (plot only, never a model input):",
        "![pca](figures/pca.png)",
    ]

    # 6. Hospital A vs B
    if len(hospitals) == 2:
        shift = stats.hospital_shift(df, measured)
        plots.smd_bars(
            shift,
            f"Hospital {hospitals[0]} minus {hospitals[1]}",
            fig_dir / "hospital_shift.png",
            dpi,
        )
        flagged = shift[shift["smd"].abs() >= config["shift_threshold"]]
        sections += [
            f"## 6. Hospital {hospitals[0]} vs {hospitals[1]}",
            f"Variables with |standardized mean difference| >= {config['shift_threshold']}, "
            "largest first (values: means over hours with a value):",
            markdown_table(flagged, digits=2),
            "![hospital shift](figures/hospital_shift.png)",
        ]

    totals = ", ".join(
        f"{h}: {g['patient_id'].nunique()} patients, {len(g)} patient-hours"
        for h, g in by_hospital.items()
    )
    intro = (
        f"Dev patients only, after the M0-3 cleaning ({totals}). Aggregates only: no patient "
        "rows. Settings in `configs/eda.yaml`."
    )
    return "\n\n".join([intro, *sections])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=EDA_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    try:
        split = load_project_split()  # dev patients only
        raw = load_patient_hours(sorted(split["hospital"].unique()), split["patient_id"])
    except (FileNotFoundError, SplitFingerprintError, IncompleteCacheError) as error:
        print(f"{error}\nRun build_cache and make_split first (see README).", file=sys.stderr)
        return 1

    df = clean(raw, load_yaml(CLEANING_CONFIG))
    generated = analyse(df, load_yaml(args.config), args.out_dir / "figures")
    readme = args.out_dir / "README.md"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else None
    title = "# Exploratory analysis: hospital A vs B (dev patients)"
    readme.write_text(merge_readme(existing, generated, title), encoding="utf-8")
    print(f"wrote {readme} and figures in {args.out_dir / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
