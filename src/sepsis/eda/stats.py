"""Aggregate statistics for the exploratory analysis (M0-4).

Every function is pure and returns aggregates (counts, percentages, quantiles), never patient
rows. Hours are rows (`hour`), never ICULOS. Some functions use `SepsisLabel` and `t_sepsis`
over the whole record: that is fine for describing the data, but none of this may ever feed a
model feature.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from sepsis.data.patients import patient_table


def _in_hour_order(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["patient_id", "hour"], kind="stable")


def _by_hospital(df: pd.DataFrame):
    return df.groupby("hospital", sort=True)


def _smd(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference (a - b) / pooled SD; NaN if either side is empty."""
    a, b = a.dropna(), b.dropna()
    if a.empty or b.empty:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def cohort_summary(df: pd.DataFrame, septic_on_arrival_rows: int) -> pd.DataFrame:
    """One row per hospital: patients, patient-hours, septic share, length of stay (rows),
    onset hour, share of septic patients septic on arrival, share of records starting late."""
    patients = patient_table(df)
    first = _in_hour_order(df).groupby("patient_id")["ICULOS"].first()
    patients["late_start"] = patients["patient_id"].map(first).gt(1)
    rows = {}
    for hospital, p in patients.groupby("hospital"):
        septic = p[p["septic"] == 1]
        # first label row = t_sepsis - 6
        on_arrival = (septic["t_sepsis"] - 6) < septic_on_arrival_rows
        rows[hospital] = {
            "patients": len(p),
            "patient_hours": int(p["n_hours"].sum()),
            "septic_pct": 100 * p["septic"].mean(),
            "hours_p25": p["n_hours"].quantile(0.25),
            "hours_median": p["n_hours"].median(),
            "hours_p75": p["n_hours"].quantile(0.75),
            "onset_hour_median": septic["t_sepsis"].median(),
            "septic_on_arrival_pct": 100 * on_arrival.mean() if len(septic) else float("nan"),
            "iculos_start_after_1_pct": 100 * p["late_start"].mean(),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "hospital"
    return out


def missingness(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per hospital and column: % of hours with a value, % of patients with at least one."""
    out = []
    for hospital, g in _by_hospital(df):
        ever = g[list(columns)].notna().groupby(g["patient_id"]).any()
        for column in columns:
            out.append(
                (
                    hospital,
                    column,
                    100 * g[column].notna().mean(),
                    100 * ever[column].mean(),
                )
            )
    return pd.DataFrame(out, columns=["hospital", "column", "pct_hours", "pct_patients"])


def measurement_gaps(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per hospital and column: hours (rows) between consecutive values of the same patient."""
    ordered = _in_hour_order(df)
    out = []
    for hospital, g in _by_hospital(ordered):
        for column in columns:
            measured = g.loc[g[column].notna(), ["patient_id", "hour"]]
            gaps = measured.groupby("patient_id")["hour"].diff().dropna()
            out.append(
                (
                    hospital,
                    column,
                    len(gaps),
                    gaps.median() if len(gaps) else float("nan"),
                    gaps.quantile(0.9) if len(gaps) else float("nan"),
                )
            )
    return pd.DataFrame(out, columns=["hospital", "column", "n_gaps", "median_hours", "p90_hours"])


def co_measurement(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Entry [i, j]: share of hours with a value of i that also have a value of j.

    The row of a column that is never measured is NaN.
    """
    measured = df[list(columns)].notna().astype(float)
    both = measured.T @ measured
    counts = np.diag(both.to_numpy())
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = both.to_numpy() / counts[:, None]
    return pd.DataFrame(matrix, index=list(columns), columns=list(columns))


def candidate_panels(matrix: pd.DataFrame, distance_threshold: float) -> dict[str, int]:
    """Cluster columns on 1 - symmetric co-measurement (average linkage).

    Returns cluster ids (1, 2, ...) for every column that is ever measured.
    """
    keep = matrix.index[matrix.notna().all(axis=1)]
    sym = (matrix.loc[keep, keep] + matrix.loc[keep, keep].T) / 2
    if len(keep) < 2:
        return {c: 1 for c in keep}
    distance = (1 - sym).clip(lower=0).to_numpy(copy=True)
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(tree, t=distance_threshold, criterion="distance")
    return dict(zip(keep, (int(x) for x in labels), strict=True))


def onset_trajectories(
    df: pd.DataFrame,
    columns: Sequence[str],
    before: int,
    after: int,
    min_patients: int,
    forward_fill: bool = False,
) -> pd.DataFrame:
    """Quantiles of each column by hours from onset (row - t_sepsis), septic patients only.

    With `forward_fill`, each patient's last value is carried forward in hour order (earlier
    hours only, as the models will see it). Points with fewer than `min_patients` patients are
    dropped.
    """
    patients = patient_table(df)
    onset = patients.set_index("patient_id")["t_sepsis"].dropna()
    septic = _in_hour_order(df[df["patient_id"].isin(onset.index)]).copy()
    if forward_fill:
        septic[list(columns)] = septic.groupby("patient_id")[list(columns)].ffill()
    septic["rel_hour"] = septic["hour"] - septic["patient_id"].map(onset)
    septic = septic[septic["rel_hour"].between(-before, after)]
    out = []
    for column in columns:
        values = septic.loc[septic[column].notna(), ["hospital", "rel_hour", "patient_id", column]]
        grouped = values.groupby(["hospital", "rel_hour"])
        stats = grouped[column].quantile([0.25, 0.5, 0.75]).unstack()
        stats.columns = ["q25", "median", "q75"]
        stats["n_patients"] = grouped["patient_id"].nunique()
        stats["column"] = column
        out.append(stats.reset_index())
    result = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if result.empty:
        return result
    result = result[result["n_patients"] >= min_patients]
    result["rel_hour"] = result["rel_hour"].astype(int)
    columns_out = ["hospital", "column", "rel_hour", "n_patients", "q25", "median", "q75"]
    return result[columns_out].reset_index(drop=True)


def septic_vs_nonseptic(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per hospital and column: label-1 hours of septic patients against all hours of patients
    who never become septic (medians and standardized mean difference)."""
    septic_ids = set(patient_table(df).query("septic == 1")["patient_id"])
    out = []
    for hospital, g in _by_hospital(df):
        is_septic_patient = g["patient_id"].isin(septic_ids)
        septic_hours = g[is_septic_patient & (g["SepsisLabel"] == 1)]
        never = g[~is_septic_patient]
        for column in columns:
            s, n = septic_hours[column], never[column]
            out.append((hospital, column, s.median(), n.median(), _smd(s, n), s.count(), n.count()))
    return pd.DataFrame(
        out,
        columns=[
            "hospital",
            "column",
            "septic_median",
            "nonseptic_median",
            "smd",
            "n_septic",
            "n_nonseptic",
        ],
    )


def correlations(df: pd.DataFrame, columns: Sequence[str], min_periods: int = 1) -> pd.DataFrame:
    """Pairwise Spearman correlation over hours where both values exist; NaN for pairs that
    share fewer than `min_periods` hours."""
    return df[list(columns)].corr(method="spearman", min_periods=min_periods)


def redundant_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Pairs (a before b in column order) with |r| >= threshold, strongest first."""
    names = list(corr.columns)
    out = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) >= threshold
    ]
    pairs = pd.DataFrame(out, columns=["a", "b", "r"])
    return pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def hospital_shift(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per column: mean in each of the two hospitals, standardized mean difference (first minus
    second) and % of hours with a value; sorted by |SMD|, largest first."""
    hospitals = sorted(df["hospital"].unique())
    if len(hospitals) != 2:
        raise ValueError(f"hospital_shift needs two hospitals, got {hospitals}")
    first, second = (df[df["hospital"] == h] for h in hospitals)
    h1, h2 = hospitals
    out = [
        {
            "column": c,
            f"mean_{h1}": first[c].mean(),
            f"mean_{h2}": second[c].mean(),
            "smd": _smd(first[c], second[c]),
            f"pct_hours_{h1}": 100 * first[c].notna().mean(),
            f"pct_hours_{h2}": 100 * second[c].notna().mean(),
        }
        for c in columns
    ]
    table = pd.DataFrame(out)
    order = table["smd"].abs().sort_values(ascending=False, na_position="last").index
    return table.loc[order].reset_index(drop=True)


def patient_pca(df: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, list[float]]:
    """Two principal components of per-patient medians (standardized, median-imputed).

    For a density plot only; it is never used as a model input.
    """
    summary = df.groupby("patient_id")[list(columns)].median()
    summary = summary.loc[:, summary.notna().any()]
    filled = summary.fillna(summary.median())
    std = filled.std(ddof=0).replace(0, 1)
    z = ((filled - filled.mean()) / std).to_numpy()
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    explained = (s**2) / (s**2).sum()
    pcs = z @ vt[:2].T
    hospital = df.groupby("patient_id")["hospital"].first()
    points = pd.DataFrame(
        {
            "patient_id": summary.index,
            "hospital": hospital.loc[summary.index].to_numpy(),
            "pc1": pcs[:, 0],
            "pc2": pcs[:, 1] if pcs.shape[1] > 1 else 0.0,
        }
    )
    return points, [float(x) for x in explained[:2]]
