"""Figures for the exploratory analysis (M0-4). Every figure shows aggregates only: histograms,
medians with quantile bands, matrices and densities, never single patients."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"A": "#1f77b4", "B": "#d62728"}


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def length_and_onset(patients: pd.DataFrame, path: Path, dpi: int) -> Path:
    """Histograms (density) of rows per patient and of onset hour, per hospital."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.arange(0, 340, 4)
    for hospital, p in patients.groupby("hospital"):
        color = COLORS.get(hospital)
        ax1.hist(
            p["n_hours"],
            bins=bins,
            density=True,
            histtype="step",
            lw=1.5,
            color=color,
            label=hospital,
        )
        ax2.hist(
            p["t_sepsis"].dropna(),
            bins=bins,
            density=True,
            histtype="step",
            lw=1.5,
            color=color,
            label=hospital,
        )
    ax1.set(title="Hours per patient (rows)", xlabel="hours", ylabel="density", xlim=(0, 200))
    ax2.set(title="Onset hour of septic patients (t_sepsis)", xlabel="hour", xlim=(0, 200))
    ax1.legend()
    return _save(fig, path, dpi)


def paired_bars(
    table: pd.DataFrame, value: str, title: str, xlabel: str, path: Path, dpi: int
) -> Path:
    """Horizontal bars of `value` per column, one bar per hospital."""
    wide = table.pivot(index="column", columns="hospital", values=value)
    wide = wide.loc[wide.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(wide) + 1.2))
    y = np.arange(len(wide))
    height = 0.8 / len(wide.columns)
    for i, hospital in enumerate(wide.columns):
        ax.barh(
            y + i * height,
            wide[hospital],
            height=height,
            color=COLORS.get(hospital),
            label=hospital,
        )
    ax.set_yticks(y + height * (len(wide.columns) - 1) / 2, wide.index)
    ax.set(title=title, xlabel=xlabel)
    ax.legend()
    return _save(fig, path, dpi)


def co_measurement_heatmaps(
    matrices: dict[str, pd.DataFrame], order: Sequence[str], path: Path, dpi: int
) -> Path:
    """P(column measured | row measured) per hospital, rows and columns in the same order."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(7.5 * len(matrices), 7))
    axes = np.atleast_1d(axes)
    for ax, (hospital, m) in zip(axes, matrices.items(), strict=True):
        m = m.reindex(index=order, columns=order)
        image = ax.imshow(m.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order)), order, fontsize=7)
        ax.set_title(f"Hospital {hospital}: P(column measured | row measured), same hour")
    fig.colorbar(image, ax=axes.tolist(), shrink=0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def trajectories(
    traj: pd.DataFrame, reference: pd.DataFrame, columns: Sequence[str], path: Path, dpi: int
) -> Path:
    """Median and IQR band by hours from onset, per hospital; dashed line: median of patients
    who never become septic."""
    columns = [c for c in columns if c in set(traj["column"])]
    ncols = 4
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows), squeeze=False)
    for ax, column in zip(axes.flat, columns, strict=False):
        for hospital, t in traj[traj["column"] == column].groupby("hospital"):
            color = COLORS.get(hospital)
            ax.plot(t["rel_hour"], t["median"], color=color, lw=1.5, label=hospital)
            ax.fill_between(t["rel_hour"], t["q25"], t["q75"], color=color, alpha=0.15)
            ref = reference.query("hospital == @hospital and column == @column")
            if not ref.empty:
                ax.axhline(ref["nonseptic_median"].iloc[0], color=color, ls="--", lw=0.8)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(column, fontsize=10)
    for ax in list(axes.flat)[len(columns) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("hours from onset (rows; 0 = t_sepsis)")
    return _save(fig, path, dpi)


def smd_bars(table: pd.DataFrame, title: str, path: Path, dpi: int) -> Path:
    """Horizontal bars of the standardized mean difference per column (one bar per hospital,
    or one bar when there is no hospital column)."""
    if "hospital" in table.columns:
        return paired_bars(
            table.assign(abs_smd=table["smd"]),
            "abs_smd",
            title,
            "standardized mean difference",
            path,
            dpi,
        )
    t = table.dropna(subset=["smd"]).sort_values("smd", key=np.abs)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(t) + 1.2))
    ax.barh(t["column"], t["smd"], color=np.where(t["smd"] > 0, COLORS["A"], COLORS["B"]))
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(title=title, xlabel="standardized mean difference")
    return _save(fig, path, dpi)


def correlation_heatmap(corr: pd.DataFrame, title: str, path: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.7)
    return _save(fig, path, dpi)


def pca_density(points: pd.DataFrame, explained: Sequence[float], path: Path, dpi: int) -> Path:
    """Hexbin density of per-patient PCA scores, one panel per hospital, shared axes."""
    hospitals = sorted(points["hospital"].unique())
    x_lim = points["pc1"].quantile([0.005, 0.995]).to_numpy()
    y_lim = points["pc2"].quantile([0.005, 0.995]).to_numpy()
    fig, axes = plt.subplots(1, len(hospitals), figsize=(5.5 * len(hospitals), 4.5), squeeze=False)
    for ax, hospital in zip(axes.flat, hospitals, strict=True):
        p = points[points["hospital"] == hospital]
        ax.hexbin(
            p["pc1"],
            p["pc2"],
            gridsize=40,
            extent=(*x_lim, *y_lim),
            mincnt=5,
            cmap="viridis",
            bins="log",
        )
        ax.set(
            title=f"Hospital {hospital}",
            xlabel=f"PC1 ({100 * explained[0]:.0f}%)",
            ylabel=f"PC2 ({100 * explained[1]:.0f}%)",
            xlim=x_lim,
            ylim=y_lim,
        )
    fig.suptitle("Per-patient medians, PCA (bins with fewer than 5 patients hidden)")
    return _save(fig, path, dpi)
