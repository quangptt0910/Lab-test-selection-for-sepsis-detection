"""Compare finished runs and decide keep or drop from their run folders (`compare-runs` skill).

    uv run python -m sepsis.eval.compare --reference b2-001 b2-011 b2-012 \
        --candidate b3-002 b3-011 b3-012

Reads only `experiments/runs/<id>/` (scores, thresholds, meta, config) and prints markdown
tables and the decision. Writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sepsis.config import PROTOCOL_CONFIG, load_yaml
from sepsis.eval.runner import flags_from_scores
from sepsis.eval.scorer import Scores

RUNS_DIR = Path("experiments/runs")
SCORE_KEYS = ("A_to_B", "B_to_A", "mix_A", "mix_B")
SEED_REPEAT_KEYS = {"id", "seed", "hypothesis", "note"}  # all a seed repeat may change
BLOCK_SECTIONS = ("core", "extras", "signatures")  # feature-block sections of features.yaml
# LOG names of the four results (runner.RESULT_NAMES) -> their `scores.json` keys.
COLUMNS = {"A→B": "A_to_B", "B→A": "B_to_A", "mix→A": "mix_A", "mix→B": "mix_B"}
MIX_MEAN = "mix mean"  # (mix→A + mix→B) / 2, the figure the keep rule decides on
# Training-side out-of-fold tuning utility per setup folder (mean over mix's outer folds).
OOF_SETUPS = {"A→B": "A_to_B", "B→A": "B_to_A", "mix": "mix"}


class CompareError(ValueError):
    """Raised when runs can't be compared (partial run, bad seed repeat, ...)."""


@dataclass(frozen=True)
class Run:
    """One finished run, as its folder records it."""

    id: str
    seed: int
    sha: str
    scores: dict[str, dict[str, float]]
    config: dict[str, Any]
    oof: dict[str, float]
    settings: dict[str, Any]  # what shapes the run beyond config.yaml (`effective_settings`)
    scorer_sha256: str
    dirty: bool


@dataclass(frozen=True)
class GroupStats:
    """One side of a comparison: per-run figures, and their means and spreads (max - min)."""

    n: int
    per_run: dict[str, dict[str, float]]
    mean: dict[str, float]
    spread: dict[str, float]
    oof_mean: dict[str, float]
    oof_spread: dict[str, float]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(runs_dir: Path, run_id: str) -> Run:
    """Read one run folder; refuse a partial run."""
    folder = Path(runs_dir) / run_id
    if run_id.endswith(".partial"):
        raise CompareError(f"{run_id} is a partial folder: the run never reached the LOG")
    if not folder.is_dir():
        raise CompareError(f"no run folder {folder}")
    scores = _read_json(folder / "scores.json")
    missing = [key for key in SCORE_KEYS if key not in scores]
    if missing:
        raise CompareError(f"{run_id} is a partial run (no {missing}): not comparable")
    meta = _read_json(folder / "meta.json")
    if meta.get("id") != run_id:
        raise CompareError(f"{folder} holds run {meta.get('id')!r}, not {run_id!r}")
    config = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    oof = {}
    for name, setup in OOF_SETUPS.items():
        folds = _read_json(folder / setup / "thresholds.json")
        oof[name] = sum(float(f["tuning_utility"]) for f in folds) / len(folds)
    return Run(
        run_id,
        int(meta["seed"]),
        str(meta["sha"]),
        scores,
        config,
        oof,
        effective_settings(folder),
        str(meta.get("scorer_sha256", "")),
        bool(meta.get("dirty", False)),
    )


def run_figures(run: Run) -> dict[str, float]:
    """The four utilities and the mix mean of one run, unrounded."""
    out = {name: float(run.scores[key]["utility"]) for name, key in COLUMNS.items()}
    out[MIX_MEAN] = (out["mix→A"] + out["mix→B"]) / 2
    return out


def _mean_spread(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = rows[0].keys()
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    spread = {k: max(r[k] for r in rows) - min(r[k] for r in rows) for k in keys}
    return mean, spread


def group_stats(runs: list[Run]) -> GroupStats:
    """Means and spreads (max - min over the seeds) of every figure, eval side and OOF."""
    per_run = {run.id: run_figures(run) for run in runs}
    mean, spread = _mean_spread(list(per_run.values()))
    oof_mean, oof_spread = _mean_spread([run.oof for run in runs])
    return GroupStats(len(runs), per_run, mean, spread, oof_mean, oof_spread)


@dataclass(frozen=True)
class Decision:
    """`keep`, `drop` or `undecided`, with the numbers behind it."""

    verdict: str
    gain: float  # candidate mix mean - reference mix mean (means over the seeds)
    margin: float  # the larger of the two mix-mean seed spreads
    reasons: list[str]


def keep_decision(
    reference: GroupStats, candidate: GroupStats, *, tolerance: float, min_seeds: int
) -> Decision:
    """The keep rule: the mix mean must rise by more than the larger seed spread, and neither
    A→B nor B→A may drop by more than `tolerance` (`selection` in configs/protocol.yaml)."""
    gain = candidate.mean[MIX_MEAN] - reference.mean[MIX_MEAN]
    margin = max(reference.spread[MIX_MEAN], candidate.spread[MIX_MEAN])
    mix_ok = gain > margin
    reasons = [f"mix mean {gain:+.4f} {'>' if mix_ok else '<='} margin {margin:.4f}"]
    cross_ok = True
    for name in ("A→B", "B→A"):
        change = candidate.mean[name] - reference.mean[name]
        if change < -tolerance:
            cross_ok = False
            reasons.append(f"{name} {change:+.4f} drops by more than {tolerance}")
    verdict = "keep" if mix_ok and cross_ok else "drop"
    if margin == 0:  # seeds that change nothing: any gain would pass
        verdict = "undecided"
        reasons.insert(0, "the mix-mean seed spread is 0 on both sides: no noise margin")
    if min(reference.n, candidate.n) < min_seeds:  # one seed can't tell a change from noise
        verdict = "undecided"
        reasons.insert(
            0,
            f"needs {min_seeds} seeds per side (reference {reference.n}, candidate "
            f"{candidate.n}): a screening result, never a keep",
        )
    return Decision(verdict, gain, margin, reasons)


def cross_run_flags(
    reference: list[Run], candidate: list[Run], limits: Mapping[str, float]
) -> list[str]:
    """The runner's red flags for every run, plus `max_utility_jump`: a candidate column that
    moves by more than it from the reference run with the same seed (`red_flags` limits)."""
    flags = []
    for run in [*reference, *candidate]:
        scores = {name: Scores(**run.scores[key]) for name, key in COLUMNS.items()}
        flags += [f"{run.id}: {flag}" for flag in flags_from_scores(scores, limits)]
    by_seed = {run.seed: run for run in reference}
    jump = float(limits["max_utility_jump"])
    for run in candidate:
        base = by_seed.get(run.seed)
        if base is None:
            continue
        new, old = run_figures(run), run_figures(base)
        for name in COLUMNS:
            change = new[name] - old[name]
            if abs(change) > jump:
                flags.append(
                    f"{run.id}: red flag: {name} utility {change:+.3f} vs {base.id} "
                    f"(same seed) moves more than {jump}"
                )
    return flags


def _enabled_only(features: Mapping[str, Any]) -> dict[str, Any]:
    """A resolved features.yaml without its switched-off blocks: those change no column."""
    out = dict(features)
    for section in BLOCK_SECTIONS:
        blocks = features.get(section) or {}
        out[section] = {name: b for name, b in blocks.items() if b.get("enabled")}
    return out


def effective_settings(folder: Path) -> dict[str, Any]:
    """The resolved configs (`resolved/`, features.yaml without its switched-off blocks) and the
    feature columns of a run folder: two seed repeats must agree on all of them."""
    settings: dict[str, Any] = {}
    for path in sorted((folder / "resolved").glob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings[path.name] = _enabled_only(value) if path.name == "features.yaml" else value
    settings["feature_columns.json"] = _read_json(folder / "feature_columns.json")
    return settings


def _setting(run: Run) -> dict[str, Any]:
    """The config without the keys a seed repeat may change."""
    config = {k: v for k, v in run.config.items() if k not in SEED_REPEAT_KEYS}
    return {**config, **run.settings}


def load_group(runs_dir: Path, run_ids: list[str], name: str) -> list[Run]:
    """The runs of one side: one change, repeated with different seeds."""
    runs = [load_run(runs_dir, run_id) for run_id in run_ids]
    seeds = [run.seed for run in runs]
    if len(set(seeds)) != len(seeds):
        raise CompareError(
            f"{name}: two runs share a seed {seeds}; seed repeats need distinct seeds"
        )
    first = _setting(runs[0])
    for run in runs[1:]:
        changed = sorted(
            k for k in first.keys() | _setting(run).keys() if first.get(k) != _setting(run).get(k)
        )
        if changed:
            raise CompareError(
                f"{name}: {run.id} is not a seed repeat of {runs[0].id}: it also changes {changed}"
            )
    return runs


def _oof_stats(stats: GroupStats) -> GroupStats:
    """The OOF figures in the shape `keep_decision` reads (OOF mix in place of the mix mean)."""
    keys = {"A→B": "A→B", "B→A": "B→A", MIX_MEAN: "mix"}
    mean = {k: stats.oof_mean[v] for k, v in keys.items()}
    spread = {k: stats.oof_spread[v] for k, v in keys.items()}
    return GroupStats(stats.n, {}, mean, spread, stats.oof_mean, stats.oof_spread)


def _row(label: str, values: list[float], sign: bool = False) -> str:
    cells = [f"{v:+.4f}" if sign else f"{v:.4f}" for v in values]
    return "| " + " | ".join([label, *cells]) + " |"


def _header(first: str, names: list[str]) -> list[str]:
    return [f"| {first} | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]


def report(
    reference: list[Run],
    candidate: list[Run],
    *,
    selection: Mapping[str, Any],
    limits: Mapping[str, float],
) -> str:
    """The markdown report: per-run figures, means, spreads, OOF, flags and the decision.

    `selection` and `limits` are the sections of configs/protocol.yaml.
    """
    ref, cand = group_stats(reference), group_stats(candidate)
    names = [*COLUMNS, MIX_MEAN]
    lines = ["## Runs", "", *_header("run | seed", names)]
    lines[-1] = "|---|---|" + "---|" * len(names)
    for run in [*reference, *candidate]:
        figures = run_figures(run)
        lines.append(_row(f"{run.id} | {run.seed}", [figures[n] for n in names]))

    lines += ["", "## Means over seeds (eval side)", "", *_header("", names)]
    lines.append(_row(f"reference (n={ref.n})", [ref.mean[n] for n in names]))
    lines.append(_row(f"candidate (n={cand.n})", [cand.mean[n] for n in names]))
    lines.append(_row("change", [cand.mean[n] - ref.mean[n] for n in names], sign=True))
    lines += ["", "Seed spread (max - min):", "", *_header("", names)]
    lines.append(_row("reference", [ref.spread[n] for n in names]))
    lines.append(_row("candidate", [cand.spread[n] for n in names]))

    oof = list(OOF_SETUPS)
    lines += ["", "## Training-side OOF (tuning utility; information only)", ""]
    lines += _header("", oof)
    lines.append(_row("reference", [ref.oof_mean[n] for n in oof]))
    lines.append(_row("candidate", [cand.oof_mean[n] for n in oof]))
    lines.append(_row("change", [cand.oof_mean[n] - ref.oof_mean[n] for n in oof], sign=True))
    lines.append(_row("spread, reference", [ref.oof_spread[n] for n in oof]))
    lines.append(_row("spread, candidate", [cand.oof_spread[n] for n in oof]))
    rule = {
        "tolerance": float(selection["cross_hospital_tolerance"]),
        "min_seeds": int(selection["min_seeds"]),
    }
    oof_decision = keep_decision(_oof_stats(ref), _oof_stats(cand), **rule)
    lines += ["", f"OOF rule: {oof_decision.verdict} ({'; '.join(oof_decision.reasons)})"]

    notes = cross_run_flags(reference, candidate, limits)
    for name, runs in (("reference", reference), ("candidate", candidate)):
        by_sha: dict[str, list[str]] = {}
        for run in runs:
            by_sha.setdefault(run.sha, []).append(run.id)
        if len(by_sha) > 1:
            spans = ", ".join(f"{sha} ({', '.join(ids)})" for sha, ids in sorted(by_sha.items()))
            notes.append(f"warning: {name} runs span commits: {spans}")
    notes += [
        f"warning: {run.id} ran on uncommitted code (dirty tree)"
        for run in [*reference, *candidate]
        if run.dirty
    ]
    seeds = [sorted(run.seed for run in runs) for runs in (reference, candidate)]
    if seeds[0] != seeds[1]:
        notes.append(f"warning: the sides use different seeds ({seeds[0]} vs {seeds[1]})")
    lines += ["", "## Flags and warnings", "", *([f"- {n}" for n in notes] or ["- none"])]

    decision = keep_decision(ref, cand, **rule)
    lines += ["", f"**Decision: {decision.verdict}**", "", *[f"- {r}" for r in decision.reasons]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", nargs="+", required=True, help="reference run ids")
    parser.add_argument("--candidate", nargs="+", required=True, help="candidate run ids")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder of run folders")
    args = parser.parse_args(argv)
    repeated = sorted(
        {
            i
            for i in [*args.reference, *args.candidate]
            if [*args.reference, *args.candidate].count(i) > 1
        }
    )
    if repeated:
        raise CompareError(f"run ids {repeated} are given more than once")
    reference = load_group(args.runs, args.reference, "reference")
    candidate = load_group(args.runs, args.candidate, "candidate")
    scorers = sorted({run.scorer_sha256 for run in [*reference, *candidate]})
    if len(scorers) > 1:
        raise CompareError(f"the runs were scored by different scorer files {scorers}")
    protocol = load_yaml(PROTOCOL_CONFIG)
    text = report(
        reference, candidate, selection=protocol["selection"], limits=protocol["red_flags"]
    )
    if hasattr(sys.stdout, "reconfigure"):  # the arrows in the column names on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
