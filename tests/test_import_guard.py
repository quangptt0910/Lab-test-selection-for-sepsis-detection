"""Feature, action and risk-model code may see data only through the view (sepsis.data.view).

An allowlist per guarded package: inside the `sepsis` package, guarded code may import only the
modules listed for it and its own package. Everything else (the loaders, the patient table, labels,
the EDA, and modules that re-export loaders such as `sepsis.data.quality`) is refused. Relative
imports are resolved first.

`risk/` receives feature rows and targets from the experiment runner (`sepsis.eval`), so it may
import `sepsis.features` (for replay-time feature rows) but never the loaders or `sepsis.labels`.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
VIEW_ONLY = ["sepsis.config", "sepsis.data.view", "sepsis.data.columns", "sepsis.data.panels"]
GUARDED = {
    "sepsis.features": VIEW_ONLY,
    "sepsis.action": VIEW_ONLY,
    "sepsis.risk": ["sepsis.config", "sepsis.features"],
}


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path, module: str) -> list[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            names += [f"{base}.{alias.name}" for alias in node.names]
    return names


def _allowed(name: str, own_package: str) -> bool:
    if name != "sepsis" and not name.startswith("sepsis."):
        return True  # third-party and standard library
    allowed = [*GUARDED[own_package], own_package]
    return any(name == ok or name.startswith(ok + ".") for ok in allowed)


def forbidden_imports(path: Path, own_package: str) -> list[str]:
    return [n for n in _imports(path, _module_name(path)) if not _allowed(n, own_package)]


def test_guarded_code_reads_data_only_through_the_view() -> None:
    # Passes trivially until features/ or action/ exist; then every file is checked.
    found = {}
    for package in GUARDED:  # every guarded package, each with its own allowlist
        for path in sorted((SRC / package.replace(".", "/")).rglob("*.py")):
            names = forbidden_imports(path, package)
            if names:
                found[str(path.relative_to(SRC))] = names
    assert found == {}


def _fake(tmp_path: Path, text: str, package: str = "features") -> Path:
    path = tmp_path / "sepsis" / package / "fake.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_the_guard_catches_forbidden_imports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(__import__(__name__), "SRC", tmp_path)
    bad = _fake(
        tmp_path,
        "from sepsis.data.splits import load_patient_hours\n"
        "import sepsis.labels.utility\n"
        "from sepsis.data.quality import CLEANING_CONFIG\n"
        "from ..data import load\n"
        "from ..labels.onset import t_sepsis\n",
    )
    assert forbidden_imports(bad, "sepsis.features") == [
        "sepsis.data.splits.load_patient_hours",
        "sepsis.labels.utility",
        "sepsis.data.quality.CLEANING_CONFIG",
        "sepsis.data.load",
        "sepsis.labels.onset.t_sepsis",
    ]


def test_the_guard_allows_the_view_and_its_own_package(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(__import__(__name__), "SRC", tmp_path)
    ok = _fake(
        tmp_path,
        "import numpy as np\n"
        "from sepsis.data.view import view_table\n"
        "from sepsis.data.columns import LABS\n"
        "from ..data.panels import lab_to_panel\n"
        "from .vitals import rolling\n"
        "from sepsis.features.clinical import shock_index\n",
    )
    assert forbidden_imports(ok, "sepsis.features") == []


def test_risk_code_may_use_features_but_never_labels_or_loaders(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(__import__(__name__), "SRC", tmp_path)
    fake = _fake(
        tmp_path,
        "from sepsis.features.builder import features_at\n"
        "from sepsis.config import load_yaml\n"
        "from .base import RiskModel\n"
        "from sepsis.labels.utility import utility_targets\n"
        "from sepsis.data.splits import load_patient_hours\n"
        "from sepsis.eval.scorer import score\n",
        package="risk",
    )
    assert forbidden_imports(fake, "sepsis.risk") == [
        "sepsis.labels.utility.utility_targets",
        "sepsis.data.splits.load_patient_hours",
        "sepsis.eval.scorer.score",
    ]
