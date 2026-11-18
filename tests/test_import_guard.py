"""Feature and action code may see data only through the view (sepsis.data.view).

An allowlist: inside the `sepsis` package, guarded code may import only the modules below and its
own package. Everything else (the loaders, the patient table, labels, the EDA, and modules that
re-export loaders such as `sepsis.data.quality`) is refused. Relative imports are resolved first.

`risk/` is not guarded here: training the risk model needs the U1 - U0 target (`sepsis.labels`).
Risk code that computes features at a row must still go through the view.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
GUARDED = ["sepsis.features", "sepsis.action"]
ALLOWED = ["sepsis.config", "sepsis.data.view", "sepsis.data.columns", "sepsis.data.panels"]


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
    allowed = [*ALLOWED, own_package]
    return any(name == ok or name.startswith(ok + ".") for ok in allowed)


def forbidden_imports(path: Path, own_package: str) -> list[str]:
    return [n for n in _imports(path, _module_name(path)) if not _allowed(n, own_package)]


def test_guarded_code_reads_data_only_through_the_view() -> None:
    # Passes trivially until features/ or action/ exist; then every file is checked.
    found = {}
    for package in GUARDED:
        for path in sorted((SRC / package.replace(".", "/")).rglob("*.py")):
            names = forbidden_imports(path, package)
            if names:
                found[str(path.relative_to(SRC))] = names
    assert found == {}


def _fake(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "sepsis" / "features" / "fake.py"
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
