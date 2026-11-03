# Sepsis decision support (PhysioNet/CinC 2019)

Hourly ICU decision-support prototype: a risk model, an action model (which lab panel to order next, or WAIT) and a guideline layer. See `CONTEXT.md` for the vocabulary and `docs/` for decisions and workflow.

## Requirements
- Python 3.11 or newer (3.13 works)
- Dependencies are declared in `pyproject.toml`:
  - runtime: numpy, pandas, pyarrow, lightgbm, scikit-learn, pyyaml
  - dev: pytest, ruff
  - optional `signatures`: iisignature (only for the B3 baseline; needs a C++ compiler on Windows)

## Install

### Option A: uv (recommended)
```bash
uv sync                      # add --extra signatures for signature features
```
This creates `.venv/`. In VS Code, run **Python: Select Interpreter** and pick `.venv\Scripts\python.exe`.

### Option B: your own env (conda or venv)
```bash
conda create -n sepsis python=3.13 -y
conda activate sepsis
pip install -e .             # installs the sepsis package and runtime dependencies
pip install pytest ruff      # dev tools
```
`pip install -e .` is required: the code lives in `src/sepsis`, so `import sepsis` fails without it.

## Data
Place the PhysioNet/CinC 2019 files here (they are never committed):
```
data/training_setA/training_setA/p*.psv
data/training_setB/training_setB/p*.psv
```

## Run
Inspect the dataset (patients per hospital, hours per patient, % septic, % missing per column):
```bash
python -m sepsis.data.inspect                # both hospitals
python -m sepsis.data.inspect --hospital A   # one hospital
```
Use `-m`, not the file path: running `src/sepsis/data/inspect.py` directly lets its name shadow the stdlib `inspect`. With uv, prefix with `uv run`. Reading about 40k files takes several minutes.

## Test and lint
Run these after every change:
```bash
ruff check src tests
ruff format --check src tests
pytest -q -m "not slow"
```
`make check` runs the same three steps, but needs `make` (not installed by default on Windows). `make test` runs the full suite, including `slow` tests that need real data. With uv, prefix each command with `uv run`.

In VS Code, add to `.vscode/settings.json` to use the Testing panel:
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["-m", "not slow", "tests"]
}
```

## Rules
Read the hard rules in `CLAUDE.md` before changing features, labels or evaluation (no leakage across hours or patients, official scorer only, never commit data).
