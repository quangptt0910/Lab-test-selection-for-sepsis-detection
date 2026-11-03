"""PostToolUse hook: format and lint a Python file right after Claude edits it.

Reads the hook event JSON from stdin. Exit code 2 sends remaining lint errors back to Claude.
Exits 0 silently for non-Python files or when uv/ruff isn't available.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    file_path = (event.get("tool_input") or {}).get("file_path", "")
    path = Path(file_path)
    if path.suffix != ".py" or not path.exists() or shutil.which("uv") is None:
        return 0

    subprocess.run(["uv", "run", "--quiet", "ruff", "format", str(path)], capture_output=True)
    result = subprocess.run(
        ["uv", "run", "--quiet", "ruff", "check", "--fix", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ruff found issues in {path}:\n{result.stdout}{result.stderr}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
