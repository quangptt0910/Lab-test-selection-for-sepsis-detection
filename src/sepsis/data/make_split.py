"""Create the project's one dev/test split and CV folds from the parquet cache (run once).

Run with ``uv run python -m sepsis.data.make_split`` after ``sepsis.data.build_cache``.
Settings come from ``configs/protocol.yaml`` and paths from ``configs/data.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sepsis.config import DATA_CONFIG, PROTOCOL_CONFIG, load_yaml
from sepsis.data.load import HOSPITALS, read_cache
from sepsis.data.patients import patient_table
from sepsis.data.splits import (
    SplitFingerprintError,
    fingerprint,
    load_split,
    make_split,
    summarize_split,
    write_split,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-config", type=Path, default=DATA_CONFIG)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_CONFIG)
    args = parser.parse_args(argv)

    paths = load_yaml(args.data_config)
    settings = load_yaml(args.protocol)["split"]
    split_path = Path(paths["split_path"])
    raw = settings.get("fingerprint")
    expected = None if raw is None else str(raw)  # YAML may parse an all-digit value as int

    if split_path.exists():
        if not expected:
            print(
                f"{split_path} already exists; the split is created once (ADR-0004).",
                file=sys.stderr,
            )
            return 1
        # Only checks the file; never rewrites it. The fingerprint covers the whole file,
        # so the default dev-only load checks it without returning test rows.
        try:
            load_split(split_path, expected)
        except SplitFingerprintError as err:
            print(f"{err}\nGet the committed split file from the repository.", file=sys.stderr)
            return 1
        print(f"{split_path} matches split.fingerprint; nothing to do.")
        return 0

    # Reads ALL patients: allowed here, because this is where the split is made.
    patients = patient_table(read_cache(HOSPITALS, Path(paths["cache_dir"])))
    split = make_split(
        patients,
        test_fraction=settings["test_fraction"],
        n_folds=settings["n_folds"],
        seed=settings["seed"],
        stratify_by=settings["stratify_by"],
    )
    actual = fingerprint(split)

    if expected and actual != expected:
        print(
            f"The rebuilt split does not match split.fingerprint in {args.protocol}\n"
            f"  expected {expected}\n  got      {actual}\n"
            "Nothing was written. Library versions (pandas, scikit-learn) or the cache may "
            "differ; use the committed split file instead of rebuilding.",
            file=sys.stderr,
        )
        return 1
    write_split(split, split_path)

    print(f"wrote {split_path}")
    print(summarize_split(split).to_string(index=False))
    if expected:
        print("\nThe rebuilt split matches split.fingerprint.")
    else:
        print(f"\nfingerprint: {actual}")
        print(f"Save it as split.fingerprint in {args.protocol}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
