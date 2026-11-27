"""Ensure the canonical dataset CSVs exist before packaging the sidecar.

The desktop build is offline at runtime, so every registered dataset must be
present in ``.cache/prebenchmark`` when PyInstaller collects the bundle. This
script downloads whatever is missing, then verifies the six files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prebenchmark import data  # noqa: E402 - repo root must be on sys.path first


def main() -> int:
    root = ROOT
    expected = root / data.DEFAULT_CACHE_SUBDIR
    print(f"dataset cache: {expected}")

    for name in data.DATASETS:
        info = data.describe_dataset(name)
        print(f"  {name}: {info.n_samples} samples, {info.n_features} features -> {info.cache_path}")

    missing = [
        name
        for name, definition in data.DATASETS.items()
        if not (expected / definition.cache_filename).is_file()
    ]
    if missing:
        print(f"error: missing datasets after staging: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("all datasets staged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
