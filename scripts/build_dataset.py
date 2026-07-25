"""Build a preprocessed dataset from a chip manifest.

Usage:
    python scripts/build_dataset.py [manifest]

`manifest` may be a JSON list ([{"id", "species"}, ...]) or a CSV (chip_id,species).
If omitted, looks for configs/manifest.json then configs/manifest.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

from gpulab.data.dataset import build_and_save


def _default_manifest() -> str:
    for candidate in ("configs/manifest.json", "configs/manifest.csv"):
        if Path(candidate).is_file():
            return candidate
    raise SystemExit(
        "No manifest found. Pass a path, or create configs/manifest.json "
        "(see configs/manifest.example.json)."
    )


def main() -> None:
    manifest = sys.argv[1] if len(sys.argv) > 1 else _default_manifest()
    out = build_and_save(manifest)
    print(f"Saved dataset to {out}")


if __name__ == "__main__":
    main()
