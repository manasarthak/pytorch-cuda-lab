"""Build a preprocessed dataset from a chip manifest.

Usage:
    python scripts/build_dataset.py configs/manifest.csv
"""

from __future__ import annotations

import sys

from gpulab.data.dataset import build_and_save


def main() -> None:
    manifest = sys.argv[1] if len(sys.argv) > 1 else "configs/manifest.csv"
    out = build_and_save(manifest)
    print(f"Saved dataset to {out}")


if __name__ == "__main__":
    main()
