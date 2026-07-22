"""Rung 1: train the XGBoost baseline on a built dataset.

Usage:
    python scripts/run_baseline.py data/processed/dataset.npz
"""

from __future__ import annotations

import sys

from mcc.data.dataset import Dataset, grouped_train_test_split
from mcc.models.classical import train_xgb


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/dataset.npz"
    ds = Dataset.load(path)
    y, classes = ds.y_int()
    train_idx, test_idx = grouped_train_test_split(ds, test_fraction=0.25, seed=0)

    print(f"{len(ds)} curves | {len(classes)} classes | "
          f"{len(train_idx)} train / {len(test_idx)} test (grouped by chip)")

    result = train_xgb(
        ds.X[train_idx], y[train_idx],
        ds.X[test_idx], y[test_idx],
        classes,
        device="cuda",
    )
    print(f"\nAccuracy: {result.accuracy:.4f}\n")
    print(result.report)
    print("Top features:")
    for name, imp in list(result.importances.items())[:8]:
        print(f"  {name:12s} {imp:.4f}")


if __name__ == "__main__":
    main()
