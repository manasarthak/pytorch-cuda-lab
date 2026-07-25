"""Build a model-ready dataset from a chip manifest.

A manifest is a CSV (``chip_id,species``) or a JSON list (``[{"id", "species"}, ...]``);
see :func:`read_manifest`. For each chip we fetch the raw curves (cached), mine
positive wells, preprocess them, and stack everything into arrays. Splitting is done
*by chip* so wells from one chip never straddle the train/test boundary (matching the
org pipeline's grouped splits).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import data_dir
from .preprocess import PreprocessConfig, preprocess_positive_curves
from .s3_source import fetch_raw_eva


@dataclass
class Dataset:
    X: np.ndarray            # (N, roi_len) float32 — one preprocessed curve per row
    species: np.ndarray      # (N,) str — class label per curve
    chip_id: np.ndarray      # (N,) str — provenance, used for grouped splitting

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.species.tolist()))

    def y_int(self) -> tuple[np.ndarray, list[str]]:
        """Integer-encoded labels plus the class list (index == label id)."""
        classes = self.classes
        lookup = {c: i for i, c in enumerate(classes)}
        y = np.array([lookup[s] for s in self.species], dtype=np.int64)
        return y, classes

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, X=self.X, species=self.species, chip_id=self.chip_id)

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        d = np.load(path, allow_pickle=True)
        return cls(X=d["X"], species=d["species"], chip_id=d["chip_id"])


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a chip manifest into a ``(chip_id, species)`` DataFrame.

    Accepts two formats, chosen by file extension:

    * ``.json`` -- a list of objects, each with ``species`` and a chip id under
      either ``id`` or ``chip_id``. Example::

          [{"id": "260128_D06...-Phusion", "species": "K. pneumoniae"}, ...]

    * ``.csv`` (or anything else) -- columns ``chip_id,species`` (``id`` also
      accepted as the id column).

    The ``id``/``chip_id`` value is used verbatim as the S3 object key stem.
    """
    path = Path(path)
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(
                f"JSON manifest must be a list of objects, got {type(records).__name__}: {path}"
            )
        rows = []
        for r in records:
            chip_id = r.get("chip_id") or r.get("id")
            species = r.get("species")
            if chip_id and species:
                rows.append({"chip_id": str(chip_id).strip(), "species": str(species).strip()})
        df = pd.DataFrame(rows, columns=["chip_id", "species"])
    else:
        df = pd.read_csv(path, dtype=str)
        if "chip_id" not in df.columns and "id" in df.columns:
            df = df.rename(columns={"id": "chip_id"})
        missing = {"chip_id", "species"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV manifest missing column(s) {sorted(missing)}: {path}")
        df = df.dropna(subset=["chip_id", "species"])

    if df.empty:
        raise ValueError(f"No valid (chip_id, species) rows in manifest: {path}")
    return df[["chip_id", "species"]].reset_index(drop=True)


def build_dataset(
    manifest_path: str | Path,
    cfg: PreprocessConfig = PreprocessConfig(),
    use_cache: bool = True,
) -> Dataset:
    """Fetch + mine + preprocess every chip in the manifest into one Dataset."""
    manifest = read_manifest(manifest_path)
    X_parts: list[np.ndarray] = []
    species_parts: list[np.ndarray] = []
    chip_parts: list[np.ndarray] = []

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="chips"):
        raw = fetch_raw_eva(row.chip_id, use_cache=use_cache)
        X, indices = preprocess_positive_curves(raw, cfg)
        if X.shape[0] == 0:
            continue
        X_parts.append(X.astype(np.float32))
        species_parts.append(np.full(X.shape[0], row.species, dtype=object))
        chip_parts.append(np.full(X.shape[0], row.chip_id, dtype=object))

    if not X_parts:
        raise RuntimeError("No positive curves found across the manifest.")

    return Dataset(
        X=np.concatenate(X_parts, axis=0),
        species=np.concatenate(species_parts).astype(str),
        chip_id=np.concatenate(chip_parts).astype(str),
    )


def grouped_train_test_split(
    ds: Dataset, test_fraction: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split curve indices into train/test by *chip*, stratified by species.

    Returns ``(train_idx, test_idx)`` — integer indices into ``ds.X``.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []

    df = pd.DataFrame({"i": np.arange(len(ds)), "chip": ds.chip_id, "species": ds.species})
    for _species, sp_group in df.groupby("species"):
        chips = sp_group["chip"].unique()
        rng.shuffle(chips)
        n_test = max(1, round(len(chips) * test_fraction)) if len(chips) > 1 else 0
        test_chips = set(chips[:n_test])
        for chip, chip_group in sp_group.groupby("chip"):
            target = test_idx if chip in test_chips else train_idx
            target.extend(chip_group["i"].tolist())

    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def build_and_save(manifest_path: str | Path, out_name: str = "dataset.npz") -> Path:
    ds = build_dataset(manifest_path)
    out_path = data_dir() / "processed" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save(out_path)
    return out_path
