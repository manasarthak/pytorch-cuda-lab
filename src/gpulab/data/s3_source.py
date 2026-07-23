"""Fetch raw melt-signal curves from S3.

Standalone: no org libraries. The blobs are gzip-compressed BSON documents, one
per well, each with an ``x`` (temperature/frame axis) and ``y`` (fluorescence)
array. The object key convention is::

    s3://{bucket}/{chip_id}/{eva_db}/{chip_id}.bson.gz

These helpers are deliberately generic (plain ``boto3`` + ``bson``) so they work
against any bucket you point them at via ``S3Config``.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np

from ..config import S3Config, data_dir


def object_key(chip_id: str, eva_db: str) -> str:
    """S3 key for a chip's raw signal blob."""
    return f"{chip_id}/{eva_db}/{chip_id}.bson.gz"


def _s3_client():
    import boto3  # imported lazily so preprocessing/tests don't require boto3

    return boto3.client("s3")


def _decode_bson_gz(raw_bytes: bytes) -> list[dict[str, Any]]:
    """Decode a gzip-compressed, concatenated-BSON blob into a list of documents."""
    from bson import decode_all  # from pymongo

    return decode_all(gzip.decompress(raw_bytes))


def download_raw_bytes(chip_id: str, cfg: S3Config) -> bytes:
    """Download the raw ``.bson.gz`` blob for a chip (no caching)."""
    key = object_key(chip_id, cfg.eva_db)
    client = _s3_client()
    resp = client.get_object(Bucket=cfg.bucket, Key=key)
    return resp["Body"].read()


def curves_from_documents(documents: list[dict[str, Any]]) -> np.ndarray:
    """Stack the per-well ``y`` arrays into a ``(n_wells, n_frames)`` float array."""
    if not documents:
        raise ValueError("No documents decoded from blob.")
    return np.asarray([doc["y"] for doc in documents], dtype=np.float64)


def fetch_raw_eva(chip_id: str, cfg: S3Config | None = None, use_cache: bool = True) -> np.ndarray:
    """Return raw curves for a chip as ``(n_wells, n_frames)``.

    Downloads from S3 once and caches the decoded array under the data dir so
    repeated runs are offline. Set ``use_cache=False`` to always re-download.
    """
    cfg = cfg or S3Config.from_env()
    cache_path = data_dir() / "raw" / f"{chip_id}.npy"
    if use_cache and cache_path.is_file():
        return np.load(cache_path)

    documents = _decode_bson_gz(download_raw_bytes(chip_id, cfg))
    curves = curves_from_documents(documents)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, curves)
    return curves
