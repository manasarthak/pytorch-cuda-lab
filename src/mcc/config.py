"""Runtime configuration, sourced from environment variables.

Nothing org-specific is hard-coded here. The bucket name and DB segment come from
the environment (see ``.env.example``), so this code is safe to publish. AWS
credentials are resolved by boto3's normal chain; we never read or store them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (no third-party dep). Only sets vars not already set."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class S3Config:
    """Where the raw BSON blobs live. Values come only from the environment."""

    bucket: str
    eva_db: str

    @classmethod
    def from_env(cls) -> "S3Config":
        _load_dotenv()
        bucket = os.environ.get("MCC_S3_BUCKET")
        if not bucket:
            raise RuntimeError(
                "MCC_S3_BUCKET is not set. Copy .env.example to .env and fill it in."
            )
        eva_db = os.environ.get("MCC_S3_EVA_DB", "signal_melt_db")
        return cls(bucket=bucket, eva_db=eva_db)


def data_dir() -> Path:
    _load_dotenv()
    d = Path(os.environ.get("MCC_DATA_DIR", "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d
