"""Runtime configuration, sourced from environment variables.

Nothing org-specific is hard-coded here. The bucket name and DB segment come from
the environment, so this code is safe to publish. AWS credentials are resolved by
boto3's normal chain; we never read or store them.

Secrets file location
---------------------
The optional dotenv file lives *outside* the repo so it can never be committed.
Point at it with the ``MCC_ENV_FILE`` environment variable (set once in your shell
or user environment), e.g. on Windows PowerShell::

    setx MCC_ENV_FILE "C:\\Users\\you\\secrets\\gpulab.env"

Resolution order for each config value:

1. A variable already set in the real environment wins (dotenv never overrides it).
2. Else the file at ``MCC_ENV_FILE`` is loaded, if that variable is set.
3. Else a ``.env`` in the current directory is loaded if present (dev convenience;
   also gitignored).

So you can keep the file anywhere you like, or skip the file entirely and just
export the variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Env var whose value is the path to the dotenv file (outside the repo).
ENV_FILE_VAR = "MCC_ENV_FILE"


def resolve_env_file() -> Path | None:
    """Return the dotenv path to load, or None. Raises if MCC_ENV_FILE is a bad path."""
    override = os.environ.get(ENV_FILE_VAR)
    if override:
        p = Path(override).expanduser()
        if not p.is_file():
            raise RuntimeError(
                f"{ENV_FILE_VAR} points to {p}, which does not exist. "
                "Fix the path or unset the variable."
            )
        return p
    fallback = Path(".env")
    return fallback if fallback.is_file() else None


def _load_dotenv() -> None:
    """Load the resolved dotenv file (if any). Only sets vars not already set."""
    p = resolve_env_file()
    if p is None:
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
                "MCC_S3_BUCKET is not set. Either export it, or point MCC_ENV_FILE "
                "at a dotenv file that defines it (see gpulab/config.py docstring)."
            )
        eva_db = os.environ.get("MCC_S3_EVA_DB", "signal_melt_db")
        return cls(bucket=bucket, eva_db=eva_db)


def data_dir() -> Path:
    _load_dotenv()
    d = Path(os.environ.get("MCC_DATA_DIR", "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d
