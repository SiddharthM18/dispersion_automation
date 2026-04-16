from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Data root — default to ./data, overrideable for tests
# ---------------------------------------------------------------------------

_data_root: Path | None = None


def set_data_root(path: Path | str | None) -> None:
    """Set the data root directory. Pass None to reset to default."""
    global _data_root
    _data_root = Path(path) if path is not None else None


def data_root() -> Path:
    """Return the current data root, defaulting to ./data."""
    if _data_root is not None:
        return _data_root
    return Path("data")


# ---------------------------------------------------------------------------
# Table paths — single Parquet file per table
# ---------------------------------------------------------------------------

def features_path() -> Path:
    return data_root() / "daily_features.parquet"


def proposals_path() -> Path:
    return data_root() / "proposals.parquet"


def outcomes_path() -> Path:
    return data_root() / "outcomes.parquet"


def decisions_path() -> Path:
    return data_root() / "decisions.parquet"


# ---------------------------------------------------------------------------
# PnL matrix paths
# ---------------------------------------------------------------------------

def pnl_matrix_dir(dt: date, product: str) -> Path:
    return data_root() / "pnl_matrices" / str(dt) / product


def pnl_matrix_path(dt: date, product: str, config_hash: str) -> Path:
    return pnl_matrix_dir(dt, product) / f"{config_hash}.parquet"


# ---------------------------------------------------------------------------
# Atomic write — write to .tmp then os.replace for crash safety
# ---------------------------------------------------------------------------

def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write a Polars DataFrame to Parquet atomically.

    Writes to {path}.tmp first, then os.replace to final path.
    This guarantees that readers never see a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        df.write_parquet(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        tmp_path.unlink(missing_ok=True)
        raise


def read_parquet_if_exists(path: Path) -> pl.DataFrame | None:
    """Read a Parquet file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    return pl.read_parquet(path)
