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
# Hive partition helpers
# ---------------------------------------------------------------------------

def _partition_dir(table_dir: Path, dt: date) -> Path:
    """Return the year=YYYY/month=MM partition directory for a date."""
    return table_dir / f"year={dt.year}" / f"month={dt.month:02d}"


def partition_path(table_dir: Path, dt: date) -> Path:
    """Return the Parquet file path within a Hive partition for a date."""
    return _partition_dir(table_dir, dt) / "data.parquet"


# ---------------------------------------------------------------------------
# Table directories — Hive-partitioned by year/month
# ---------------------------------------------------------------------------

def features_dir() -> Path:
    return data_root() / "daily_features"


def proposals_dir() -> Path:
    return data_root() / "proposals"


def outcomes_dir() -> Path:
    return data_root() / "outcomes"


def decisions_dir() -> Path:
    return data_root() / "decisions"


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
    """Read a single Parquet file, returning None if it doesn't exist."""
    if not path.exists():
        return None
    return pl.read_parquet(path)


def scan_table_dir(table_dir: Path) -> pl.DataFrame | None:
    """Scan all Parquet files in a Hive-partitioned table directory.

    Returns None if the directory doesn't exist or contains no parquet files.
    The year=/month= partition columns are NOT included in the result —
    dates are already stored as columns in the data itself.
    """
    if not table_dir.exists():
        return None
    parquet_files = sorted(table_dir.glob("**/data.parquet"))
    if not parquet_files:
        return None
    return pl.read_parquet(
        parquet_files,
        hive_partitioning=False,
    )
