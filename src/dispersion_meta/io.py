from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from . import paths, schemas


# ===================================================================
# Daily features
# ===================================================================

def write_features(df: pl.DataFrame) -> None:
    """Write daily features. Merges with existing data, overwriting on (date, product)."""
    df = schemas.validate_for_write(df, "daily_features")
    existing = paths.read_parquet_if_exists(paths.features_path())
    if existing is not None:
        schemas.assert_schema_compatible(existing, "daily_features")
        # Remove rows for dates+products being overwritten
        keys = df.select("date", "product")
        existing = existing.join(keys, on=["date", "product"], how="anti")
        df = pl.concat([existing, df], how="vertical_relaxed")
    paths.atomic_write_parquet(df.sort("date", "product"), paths.features_path())


def read_features(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Read features with optional date range and product filters."""
    df = paths.read_parquet_if_exists(paths.features_path())
    if df is None:
        return None
    schemas.assert_schema_compatible(df, "daily_features")
    if start_date is not None:
        df = df.filter(pl.col("date") >= start_date)
    if end_date is not None:
        df = df.filter(pl.col("date") <= end_date)
    if products is not None:
        df = df.filter(pl.col("product").is_in(products))
    return df


# ===================================================================
# Proposals
# ===================================================================

def write_proposals(
    df: pl.DataFrame,
    pnl_matrices: dict[tuple[date, str, str], np.ndarray] | None = None,
) -> None:
    """Write proposals and optionally store PnL matrices.

    Three-step write:
    1. Write PnL matrices for proposed configs (best/alt/explore)
    2. Write proposal rows (merged with existing)
    3. Orphan sweep — remove matrix files not in today's proposals

    pnl_matrices keys are (date, product, config_hash).
    """
    df = schemas.validate_for_write(df, "proposals")

    # Step 1: write PnL matrices
    if pnl_matrices:
        for (dt, product, config_hash), matrix in pnl_matrices.items():
            mat_df = pl.DataFrame(
                {f"col_{i}": matrix[:, i] for i in range(matrix.shape[1])},
                schema={f"col_{i}": pl.Float64 for i in range(matrix.shape[1])},
            )
            mat_path = paths.pnl_matrix_path(dt, product, config_hash)
            paths.atomic_write_parquet(mat_df, mat_path)

    # Step 2: write proposal rows
    existing = paths.read_parquet_if_exists(paths.proposals_path())
    if existing is not None:
        schemas.assert_schema_compatible(existing, "proposals")
        keys = df.select("date", "product", "config_hash")
        existing = existing.join(keys, on=["date", "product", "config_hash"], how="anti")
        df = pl.concat([existing, df], how="vertical_relaxed")
    paths.atomic_write_parquet(df.sort("date", "product", "config_hash"), paths.proposals_path())

    # Step 3: orphan sweep per (date, product)
    if pnl_matrices:
        _orphan_sweep(pnl_matrices, df)


def _orphan_sweep(
    pnl_matrices: dict[tuple[date, str, str], np.ndarray],
    all_proposals: pl.DataFrame,
) -> None:
    """Remove matrix files not in current proposals and leftover .tmp files."""
    # Collect (date, product) pairs that had matrices written
    day_products: set[tuple[date, str]] = set()
    for dt, product, _ in pnl_matrices:
        day_products.add((dt, product))

    for dt, product in day_products:
        mat_dir = paths.pnl_matrix_dir(dt, product)
        if not mat_dir.exists():
            continue

        # Get config hashes for this date+product from proposals
        valid_hashes = set(
            all_proposals.filter(
                (pl.col("date") == dt) & (pl.col("product") == product)
            )["config_hash"].to_list()
        )

        for f in mat_dir.iterdir():
            # Clean up .tmp files
            if f.suffix == ".tmp" or f.name.endswith(".parquet.tmp"):
                f.unlink(missing_ok=True)
                continue
            # Remove orphaned matrix files
            if f.suffix == ".parquet":
                file_hash = f.stem
                if file_hash not in valid_hashes:
                    f.unlink(missing_ok=True)


def read_proposals(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
    families: list[str] | None = None,
    proposal_types: list[str] | None = None,
) -> pl.DataFrame | None:
    """Read proposals with optional filters."""
    df = paths.read_parquet_if_exists(paths.proposals_path())
    if df is None:
        return None
    schemas.assert_schema_compatible(df, "proposals")
    if start_date is not None:
        df = df.filter(pl.col("date") >= start_date)
    if end_date is not None:
        df = df.filter(pl.col("date") <= end_date)
    if products is not None:
        df = df.filter(pl.col("product").is_in(products))
    if families is not None:
        df = df.filter(pl.col("family").is_in(families))
    if proposal_types is not None:
        df = df.filter(pl.col("proposal_type").is_in(proposal_types))
    return df


def read_pnl_matrix(dt: date, product: str, config_hash: str) -> np.ndarray | None:
    """Read a single PnL matrix by (date, product, config_hash)."""
    mat_path = paths.pnl_matrix_path(dt, product, config_hash)
    if not mat_path.exists():
        return None
    mat_df = pl.read_parquet(mat_path)
    return mat_df.to_numpy()


# ===================================================================
# Outcomes
# ===================================================================

def write_outcomes(df: pl.DataFrame) -> None:
    """Write outcomes. Merges with existing, overwriting on (date, product, config_hash)."""
    if len(df) == 0:
        raise ValueError("Cannot write empty outcomes DataFrame")
    df = schemas.validate_for_write(df, "outcomes")
    existing = paths.read_parquet_if_exists(paths.outcomes_path())
    if existing is not None:
        schemas.assert_schema_compatible(existing, "outcomes")
        keys = df.select("date", "product", "config_hash")
        existing = existing.join(keys, on=["date", "product", "config_hash"], how="anti")
        df = pl.concat([existing, df], how="vertical_relaxed")
    paths.atomic_write_parquet(df.sort("date", "product", "config_hash"), paths.outcomes_path())


def read_outcomes(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Read outcomes with optional filters."""
    df = paths.read_parquet_if_exists(paths.outcomes_path())
    if df is None:
        return None
    schemas.assert_schema_compatible(df, "outcomes")
    if start_date is not None:
        df = df.filter(pl.col("date") >= start_date)
    if end_date is not None:
        df = df.filter(pl.col("date") <= end_date)
    if products is not None:
        df = df.filter(pl.col("product").is_in(products))
    return df


# ===================================================================
# Decisions
# ===================================================================

def append_decisions(df: pl.DataFrame) -> None:
    """Append decision rows. Validates decision values and datetime awareness.

    Decisions are append-only: a (date, product, config_hash) may have multiple rows
    if the user changes their mind. Latest by decided_at_utc wins when collapsed.
    """
    df = _validate_decisions(df)
    df = schemas.validate_for_write(df, "decisions")
    existing = paths.read_parquet_if_exists(paths.decisions_path())
    if existing is not None:
        schemas.assert_schema_compatible(existing, "decisions")
        df = pl.concat([existing, df], how="vertical_relaxed")
    paths.atomic_write_parquet(df.sort("decided_at_utc"), paths.decisions_path())


def _validate_decisions(df: pl.DataFrame) -> pl.DataFrame:
    """Validate decision-specific constraints before schema validation."""
    # Validate decision values
    if "decision" in df.columns:
        invalid = set(df["decision"].unique().to_list()) - schemas.VALID_DECISIONS
        if invalid:
            raise ValueError(f"Invalid decision values: {invalid}")

    # modified decision requires modified_config_hash
    if "decision" in df.columns and "modified_config_hash" in df.columns:
        modified_rows = df.filter(pl.col("decision") == "modified")
        if modified_rows["modified_config_hash"].null_count() > 0:
            raise ValueError("decision='modified' requires modified_config_hash to be set")
        non_modified = df.filter(pl.col("decision") != "modified")
        if non_modified["modified_config_hash"].null_count() != len(non_modified):
            raise ValueError("modified_config_hash must be null when decision is not 'modified'")

    # Reject naive datetimes
    if "decided_at_utc" in df.columns:
        dtype = df["decided_at_utc"].dtype
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            raise ValueError(
                "decided_at_utc must be timezone-aware UTC; got naive datetime"
            )

    return df


def read_decisions_raw(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Read the full decision audit log."""
    df = paths.read_parquet_if_exists(paths.decisions_path())
    if df is None:
        return None
    schemas.assert_schema_compatible(df, "decisions")
    if start_date is not None:
        df = df.filter(pl.col("date") >= start_date)
    if end_date is not None:
        df = df.filter(pl.col("date") <= end_date)
    if products is not None:
        df = df.filter(pl.col("product").is_in(products))
    return df


def read_decisions_latest(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Read decisions collapsed to one row per (date, product, config_hash) by latest timestamp."""
    df = read_decisions_raw(start_date=start_date, end_date=end_date, products=products)
    if df is None:
        return None
    return (
        df.sort("decided_at_utc")
        .group_by(["date", "product", "config_hash"])
        .last()
    )
