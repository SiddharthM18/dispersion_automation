from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from . import paths, schemas


# ---------------------------------------------------------------------------
# Internal helpers for partition-scoped writes
# ---------------------------------------------------------------------------

def _dates_in_df(df: pl.DataFrame) -> set[date]:
    """Extract unique dates from a DataFrame."""
    return set(df["date"].unique().to_list())


def _write_partitioned(
    df: pl.DataFrame,
    table_dir_fn: callable,
    table_name: str,
    key_columns: list[str],
) -> None:
    """Write a DataFrame into Hive-partitioned Parquet files.

    For each distinct year/month partition in the new data:
    1. Read the existing partition file (if any)
    2. Anti-join to remove rows being overwritten
    3. Concat and write back
    """
    df = schemas.validate_for_write(df, table_name)
    table_dir = table_dir_fn()

    # Group new data by partition (year, month)
    partitions: dict[tuple[int, int], pl.DataFrame] = {}
    for dt in _dates_in_df(df):
        key = (dt.year, dt.month)
        part = df.filter(
            (pl.col("date").dt.year() == dt.year)
            & (pl.col("date").dt.month() == dt.month)
        )
        if key in partitions:
            partitions[key] = pl.concat([partitions[key], part], how="vertical_relaxed")
        else:
            partitions[key] = part

    for (year, month), part_df in partitions.items():
        # Deduplicate within the partition being written
        part_df = part_df.unique(subset=key_columns, keep="last")

        ref_date = date(year, month, 1)
        part_path = paths.partition_path(table_dir, ref_date)
        existing = paths.read_parquet_if_exists(part_path)

        if existing is not None:
            schemas.assert_schema_compatible(existing, table_name)
            keys = part_df.select(key_columns)
            existing = existing.join(keys, on=key_columns, how="anti")
            part_df = pl.concat([existing, part_df], how="vertical_relaxed")

        paths.atomic_write_parquet(
            part_df.sort(key_columns), part_path
        )


def _read_partitioned(
    table_dir_fn: callable,
    table_name: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Read from a Hive-partitioned table directory with optional filters."""
    df = paths.scan_table_dir(table_dir_fn())
    if df is None:
        return None
    schemas.assert_schema_compatible(df, table_name)
    if start_date is not None:
        df = df.filter(pl.col("date") >= start_date)
    if end_date is not None:
        df = df.filter(pl.col("date") <= end_date)
    if products is not None:
        df = df.filter(pl.col("product").is_in(products))
    return df


# ===================================================================
# Daily features
# ===================================================================

def write_features(df: pl.DataFrame) -> None:
    """Write daily features. Merges with existing data, overwriting on (date, product)."""
    _write_partitioned(df, paths.features_dir, "daily_features", ["date", "product"])


def read_features(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Read features with optional date range and product filters."""
    return _read_partitioned(
        paths.features_dir, "daily_features",
        start_date=start_date, end_date=end_date, products=products,
    )


# ===================================================================
# Proposals
# ===================================================================

def _validate_proposals(
    df: pl.DataFrame,
    pnl_matrices: dict[tuple[date, str, str], np.ndarray] | None = None,
) -> None:
    """Validate semantic constraints on proposals that schema validation can't catch.

    Runs before any writes so bad data fails early with specific error messages
    rather than producing silently corrupt rows.
    """
    errors: list[str] = []

    # family must be valid
    if "family" in df.columns:
        invalid = set(df["family"].unique().to_list()) - schemas.VALID_FAMILIES
        if invalid:
            errors.append(f"Invalid family values: {sorted(invalid)}")

    # proposal_type must be valid
    if "proposal_type" in df.columns:
        invalid = set(df["proposal_type"].unique().to_list()) - schemas.VALID_PROPOSAL_TYPES
        if invalid:
            errors.append(f"Invalid proposal_type values: {sorted(invalid)}")

    # solver_status must be valid
    if "solver_status" in df.columns:
        invalid = set(df["solver_status"].unique().to_list()) - schemas.VALID_SOLVER_STATUSES
        if invalid:
            errors.append(f"Invalid solver_status values: {sorted(invalid)}")

    # Per-row checks: weights/column_names length match, n_names consistency
    for i, row in enumerate(df.iter_rows(named=True)):
        weights = row.get("weights")
        col_names = row.get("column_names")

        if weights is not None and col_names is not None:
            w_list = weights.to_list() if isinstance(weights, pl.Series) else weights
            c_list = col_names.to_list() if isinstance(col_names, pl.Series) else col_names

            if len(w_list) != len(c_list):
                errors.append(
                    f"Row {i}: len(weights)={len(w_list)} != "
                    f"len(column_names)={len(c_list)}"
                )

    # PnL matrix shape must match column_names when provided
    if pnl_matrices:
        for (dt, product, config_hash), matrix in pnl_matrices.items():
            if matrix.ndim != 2:
                errors.append(
                    f"PnL matrix ({dt}, {product}, {config_hash}): "
                    f"expected 2D, got {matrix.ndim}D"
                )
                continue

            match = df.filter(
                (pl.col("date") == dt)
                & (pl.col("product") == product)
                & (pl.col("config_hash") == config_hash)
            )
            if len(match) > 0:
                col_names = match["column_names"][0]
                c_list = col_names.to_list() if isinstance(col_names, pl.Series) else col_names
                if matrix.shape[1] != len(c_list):
                    errors.append(
                        f"PnL matrix ({dt}, {product}, {config_hash}): "
                        f"matrix cols={matrix.shape[1]} != "
                        f"len(column_names)={len(c_list)}"
                    )

    if errors:
        raise ValueError(
            f"Proposal validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def write_proposals(
    df: pl.DataFrame,
    pnl_matrices: dict[tuple[date, str, str], np.ndarray] | None = None,
) -> None:
    """Write proposals and optionally store PnL matrices.

    Three-step write:
    1. Write PnL matrices for proposed configs (best/alt/explore)
    2. Write proposal rows (partitioned by year/month)
    3. Orphan sweep — remove matrix files not in today's proposals

    pnl_matrices keys are (date, product, config_hash).
    """
    # Semantic validation first — fail early with specific messages
    _validate_proposals(df, pnl_matrices)
    # Schema validation
    schemas.validate_for_write(df, "proposals")

    # Step 1: write PnL matrices
    if pnl_matrices:
        for (dt, product, config_hash), matrix in pnl_matrices.items():
            mat_df = pl.DataFrame(
                {f"col_{i}": matrix[:, i] for i in range(matrix.shape[1])},
                schema={f"col_{i}": pl.Float64 for i in range(matrix.shape[1])},
            )
            mat_path = paths.pnl_matrix_path(dt, product, config_hash)
            paths.atomic_write_parquet(mat_df, mat_path)

    # Step 2: write proposal rows (partitioned)
    _write_partitioned(df, paths.proposals_dir, "proposals", ["date", "product", "config_hash"])

    # Step 3: orphan sweep per (date, product)
    if pnl_matrices:
        # Re-read the full proposals for affected dates to get valid hashes
        all_proposals = paths.scan_table_dir(paths.proposals_dir())
        if all_proposals is not None:
            _orphan_sweep(pnl_matrices, all_proposals)


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
    df = _read_partitioned(
        paths.proposals_dir, "proposals",
        start_date=start_date, end_date=end_date, products=products,
    )
    if df is None:
        return None
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
    _write_partitioned(df, paths.outcomes_dir, "outcomes", ["date", "product", "config_hash"])


def read_outcomes(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Read outcomes with optional filters."""
    return _read_partitioned(
        paths.outcomes_dir, "outcomes",
        start_date=start_date, end_date=end_date, products=products,
    )


# ===================================================================
# Decisions
# ===================================================================

def append_decisions(df: pl.DataFrame) -> None:
    """Append decision rows. Validates decision values and datetime awareness.

    Decisions are append-only: a (date, product, config_hash) may have multiple rows
    if the user changes their mind. Latest by decided_at_utc wins when collapsed.

    Unlike other tables, decisions are truly append-only — no anti-join dedup.
    Each partition file accumulates all decisions for that month.
    """
    df = _validate_decisions(df)
    df = schemas.validate_for_write(df, "decisions")
    table_dir = paths.decisions_dir()

    # Group by partition and append to each
    for dt in _dates_in_df(df):
        ref_date = date(dt.year, dt.month, 1)
        part_path = paths.partition_path(table_dir, ref_date)
        part_df = df.filter(
            (pl.col("date").dt.year() == dt.year)
            & (pl.col("date").dt.month() == dt.month)
        )

        existing = paths.read_parquet_if_exists(part_path)
        if existing is not None:
            schemas.assert_schema_compatible(existing, "decisions")
            part_df = pl.concat([existing, part_df], how="vertical_relaxed")

        paths.atomic_write_parquet(part_df.sort("decided_at_utc"), part_path)


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
    return _read_partitioned(
        paths.decisions_dir, "decisions",
        start_date=start_date, end_date=end_date, products=products,
    )


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
