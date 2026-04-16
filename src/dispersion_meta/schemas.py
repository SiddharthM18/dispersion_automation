from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Products & families — canonical string identifiers
# ---------------------------------------------------------------------------

VALID_PRODUCTS = frozenset({"corridor_var", "vol", "gamma", "dngamma"})
VALID_FAMILIES = frozenset({"max_mean", "max_min", "max_sharpe", "min_drawdown", "composite"})
VALID_PROPOSAL_TYPES = frozenset({"best", "alt", "explore", "candidate"})
VALID_DECISIONS = frozenset({"accepted", "rejected", "modified"})
VALID_META_SCORE_MODES = frozenset({"research", "continuous"})
VALID_SOLVER_STATUSES = frozenset({"optimal", "optimal_inaccurate", "infeasible", "unbounded", "solver_error"})

# ---------------------------------------------------------------------------
# Schema definitions — versioned
# ---------------------------------------------------------------------------

DAILY_FEATURES_V1: dict[str, pl.DataType] = {
    "date": pl.Date,
    "product": pl.Utf8,
    "vix_level": pl.Float64,
    "vix_20d_change": pl.Float64,
    "avg_pairwise_corr": pl.Float64,
    "dispersion_pnl_20d": pl.Float64,
    "skew_steepness": pl.Float64,
    "term_structure_slope": pl.Float64,
    "earnings_density_21d": pl.Float64,
    "_table_schema_version": pl.Int32,
}

PROPOSALS_V1: dict[str, pl.DataType] = {
    "date": pl.Date,
    "product": pl.Utf8,
    "config_hash": pl.Utf8,
    "config_json": pl.Utf8,
    "config_schema_version": pl.Int32,
    "family": pl.Utf8,
    "proposal_type": pl.Utf8,
    "thompson_sample_value": pl.Float64,
    "posterior_mean": pl.Float64,
    "posterior_std": pl.Float64,
    "weights": pl.List(pl.Float64),
    "column_names": pl.List(pl.Utf8),
    "n_names": pl.Int32,
    "in_sample_sharpe": pl.Float64,
    "in_sample_mean_pnl": pl.Float64,
    "in_sample_max_dd": pl.Float64,
    "solver_status": pl.Utf8,
    "solve_time_seconds": pl.Float64,
    "pnl_matrix_path": pl.Utf8,
    "_table_schema_version": pl.Int32,
}

OUTCOMES_V1: dict[str, pl.DataType] = {
    "date": pl.Date,
    "product": pl.Utf8,
    "config_hash": pl.Utf8,
    "eval_date": pl.Date,
    "forward_window_days": pl.Int32,
    "forward_5d_pnl": pl.Float64,
    "forward_5d_mean_return": pl.Float64,
    "forward_realized_vol_21d": pl.Float64,
    "forward_sharpe": pl.Float64,
    "turnover_vs_prev_best": pl.Float64,
    "meta_score": pl.Float64,
    "meta_score_mode": pl.Utf8,
    "lambda_used": pl.Float64,
    "_table_schema_version": pl.Int32,
}

DECISIONS_V1: dict[str, pl.DataType] = {
    "date": pl.Date,
    "product": pl.Utf8,
    "config_hash": pl.Utf8,
    "decision": pl.Utf8,
    "decided_at_utc": pl.Datetime("us", "UTC"),
    "notes": pl.Utf8,
    "modified_config_hash": pl.Utf8,
    "_table_schema_version": pl.Int32,
}

# Current aliases
DAILY_FEATURES = DAILY_FEATURES_V1
PROPOSALS = PROPOSALS_V1
OUTCOMES = OUTCOMES_V1
DECISIONS = DECISIONS_V1

# Version numbers for each table
DAILY_FEATURES_VERSION = 1
PROPOSALS_VERSION = 1
OUTCOMES_VERSION = 1
DECISIONS_VERSION = 1

_TABLE_VERSIONS: dict[str, int] = {
    "daily_features": DAILY_FEATURES_VERSION,
    "proposals": PROPOSALS_VERSION,
    "outcomes": OUTCOMES_VERSION,
    "decisions": DECISIONS_VERSION,
}

_TABLE_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "daily_features": DAILY_FEATURES,
    "proposals": PROPOSALS,
    "outcomes": OUTCOMES,
    "decisions": DECISIONS,
}

# Nullable columns — these may legitimately contain null values
_NULLABLE_COLUMNS: dict[str, frozenset[str]] = {
    "daily_features": frozenset(),
    "proposals": frozenset({
        "thompson_sample_value", "posterior_mean", "posterior_std", "pnl_matrix_path",
    }),
    "outcomes": frozenset({"turnover_vs_prev_best"}),
    "decisions": frozenset({"notes", "modified_config_hash"}),
}


def validate_for_write(df: pl.DataFrame, table_name: str) -> pl.DataFrame:
    """Validate and prepare a DataFrame for writing.

    Injects _table_schema_version, casts to declared dtypes, and rejects
    extra/missing columns. Callers must NOT set _table_schema_version themselves
    — that's owned by this function.
    """
    schema = _TABLE_SCHEMAS[table_name]
    version = _TABLE_VERSIONS[table_name]

    # Caller must not set version themselves
    if "_table_schema_version" in df.columns and df["_table_schema_version"].drop_nulls().len() > 0:
        raise ValueError(
            "_table_schema_version must not be set by callers; the schema layer owns it"
        )

    # Drop the version column if present (empty/null), we'll inject it
    if "_table_schema_version" in df.columns:
        df = df.drop("_table_schema_version")

    # Check for missing columns (excluding version, which we inject)
    expected = set(schema.keys()) - {"_table_schema_version"}
    actual = set(df.columns)
    missing = expected - actual
    extra = actual - expected

    if missing:
        raise ValueError(f"Missing columns for {table_name}: {sorted(missing)}")
    if extra:
        raise ValueError(f"Extra columns for {table_name}: {sorted(extra)}")

    # Inject version
    df = df.with_columns(pl.lit(version).cast(pl.Int32).alias("_table_schema_version"))

    # Cast to declared dtypes and reorder to schema column order
    cast_exprs = []
    for col_name, dtype in schema.items():
        cast_exprs.append(pl.col(col_name).cast(dtype))
    df = df.select(cast_exprs)

    return df


def assert_schema_compatible(df: pl.DataFrame, table_name: str) -> None:
    """Check that a read DataFrame is compatible with the current schema.

    Lenient: tolerates extra columns (forward compat) but rejects missing ones.
    """
    schema = _TABLE_SCHEMAS[table_name]
    expected = set(schema.keys())
    actual = set(df.columns)
    missing = expected - actual

    if missing:
        raise ValueError(
            f"Read {table_name} is missing required columns: {sorted(missing)}"
        )
