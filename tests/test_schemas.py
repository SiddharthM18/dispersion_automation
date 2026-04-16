from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from dispersion_meta.schemas import (
    assert_schema_compatible,
    validate_for_write,
)


def _minimal_features_df() -> pl.DataFrame:
    return pl.DataFrame({
        "date": [date(2024, 1, 2)],
        "product": ["vol"],
        "vix_level": [20.0],
        "vix_20d_change": [1.0],
        "avg_pairwise_corr": [0.4],
        "dispersion_pnl_20d": [0.01],
        "skew_steepness": [0.1],
        "term_structure_slope": [0.05],
        "earnings_density_21d": [0.2],
    })


class TestValidateForWrite:
    def test_happy_path(self):
        df = _minimal_features_df()
        result = validate_for_write(df, "daily_features")
        assert "_table_schema_version" in result.columns
        assert result["_table_schema_version"][0] == 1

    def test_missing_column_raises(self):
        df = _minimal_features_df().drop("vix_level")
        with pytest.raises(ValueError, match="Missing columns"):
            validate_for_write(df, "daily_features")

    def test_extra_column_raises(self):
        df = _minimal_features_df().with_columns(pl.lit(99).alias("bogus"))
        with pytest.raises(ValueError, match="Extra columns"):
            validate_for_write(df, "daily_features")

    def test_caller_setting_version_raises(self):
        df = _minimal_features_df().with_columns(
            pl.lit(1).cast(pl.Int32).alias("_table_schema_version")
        )
        with pytest.raises(ValueError, match="must not be set by callers"):
            validate_for_write(df, "daily_features")

    def test_column_order_matches_schema(self):
        df = _minimal_features_df()
        result = validate_for_write(df, "daily_features")
        from dispersion_meta.schemas import DAILY_FEATURES
        assert list(result.columns) == list(DAILY_FEATURES.keys())

    def test_dtype_casting(self):
        """Int columns should be cast to Float64 where schema expects it."""
        df = _minimal_features_df().with_columns(
            pl.col("vix_level").cast(pl.Int64)
        )
        result = validate_for_write(df, "daily_features")
        assert result["vix_level"].dtype == pl.Float64


class TestAssertSchemaCompatible:
    def test_happy_path(self):
        df = validate_for_write(_minimal_features_df(), "daily_features")
        assert_schema_compatible(df, "daily_features")  # should not raise

    def test_extra_columns_tolerated(self):
        """Lenient read: extra columns are fine (forward compat)."""
        df = validate_for_write(_minimal_features_df(), "daily_features")
        df = df.with_columns(pl.lit("extra").alias("future_col"))
        assert_schema_compatible(df, "daily_features")  # should not raise

    def test_missing_column_raises(self):
        df = validate_for_write(_minimal_features_df(), "daily_features")
        df = df.drop("vix_level")
        with pytest.raises(ValueError, match="missing required columns"):
            assert_schema_compatible(df, "daily_features")
