from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

from dispersion_meta import io, paths


@pytest.fixture(autouse=True)
def tmp_data_root(tmp_path):
    """Redirect all IO to a temp directory for test isolation."""
    paths.set_data_root(tmp_path)
    yield tmp_path
    paths.set_data_root(None)


def _features_df(dt: date = date(2024, 1, 2), product: str = "vol") -> pl.DataFrame:
    return pl.DataFrame({
        "date": [dt],
        "product": [product],
        "vix_level": [20.0],
        "vix_20d_change": [1.0],
        "avg_pairwise_corr": [0.4],
        "dispersion_pnl_20d": [0.01],
        "skew_steepness": [0.1],
        "term_structure_slope": [0.05],
        "earnings_density_21d": [0.2],
    })


def _proposals_df(
    dt: date = date(2024, 1, 2),
    product: str = "vol",
    config_hash: str = "abc123def456789a",
) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [dt],
        "product": [product],
        "config_hash": [config_hash],
        "config_json": ['{"objective":"max_sharpe"}'],
        "config_schema_version": [1],
        "family": ["max_sharpe"],
        "proposal_type": ["best"],
        "thompson_sample_value": [0.5],
        "posterior_mean": [0.3],
        "posterior_std": [0.2],
        "weights": [[0.5, 0.3, 0.2]],
        "column_names": [["A", "B", "C"]],
        "n_names": [3],
        "in_sample_sharpe": [1.5],
        "in_sample_mean_pnl": [0.01],
        "in_sample_max_dd": [-0.03],
        "solver_status": ["optimal"],
        "solve_time_seconds": [0.1],
        "pnl_matrix_path": [f"pnl_matrices/{dt}/{product}/{config_hash}.parquet"],
    })


def _outcomes_df(
    dt: date = date(2024, 1, 2),
    product: str = "vol",
    config_hash: str = "abc123def456789a",
) -> pl.DataFrame:
    return pl.DataFrame({
        "date": [dt],
        "product": [product],
        "config_hash": [config_hash],
        "eval_date": [date(2024, 1, 9)],
        "forward_window_days": [5],
        "forward_5d_pnl": [0.05],
        "forward_5d_mean_return": [0.01],
        "forward_realized_vol_21d": [0.15],
        "forward_sharpe": [0.0667],
        "turnover_vs_prev_best": [None],
        "meta_score": [0.0667],
        "meta_score_mode": ["research"],
        "lambda_used": [0.0],
    })


# ===================================================================
# Features
# ===================================================================

class TestFeatures:
    def test_round_trip(self):
        df = _features_df()
        io.write_features(df)
        result = io.read_features()
        assert result is not None
        assert len(result) == 1
        assert result["product"][0] == "vol"

    def test_missing_date_returns_none(self):
        result = io.read_features()
        assert result is None

    def test_idempotent_rewrite(self):
        df = _features_df()
        io.write_features(df)
        io.write_features(df)  # same data again
        result = io.read_features()
        assert len(result) == 1

    def test_merge_different_dates(self):
        io.write_features(_features_df(date(2024, 1, 2)))
        io.write_features(_features_df(date(2024, 1, 3)))
        result = io.read_features()
        assert len(result) == 2

    def test_product_filter(self):
        io.write_features(_features_df(product="vol"))
        io.write_features(_features_df(product="gamma"))
        result = io.read_features(products=["vol"])
        assert len(result) == 1

    def test_date_range_filter(self):
        io.write_features(_features_df(date(2024, 1, 2)))
        io.write_features(_features_df(date(2024, 1, 10)))
        result = io.read_features(start_date=date(2024, 1, 5))
        assert len(result) == 1


# ===================================================================
# Proposals
# ===================================================================

class TestProposals:
    def test_round_trip(self):
        df = _proposals_df()
        io.write_proposals(df)
        result = io.read_proposals()
        assert result is not None
        assert len(result) == 1

    def test_pnl_matrix_round_trip(self):
        dt = date(2024, 1, 2)
        product = "vol"
        config_hash = "abc123def456789a"
        matrix = np.random.default_rng(0).normal(size=(100, 3))

        df = _proposals_df(dt, product, config_hash)
        io.write_proposals(df, pnl_matrices={(dt, product, config_hash): matrix})

        loaded = io.read_pnl_matrix(dt, product, config_hash)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, matrix)

    def test_orphan_sweep(self, tmp_data_root):
        dt = date(2024, 1, 2)
        product = "vol"
        good_hash = "abc123def456789a"
        orphan_hash = "orphan1234567890"

        # Write an orphan matrix first
        orphan_path = paths.pnl_matrix_path(dt, product, orphan_hash)
        orphan_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"col_0": [1.0], "col_1": [2.0], "col_2": [3.0]}).write_parquet(orphan_path)
        assert orphan_path.exists()

        # Write proposals — only good_hash, should sweep orphan
        df = _proposals_df(dt, product, good_hash)
        io.write_proposals(df, pnl_matrices={(dt, product, good_hash): np.ones((10, 3))})

        assert not orphan_path.exists()
        assert paths.pnl_matrix_path(dt, product, good_hash).exists()

    def test_tmp_cleanup(self, tmp_data_root):
        dt = date(2024, 1, 2)
        product = "vol"
        config_hash = "abc123def456789a"

        # Create a leftover .tmp file
        mat_dir = paths.pnl_matrix_dir(dt, product)
        mat_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = mat_dir / "leftover.parquet.tmp"
        tmp_file.write_text("junk")

        df = _proposals_df(dt, product, config_hash)
        io.write_proposals(df, pnl_matrices={(dt, product, config_hash): np.ones((10, 3))})

        assert not tmp_file.exists()

    def test_family_filter(self):
        df = _proposals_df()
        io.write_proposals(df)
        result = io.read_proposals(families=["max_mean"])
        assert len(result) == 0

    def test_invalid_family_raises(self):
        df = _proposals_df()
        df = df.with_columns(pl.lit("bogus_family").alias("family"))
        with pytest.raises(ValueError, match="Invalid family"):
            io.write_proposals(df)

    def test_invalid_proposal_type_raises(self):
        df = _proposals_df()
        df = df.with_columns(pl.lit("invalid_type").alias("proposal_type"))
        with pytest.raises(ValueError, match="Invalid proposal_type"):
            io.write_proposals(df)

    def test_invalid_solver_status_raises(self):
        df = _proposals_df()
        df = df.with_columns(pl.lit("crashed").alias("solver_status"))
        with pytest.raises(ValueError, match="Invalid solver_status"):
            io.write_proposals(df)

    def test_weights_column_names_length_mismatch_raises(self):
        df = _proposals_df()
        df = df.with_columns(pl.Series("weights", [[0.5, 0.5]]))  # 2 weights, 3 col_names
        with pytest.raises(ValueError, match="len\\(weights\\).*len\\(column_names\\)"):
            io.write_proposals(df)

    def test_pnl_matrix_not_2d_raises(self):
        dt = date(2024, 1, 2)
        product = "vol"
        config_hash = "abc123def456789a"
        df = _proposals_df(dt, product, config_hash)
        bad_matrix = np.ones((10,))  # 1D
        with pytest.raises(ValueError, match="expected 2D"):
            io.write_proposals(df, pnl_matrices={(dt, product, config_hash): bad_matrix})

    def test_pnl_matrix_cols_mismatch_raises(self):
        dt = date(2024, 1, 2)
        product = "vol"
        config_hash = "abc123def456789a"
        df = _proposals_df(dt, product, config_hash)  # 3 column_names
        bad_matrix = np.ones((10, 5))  # 5 cols != 3
        with pytest.raises(ValueError, match="matrix cols.*len\\(column_names\\)"):
            io.write_proposals(df, pnl_matrices={(dt, product, config_hash): bad_matrix})

    def test_candidate_nullable_fields(self):
        df = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "product": ["vol"],
            "config_hash": ["abc123def456789a"],
            "config_json": ['{"objective":"max_mean"}'],
            "config_schema_version": [1],
            "family": ["max_mean"],
            "proposal_type": ["candidate"],
            "thompson_sample_value": [None],
            "posterior_mean": [None],
            "posterior_std": [None],
            "weights": [[0.5, 0.5]],
            "column_names": [["A", "B"]],
            "n_names": [2],
            "in_sample_sharpe": [1.0],
            "in_sample_mean_pnl": [0.005],
            "in_sample_max_dd": [-0.02],
            "solver_status": ["optimal"],
            "solve_time_seconds": [0.05],
            "pnl_matrix_path": [None],
        })
        io.write_proposals(df)
        result = io.read_proposals()
        assert result["thompson_sample_value"].null_count() == 1
        assert result["pnl_matrix_path"].null_count() == 1


# ===================================================================
# Outcomes
# ===================================================================

class TestOutcomes:
    def test_round_trip(self):
        df = _outcomes_df()
        io.write_outcomes(df)
        result = io.read_outcomes()
        assert result is not None
        assert len(result) == 1

    def test_empty_raises(self):
        df = _outcomes_df().head(0)
        with pytest.raises(ValueError, match="empty"):
            io.write_outcomes(df)

    def test_merge_different_dates(self):
        io.write_outcomes(_outcomes_df(date(2024, 1, 2)))
        io.write_outcomes(_outcomes_df(date(2024, 1, 3)))
        result = io.read_outcomes()
        assert len(result) == 2


# ===================================================================
# Decisions
# ===================================================================

class TestDecisions:
    def _decision_df(self, decision="accepted", notes=None, modified_hash=None):
        return pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "product": ["vol"],
            "config_hash": ["abc123def456789a"],
            "decision": [decision],
            "decided_at_utc": [datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)],
            "notes": [notes],
            "modified_config_hash": [modified_hash],
        })

    def test_round_trip(self):
        io.append_decisions(self._decision_df())
        result = io.read_decisions_raw()
        assert result is not None
        assert len(result) == 1

    def test_append_only(self):
        io.append_decisions(self._decision_df("accepted"))
        df2 = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "product": ["vol"],
            "config_hash": ["abc123def456789a"],
            "decision": ["rejected"],
            "decided_at_utc": [datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc)],
            "notes": ["changed my mind"],
            "modified_config_hash": [None],
        })
        io.append_decisions(df2)
        raw = io.read_decisions_raw()
        assert len(raw) == 2

    def test_latest_collapses(self):
        io.append_decisions(self._decision_df("accepted"))
        df2 = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "product": ["vol"],
            "config_hash": ["abc123def456789a"],
            "decision": ["rejected"],
            "decided_at_utc": [datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc)],
            "notes": [None],
            "modified_config_hash": [None],
        })
        io.append_decisions(df2)
        latest = io.read_decisions_latest()
        assert len(latest) == 1
        assert latest["decision"][0] == "rejected"

    def test_invalid_decision_raises(self):
        df = self._decision_df("invalid_decision")
        with pytest.raises(ValueError, match="Invalid decision"):
            io.append_decisions(df)

    def test_modified_requires_hash(self):
        df = self._decision_df("modified", modified_hash=None)
        with pytest.raises(ValueError, match="modified_config_hash"):
            io.append_decisions(df)

    def test_modified_with_hash_ok(self):
        df = self._decision_df("modified", modified_hash="newcfg1234567890")
        io.append_decisions(df)
        result = io.read_decisions_raw()
        assert len(result) == 1

    def test_non_modified_with_hash_raises(self):
        df = self._decision_df("accepted", modified_hash="shouldnt_be_here")
        with pytest.raises(ValueError, match="modified_config_hash must be null"):
            io.append_decisions(df)

    def test_naive_datetime_raises(self):
        df = pl.DataFrame({
            "date": [date(2024, 1, 2)],
            "product": ["vol"],
            "config_hash": ["abc123def456789a"],
            "decision": ["accepted"],
            "decided_at_utc": [datetime(2024, 1, 2, 10, 0)],  # naive!
            "notes": [None],
            "modified_config_hash": [None],
        })
        with pytest.raises(ValueError, match="timezone-aware"):
            io.append_decisions(df)
