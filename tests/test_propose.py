"""Tests for propose.py — daily proposal orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from dispersion_meta import io, schemas
from dispersion_meta.paths import set_data_root
from dispersion_meta.propose import propose_today


# ---------------------------------------------------------------------------
# Fake optimizer result for fast, deterministic tests
# ---------------------------------------------------------------------------

@dataclass
class FakeResult:
    weights: np.ndarray
    n_names: int
    selected_names: list[int]
    sharpe: float = 1.5
    mean_pnl: float = 0.01
    max_drawdown: float = -0.03
    status: str = "optimal"
    solve_time: float = 0.1


def _fake_run_optimizer(config, pnl_matrix, column_names):
    n = pnl_matrix.shape[1]
    w = np.ones(n) / n
    return FakeResult(
        weights=w,
        n_names=n,
        selected_names=list(range(n)),
    )


def _fake_run_optimizer_infeasible(config, pnl_matrix, column_names):
    return FakeResult(
        weights=np.array([np.nan]),
        n_names=0,
        selected_names=[],
        status="infeasible",
    )


_infeasible_call_count = 0


def _fake_run_optimizer_flaky(config, pnl_matrix, column_names):
    """First 2 calls return infeasible, third succeeds."""
    global _infeasible_call_count
    _infeasible_call_count += 1
    if _infeasible_call_count % 3 != 0:
        return _fake_run_optimizer_infeasible(config, pnl_matrix, column_names)
    return _fake_run_optimizer(config, pnl_matrix, column_names)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path):
    set_data_root(tmp_path)
    yield
    set_data_root(None)


@pytest.fixture()
def features():
    return {
        "vix_level": 18.5,
        "vix_20d_change": -0.02,
        "avg_pairwise_corr": 0.42,
        "dispersion_pnl_20d": 0.013,
        "skew_steepness": 0.08,
        "term_structure_slope": 0.015,
        "earnings_density_21d": 0.07,
    }


@pytest.fixture()
def pnl_matrices():
    rng = np.random.default_rng(42)
    products = sorted(schemas.VALID_PRODUCTS)
    return {p: rng.normal(0.0, 0.01, size=(60, 30)) for p in products}


@pytest.fixture()
def column_names():
    return [f"S{i:02d}" for i in range(30)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProposeToday:
    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_basic_proposal_count(self, mock_opt, features, pnl_matrices, column_names):
        """Should produce 4 proposals per product (best + 2 alt + 1 explore)."""
        result = propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        n_products = len(pnl_matrices)
        assert result["n_proposals_total"] == 4 * n_products
        assert result["n_products"] == n_products

    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_proposals_written_to_disk(self, mock_opt, features, pnl_matrices, column_names):
        propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        proposals = io.read_proposals()
        assert proposals is not None
        assert len(proposals) == 4 * len(pnl_matrices)

    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_features_written_to_disk(self, mock_opt, features, pnl_matrices, column_names):
        propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        feats = io.read_features()
        assert feats is not None
        assert len(feats) == len(pnl_matrices)

    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_proposal_types_per_product(self, mock_opt, features, pnl_matrices, column_names):
        propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        proposals = io.read_proposals()
        for product in pnl_matrices:
            prod_props = proposals.filter(pl.col("product") == product)
            types = prod_props["proposal_type"].to_list()
            assert types.count("best") == 1
            assert types.count("alt") == 2
            assert types.count("explore") == 1

    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_all_products_present(self, mock_opt, features, pnl_matrices, column_names):
        propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        proposals = io.read_proposals()
        products_in_proposals = set(proposals["product"].unique().to_list())
        assert products_in_proposals == set(pnl_matrices.keys())

    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_summary_structure(self, mock_opt, features, pnl_matrices, column_names):
        result = propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        assert "today" in result
        assert "n_products" in result
        assert "n_proposals_total" in result
        assert "products" in result
        for product, info in result["products"].items():
            assert "n_proposals" in info
            assert "families" in info
            assert "types" in info
            assert "diagnostics" in info

    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_solver_status_stored(self, mock_opt, features, pnl_matrices, column_names):
        propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        proposals = io.read_proposals()
        statuses = set(proposals["solver_status"].unique().to_list())
        assert statuses == {"optimal"}

    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_weights_length_matches_columns(self, mock_opt, features, pnl_matrices, column_names):
        propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        proposals = io.read_proposals()
        for row in proposals.iter_rows(named=True):
            w = row["weights"]
            c = row["column_names"]
            w_list = w.to_list() if isinstance(w, pl.Series) else w
            c_list = c.to_list() if isinstance(c, pl.Series) else c
            assert len(w_list) == len(c_list) == 30


class TestInfeasibilityRetry:
    @patch(
        "dispersion_meta.propose._run_optimizer",
        side_effect=_fake_run_optimizer_infeasible,
    )
    def test_all_infeasible_skips_slot(self, mock_opt, features, pnl_matrices, column_names):
        """If all retry attempts fail, that proposal slot is skipped."""
        result = propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        assert result["n_proposals_total"] == 0

    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer_flaky)
    def test_retry_recovers(self, mock_opt, features, pnl_matrices, column_names):
        """Flaky solver (fails 2/3 times) should still produce proposals via retry."""
        global _infeasible_call_count
        _infeasible_call_count = 0

        result = propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        # Some proposals should succeed via retries
        assert result["n_proposals_total"] > 0

    @patch("dispersion_meta.propose._run_optimizer", side_effect=Exception("boom"))
    def test_exception_does_not_crash(self, mock_opt, features, pnl_matrices, column_names):
        """Optimizer exceptions should be caught and logged, not crash the cycle."""
        result = propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        assert result["n_proposals_total"] == 0


class TestInputValidation:
    def test_missing_feature_raises(self, pnl_matrices, column_names):
        with pytest.raises(ValueError, match="Missing features"):
            propose_today(
                today=date(2026, 4, 16),
                features={"vix_level": 18.5},  # missing 6 features
                pnl_matrices=pnl_matrices,
                column_names=column_names,
            )

    def test_invalid_product_raises(self, features, column_names):
        with pytest.raises(ValueError, match="Invalid products"):
            propose_today(
                today=date(2026, 4, 16),
                features=features,
                pnl_matrices={"bogus_product": np.zeros((10, 5))},
                column_names=[f"S{i}" for i in range(5)],
            )

    def test_column_count_mismatch_raises(self, features, pnl_matrices, column_names):
        with pytest.raises(ValueError, match="columns"):
            propose_today(
                today=date(2026, 4, 16),
                features=features,
                pnl_matrices=pnl_matrices,
                column_names=["A", "B"],  # 2, not 30
            )


class TestColdStart:
    @patch("dispersion_meta.propose._run_optimizer", side_effect=_fake_run_optimizer)
    def test_day1_no_training_data(self, mock_opt, features, pnl_matrices, column_names):
        """Day 1 with no historical data should still produce proposals (from prior)."""
        result = propose_today(
            today=date(2026, 4, 16),
            features=features,
            pnl_matrices=pnl_matrices,
            column_names=column_names,
            seed=42,
        )
        assert result["n_proposals_total"] == 4 * len(pnl_matrices)
