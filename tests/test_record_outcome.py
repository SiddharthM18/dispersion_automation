"""Tests for record_outcome.py — T+5 outcome computation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from dispersion_meta import io, schemas
from dispersion_meta.paths import set_data_root
from dispersion_meta.record_outcome import record_outcomes
from dispersion_meta.meta_score import MetaScoreConfig


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path):
    set_data_root(tmp_path)
    yield
    set_data_root(None)


def _write_test_proposals(
    propose_date: date,
    products: list[str],
    n_names: int = 10,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Write minimal proposals for testing outcome recording."""
    if rng is None:
        rng = np.random.default_rng(42)

    rows = []
    for product in products:
        for i, (fam, ptype) in enumerate([
            ("max_mean", "best"),
            ("max_sharpe", "alt"),
            ("min_drawdown", "alt"),
            ("composite", "explore"),
        ]):
            w = rng.dirichlet(np.ones(n_names)).tolist()
            rows.append({
                "date": propose_date,
                "product": product,
                "config_hash": f"hash_{product}_{i}",
                "config_json": f'{{"objective": "{fam}"}}',
                "config_schema_version": 1,
                "family": fam,
                "proposal_type": ptype,
                "thompson_sample_value": rng.normal(),
                "posterior_mean": rng.normal(),
                "posterior_std": abs(rng.normal()),
                "weights": w,
                "column_names": [f"S{j:02d}" for j in range(n_names)],
                "n_names": n_names,
                "in_sample_sharpe": 1.5,
                "in_sample_mean_pnl": 0.01,
                "in_sample_max_dd": -0.03,
                "solver_status": "optimal",
                "solve_time_seconds": 0.1,
                "pnl_matrix_path": None,
            })

    df = pl.DataFrame(rows)
    io.write_proposals(df)
    return df


class TestRecordOutcomes:
    def test_basic_round_trip(self):
        products = ["corridor_var", "vol"]
        propose_date = date(2026, 4, 9)
        eval_date = date(2026, 4, 16)
        n_names = 10

        _write_test_proposals(propose_date, products, n_names=n_names)

        rng = np.random.default_rng(99)
        forward_returns = {p: rng.normal(0.0, 0.01, (5, n_names)) for p in products}
        trailing_vol = {p: 0.015 for p in products}
        column_names = [f"S{j:02d}" for j in range(n_names)]

        outcomes = record_outcomes(
            propose_date=propose_date,
            eval_date=eval_date,
            forward_returns=forward_returns,
            trailing_vol=trailing_vol,
            column_names=column_names,
        )

        assert len(outcomes) == 4 * len(products)

        # Verify written to disk
        stored = io.read_outcomes()
        assert stored is not None
        assert len(stored) == len(outcomes)

    def test_outcome_fields(self):
        products = ["corridor_var"]
        propose_date = date(2026, 4, 9)
        eval_date = date(2026, 4, 16)
        n_names = 10

        _write_test_proposals(propose_date, products, n_names=n_names)

        rng = np.random.default_rng(99)
        forward_returns = {"corridor_var": rng.normal(0.0, 0.01, (5, n_names))}
        trailing_vol = {"corridor_var": 0.015}

        outcomes = record_outcomes(
            propose_date=propose_date,
            eval_date=eval_date,
            forward_returns=forward_returns,
            trailing_vol=trailing_vol,
            column_names=[f"S{j:02d}" for j in range(n_names)],
        )

        required_cols = [
            "date", "product", "config_hash", "eval_date",
            "forward_window_days", "forward_5d_pnl", "forward_5d_mean_return",
            "forward_realized_vol_21d", "forward_sharpe", "meta_score",
            "meta_score_mode", "lambda_used",
        ]
        for col in required_cols:
            assert col in outcomes.columns, f"Missing column: {col}"

    def test_forward_pnl_math(self):
        """Verify forward_5d_pnl = sum(daily_returns @ weights)."""
        products = ["corridor_var"]
        propose_date = date(2026, 4, 9)
        n_names = 5

        _write_test_proposals(propose_date, products, n_names=n_names)

        # Deterministic returns
        fwd = np.ones((5, n_names)) * 0.01
        forward_returns = {"corridor_var": fwd}
        trailing_vol = {"corridor_var": 0.015}

        outcomes = record_outcomes(
            propose_date=propose_date,
            eval_date=date(2026, 4, 16),
            forward_returns=forward_returns,
            trailing_vol=trailing_vol,
            column_names=[f"S{j:02d}" for j in range(n_names)],
        )

        for row in outcomes.iter_rows(named=True):
            # With uniform returns of 0.01 and weights summing to 1,
            # daily portfolio return = 0.01, 5-day PnL = 0.05
            assert abs(row["forward_5d_pnl"] - 0.05) < 1e-10
            assert abs(row["forward_5d_mean_return"] - 0.01) < 1e-10

    def test_meta_score_uses_trailing_vol(self):
        """Meta score should use the provided trailing vol."""
        products = ["corridor_var"]
        propose_date = date(2026, 4, 9)
        n_names = 5

        _write_test_proposals(propose_date, products, n_names=n_names)

        fwd = np.ones((5, n_names)) * 0.01
        forward_returns = {"corridor_var": fwd}

        # Research mode: meta_score = forward_sharpe = mean_return / vol
        outcomes = record_outcomes(
            propose_date=propose_date,
            eval_date=date(2026, 4, 16),
            forward_returns=forward_returns,
            trailing_vol={"corridor_var": 0.02},
            column_names=[f"S{j:02d}" for j in range(n_names)],
        )

        for row in outcomes.iter_rows(named=True):
            expected_sharpe = 0.01 / 0.02  # mean_return / vol
            assert abs(row["forward_sharpe"] - expected_sharpe) < 1e-10
            assert row["meta_score_mode"] == "research"

    def test_no_proposals_raises(self):
        with pytest.raises(ValueError, match="No proposals found"):
            record_outcomes(
                propose_date=date(2026, 1, 1),
                eval_date=date(2026, 1, 8),
                forward_returns={"corridor_var": np.zeros((5, 10))},
                trailing_vol={"corridor_var": 0.01},
                column_names=[f"S{j:02d}" for j in range(10)],
            )

    def test_eval_date_stored(self):
        products = ["corridor_var"]
        propose_date = date(2026, 4, 9)
        eval_date = date(2026, 4, 16)
        n_names = 5

        _write_test_proposals(propose_date, products, n_names=n_names)

        outcomes = record_outcomes(
            propose_date=propose_date,
            eval_date=eval_date,
            forward_returns={"corridor_var": np.ones((5, n_names)) * 0.01},
            trailing_vol={"corridor_var": 0.015},
            column_names=[f"S{j:02d}" for j in range(n_names)],
        )

        for row in outcomes.iter_rows(named=True):
            assert row["eval_date"] == eval_date
            assert row["forward_window_days"] == 5
