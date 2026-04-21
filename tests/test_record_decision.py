"""Tests for record_decision.py."""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from dispersion_meta import io
from dispersion_meta.paths import set_data_root
from dispersion_meta.record_decision import record_decision


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path):
    set_data_root(tmp_path)
    yield
    set_data_root(None)


def _write_test_proposal(propose_date: date, product: str, config_hash: str) -> None:
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(10)).tolist()
    df = pl.DataFrame([{
        "date": propose_date,
        "product": product,
        "config_hash": config_hash,
        "config_json": '{"objective": "max_mean"}',
        "config_schema_version": 1,
        "family": "max_mean",
        "proposal_type": "best",
        "thompson_sample_value": 1.0,
        "posterior_mean": 0.5,
        "posterior_std": 0.3,
        "weights": w,
        "column_names": [f"S{j:02d}" for j in range(10)],
        "n_names": 10,
        "in_sample_sharpe": 1.5,
        "in_sample_mean_pnl": 0.01,
        "in_sample_max_dd": -0.03,
        "solver_status": "optimal",
        "solve_time_seconds": 0.1,
        "pnl_matrix_path": None,
    }])
    io.write_proposals(df)


class TestRecordDecision:
    def test_accept(self):
        propose_date = date(2026, 4, 9)
        _write_test_proposal(propose_date, "corridor_var", "abc123")

        record_decision(
            propose_date=propose_date,
            config_hash="abc123",
            product="corridor_var",
            decision="accepted",
        )

        decisions = io.read_decisions_raw()
        assert decisions is not None
        assert len(decisions) == 1
        assert decisions["decision"][0] == "accepted"

    def test_reject_with_notes(self):
        propose_date = date(2026, 4, 9)
        _write_test_proposal(propose_date, "corridor_var", "abc123")

        record_decision(
            propose_date=propose_date,
            config_hash="abc123",
            product="corridor_var",
            decision="rejected",
            notes="Too aggressive for current regime",
        )

        decisions = io.read_decisions_raw()
        assert decisions["notes"][0] == "Too aggressive for current regime"

    def test_modified_with_hash(self):
        propose_date = date(2026, 4, 9)
        _write_test_proposal(propose_date, "corridor_var", "abc123")

        record_decision(
            propose_date=propose_date,
            config_hash="abc123",
            product="corridor_var",
            decision="modified",
            modified_config_hash="def456",
        )

        decisions = io.read_decisions_raw()
        assert decisions["decision"][0] == "modified"
        assert decisions["modified_config_hash"][0] == "def456"

    def test_modified_without_hash_raises(self):
        propose_date = date(2026, 4, 9)
        _write_test_proposal(propose_date, "corridor_var", "abc123")

        with pytest.raises(ValueError):
            record_decision(
                propose_date=propose_date,
                config_hash="abc123",
                product="corridor_var",
                decision="modified",
                # missing modified_config_hash
            )

    def test_invalid_decision_raises(self):
        propose_date = date(2026, 4, 9)
        _write_test_proposal(propose_date, "corridor_var", "abc123")

        with pytest.raises(ValueError):
            record_decision(
                propose_date=propose_date,
                config_hash="abc123",
                product="corridor_var",
                decision="maybe",
            )

    def test_no_proposal_raises(self):
        with pytest.raises(ValueError, match="No proposals found"):
            record_decision(
                propose_date=date(2026, 1, 1),
                config_hash="abc123",
                product="corridor_var",
                decision="accepted",
            )

    def test_wrong_config_hash_raises(self):
        propose_date = date(2026, 4, 9)
        _write_test_proposal(propose_date, "corridor_var", "abc123")

        with pytest.raises(ValueError, match="No proposal with config_hash"):
            record_decision(
                propose_date=propose_date,
                config_hash="wrong_hash",
                product="corridor_var",
                decision="accepted",
            )

    def test_multiple_decisions_append(self):
        propose_date = date(2026, 4, 9)
        _write_test_proposal(propose_date, "corridor_var", "abc123")

        record_decision(
            propose_date=propose_date,
            config_hash="abc123",
            product="corridor_var",
            decision="rejected",
        )
        record_decision(
            propose_date=propose_date,
            config_hash="abc123",
            product="corridor_var",
            decision="accepted",
            notes="Changed mind",
        )

        raw = io.read_decisions_raw()
        assert len(raw) == 2

        latest = io.read_decisions_latest()
        assert len(latest) == 1
        assert latest["decision"][0] == "accepted"
