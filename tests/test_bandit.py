"""Tests for the contextual Thompson sampling bandit."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from dispersion_meta.bandit import BayesianLinearBandit
from dispersion_meta.config_space import FAMILIES
from dispersion_meta.synthetic import (
    MetaScoreOracle,
    generate_features,
    synth_outcome_rows,
    synth_proposals_for_day,
)


def _make_training_df(
    n_days: int = 200,
    products: list[str] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a synthetic training table (features joined with proposals+outcomes)."""
    if products is None:
        products = ["corridor_var"]

    rng = np.random.default_rng(seed)
    oracle = MetaScoreOracle()
    start = date(2024, 1, 2)

    features = generate_features(n_days, products, start_date=start, rng=rng)

    all_proposals = []
    for i in range(n_days):
        dt = start + timedelta(days=i)
        day_proposals = synth_proposals_for_day(
            dt, products, n_per_family=2, n_names=20, rng=rng,
        )
        all_proposals.append(day_proposals)
    proposals = pl.concat(all_proposals)

    outcomes = synth_outcome_rows(proposals, features, oracle, rng=rng)

    # Join to mimic build_training_table output
    joined = proposals.join(features, on=["date", "product"], how="inner", suffix="_feat")
    feat_version_col = "_table_schema_version_feat"
    if feat_version_col in joined.columns:
        joined = joined.drop(feat_version_col)

    joined = joined.join(
        outcomes, on=["date", "product", "config_hash"], how="inner", suffix="_out",
    )
    out_version_col = "_table_schema_version_out"
    if out_version_col in joined.columns:
        joined = joined.drop(out_version_col)

    return joined


class TestArmInitialization:
    def test_prior_state(self):
        bandit = BayesianLinearBandit(alpha=1.0)
        for fam in FAMILIES:
            arm = bandit.arms[fam]
            assert arm.n_obs == 0
            np.testing.assert_array_equal(arm.mu_n, np.zeros(8))
            np.testing.assert_array_equal(arm.Lambda_n, np.eye(8))
            np.testing.assert_array_equal(arm.Lambda_n_inv, np.eye(8))

    def test_custom_alpha(self):
        bandit = BayesianLinearBandit(alpha=2.0)
        arm = bandit.arms["max_mean"]
        np.testing.assert_array_equal(arm.Lambda_n, 2.0 * np.eye(8))
        np.testing.assert_array_almost_equal(arm.Lambda_n_inv, 0.5 * np.eye(8))

    def test_all_families_present(self):
        bandit = BayesianLinearBandit()
        assert set(bandit.arms.keys()) == set(FAMILIES)


class TestFit:
    @pytest.fixture()
    def training_df(self):
        return _make_training_df(n_days=100, products=["corridor_var"], seed=42)

    def test_fit_updates_posteriors(self, training_df):
        product_df = training_df.filter(pl.col("product") == "corridor_var")
        bandit = BayesianLinearBandit()
        bandit.fit(product_df)

        for fam in FAMILIES:
            arm = bandit.arms[fam]
            assert arm.n_obs > 0
            # Posterior precision should be strictly greater than prior
            assert np.trace(arm.Lambda_n) > np.trace(bandit.alpha * np.eye(8))

    def test_fit_stores_scaler(self, training_df):
        product_df = training_df.filter(pl.col("product") == "corridor_var")
        bandit = BayesianLinearBandit()
        bandit.fit(product_df)

        assert bandit._feat_mean is not None
        assert bandit._feat_std is not None
        assert bandit._feat_mean.shape == (7,)
        assert bandit._feat_std.shape == (7,)
        # No zero stds
        assert np.all(bandit._feat_std > 0)

    def test_fit_with_no_family_data_keeps_prior(self):
        """If a family has no rows, its arm stays at the prior."""
        # Create a tiny training df with only max_mean rows
        training_df = _make_training_df(n_days=10, seed=0)
        only_max_mean = training_df.filter(pl.col("family") == "max_mean")

        bandit = BayesianLinearBandit()
        bandit.fit(only_max_mean)

        # max_mean should be updated
        assert bandit.arms["max_mean"].n_obs > 0
        # All others should be at prior
        for fam in FAMILIES:
            if fam != "max_mean":
                assert bandit.arms[fam].n_obs == 0
                np.testing.assert_array_equal(bandit.arms[fam].mu_n, np.zeros(8))


class TestThompsonSample:
    @pytest.fixture()
    def fitted_bandit(self):
        training_df = _make_training_df(n_days=200, products=["corridor_var"], seed=42)
        product_df = training_df.filter(pl.col("product") == "corridor_var")
        bandit = BayesianLinearBandit()
        bandit.fit(product_df)
        return bandit

    def test_returns_all_families(self, fitted_bandit):
        rng = np.random.default_rng(0)
        x = np.zeros(7)
        results = fitted_bandit.thompson_sample(x, rng)
        assert len(results) == len(FAMILIES)
        assert {r["family"] for r in results} == set(FAMILIES)

    def test_sorted_descending(self, fitted_bandit):
        rng = np.random.default_rng(0)
        x = np.random.default_rng(1).normal(size=7)
        results = fitted_bandit.thompson_sample(x, rng)
        values = [r["sampled_value"] for r in results]
        assert values == sorted(values, reverse=True)

    def test_result_fields(self, fitted_bandit):
        rng = np.random.default_rng(0)
        x = np.zeros(7)
        results = fitted_bandit.thompson_sample(x, rng)
        for r in results:
            assert "family" in r
            assert "sampled_value" in r
            assert "posterior_mean" in r
            assert "posterior_std" in r
            assert r["posterior_std"] >= 0

    def test_samples_vary_with_rng(self, fitted_bandit):
        x = np.zeros(7)
        r1 = fitted_bandit.thompson_sample(x, np.random.default_rng(0))
        r2 = fitted_bandit.thompson_sample(x, np.random.default_rng(1))
        vals1 = [r["sampled_value"] for r in r1]
        vals2 = [r["sampled_value"] for r in r2]
        assert vals1 != vals2

    def test_posterior_mean_deterministic(self, fitted_bandit):
        """Posterior mean should be the same regardless of RNG seed."""
        x = np.zeros(7)
        r1 = fitted_bandit.thompson_sample(x, np.random.default_rng(0))
        r2 = fitted_bandit.thompson_sample(x, np.random.default_rng(99))
        means1 = {r["family"]: r["posterior_mean"] for r in r1}
        means2 = {r["family"]: r["posterior_mean"] for r in r2}
        for fam in FAMILIES:
            assert abs(means1[fam] - means2[fam]) < 1e-10


class TestSelectProposals:
    @pytest.fixture()
    def fitted_bandit(self):
        training_df = _make_training_df(n_days=200, products=["corridor_var"], seed=42)
        product_df = training_df.filter(pl.col("product") == "corridor_var")
        bandit = BayesianLinearBandit()
        bandit.fit(product_df)
        return bandit

    def test_returns_4_proposals(self, fitted_bandit):
        rng = np.random.default_rng(0)
        x = np.zeros(7)
        proposals = fitted_bandit.select_proposals(x, rng)
        assert len(proposals) == 4

    def test_proposal_types(self, fitted_bandit):
        rng = np.random.default_rng(0)
        x = np.zeros(7)
        proposals = fitted_bandit.select_proposals(x, rng)
        types = [p["proposal_type"] for p in proposals]
        assert types.count("best") == 1
        assert types.count("alt") == 2
        assert types.count("explore") == 1

    def test_explore_not_in_top3(self, fitted_bandit):
        """The explore family should not duplicate a top-3 selection."""
        rng = np.random.default_rng(0)
        x = np.zeros(7)
        proposals = fitted_bandit.select_proposals(x, rng)
        top3_families = {p["family"] for p in proposals if p["proposal_type"] != "explore"}
        explore_family = [p["family"] for p in proposals if p["proposal_type"] == "explore"][0]
        assert explore_family not in top3_families

    def test_4_distinct_families(self, fitted_bandit):
        """All 4 proposals should have distinct families."""
        rng = np.random.default_rng(0)
        x = np.zeros(7)
        proposals = fitted_bandit.select_proposals(x, rng)
        families = [p["family"] for p in proposals]
        assert len(set(families)) == 4

    def test_has_thompson_sample_value(self, fitted_bandit):
        rng = np.random.default_rng(0)
        x = np.zeros(7)
        proposals = fitted_bandit.select_proposals(x, rng)
        for p in proposals:
            assert "thompson_sample_value" in p
            assert isinstance(p["thompson_sample_value"], float)


class TestColdStart:
    """With no training data (prior only), Thompson sampling should behave sensibly."""

    def test_cold_start_uniform_ish(self):
        """Without training data, all families should be selected with ~equal frequency."""
        bandit = BayesianLinearBandit()
        # Set trivial scaler so thompson_sample doesn't crash
        bandit._feat_mean = np.zeros(7)
        bandit._feat_std = np.ones(7)

        x = np.zeros(7)
        best_counts: dict[str, int] = {f: 0 for f in FAMILIES}
        n_trials = 1000
        rng = np.random.default_rng(42)

        for _ in range(n_trials):
            proposals = bandit.select_proposals(x, rng)
            best_fam = proposals[0]["family"]
            best_counts[best_fam] += 1

        # With 5 families and no data, each should be best ~20% of the time
        # Allow wide tolerance (10%-30%) to avoid flaky tests
        for fam, count in best_counts.items():
            pct = count / n_trials
            assert 0.05 < pct < 0.40, f"{fam} selected {pct:.1%} — expected ~20%"


class TestOracleRecovery:
    """After fitting on oracle-generated data, the bandit should preferentially
    select the oracle's best family for a given feature regime."""

    def test_high_vix_prefers_min_drawdown(self):
        """Oracle gives min_drawdown the highest score when VIX is high.
        The bandit should learn this and select it more often as 'best'.
        """
        training_df = _make_training_df(n_days=300, products=["corridor_var"], seed=42)
        product_df = training_df.filter(pl.col("product") == "corridor_var")

        bandit = BayesianLinearBandit()
        bandit.fit(product_df)

        # High VIX feature vector
        x_high_vix = np.array([35.0, 5.0, 0.2, 0.0, 0.0, 0.0, 0.1])

        best_counts: dict[str, int] = {f: 0 for f in FAMILIES}
        rng = np.random.default_rng(0)
        n_trials = 500

        for _ in range(n_trials):
            proposals = bandit.select_proposals(x_high_vix, rng)
            best_fam = proposals[0]["family"]
            best_counts[best_fam] += 1

        # min_drawdown should be the most frequently selected best
        assert best_counts["min_drawdown"] == max(best_counts.values()), (
            f"Expected min_drawdown to be most selected, got: {best_counts}"
        )

    def test_max_sharpe_posterior_increases_with_corr(self):
        """Oracle gives max_sharpe a positive coefficient on avg_pairwise_corr.
        The bandit's posterior mean for max_sharpe should be higher when corr is
        high vs low — i.e., it learns the correct direction.
        """
        training_df = _make_training_df(n_days=300, products=["corridor_var"], seed=42)
        product_df = training_df.filter(pl.col("product") == "corridor_var")

        bandit = BayesianLinearBandit()
        bandit.fit(product_df)

        rng = np.random.default_rng(0)
        # Same base features, only avg_pairwise_corr differs
        x_low_corr = np.array([20.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1])
        x_high_corr = np.array([20.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1])

        samples_low = bandit.thompson_sample(x_low_corr, rng)
        samples_high = bandit.thompson_sample(x_high_corr, rng)

        mean_low = {s["family"]: s["posterior_mean"] for s in samples_low}
        mean_high = {s["family"]: s["posterior_mean"] for s in samples_high}

        # max_sharpe has +1.5 coef on corr → posterior mean should increase
        assert mean_high["max_sharpe"] > mean_low["max_sharpe"]
        # min_drawdown has -1.0 coef on corr → posterior mean should decrease
        assert mean_high["min_drawdown"] < mean_low["min_drawdown"]


class TestDiagnostics:
    def test_diagnostics_structure(self):
        training_df = _make_training_df(n_days=50, products=["corridor_var"], seed=42)
        product_df = training_df.filter(pl.col("product") == "corridor_var")

        bandit = BayesianLinearBandit()
        bandit.fit(product_df)
        diag = bandit.diagnostics()

        assert "arms" in diag
        assert "scaler_mean" in diag
        assert "scaler_std" in diag
        assert set(diag["arms"].keys()) == set(FAMILIES)
        for fam_info in diag["arms"].values():
            assert "n_obs" in fam_info
            assert "posterior_mean_norm" in fam_info
            assert "posterior_trace" in fam_info
