from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from dispersion_meta import paths
from dispersion_meta.meta_score import MetaScoreConfig
from dispersion_meta.schemas import VALID_FAMILIES, VALID_PRODUCTS
from dispersion_meta.synthetic import (
    FEATURE_NAMES,
    MetaScoreOracle,
    generate_features,
    populate_synthetic_history,
    synth_proposals_for_day,
)


@pytest.fixture(autouse=True)
def tmp_data_root(tmp_path):
    paths.set_data_root(tmp_path)
    yield tmp_path
    paths.set_data_root(None)


class TestMetaScoreOracle:
    def test_expected_score_deterministic(self):
        oracle = MetaScoreOracle()
        features = np.array([20.0, 1.0, 0.4, 0.01, 0.1, 0.05, 0.2])
        s1 = oracle.expected_score(features, "max_sharpe", "vol")
        s2 = oracle.expected_score(features, "max_sharpe", "vol")
        assert s1 == s2

    def test_different_families_different_scores(self):
        oracle = MetaScoreOracle()
        features = np.array([20.0, 1.0, 0.4, 0.01, 0.1, 0.05, 0.2])
        scores = {f: oracle.expected_score(features, f, "vol") for f in VALID_FAMILIES}
        # Not all the same
        assert len(set(scores.values())) > 1

    def test_sample_score_adds_noise(self):
        oracle = MetaScoreOracle()
        features = np.array([20.0, 1.0, 0.4, 0.01, 0.1, 0.05, 0.2])
        rng = np.random.default_rng(42)
        samples = [oracle.sample_score(features, "max_sharpe", "vol", rng) for _ in range(100)]
        # Should have variance
        assert np.std(samples) > 0.1


class TestGenerateFeatures:
    def test_shape(self):
        products = ["vol", "gamma"]
        df = generate_features(10, products)
        assert len(df) == 10 * 2  # 10 days × 2 products

    def test_columns(self):
        df = generate_features(5, ["vol"])
        for col in FEATURE_NAMES:
            assert col in df.columns


class TestSynthProposals:
    def test_shape(self):
        products = ["vol", "gamma"]
        df = synth_proposals_for_day(date(2024, 1, 2), products, n_per_family=10)
        # 5 families × 10 per family × 2 products = 100
        assert len(df) == 5 * 10 * 2

    def test_proposal_types(self):
        df = synth_proposals_for_day(date(2024, 1, 2), ["vol"], n_per_family=5)
        types = df["proposal_type"].unique().to_list()
        assert "best" in types
        assert "alt" in types
        assert "candidate" in types


class TestPopulateSyntheticHistory:
    def test_60_days_row_count(self):
        data = populate_synthetic_history(
            n_days=60, n_per_family=10, n_names=20, seed=42
        )
        proposals = data["proposals"]
        # 60 days × 5 products × 5 families × 10 per family = 15000
        assert len(proposals) == 60 * 5 * 5 * 10

    def test_all_families_present(self):
        data = populate_synthetic_history(n_days=10, n_per_family=5, seed=42)
        families = set(data["proposals"]["family"].unique().to_list())
        assert families == VALID_FAMILIES

    def test_all_products_present(self):
        data = populate_synthetic_history(n_days=10, n_per_family=5, seed=42)
        products = set(data["proposals"]["product"].unique().to_list())
        assert products == VALID_PRODUCTS

    def test_outcomes_match_proposals(self):
        data = populate_synthetic_history(n_days=10, n_per_family=5, seed=42)
        assert len(data["outcomes"]) == len(data["proposals"])

    def test_continuous_mode_different_scores(self):
        """Switching to continuous mode produces different meta_scores but same forward_sharpe."""
        data_research = populate_synthetic_history(n_days=10, n_per_family=5, seed=42)
        paths.set_data_root(paths.data_root().parent / "continuous_test")
        data_continuous = populate_synthetic_history(
            n_days=10, n_per_family=5, seed=42,
            meta_score_config=MetaScoreConfig(mode="continuous", lambda_turnover=0.3),
        )

        # forward_sharpe should be identical (same seed, same oracle)
        r_sharpe = data_research["outcomes"]["forward_sharpe"].to_list()
        c_sharpe = data_continuous["outcomes"]["forward_sharpe"].to_list()
        for r, c in zip(r_sharpe, c_sharpe):
            assert abs(r - c) < 1e-10

        # meta_scores should differ (continuous has turnover penalty)
        r_scores = data_research["outcomes"]["meta_score"].to_list()
        c_scores = data_continuous["outcomes"]["meta_score"].to_list()
        diffs = [abs(r - c) for r, c in zip(r_scores, c_scores)]
        assert max(diffs) > 0.01  # at least some should differ


class TestOracleLearnability:
    """Verify the oracle's signal is recoverable from noisy data.

    This is the contract that synthetic data is fit-for-purpose for bandit development.
    Fit per-family linear regression on synthetic data and verify the largest-magnitude
    oracle coefficient is recovered with the correct sign.
    """

    def test_coefficient_recovery(self):
        oracle = MetaScoreOracle()
        rng = np.random.default_rng(123)
        n_samples = 2000
        product = "corridor_var"

        for family in VALID_FAMILIES:
            # Generate random features and noisy outcomes
            X = rng.normal(size=(n_samples, len(FEATURE_NAMES)))
            y = np.array([
                oracle.sample_score(X[i], family, product, rng)
                for i in range(n_samples)
            ])

            reg = LinearRegression().fit(X, y)

            # Find the oracle's largest-magnitude coefficient
            true_coef = oracle.coefficients[family]
            max_idx = np.argmax(np.abs(true_coef))
            true_sign = np.sign(true_coef[max_idx])
            recovered_sign = np.sign(reg.coef_[max_idx])

            assert true_sign == recovered_sign, (
                f"Family {family}: oracle coef[{max_idx}] ({FEATURE_NAMES[max_idx]}) "
                f"sign mismatch: true={true_sign}, recovered={recovered_sign}"
            )

            # Also check that the recovered coefficient magnitude is reasonable
            scale = oracle.product_scales.get(product, 1.0)
            expected_magnitude = abs(true_coef[max_idx]) * scale
            recovered_magnitude = abs(reg.coef_[max_idx])
            assert recovered_magnitude > expected_magnitude * 0.3, (
                f"Family {family}: recovered magnitude {recovered_magnitude:.3f} "
                f"too small vs expected {expected_magnitude:.3f}"
            )
