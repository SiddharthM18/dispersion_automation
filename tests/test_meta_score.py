from __future__ import annotations

import pytest

from dispersion_meta.meta_score import MetaScoreConfig, compute_meta_score


class TestMetaScoreConfig:
    def test_research_default(self):
        cfg = MetaScoreConfig()
        assert cfg.mode == "research"
        assert cfg.lambda_turnover == 0.0

    def test_continuous_requires_positive_lambda(self):
        with pytest.raises(ValueError, match="continuous mode requires lambda_turnover > 0"):
            MetaScoreConfig(mode="continuous", lambda_turnover=0.0)

    def test_continuous_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="continuous mode requires lambda_turnover > 0"):
            MetaScoreConfig(mode="continuous", lambda_turnover=-0.1)

    def test_research_nonzero_lambda_raises(self):
        with pytest.raises(ValueError, match="research mode ignores lambda_turnover"):
            MetaScoreConfig(mode="research", lambda_turnover=0.5)

    def test_continuous_valid(self):
        cfg = MetaScoreConfig(mode="continuous", lambda_turnover=0.3)
        assert cfg.lambda_turnover == 0.3


class TestComputeMetaScore:
    def test_research_mode_null_turnover(self):
        result = compute_meta_score(
            config=MetaScoreConfig(),
            forward_5d_mean_return=0.01,
            forward_realized_vol_21d=0.15,
            weights=[0.5, 0.5],
            prev_best_weights=None,
        )
        assert result["turnover_vs_prev_best"] is None
        assert result["meta_score_mode"] == "research"
        assert result["lambda_used"] == 0.0
        assert abs(result["forward_sharpe"] - 0.01 / 0.15) < 1e-10
        assert result["meta_score"] == result["forward_sharpe"]

    def test_continuous_mode_with_prev_weights(self):
        cfg = MetaScoreConfig(mode="continuous", lambda_turnover=0.3)
        result = compute_meta_score(
            config=cfg,
            forward_5d_mean_return=0.01,
            forward_realized_vol_21d=0.15,
            weights=[0.6, 0.4],
            prev_best_weights=[0.5, 0.5],
        )
        expected_sharpe = 0.01 / 0.15
        expected_turnover = abs(0.6 - 0.5) + abs(0.4 - 0.5)  # 0.2
        expected_score = expected_sharpe - 0.3 * expected_turnover

        assert result["turnover_vs_prev_best"] == pytest.approx(expected_turnover)
        assert result["meta_score"] == pytest.approx(expected_score)
        assert result["meta_score_mode"] == "continuous"
        assert result["lambda_used"] == 0.3

    def test_continuous_cold_start_no_prev(self):
        """When prev_best_weights is None, turnover is vs zero baseline."""
        cfg = MetaScoreConfig(mode="continuous", lambda_turnover=0.3)
        result = compute_meta_score(
            config=cfg,
            forward_5d_mean_return=0.01,
            forward_realized_vol_21d=0.15,
            weights=[0.6, 0.4],
            prev_best_weights=None,
        )
        # Turnover vs zeros = sum(|w|) = 1.0
        assert result["turnover_vs_prev_best"] == pytest.approx(1.0)
        assert result["meta_score_mode"] == "continuous"

    def test_zero_vol_guard(self):
        result = compute_meta_score(
            config=MetaScoreConfig(),
            forward_5d_mean_return=0.01,
            forward_realized_vol_21d=0.0,
            weights=[0.5, 0.5],
            prev_best_weights=None,
        )
        assert result["forward_sharpe"] == 0.0

    def test_result_has_all_outcome_fields(self):
        """The returned dict must have every field the outcomes schema expects from it."""
        result = compute_meta_score(
            config=MetaScoreConfig(),
            forward_5d_mean_return=0.01,
            forward_realized_vol_21d=0.15,
            weights=[0.5, 0.5],
            prev_best_weights=None,
        )
        expected_keys = {
            "forward_sharpe", "turnover_vs_prev_best",
            "meta_score", "meta_score_mode", "lambda_used",
        }
        assert set(result.keys()) == expected_keys
