"""Single source of truth for the meta-score formula.

Both synthetic data generation and (eventually) real outcome recording must call
compute_meta_score() rather than computing the score inline. This isolation means
the formula can be changed by editing one file, and all components are stored so
meta_scores can be recomputed under a new formula without re-running anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MetaScoreConfig:
    """Configuration for meta-score computation.

    Two modes:
    - research (default): meta_score = forward_sharpe. No turnover penalty.
      Matches the cherry-pick workflow where the user selects which proposals to act on.
    - continuous: meta_score = forward_sharpe - lambda * L1_turnover.
      Use if committing to act on every 'best' proposal.
    """

    mode: Literal["research", "continuous"] = "research"
    lambda_turnover: float = 0.0

    def __post_init__(self) -> None:
        if self.mode == "continuous" and self.lambda_turnover <= 0:
            raise ValueError(
                "continuous mode requires lambda_turnover > 0; "
                "use research mode if you don't want a turnover penalty"
            )
        if self.mode == "research" and self.lambda_turnover != 0.0:
            raise ValueError(
                "research mode ignores lambda_turnover; set it to 0.0 explicitly"
            )


DEFAULT_META_SCORE_CONFIG = MetaScoreConfig(mode="research", lambda_turnover=0.0)


def compute_meta_score(
    *,
    config: MetaScoreConfig,
    forward_5d_mean_return: float,
    forward_realized_vol_21d: float,
    weights: list[float],
    prev_best_weights: list[float] | None,
) -> dict:
    """Compute meta_score and all its components.

    Returns a dict with keys ready to merge into an outcomes row:
      forward_sharpe, turnover_vs_prev_best (nullable),
      meta_score, meta_score_mode, lambda_used.

    In 'research' mode: meta_score = forward_sharpe; turnover is null.
    In 'continuous' mode: meta_score = forward_sharpe - lambda * L1_turnover.
        If prev_best_weights is None (no prior 'best'), turnover is computed
        vs a flat-zero baseline (i.e., L1 norm of current weights) — a reasonable
        cold-start convention since the first proposal has no predecessor.
    """
    # Guard against zero/near-zero vol
    if abs(forward_realized_vol_21d) < 1e-12:
        forward_sharpe = 0.0
    else:
        forward_sharpe = forward_5d_mean_return / forward_realized_vol_21d

    if config.mode == "research":
        return {
            "forward_sharpe": forward_sharpe,
            "turnover_vs_prev_best": None,
            "meta_score": forward_sharpe,
            "meta_score_mode": "research",
            "lambda_used": 0.0,
        }

    # continuous mode
    w = np.array(weights, dtype=np.float64)
    if prev_best_weights is not None:
        w_prev = np.array(prev_best_weights, dtype=np.float64)
    else:
        # Cold start: turnover vs zero baseline
        w_prev = np.zeros_like(w)

    turnover = float(np.sum(np.abs(w - w_prev)))
    meta_score = forward_sharpe - config.lambda_turnover * turnover

    return {
        "forward_sharpe": forward_sharpe,
        "turnover_vs_prev_best": turnover,
        "meta_score": meta_score,
        "meta_score_mode": "continuous",
        "lambda_used": config.lambda_turnover,
    }
