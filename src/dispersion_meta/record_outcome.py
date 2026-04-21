"""T+5 outcome recording.

Reads proposals for a given date, computes forward realized stats using
user-provided forward returns and trailing vol, and writes outcomes.
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl

from . import io
from .meta_score import DEFAULT_META_SCORE_CONFIG, MetaScoreConfig, compute_meta_score
from .training_table import latest_best_weights

logger = logging.getLogger(__name__)


def record_outcomes(
    *,
    propose_date: date,
    eval_date: date,
    forward_returns: dict[str, np.ndarray],
    trailing_vol: dict[str, float],
    column_names: list[str],
    meta_score_config: MetaScoreConfig = DEFAULT_META_SCORE_CONFIG,
) -> pl.DataFrame:
    """Compute and write outcomes for proposals from propose_date.

    Parameters
    ----------
    propose_date : date
        Which day's proposals to evaluate.
    eval_date : date
        The evaluation date (should be propose_date + ~5 trading days).
    forward_returns : dict[str, np.ndarray]
        Per product, (forward_days, N) daily returns for the forward window.
    trailing_vol : dict[str, float]
        Per product, 21-day trailing realized vol.
    column_names : list[str]
        N ticker names matching the columns in forward_returns.
    meta_score_config : MetaScoreConfig
        Score computation mode (research or continuous).

    Returns
    -------
    pl.DataFrame
        The outcomes DataFrame that was written.
    """
    proposals = io.read_proposals(
        start_date=propose_date, end_date=propose_date,
    )
    if proposals is None or len(proposals) == 0:
        raise ValueError(f"No proposals found for {propose_date}")

    rows = []
    for row in proposals.iter_rows(named=True):
        product = row["product"]

        if product not in forward_returns:
            logger.warning("No forward returns for product %s — skipping", product)
            continue
        if product not in trailing_vol:
            logger.warning("No trailing vol for product %s — skipping", product)
            continue

        weights = row["weights"]
        if isinstance(weights, pl.Series):
            weights = weights.to_list()

        fwd = forward_returns[product]  # (forward_days, N)
        forward_days = fwd.shape[0]
        w = np.array(weights)

        # Portfolio return per day: fwd @ w → (forward_days,)
        daily_portfolio_returns = fwd @ w
        forward_5d_pnl = float(np.sum(daily_portfolio_returns))
        forward_5d_mean_return = float(np.mean(daily_portfolio_returns))
        forward_realized_vol_21d = trailing_vol[product]

        # Previous best weights for turnover computation
        prev_best = latest_best_weights(product)

        score = compute_meta_score(
            config=meta_score_config,
            forward_5d_mean_return=forward_5d_mean_return,
            forward_realized_vol_21d=forward_realized_vol_21d,
            weights=weights,
            prev_best_weights=prev_best,
        )

        rows.append({
            "date": propose_date,
            "product": product,
            "config_hash": row["config_hash"],
            "eval_date": eval_date,
            "forward_window_days": forward_days,
            "forward_5d_pnl": forward_5d_pnl,
            "forward_5d_mean_return": forward_5d_mean_return,
            "forward_realized_vol_21d": forward_realized_vol_21d,
            **score,
        })

    if not rows:
        raise ValueError(
            f"No outcomes could be computed for proposals on {propose_date}"
        )

    outcomes_df = pl.DataFrame(rows)
    io.write_outcomes(outcomes_df)
    return outcomes_df
