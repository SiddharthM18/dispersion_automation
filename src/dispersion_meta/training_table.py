"""Joins for the bandit — the boundary between data layer and learning layer.

build_training_table() produces what the bandit consumes: one row per
(date, product, config_hash) with features, proposal info, and realized
forward outcomes joined together, filtered by a strict walk-forward as_of date.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from . import io


def build_training_table(
    *,
    as_of: date,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Inner-join proposals, features, and outcomes with walk-forward filtering.

    The as_of parameter enforces strict before-date filtering: only rows where
    eval_date <= as_of are included. This is load-bearing — without it the
    bandit would train on future data.
    """
    proposals = io.read_proposals(products=products)
    features = io.read_features(products=products)
    outcomes = io.read_outcomes(products=products)

    if proposals is None or features is None or outcomes is None:
        return None

    # Walk-forward filter: only outcomes that have been realized by as_of
    outcomes = outcomes.filter(pl.col("eval_date") <= as_of)

    if len(outcomes) == 0:
        return None

    # Join proposals with features on (date, product)
    joined = proposals.join(
        features,
        on=["date", "product"],
        how="inner",
        suffix="_feat",
    )

    # Drop the duplicated version column from features
    feat_version_col = "_table_schema_version_feat"
    if feat_version_col in joined.columns:
        joined = joined.drop(feat_version_col)

    # Join with outcomes on (date, product, config_hash)
    joined = joined.join(
        outcomes,
        on=["date", "product", "config_hash"],
        how="inner",
        suffix="_out",
    )

    # Drop duplicated version column from outcomes
    out_version_col = "_table_schema_version_out"
    if out_version_col in joined.columns:
        joined = joined.drop(out_version_col)

    return joined


def build_full_table(
    *,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Left-join proposals with features and outcomes — includes pending proposals."""
    proposals = io.read_proposals(products=products)
    features = io.read_features(products=products)
    outcomes = io.read_outcomes(products=products)

    if proposals is None:
        return None

    joined = proposals
    if features is not None:
        joined = joined.join(
            features, on=["date", "product"], how="left", suffix="_feat"
        )
        feat_version_col = "_table_schema_version_feat"
        if feat_version_col in joined.columns:
            joined = joined.drop(feat_version_col)

    if outcomes is not None:
        joined = joined.join(
            outcomes, on=["date", "product", "config_hash"], how="left", suffix="_out"
        )
        out_version_col = "_table_schema_version_out"
        if out_version_col in joined.columns:
            joined = joined.drop(out_version_col)

    return joined


def pending_proposals(
    *,
    products: list[str] | None = None,
) -> pl.DataFrame | None:
    """Return proposals that don't yet have outcomes (forward window hasn't elapsed)."""
    proposals = io.read_proposals(products=products)
    outcomes = io.read_outcomes(products=products)

    if proposals is None:
        return None

    if outcomes is None:
        return proposals

    # Anti-join: proposals without matching outcomes
    return proposals.join(
        outcomes.select("date", "product", "config_hash"),
        on=["date", "product", "config_hash"],
        how="anti",
    )


def latest_best_weights(product: str) -> list[float] | None:
    """Return the weights from the most recent 'best' proposal for a product.

    Used by compute_meta_score in continuous mode to compute turnover
    vs the previous day's best proposed basket.
    """
    proposals = io.read_proposals(products=[product], proposal_types=["best"])
    if proposals is None or len(proposals) == 0:
        return None

    latest = proposals.sort("date").tail(1)
    weights = latest["weights"][0].to_list()
    return weights
