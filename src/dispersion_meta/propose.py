"""Daily proposal orchestration.

Ties together the bandit, config space, optimizer, and data layer:
1. Write today's features
2. Load training table and fit bandit posteriors
3. Thompson sample → select families → sample configs → solve
4. Write proposals + PnL matrices
"""
from __future__ import annotations

import logging
from datetime import date

import numpy as np
import polars as pl

from . import io, schemas
from .bandit import BayesianLinearBandit
from .config_space import FAMILIES, sample_config
from .hashing import hash_and_serialize
from .synthetic import FEATURE_NAMES
from .training_table import build_training_table

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def propose_today(
    *,
    today: date,
    features: dict[str, float],
    pnl_matrices: dict[str, np.ndarray],
    column_names: list[str],
    alpha: float = 1.0,
    sigma_sq: float = 0.16,
    seed: int | None = None,
) -> dict:
    """Run the full daily proposal cycle.

    Parameters
    ----------
    today : date
        Trading date.
    features : dict[str, float]
        7 feature values keyed by FEATURE_NAMES.
    pnl_matrices : dict[str, np.ndarray]
        One (T, N) PnL matrix per product.
    column_names : list[str]
        N ticker names, same across products.
    alpha : float
        Ridge prior strength for the bandit.
    sigma_sq : float
        Noise variance for the bandit.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Summary with per-product proposal info and diagnostics.
    """
    products = sorted(pnl_matrices.keys())
    _validate_inputs(features, products, pnl_matrices, column_names)

    rng = np.random.default_rng(seed)

    # Step 1: write today's features (one row per product)
    _write_features(today, features, products)

    # Step 2: load training table
    training = build_training_table(as_of=today, products=products)

    # Step 3-4: per-product bandit + proposals
    x_today = np.array([features[f] for f in FEATURE_NAMES])
    all_proposal_rows: list[dict] = []
    all_pnl_mats: dict[tuple[date, str, str], np.ndarray] = {}
    summary: dict[str, dict] = {}

    for product in products:
        bandit = BayesianLinearBandit(
            families=FAMILIES, alpha=alpha, sigma_sq=sigma_sq,
        )

        # Fit bandit (or use prior if no training data)
        if training is not None:
            product_training = training.filter(pl.col("product") == product)
            if len(product_training) > 0:
                bandit.fit(product_training)
            else:
                _set_trivial_scaler(bandit, x_today)
        else:
            _set_trivial_scaler(bandit, x_today)

        # Thompson sample and select families
        proposals = bandit.select_proposals(x_today, rng)

        product_proposals = []
        for prop in proposals:
            row = _solve_proposal(
                prop=prop,
                product=product,
                today=today,
                pnl_matrix=pnl_matrices[product],
                column_names=column_names,
                rng=rng,
            )
            if row is not None:
                product_proposals.append(row)
                all_pnl_mats[(today, product, row["config_hash"])] = pnl_matrices[product]

        all_proposal_rows.extend(product_proposals)
        summary[product] = {
            "n_proposals": len(product_proposals),
            "families": [p["family"] for p in product_proposals],
            "types": [p["proposal_type"] for p in product_proposals],
            "diagnostics": bandit.diagnostics(),
        }

    # Step 5: write all proposals
    if all_proposal_rows:
        df = pl.DataFrame(all_proposal_rows)
        io.write_proposals(df, pnl_matrices=all_pnl_mats)

    return {
        "today": today,
        "n_products": len(products),
        "n_proposals_total": len(all_proposal_rows),
        "products": summary,
    }


def _validate_inputs(
    features: dict[str, float],
    products: list[str],
    pnl_matrices: dict[str, np.ndarray],
    column_names: list[str],
) -> None:
    """Validate inputs before running the proposal cycle."""
    missing_feats = set(FEATURE_NAMES) - set(features.keys())
    if missing_feats:
        raise ValueError(f"Missing features: {sorted(missing_feats)}")

    invalid_products = set(products) - schemas.VALID_PRODUCTS
    if invalid_products:
        raise ValueError(f"Invalid products: {sorted(invalid_products)}")

    for product, matrix in pnl_matrices.items():
        if matrix.ndim != 2:
            raise ValueError(f"PnL matrix for {product} must be 2D, got {matrix.ndim}D")
        if matrix.shape[1] != len(column_names):
            raise ValueError(
                f"PnL matrix for {product} has {matrix.shape[1]} columns, "
                f"expected {len(column_names)} (len(column_names))"
            )


def _write_features(today: date, features: dict[str, float], products: list[str]) -> None:
    """Write one feature row per product for today."""
    rows = []
    for product in products:
        row = {"date": today, "product": product}
        row.update(features)
        rows.append(row)
    io.write_features(pl.DataFrame(rows))


def _set_trivial_scaler(bandit: BayesianLinearBandit, x_today: np.ndarray) -> None:
    """Set a trivial scaler (mean=x_today, std=1) for cold-start.

    When there's no training data, we can't compute feature statistics.
    Using x_today as mean and 1.0 as std means today's standardized
    features will be all zeros — Thompson samples come from the prior,
    giving approximately uniform family selection. This is correct
    cold-start behavior.
    """
    bandit._feat_mean = x_today.copy()
    bandit._feat_std = np.ones_like(x_today)


def _solve_proposal(
    *,
    prop: dict,
    product: str,
    today: date,
    pnl_matrix: np.ndarray,
    column_names: list[str],
    rng: np.random.Generator,
) -> dict | None:
    """Sample a config from the family grid, solve, and build a proposal row.

    Retries up to MAX_RETRIES times on solver failure.
    """
    family = prop["family"]

    for attempt in range(1 + MAX_RETRIES):
        config = sample_config(family, rng)
        config_hash, config_json = hash_and_serialize(config)

        try:
            result = _run_optimizer(config, pnl_matrix, column_names)
        except Exception:
            logger.warning(
                "Optimizer raised for %s/%s (attempt %d/%d)",
                product, family, attempt + 1, 1 + MAX_RETRIES,
                exc_info=True,
            )
            continue

        status = result.status
        if status in ("optimal", "optimal_inaccurate"):
            weights = result.weights.tolist()
            return {
                "date": today,
                "product": product,
                "config_hash": config_hash,
                "config_json": config_json,
                "config_schema_version": 1,
                "family": family,
                "proposal_type": prop["proposal_type"],
                "thompson_sample_value": prop.get("thompson_sample_value"),
                "posterior_mean": prop.get("posterior_mean"),
                "posterior_std": prop.get("posterior_std"),
                "weights": weights,
                "column_names": column_names,
                "n_names": int(result.n_names),
                "in_sample_sharpe": float(result.sharpe),
                "in_sample_mean_pnl": float(result.mean_pnl),
                "in_sample_max_dd": float(result.max_drawdown),
                "solver_status": status,
                "solve_time_seconds": float(result.solve_time),
                "pnl_matrix_path": f"pnl_matrices/{today}/{product}/{config_hash}.parquet",
            }

        logger.warning(
            "Solver returned %s for %s/%s (attempt %d/%d)",
            status, product, family, attempt + 1, 1 + MAX_RETRIES,
        )

    logger.warning(
        "All %d attempts failed for %s/%s — skipping proposal slot",
        1 + MAX_RETRIES, product, family,
    )
    return None


def _run_optimizer(
    config: dict, pnl_matrix: np.ndarray, column_names: list[str]
) -> object:
    """Call DispersionOptimizer.from_config and solve.

    Isolated for easy mocking in tests.
    """
    from dispersion_optimization.optimizer import DispersionOptimizer

    opt = DispersionOptimizer.from_config(config, pnl_matrix, column_names=column_names)
    return opt.solve()
