"""Synthetic data generator with a known ground-truth oracle.

The oracle defines the true relationship between (features, family, product) → expected
meta_score. The bandit (next session) must NEVER see the oracle directly — it only sees
noisy outcomes. The learnability test verifies that the oracle's signal is recoverable
from noisy data via per-family linear regression.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from . import io, schemas
from .hashing import hash_and_serialize
from .meta_score import DEFAULT_META_SCORE_CONFIG, MetaScoreConfig, compute_meta_score


# ---------------------------------------------------------------------------
# Oracle — per-family linear coefficients over the feature vector
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "vix_level",
    "vix_20d_change",
    "avg_pairwise_corr",
    "dispersion_pnl_20d",
    "skew_steepness",
    "term_structure_slope",
    "earnings_density_21d",
]

# Each family has a coefficient vector over the 7 features.
# These encode which regimes each family thrives in.
ORACLE_COEFFICIENTS: dict[str, np.ndarray] = {
    # max_sharpe: prefers moderate VIX (negative), high pairwise corr (large positive),
    # avoids earnings noise (negative on earnings_density)
    "max_sharpe": np.array([-0.8, -0.2, 1.5, 0.3, 0.1, 0.1, -0.7]),
    # max_mean: strong positive on dispersion_pnl_20d (momentum), weak elsewhere
    "max_mean": np.array([0.1, 0.0, 0.2, 1.8, 0.1, 0.0, -0.1]),
    # max_min: moderate positive on pairwise corr and dispersion momentum
    "max_min": np.array([-0.3, -0.1, 0.9, 1.0, 0.2, 0.1, -0.4]),
    # min_drawdown: positive on vix_level and vix_20d_change (likes high/rising vol),
    # negative on pairwise corr
    "min_drawdown": np.array([1.2, 0.9, -1.0, 0.1, 0.2, 0.1, -0.3]),
    # composite: large positive on skew_steepness and term_structure_slope
    "composite": np.array([0.1, 0.0, 0.2, 0.3, 1.5, 1.3, 0.0]),
}

# Per-product scaling — different products have different base levels and sensitivities
PRODUCT_SCALES: dict[str, float] = {
    "corridor_var": 1.0,
    "vol": 0.8,
    "gamma": 1.1,
    "dngamma": 0.9,
}

PRODUCT_OFFSETS: dict[str, float] = {
    "corridor_var": 0.0,
    "vol": 0.1,
    "gamma": -0.05,
    "dngamma": 0.15,
}

NOISE_STD = 0.4


class MetaScoreOracle:
    """Ground-truth oracle: maps (features, family, product) → expected meta_score.

    The bandit never sees this directly. Synthetic outcomes are sampled by adding
    Gaussian noise to the oracle's expected value.
    """

    def __init__(
        self,
        coefficients: dict[str, np.ndarray] | None = None,
        product_scales: dict[str, float] | None = None,
        product_offsets: dict[str, float] | None = None,
        noise_std: float = NOISE_STD,
    ) -> None:
        self.coefficients = coefficients or ORACLE_COEFFICIENTS
        self.product_scales = product_scales or PRODUCT_SCALES
        self.product_offsets = product_offsets or PRODUCT_OFFSETS
        self.noise_std = noise_std

    def expected_score(
        self, features: np.ndarray, family: str, product: str
    ) -> float:
        """Return the true expected meta_score (no noise)."""
        coef = self.coefficients[family]
        scale = self.product_scales.get(product, 1.0)
        offset = self.product_offsets.get(product, 0.0)
        return float(np.dot(coef, features) * scale + offset)

    def sample_score(
        self, features: np.ndarray, family: str, product: str, rng: np.random.Generator
    ) -> float:
        """Sample a noisy meta_score."""
        expected = self.expected_score(features, family, product)
        return expected + rng.normal(0.0, self.noise_std)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_features(
    n_days: int,
    products: list[str],
    start_date: date = date(2024, 1, 2),
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Generate synthetic daily features for multiple products."""
    if rng is None:
        rng = np.random.default_rng(42)

    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    rows = []
    for dt in dates:
        # Base features for this day (shared macro)
        base = {
            "vix_level": rng.normal(20.0, 5.0),
            "vix_20d_change": rng.normal(0.0, 3.0),
            "term_structure_slope": rng.normal(0.0, 0.5),
        }
        for product in products:
            rows.append({
                "date": dt,
                "product": product,
                "vix_level": base["vix_level"],
                "vix_20d_change": base["vix_20d_change"],
                "avg_pairwise_corr": rng.normal(0.4, 0.15),
                "dispersion_pnl_20d": rng.normal(0.0, 1.0),
                "skew_steepness": rng.normal(0.0, 0.5),
                "term_structure_slope": base["term_structure_slope"],
                "earnings_density_21d": rng.uniform(0.0, 0.5),
            })

    return pl.DataFrame(rows)


def _make_config(family: str, rng: np.random.Generator) -> dict:
    """Generate a random optimizer config for a given family."""
    config: dict = {"objective": family}
    if family == "max_sharpe":
        config["target_std"] = round(rng.uniform(0.02, 0.10), 4)
    elif family == "composite":
        n_components = rng.integers(1, 4)
        config["components"] = [
            {
                "stat": rng.choice(["mean", "min", "std"]),
                "period": rng.choice(["3m", "6m", "1y"]),
                "weight": round(rng.uniform(0.1, 2.0), 3),
            }
            for _ in range(int(n_components))
        ]
    # Common optional params
    if rng.random() > 0.5:
        config["max_weight"] = round(rng.uniform(0.05, 0.3), 3)
    if rng.random() > 0.7:
        config["min_names"] = int(rng.integers(3, 10))
    return config


def synth_proposals_for_day(
    dt: date,
    products: list[str],
    n_per_family: int = 10,
    n_names: int = 20,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Generate synthetic proposals for one day across all products.

    Creates n_per_family configs per family per product. The first config per family
    is tagged 'best', second 'alt', and the rest 'candidate'.
    """
    if rng is None:
        rng = np.random.default_rng()

    families = sorted(schemas.VALID_FAMILIES)
    rows = []

    for product in products:
        for family in families:
            for i in range(n_per_family):
                config = _make_config(family, rng)
                config_hash, config_json = hash_and_serialize(config)

                # Generate random weights (simplex)
                raw_w = rng.exponential(1.0, size=n_names)
                weights = (raw_w / raw_w.sum()).tolist()
                column_names = [f"S{j:02d}" for j in range(n_names)]

                if i == 0:
                    proposal_type = "best"
                elif i == 1:
                    proposal_type = "alt"
                else:
                    proposal_type = "candidate"

                rows.append({
                    "date": dt,
                    "product": product,
                    "config_hash": config_hash,
                    "config_json": config_json,
                    "config_schema_version": 1,
                    "family": family,
                    "proposal_type": proposal_type,
                    "thompson_sample_value": None if proposal_type == "candidate" else rng.normal(0.0, 1.0),
                    "posterior_mean": None if proposal_type == "candidate" else rng.normal(0.0, 0.5),
                    "posterior_std": None if proposal_type == "candidate" else abs(rng.normal(0.5, 0.2)),
                    "weights": weights,
                    "column_names": column_names,
                    "n_names": n_names,
                    "in_sample_sharpe": rng.normal(1.5, 0.5),
                    "in_sample_mean_pnl": rng.normal(0.01, 0.005),
                    "in_sample_max_dd": -abs(rng.normal(0.03, 0.01)),
                    "solver_status": "optimal",
                    "solve_time_seconds": rng.exponential(0.5),
                    "pnl_matrix_path": None if proposal_type == "candidate" else f"pnl_matrices/{dt}/{product}/{config_hash}.parquet",
                })

    return pl.DataFrame(rows)


def synth_outcome_rows(
    proposals: pl.DataFrame,
    features: pl.DataFrame,
    oracle: MetaScoreOracle,
    meta_score_config: MetaScoreConfig = DEFAULT_META_SCORE_CONFIG,
    forward_days: int = 5,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """Generate synthetic outcomes for proposals using the oracle.

    MUST call compute_meta_score rather than computing the score inline —
    this ensures the formula is always sourced from meta_score.py.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows = []
    for row in proposals.iter_rows(named=True):
        dt = row["date"]
        product = row["product"]

        # Get features for this (date, product)
        feat_row = features.filter(
            (pl.col("date") == dt) & (pl.col("product") == product)
        )
        if len(feat_row) == 0:
            continue

        feat_values = np.array([
            feat_row[col][0] for col in FEATURE_NAMES
        ])

        family = row["family"]

        # Oracle gives the expected forward_sharpe; we decompose into components
        noisy_sharpe = oracle.sample_score(feat_values, family, product, rng)

        # Decompose: forward_sharpe = forward_5d_mean_return / forward_realized_vol_21d
        # Generate a plausible vol, then back out the mean return
        forward_vol = abs(rng.normal(0.15, 0.03))
        forward_mean = noisy_sharpe * forward_vol
        forward_pnl = forward_mean * forward_days

        weights = row["weights"]
        if isinstance(weights, pl.Series):
            weights = weights.to_list()

        # Use compute_meta_score — single source of truth
        score_components = compute_meta_score(
            config=meta_score_config,
            forward_5d_mean_return=forward_mean,
            forward_realized_vol_21d=forward_vol,
            weights=weights,
            prev_best_weights=None,
        )

        eval_date = dt + timedelta(days=forward_days + 2)  # +2 for weekends approx

        rows.append({
            "date": dt,
            "product": product,
            "config_hash": row["config_hash"],
            "eval_date": eval_date,
            "forward_window_days": forward_days,
            "forward_5d_pnl": forward_pnl,
            "forward_5d_mean_return": forward_mean,
            "forward_realized_vol_21d": forward_vol,
            **score_components,
        })

    return pl.DataFrame(rows)


def populate_synthetic_history(
    n_days: int = 60,
    products: list[str] | None = None,
    n_per_family: int = 10,
    n_names: int = 20,
    meta_score_config: MetaScoreConfig = DEFAULT_META_SCORE_CONFIG,
    seed: int = 42,
) -> dict[str, pl.DataFrame]:
    """Generate and persist a complete synthetic history.

    Returns dict with keys: features, proposals, outcomes.
    """
    if products is None:
        products = sorted(schemas.VALID_PRODUCTS)

    rng = np.random.default_rng(seed)
    oracle = MetaScoreOracle()

    start_date = date(2024, 1, 2)
    features = generate_features(n_days, products, start_date=start_date, rng=rng)

    all_proposals = []
    for i in range(n_days):
        dt = start_date + timedelta(days=i)
        day_proposals = synth_proposals_for_day(
            dt, products, n_per_family=n_per_family, n_names=n_names, rng=rng
        )
        all_proposals.append(day_proposals)

    proposals = pl.concat(all_proposals)
    outcomes = synth_outcome_rows(
        proposals, features, oracle,
        meta_score_config=meta_score_config, rng=rng,
    )

    # Write to disk
    io.write_features(features)
    io.write_proposals(proposals)
    io.write_outcomes(outcomes)

    return {"features": features, "proposals": proposals, "outcomes": outcomes}
