"""Contextual Thompson sampling bandit with Bayesian linear regression per arm.

One BayesianLinearBandit instance per product.  Each of the 5 objective
families is an arm.  The model maps an 8-dimensional feature vector
(7 raw features + intercept) to expected meta_score via a Bayesian
linear model with a conjugate normal prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .config_space import FAMILIES
from .synthetic import FEATURE_NAMES


@dataclass
class ArmState:
    """Posterior state for one family arm."""

    family: str
    n_obs: int
    mu_n: np.ndarray        # (d,) posterior mean of weights
    Lambda_n: np.ndarray     # (d, d) posterior precision matrix
    Lambda_n_inv: np.ndarray  # (d, d) cached posterior covariance


class BayesianLinearBandit:
    """Contextual Thompson sampling bandit with per-family Bayesian linear models.

    One instance per product.  Fits on the product-filtered training table.
    At propose time, Thompson-samples each arm and selects 4 proposals:
    1 best, 2 alt, 1 explore (highest posterior variance).
    """

    def __init__(
        self,
        families: list[str] | None = None,
        n_features: int = 7,
        alpha: float = 1.0,
        sigma_sq: float = 0.16,
    ) -> None:
        self.families = families or FAMILIES
        self.n_features = n_features
        self.d = n_features + 1  # +1 for intercept
        self.alpha = alpha
        self.sigma_sq = sigma_sq

        # Scaler params — set during fit()
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None

        # Per-arm posterior state — initialized to prior
        self.arms: dict[str, ArmState] = {}
        for fam in self.families:
            self._init_arm(fam)

    def _init_arm(self, family: str) -> None:
        """Initialize an arm to the prior."""
        Lambda_0 = self.alpha * np.eye(self.d)
        Lambda_0_inv = (1.0 / self.alpha) * np.eye(self.d)
        mu_0 = np.zeros(self.d)
        self.arms[family] = ArmState(
            family=family,
            n_obs=0,
            mu_n=mu_0,
            Lambda_n=Lambda_0.copy(),
            Lambda_n_inv=Lambda_0_inv.copy(),
        )

    def _standardize(self, X_raw: np.ndarray) -> np.ndarray:
        """Standardize raw features using stored scaler params.

        X_raw: (n, n_features) or (n_features,)
        Returns same shape, standardized.
        """
        if self._feat_mean is None or self._feat_std is None:
            raise RuntimeError("Bandit has not been fit — no scaler params available")
        return (X_raw - self._feat_mean) / self._feat_std

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Prepend a column of ones for the intercept term.

        X: (n, n_features) → (n, d)  or  (n_features,) → (d,)
        """
        if X.ndim == 1:
            return np.concatenate([[1.0], X])
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, training_df: pl.DataFrame, feature_columns: list[str] | None = None) -> None:
        """Fit all arm posteriors from a training table filtered to one product.

        1. Compute and store scaler params from training features
        2. Standardize features, add intercept
        3. For each family with training data, compute posterior
        4. Arms with no data keep the prior
        """
        if feature_columns is None:
            feature_columns = FEATURE_NAMES

        # Extract feature matrix and compute scaler
        X_raw = training_df.select(feature_columns).to_numpy()
        self._feat_mean = X_raw.mean(axis=0)
        self._feat_std = X_raw.std(axis=0)
        # Guard against zero-std features (constant columns)
        self._feat_std = np.where(self._feat_std < 1e-12, 1.0, self._feat_std)

        X_scaled = self._standardize(X_raw)
        X = self._add_intercept(X_scaled)  # (n, d)

        for fam in self.families:
            fam_mask = (training_df["family"] == fam).to_numpy()
            n_obs = int(fam_mask.sum())

            if n_obs == 0:
                self._init_arm(fam)
                continue

            X_fam = X[fam_mask]  # (n_fam, d)
            y_fam = training_df.filter(pl.col("family") == fam)["meta_score"].to_numpy()

            # Posterior: Lambda_n = Lambda_0 + (1/sigma^2) X^T X
            Lambda_0 = self.alpha * np.eye(self.d)
            mu_0 = np.zeros(self.d)

            XtX = X_fam.T @ X_fam
            Xty = X_fam.T @ y_fam

            Lambda_n = Lambda_0 + (1.0 / self.sigma_sq) * XtX
            # Jitter for numerical stability
            Lambda_n += 1e-6 * np.eye(self.d)
            Lambda_n_inv = np.linalg.inv(Lambda_n)

            mu_n = Lambda_n_inv @ (Lambda_0 @ mu_0 + (1.0 / self.sigma_sq) * Xty)

            self.arms[fam] = ArmState(
                family=fam,
                n_obs=n_obs,
                mu_n=mu_n,
                Lambda_n=Lambda_n,
                Lambda_n_inv=Lambda_n_inv,
            )

    def fit_with_scaler(
        self,
        training_df: pl.DataFrame,
        feat_mean: np.ndarray,
        feat_std: np.ndarray,
        feature_columns: list[str] | None = None,
    ) -> None:
        """Fit posteriors using externally provided scaler params.

        Useful when you want consistent standardization across products.
        """
        if feature_columns is None:
            feature_columns = FEATURE_NAMES

        self._feat_mean = feat_mean
        self._feat_std = np.where(feat_std < 1e-12, 1.0, feat_std)

        X_raw = training_df.select(feature_columns).to_numpy()
        X_scaled = self._standardize(X_raw)
        X = self._add_intercept(X_scaled)

        for fam in self.families:
            fam_mask = (training_df["family"] == fam).to_numpy()
            n_obs = int(fam_mask.sum())

            if n_obs == 0:
                self._init_arm(fam)
                continue

            X_fam = X[fam_mask]
            y_fam = training_df.filter(pl.col("family") == fam)["meta_score"].to_numpy()

            Lambda_0 = self.alpha * np.eye(self.d)
            mu_0 = np.zeros(self.d)
            XtX = X_fam.T @ X_fam
            Xty = X_fam.T @ y_fam

            Lambda_n = Lambda_0 + (1.0 / self.sigma_sq) * XtX
            Lambda_n += 1e-6 * np.eye(self.d)
            Lambda_n_inv = np.linalg.inv(Lambda_n)
            mu_n = Lambda_n_inv @ (Lambda_0 @ mu_0 + (1.0 / self.sigma_sq) * Xty)

            self.arms[fam] = ArmState(
                family=fam, n_obs=n_obs, mu_n=mu_n,
                Lambda_n=Lambda_n, Lambda_n_inv=Lambda_n_inv,
            )

    def thompson_sample(self, x_today: np.ndarray, rng: np.random.Generator) -> list[dict]:
        """Draw Thompson samples for all arms given today's raw features.

        x_today: raw 7-dimensional feature vector (pre-standardization).
        Returns list of dicts sorted by sampled value descending.
        """
        x_std = self._standardize(x_today)
        x = self._add_intercept(x_std)  # (d,)

        results = []
        for fam in self.families:
            arm = self.arms[fam]
            # Sample w_tilde ~ N(mu_n, Lambda_n^{-1})
            w_tilde = rng.multivariate_normal(arm.mu_n, arm.Lambda_n_inv)
            sampled_value = float(x @ w_tilde)
            posterior_mean = float(x @ arm.mu_n)
            posterior_std = float(np.sqrt(x @ arm.Lambda_n_inv @ x))

            results.append({
                "family": fam,
                "sampled_value": sampled_value,
                "posterior_mean": posterior_mean,
                "posterior_std": posterior_std,
            })

        results.sort(key=lambda r: r["sampled_value"], reverse=True)
        return results

    def select_proposals(self, x_today: np.ndarray, rng: np.random.Generator) -> list[dict]:
        """Select 4 proposals: 1 best, 2 alt, 1 explore.

        The explore slot goes to the family with highest posterior variance
        (x^T Lambda_n^{-1} x) not already in the top 3.
        """
        samples = self.thompson_sample(x_today, rng)

        # Top 3 by sampled value → best + 2 alt
        top3 = samples[:3]
        top3_families = {s["family"] for s in top3}

        proposals = [
            {**top3[0], "proposal_type": "best",
             "thompson_sample_value": top3[0]["sampled_value"]},
            {**top3[1], "proposal_type": "alt",
             "thompson_sample_value": top3[1]["sampled_value"]},
            {**top3[2], "proposal_type": "alt",
             "thompson_sample_value": top3[2]["sampled_value"]},
        ]

        # Explore: highest posterior_std not already selected
        remaining = [s for s in samples if s["family"] not in top3_families]
        if remaining:
            explore = max(remaining, key=lambda s: s["posterior_std"])
        else:
            # All 5 families in top 3 is impossible (only 3 slots, 5 families),
            # but handle gracefully: pick the one with highest posterior_std from top3
            explore = max(samples[3:], key=lambda s: s["posterior_std"])

        proposals.append({
            **explore, "proposal_type": "explore",
            "thompson_sample_value": explore["sampled_value"],
        })

        return proposals

    def diagnostics(self) -> dict:
        """Return diagnostic info for logging."""
        arm_info = {}
        for fam, arm in self.arms.items():
            arm_info[fam] = {
                "n_obs": arm.n_obs,
                "posterior_mean_norm": float(np.linalg.norm(arm.mu_n)),
                "posterior_trace": float(np.trace(arm.Lambda_n_inv)),
            }
        return {
            "arms": arm_info,
            "scaler_mean": self._feat_mean.tolist() if self._feat_mean is not None else None,
            "scaler_std": self._feat_std.tolist() if self._feat_std is not None else None,
        }
