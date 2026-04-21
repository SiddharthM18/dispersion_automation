"""Discrete config search space for the bandit's two-tier selection.

The bandit picks a family (arm); within that family, a config is sampled
uniformly from a small grid.  Every config dict here is valid input to
``DispersionOptimizer.from_config()``.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Grid definitions — one list of config dicts per family
# ---------------------------------------------------------------------------

CONFIG_GRID: dict[str, list[dict]] = {
    "max_mean": [
        {
            "objective": "max_mean",
            "weight_bounds": {"min": 0.0, "max": m},
            "cardinality": {"method": "mip", "min_names": lo, "max_names": hi},
        }
        for m in [0.05, 0.08, 0.10, 0.15]
        for lo, hi in [(10, 20), (15, 30), (20, 40)]
    ],
    "max_min": [
        {
            "objective": "max_min",
            "weight_bounds": {"min": 0.0, "max": m},
            "cardinality": {"method": "mip", "min_names": lo, "max_names": hi},
        }
        for m in [0.05, 0.08, 0.10, 0.15]
        for lo, hi in [(10, 20), (15, 30), (20, 40)]
    ],
    "max_sharpe": [
        {
            "objective": "max_sharpe",
            "target_std": ts,
            "weight_bounds": {"min": 0.0, "max": m},
            "cardinality": {"method": "mip", "min_names": lo, "max_names": hi},
        }
        for ts in [0.01, 0.02, 0.03, 0.05]
        for m in [0.05, 0.08, 0.10]
        for lo, hi in [(10, 20), (15, 30), (20, 40)]
    ],
    "min_drawdown": [
        {
            "objective": "min_drawdown",
            "weight_bounds": {"min": 0.0, "max": m},
            "cardinality": {"method": "mip", "min_names": lo, "max_names": hi},
        }
        for m in [0.05, 0.08, 0.10, 0.15]
        for lo, hi in [(10, 20), (15, 30), (20, 40)]
    ],
    "composite": [
        {
            "objective": "composite",
            "components": comps,
            "weight_bounds": {"min": 0.0, "max": m},
            "cardinality": {"method": "mip", "min_names": lo, "max_names": hi},
        }
        for comps in [
            [
                {"stat": "mean", "period": "3m", "weight": 0.5},
                {"stat": "min", "period": "1m", "weight": 0.5},
            ],
            [
                {"stat": "sum", "period": "full", "weight": 1.0},
                {"stat": "std", "period": "1y", "weight": -0.3},
            ],
            [
                {"stat": "mean", "period": "1m", "weight": 1.0},
                {"stat": "mean", "period": "3m", "weight": 2.0},
                {"stat": "mean", "period": "1y", "weight": 3.0},
            ],
        ]
        for m in [0.05, 0.08, 0.10]
        for lo, hi in [(10, 20), (15, 30), (20, 40)]
    ],
}

FAMILIES: list[str] = sorted(CONFIG_GRID.keys())


def sample_config(family: str, rng: np.random.Generator) -> dict:
    """Uniformly sample one config from the family's grid."""
    grid = CONFIG_GRID[family]
    idx = rng.integers(0, len(grid))
    return grid[idx]


def all_configs(family: str) -> list[dict]:
    """Return the full grid for a family."""
    return CONFIG_GRID[family]


def grid_size(family: str) -> int:
    """Number of configs in a family's grid."""
    return len(CONFIG_GRID[family])


def total_grid_size() -> int:
    """Total configs across all families."""
    return sum(len(v) for v in CONFIG_GRID.values())
