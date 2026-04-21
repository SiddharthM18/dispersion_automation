"""Tests for config_space — grid structure and optimizer compatibility."""
from __future__ import annotations

import numpy as np
import pytest

from dispersion_meta.config_space import (
    CONFIG_GRID,
    FAMILIES,
    all_configs,
    grid_size,
    sample_config,
    total_grid_size,
)
from dispersion_meta.schemas import VALID_FAMILIES


class TestGridStructure:
    def test_families_match_schema(self):
        assert set(CONFIG_GRID.keys()) == VALID_FAMILIES

    def test_families_sorted(self):
        assert FAMILIES == sorted(FAMILIES)

    def test_no_empty_grids(self):
        for family, configs in CONFIG_GRID.items():
            assert len(configs) > 0, f"{family} grid is empty"

    def test_expected_grid_sizes(self):
        # max_mean: 4 max_weight × 3 cardinality = 12
        assert grid_size("max_mean") == 12
        # max_min: same
        assert grid_size("max_min") == 12
        # min_drawdown: same
        assert grid_size("min_drawdown") == 12
        # max_sharpe: 4 target_std × 3 max_weight × 3 cardinality = 36
        assert grid_size("max_sharpe") == 36
        # composite: 3 component sets × 3 max_weight × 3 cardinality = 27
        assert grid_size("composite") == 27

    def test_total_grid_size(self):
        assert total_grid_size() == 12 + 12 + 36 + 12 + 27

    def test_all_configs_returns_full_grid(self):
        for family in FAMILIES:
            assert len(all_configs(family)) == grid_size(family)

    def test_every_config_has_objective(self):
        for family, configs in CONFIG_GRID.items():
            for cfg in configs:
                assert cfg["objective"] == family

    def test_every_config_has_weight_bounds(self):
        for configs in CONFIG_GRID.values():
            for cfg in configs:
                wb = cfg["weight_bounds"]
                assert wb["min"] == 0.0
                assert wb["max"] > 0.0

    def test_every_config_has_cardinality(self):
        for configs in CONFIG_GRID.values():
            for cfg in configs:
                card = cfg["cardinality"]
                assert card["method"] == "mip"
                assert card["min_names"] < card["max_names"]

    def test_max_sharpe_has_target_std(self):
        for cfg in CONFIG_GRID["max_sharpe"]:
            assert "target_std" in cfg
            assert cfg["target_std"] > 0

    def test_composite_has_components(self):
        for cfg in CONFIG_GRID["composite"]:
            assert "components" in cfg
            assert len(cfg["components"]) >= 1


class TestSampling:
    def test_sample_returns_valid_config(self):
        rng = np.random.default_rng(0)
        for family in FAMILIES:
            cfg = sample_config(family, rng)
            assert cfg["objective"] == family
            assert cfg in CONFIG_GRID[family]

    def test_sample_covers_grid(self):
        """With enough samples, every config in a small grid should appear."""
        rng = np.random.default_rng(42)
        family = "max_mean"  # 12 configs
        seen = set()
        for _ in range(500):
            cfg = sample_config(family, rng)
            seen.add(id_config(cfg))
        assert len(seen) == grid_size(family)

    def test_sample_invalid_family_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(KeyError):
            sample_config("bogus_family", rng)


class TestOptimizerCompatibility:
    """Smoke test: every config in the grid must be accepted by from_config.

    We validate that from_config parses without raising and that solve()
    returns a result (any status).  Some configs may return solver_error
    or infeasible on random noise — that's expected and handled by the
    propose script's retry logic.  The test for *optimal* solves uses
    the linear families (max_mean, max_min, min_drawdown) which are
    pure LP+MIP and reliably solvable.
    """

    @pytest.fixture()
    def pnl_matrix(self):
        rng = np.random.default_rng(99)
        # 60 rows (days), 50 columns (names) — big enough for max_names=40
        return rng.normal(0.0, 0.01, size=(60, 50))

    @pytest.mark.parametrize("family", FAMILIES)
    def test_all_configs_accepted(self, family, pnl_matrix):
        """from_config must not raise for any config in the grid."""
        from dispersion_optimization.optimizer import DispersionOptimizer

        for cfg in all_configs(family):
            opt = DispersionOptimizer.from_config(cfg, pnl_matrix)
            result = opt.solve()
            # Result object must always be returned, regardless of status
            assert result.status in (
                "optimal", "optimal_inaccurate", "infeasible", "solver_error",
            )

    @pytest.mark.parametrize("family", ["max_mean", "max_min", "min_drawdown"])
    def test_linear_families_solve_optimal(self, family, pnl_matrix):
        """Linear objective families should solve to optimal on well-formed data."""
        from dispersion_optimization.optimizer import DispersionOptimizer

        for cfg in all_configs(family):
            opt = DispersionOptimizer.from_config(cfg, pnl_matrix)
            result = opt.solve()
            assert result.status in ("optimal", "optimal_inaccurate")
            assert result.weights.shape == (pnl_matrix.shape[1],)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def id_config(cfg: dict) -> str:
    """Deterministic string id for a config dict (for set membership)."""
    import json
    return json.dumps(cfg, sort_keys=True)
