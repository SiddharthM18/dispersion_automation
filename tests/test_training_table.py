from __future__ import annotations

from datetime import date

import pytest

from dispersion_meta import paths
from dispersion_meta.synthetic import populate_synthetic_history


@pytest.fixture(autouse=True)
def tmp_data_root(tmp_path):
    paths.set_data_root(tmp_path)
    yield tmp_path
    paths.set_data_root(None)


class TestTrainingTable:
    def _populate(self, n_days=20, seed=42):
        return populate_synthetic_history(
            n_days=n_days, n_per_family=5, n_names=10, seed=seed
        )

    def test_basic_join(self):
        from dispersion_meta.training_table import build_training_table

        self._populate()
        result = build_training_table(as_of=date(2025, 1, 1))
        assert result is not None
        assert len(result) > 0
        # Should have columns from all three tables
        assert "vix_level" in result.columns  # from features
        assert "family" in result.columns  # from proposals
        assert "meta_score" in result.columns  # from outcomes

    def test_walk_forward_filter(self):
        from dispersion_meta.training_table import build_training_table

        self._populate(n_days=30)
        # Very early as_of should give few/no rows
        early = build_training_table(as_of=date(2024, 1, 1))
        # Late as_of should give more rows
        late = build_training_table(as_of=date(2025, 1, 1))

        early_len = 0 if early is None else len(early)
        late_len = 0 if late is None else len(late)
        assert late_len >= early_len

    def test_product_filter(self):
        from dispersion_meta.training_table import build_training_table

        self._populate()
        result = build_training_table(as_of=date(2025, 1, 1), products=["vol"])
        assert result is not None
        assert set(result["product"].unique().to_list()) == {"vol"}

    def test_pending_proposals(self):
        from dispersion_meta.training_table import pending_proposals

        self._populate()
        # Since all outcomes are written, pending should be empty or None
        pending = pending_proposals()
        if pending is not None:
            assert len(pending) == 0

    def test_latest_best_weights(self):
        from dispersion_meta.training_table import latest_best_weights

        self._populate()
        weights = latest_best_weights("vol")
        assert weights is not None
        assert len(weights) == 10  # n_names=10
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_build_full_table(self):
        from dispersion_meta.training_table import build_full_table

        self._populate()
        result = build_full_table()
        assert result is not None
        assert len(result) > 0

    def test_all_families_present(self):
        from dispersion_meta.training_table import build_training_table
        from dispersion_meta.schemas import VALID_FAMILIES

        self._populate(n_days=30)
        result = build_training_table(as_of=date(2025, 1, 1))
        assert result is not None
        families_in_data = set(result["family"].unique().to_list())
        assert families_in_data == VALID_FAMILIES
