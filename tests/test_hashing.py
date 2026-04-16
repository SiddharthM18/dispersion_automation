from __future__ import annotations

import numpy as np
import pytest

from dispersion_meta.hashing import canonicalize_config, hash_and_serialize, hash_config


class TestCanonicalize:
    def test_key_order_invariance(self):
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        assert canonicalize_config(a) == canonicalize_config(b)

    def test_float_rounding(self):
        """Floats that differ only beyond 10 decimals hash the same."""
        a = {"x": 0.1 + 0.2}
        b = {"x": 0.3}
        assert canonicalize_config(a) == canonicalize_config(b)

    def test_none_equivalence(self):
        """None values are dropped — {a: 1, b: None} == {a: 1}."""
        a = {"a": 1, "b": None}
        b = {"a": 1}
        assert canonicalize_config(a) == canonicalize_config(b)

    def test_bool_vs_int_distinction(self):
        """bool and int must hash differently despite isinstance(True, int) being True."""
        a = {"x": True}
        b = {"x": 1}
        assert canonicalize_config(a) != canonicalize_config(b)

    def test_nested_dict(self):
        a = {"outer": {"b": 1, "a": 2}}
        b = {"outer": {"a": 2, "b": 1}}
        assert canonicalize_config(a) == canonicalize_config(b)

    def test_list_preserved(self):
        a = {"x": [3, 1, 2]}
        b = {"x": [1, 2, 3]}
        assert canonicalize_config(a) != canonicalize_config(b)

    def test_bytes_raises(self):
        with pytest.raises(TypeError, match="Cannot canonicalize type bytes"):
            canonicalize_config({"x": b"data"})

    def test_numpy_array_raises(self):
        with pytest.raises(TypeError):
            canonicalize_config({"x": np.array([1, 2, 3])})

    def test_set_raises(self):
        with pytest.raises(TypeError):
            canonicalize_config({"x": {1, 2, 3}})

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            canonicalize_config({"x": float("nan")})

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            canonicalize_config({"x": float("inf")})

    def test_numpy_scalar_same_as_python(self):
        """numpy scalars should hash the same as their Python equivalents."""
        a = {"x": np.float64(1.5)}
        b = {"x": 1.5}
        assert canonicalize_config(a) == canonicalize_config(b)

    def test_numpy_int_same_as_python(self):
        a = {"x": np.int64(42)}
        b = {"x": 42}
        assert canonicalize_config(a) == canonicalize_config(b)


class TestHashConfig:
    def test_deterministic(self):
        config = {"objective": "max_sharpe", "target_std": 0.05}
        assert hash_config(config) == hash_config(config)

    def test_length(self):
        h = hash_config({"a": 1})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_configs_different_hashes(self):
        h1 = hash_config({"objective": "max_sharpe"})
        h2 = hash_config({"objective": "max_mean"})
        assert h1 != h2


class TestHashAndSerialize:
    def test_round_trip(self):
        config = {"objective": "max_sharpe", "target_std": 0.05}
        config_hash, config_json = hash_and_serialize(config)
        assert len(config_hash) == 16
        assert isinstance(config_json, str)
        # Re-hashing the same config gives the same result
        h2, j2 = hash_and_serialize(config)
        assert config_hash == h2
        assert config_json == j2
