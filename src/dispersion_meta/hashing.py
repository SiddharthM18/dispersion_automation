from __future__ import annotations

import hashlib
import json
import math

import numpy as np

# Schema version included in the hash payload so old hashes stay stable
# when config schema is extended.
_HASH_SCHEMA_VERSION = 1


def _canonicalize(obj: object) -> object:
    """Recursively canonicalize a config value for deterministic hashing.

    - Sort dict keys, drop None values (treat as absent).
    - Round floats to 10 decimals.
    - Check bool before int because isinstance(True, int) is True in Python.
    - Raise on non-canonicalizable types (bytes, sets, numpy arrays).
    - Accept numpy scalars (they subclass Python numeric types).
    """
    if obj is None:
        return None

    # bool MUST come before int — isinstance(True, int) is True
    if isinstance(obj, bool) and not isinstance(obj, np.bool_):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)  # collapse numpy int subtypes

    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Cannot canonicalize non-finite float: {obj}")
        return round(val, 10)

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in sorted(obj.items()) if v is not None}

    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]

    # Reject types that can't be deterministically serialized
    type_name = type(obj).__name__
    raise TypeError(
        f"Cannot canonicalize type {type_name}; "
        "config values must be str, int, float, bool, None, list, or dict"
    )


def canonicalize_config(config: dict) -> str:
    """Return the canonical JSON string for a config dict.

    This is the exact string that gets hashed — store it in config_json
    so the hash is always reproducible.
    """
    canonical = _canonicalize(config)
    # Wrap with schema version so hashes are stable across schema changes
    payload = {"_v": _HASH_SCHEMA_VERSION, "config": canonical}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def hash_config(config: dict) -> str:
    """SHA-256 of the canonical config, truncated to 16 hex chars (64 bits)."""
    canonical_json = canonicalize_config(config)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


def hash_and_serialize(config: dict) -> tuple[str, str]:
    """Return (config_hash, config_json) for a config dict.

    Convenience wrapper — the config_json is the exact string that was hashed,
    so storing both guarantees reproducibility.
    """
    config_json = canonicalize_config(config)
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:16]
    return config_hash, config_json
