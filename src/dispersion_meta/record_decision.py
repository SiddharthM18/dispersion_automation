"""Accept/reject decision recording.

Thin wrapper around io.append_decisions with input validation.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from . import io


def record_decision(
    *,
    propose_date: date,
    config_hash: str,
    product: str,
    decision: str,
    notes: str | None = None,
    modified_config_hash: str | None = None,
) -> None:
    """Record a single accept/reject/modified decision.

    Parameters
    ----------
    propose_date : date
        The date of the proposal being decided on.
    config_hash : str
        The config_hash of the proposal.
    product : str
        Product identifier.
    decision : str
        One of 'accepted', 'rejected', 'modified'.
    notes : str | None
        Optional free-text notes.
    modified_config_hash : str | None
        Required when decision='modified'; the hash of the replacement config.
    """
    # Validate proposal exists
    proposals = io.read_proposals(
        start_date=propose_date, end_date=propose_date, products=[product],
    )
    if proposals is None or len(proposals) == 0:
        raise ValueError(
            f"No proposals found for {propose_date} / {product}"
        )

    match = proposals.filter(pl.col("config_hash") == config_hash)
    if len(match) == 0:
        raise ValueError(
            f"No proposal with config_hash={config_hash} found for "
            f"{propose_date} / {product}"
        )

    row = {
        "date": propose_date,
        "product": product,
        "config_hash": config_hash,
        "decision": decision,
        "decided_at_utc": datetime.now(timezone.utc),
        "notes": notes,
        "modified_config_hash": modified_config_hash,
    }

    io.append_decisions(pl.DataFrame([row]))
