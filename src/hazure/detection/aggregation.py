"""Checking that an aggregation measures what a detector needs of it."""

from __future__ import annotations

from typing import Final

#: Statistics that summarise where a window sits, for :class:`SpikeDetector`.
_CENTRE_AGGS: Final = ("median", "mean")


#: Statistics that summarise how much a window varies, for
#: :class:`VolatilityShiftDetector`.
_SPREAD_AGGS: Final = ("std", "var", "iqr", "idr")


def _check_agg(agg: object, allowed: tuple[str, ...], detector: str) -> None:
    """Reject an aggregation that does not measure what the detector needs."""
    if agg not in allowed:
        msg = f"{detector} agg={agg!r} must be one of {list(allowed)}."
        raise ValueError(msg)
