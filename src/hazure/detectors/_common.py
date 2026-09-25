"""The choices the ready-made detectors make in common, made in one place.

Each of them pairs a scorer with a threshold, and while the scorers differ the
threshold is very often the same decision taken for the same reason. Stating it
here keeps the reason attached to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from hazure.thresholds import IqrThreshold, SignedThreshold

if TYPE_CHECKING:
    from hazure.thresholds import Factor, Side

#: Statistics that summarise where a window sits, for spike detection.
CENTRE_AGGS: Final = ("median", "mean")


#: Statistics that summarise how much a window varies, for volatility shifts.
SPREAD_AGGS: Final = ("std", "var", "iqr", "idr")


def check_agg(agg: object, allowed: tuple[str, ...], detector: str) -> None:
    """Reject an aggregation that does not measure what the detector needs.

    Which aggregations are admissible depends on the detector, so the message
    names it rather than only the parameter — the same ``agg="std"`` is right for
    one detector and wrong for another, and the reason is the detector.
    """
    if agg not in allowed:
        msg = f"{detector} agg={agg!r} must be one of {list(allowed)}."
        raise ValueError(msg)


def upper_fence(factor: Factor) -> IqrThreshold:
    """Return the one-sided fence the unsigned detectors end with.

    Their scorers report severity, not direction: a squared reconstruction
    error, the size of a change, a distance to the nearest neighbour. Small is
    the ordinary case and there is nothing to notice below the training
    quartiles, so only the upper tail carries a verdict and the lower fence is
    left off.

    Parameters
    ----------
    factor
        Inter-quartile-range factor for the upper fence, or ``None`` to leave
        the score unbounded above as well.

    Returns
    -------
    IqrThreshold
        A fence bounded above only.
    """
    return IqrThreshold(factor=(None, factor))


def signed_fence(factor: Factor, side: Side) -> SignedThreshold:
    """Return the fence the signed detectors end with.

    The same one-sided fence, applied to the magnitude of a signed score, with
    the direction of the score deciding whether a large one is reported.

    Parameters
    ----------
    factor
        Inter-quartile-range factor for the fence on the magnitude.
    side
        Which direction to report.

    Returns
    -------
    SignedThreshold
        The fence, wrapped to read the sign.
    """
    return SignedThreshold(upper_fence(factor), side=side)
