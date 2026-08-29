"""The choices the compound detectors make in common, made in one place.

Each of them assembles a scorer and a threshold in ``_build``, and while the
scorers differ the threshold is very often the same decision taken for the same
reason. Stating it here keeps the reason attached to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from hazure.thresholds.fence import Factor

#: Statistics that summarise where a window sits, for :class:`SpikeDetector`.
_CENTRE_AGGS: Final = ("median", "mean")


#: Statistics that summarise how much a window varies, for
#: :class:`VolatilityShiftDetector`.
_SPREAD_AGGS: Final = ("std", "var", "iqr", "idr")


def _check_agg(agg: object, allowed: tuple[str, ...], detector: str) -> None:
    """Reject an aggregation that does not measure what the detector needs.

    Which aggregations are admissible depends on the detector, so the message
    names it rather than only the parameter — the same ``agg="std"`` is right for
    one detector and wrong for another, and the reason is the detector.
    """
    if agg not in allowed:
        msg = f"{detector} agg={agg!r} must be one of {list(allowed)}."
        raise ValueError(msg)


def _upper_iqr_threshold(factor: Factor) -> IqrThreshold:
    """Return the one-sided fence the unsigned compound detectors end with.

    Their scorers report severity, not direction: a squared reconstruction
    error, an absolute residual, the size of a shift. Small is the ordinary
    case and there is nothing to notice below the training quartiles, so only
    the upper tail carries a verdict and the lower fence is left off.

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
