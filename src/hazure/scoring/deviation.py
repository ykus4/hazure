"""Distance from a learned centre, in units of a learned spread.

Both statistics are fitted once and reused, so the yardstick does not move
when the data goes wrong.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np

from hazure import BaseScorer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "DeviationScorer",
]


#: Where :class:`DeviationScorer` puts the centre of normal.
Centre = Literal["median", "mean"]


#: What :class:`DeviationScorer` measures the deviation in units of.
Scale = Literal["iqr", "idr", "mad", "std"]


#: Scales a median absolute deviation into a standard-deviation estimate; see
#: :data:`hazure.thresholds.MAD_SCALE`.
_MAD_SCALE = 1.482602218505602


class DeviationScorer(BaseScorer):
    """Score each point by its signed distance from a learned centre.

    A robust z-score: ``(x - centre) / scale``, where both are learned once from
    the training series. With a median centre and an inter-quartile-range scale,
    neither estimate is moved by the outliers being looked for, which a mean and
    a standard deviation both are — a single value a thousand times too large
    inflates the scale enough to hide itself.

    The score keeps its sign, so a detector can act on excursions in one
    direction only.

    Parameters
    ----------
    center
        Where normal sits: ``"median"`` or ``"mean"``.
    scale
        What one unit of deviation is: ``"iqr"`` (quartile spread), ``"idr"``
        (10th-to-90th-percentile spread), ``"mad"`` (median absolute deviation,
        scaled to estimate a standard deviation) or ``"std"``.

    Attributes
    ----------
    center_ : float
        The learned centre.
    scale_ : float
        The learned scale. Zero for a constant training series.

    Raises
    ------
    ValueError
        ``center`` or ``scale`` is not one of the listed choices.

    Notes
    -----
    A constant training series has no spread, so every point equal to the centre
    scores 0 and anything else scores infinity. That is the honest reading: with
    no observed variation, any change at all is unprecedented.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [10.0, 12.0, 11.0, 13.0, 12.0, 10.0, 40.0])
    >>> scorer = DeviationScorer().fit(ts)
    >>> (scorer.center_, scorer.scale_)
    (12.0, 2.0)
    >>> scorer.score(ts).values.ravel()
    array([-1. ,  0. , -0.5,  0.5,  0. , -1. , 14. ])
    """

    center_: float
    scale_: float

    def __init__(self, center: Centre = "median", scale: Scale = "iqr") -> None:
        _check_choice(center, ("median", "mean"), "center")
        _check_choice(scale, ("iqr", "idr", "mad", "std"), "scale")
        self.center = center
        self.scale = scale

    def _learn(self, ts: TimeSeries) -> None:
        _check_choice(self.center, ("median", "mean"), "center")
        _check_choice(self.scale, ("iqr", "idr", "mad", "std"), "scale")

        column = ts.values[:, 0]
        observed = column[~np.isnan(column)]
        if observed.size == 0:
            self.center_ = self.scale_ = math.nan
            return

        self.center_ = float(
            np.median(observed) if self.center == "median" else observed.mean()
        )
        self.scale_ = _spread(observed, self.scale, self.center_)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        deviation = ts.values[:, 0] - self.center_
        if self.scale_ == 0.0:
            # No observed variation: a point on the centre is unremarkable and
            # anything else is unprecedented. Chosen rather than divided by zero,
            # which would turn a point on the centre into 0/0.
            scores = np.where(deviation > 0.0, np.inf, -np.inf)
            scores = np.where(deviation == 0.0, 0.0, scores)
            scores[np.isnan(deviation)] = np.nan
            return ts.wrap(scores)
        return ts.wrap(deviation / self.scale_)


def _spread(observed: NDArray[np.float64], scale: Scale, centre: float) -> float:
    """Measure the dispersion of ``observed`` the way ``scale`` asks for."""
    if scale == "std":
        # ddof=1, and a single observation therefore has no spread rather than a
        # spread of zero; treat that as zero so scoring stays defined.
        return 0.0 if observed.size < 2 else float(observed.std(ddof=1))
    if scale == "mad":
        return _MAD_SCALE * float(np.median(np.abs(observed - np.median(observed))))
    edges = (0.25, 0.75) if scale == "iqr" else (0.1, 0.9)
    low, high = np.quantile(observed, edges)
    return float(high - low)


def _check_choice(value: object, allowed: tuple[str, ...], name: str) -> None:
    """Reject a parameter that is not one of a small set of names."""
    if value not in allowed:
        msg = f"{name}={value!r} is not one of {list(allowed)}."
        raise ValueError(msg)
