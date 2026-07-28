"""How far outside a rolling quantile band each point falls."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from hazure import BaseScorer, rolling

if TYPE_CHECKING:
    from hazure import TimeSeries
    from hazure._core.window import Window

__all__ = [
    "RollingQuantileScorer",
]


class RollingQuantileScorer(BaseScorer):
    """Score each point by how far outside a rolling quantile band it sits.

    The band is the ``low`` and ``high`` quantiles of the window ending at each
    point; the score is the distance beyond whichever edge was crossed, and zero
    inside the band. Two order statistics per point, nothing fitted.

    What this buys over a quantile of the whole series is a normal range that
    moves. A series that drifts upwards over a month spends the second half of it
    above any global upper quantile, and a global rule then reports the drift
    rather than the anomalies; a rolling band follows the level and keeps
    reporting only departures from it.

    Parameters
    ----------
    window
        Observations (``int``) or duration defining the band. Wide enough that
        the quantiles are estimable, narrow enough to track the drift.
    low
        Lower quantile of the band, in ``[0, 1]``.
    high
        Upper quantile of the band, in ``[0, 1]``.

    Raises
    ------
    ValueError
        A quantile falls outside ``[0, 1]``, or ``low`` exceeds ``high``.

    Notes
    -----
    The window trails, and it includes the point being scored. A point extreme
    enough to move the band's own edge therefore scores lower than it otherwise
    would, which is a conservative bias — the band is harder to leave, not easier.
    Widening the window weakens the effect.

    The score is a magnitude in the units of the series, and it is zero, not NaN,
    inside the band: "inside the normal range" is a measurement, not a missing
    one.

    Nothing is learned; :meth:`fit` is optional.

    Examples
    --------
    A series drifting upwards with one point that breaks the drift:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.arange(40.0) + np.tile([0.0, 0.5, -0.5, 0.2], 10)
    >>> values[26] += 12.0
    >>> time = np.arange("2024-01-01", "2024-02-10", dtype="datetime64[D]")
    >>> scores = RollingQuantileScorer(window=10).score(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> int(np.nanargmax(scores.values.ravel()))
    26
    """

    trainable: ClassVar[bool] = False

    def __init__(self, window: Window, low: float = 0.05, high: float = 0.95) -> None:
        _check_band(low, high)
        self.window = window
        self.low = low
        self.high = high

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        _check_band(self.low, self.high)
        column = ts.values[:, 0]
        lower = rolling(column, self.window, "quantile", time=ts.time, q=self.low)
        upper = rolling(column, self.window, "quantile", time=ts.time, q=self.high)
        # NaN in either edge or in the point itself propagates through both
        # maxima, so an unmeasurable band stays unknown rather than reading as
        # "inside".
        outside = np.maximum(column - upper, lower - column)
        return ts.wrap(np.maximum(outside, 0.0))


def _check_band(low: float, high: float) -> None:
    """Reject a quantile band that does not describe an interval.

    Parameters
    ----------
    low
        Lower quantile.
    high
        Upper quantile.

    Raises
    ------
    ValueError
        A quantile is outside ``[0, 1]``, or the pair is inverted.
    """
    for name, value in (("low", low), ("high", high)):
        if not 0.0 <= value <= 1.0:
            msg = (
                f"{name}={value} is not a quantile; it must lie in [0, 1]. For an "
                f"absolute band use a fixed threshold on the values themselves."
            )
            raise ValueError(msg)
    if low > high:
        msg = f"low={low} is above high={high}, which describes an empty band."
        raise ValueError(msg)
