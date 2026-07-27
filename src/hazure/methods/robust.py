"""Local robust statistics: the Hampel filter and a rolling quantile band.

Both methods here judge a point against its own neighbourhood rather than against
the whole series, and both do it with order statistics rather than moments. That
combination is what makes them work on data a global rule mishandles:

* a **drifting** series has no single normal range, so a global centre is wrong
  almost everywhere, while a rolling one follows the drift;
* a **contaminated** window still has a usable median and median absolute
  deviation, where a mean and a standard deviation are pulled towards the very
  outlier they are meant to expose — one value a thousand times too large widens
  the standard deviation enough to hide itself.

Both are cheap: two rolling order statistics each, and no model to fit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from hazure import BaseScorer, rolling
from hazure.detection import ScoreDetector
from hazure.thresholds import MAD_SCALE, FixedThreshold

if TYPE_CHECKING:
    from hazure import TimeSeries
    from hazure._core.window import Window

__all__ = ["HampelDetector", "HampelScorer", "RollingQuantileScorer"]


class HampelScorer(BaseScorer):
    """Score each point by its distance from the local median, in local MADs.

    The Hampel filter [1]_. For every point, take the median of the window
    around it as the local notion of normal and the median absolute deviation of
    that window as the local notion of scale, then report

    ``|x - local median| / (1.4826 * local MAD)``.

    Both estimates are order statistics, so a single wild value moves neither:
    that is what lets the filter measure an outlier against a scale the outlier
    did not inflate.

    Parameters
    ----------
    window
        Observations (``int``) or duration (``"7d"``, ``timedelta``) making up
        each point's neighbourhood. Wide enough to estimate a scale from, narrow
        enough that the level is locally constant.
    center
        Centre the window on each point rather than trailing it. Centred is the
        default because a filter is normally applied retrospectively, and a
        two-sided neighbourhood judges a point without the level shift of the
        point itself dragging the centre. Set False for a causal score that uses
        only the past.

    Raises
    ------
    ValueError
        The window is not positive, or a duration window is used on a series
        whose time axis cannot support it.

    Notes
    -----
    The factor 1.4826 turns a median absolute deviation into an estimate of the
    standard deviation of a normal sample, so the score reads on the familiar
    "number of sigmas" scale: for X ~ N(mu, sigma) the median of ``|X - mu|`` is
    ``0.6745 * sigma``, and 1 / 0.6745 = 1.4826. Without it, a score of 3 would
    mean three MADs, which is only two standard deviations, and every published
    Hampel threshold would read wrong.

    A window with no spread at all — a locally constant stretch — has no scale, so
    the distance cannot be expressed in units of it. Those points score NaN,
    "undefined", rather than infinity: a flat window is a statement about the
    absence of information, not about the size of a departure.

    The spread is the rolling median of ``|x - local median|``, which keeps both
    passes to a single vectorised rolling call over bounded memory. Where the level
    is locally stable — the case the filter is designed for — every point in a
    window shares that window's median, and this agrees exactly with the classical
    definition of taking the deviations from one window's own centre.

    Points whose window holds fewer than a full complement of observations score
    NaN, which for a centred window is a margin of half a window at each end.
    Whether a point is scored is decided by the *centre*: the spread is taken from
    however many deviations its window contains, so the second pass does not widen
    that margin to a whole window.

    Nothing is learned; :meth:`fit` is optional.

    References
    ----------
    .. [1] F. R. Hampel, "The Influence Curve and its Role in Robust
       Estimation", Journal of the American Statistical Association 69(346),
       1974, pp. 383-393.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([10.0, 11.0, 12.0, 11.0], 8)
    >>> values[17] = 40.0
    >>> time = np.arange("2024-01-01", "2024-02-02", dtype="datetime64[D]")
    >>> scores = HampelScorer().score(TimeSeries.from_arrays(time, values))
    >>> int(np.nanargmax(scores.values.ravel()))
    17
    """

    trainable: ClassVar[bool] = False

    def __init__(self, window: Window = 7, center: bool = True) -> None:
        self.window = window
        self.center = center

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        column = ts.values[:, 0]
        centre = rolling(
            column, self.window, "median", time=ts.time, center=self.center
        )
        deviation = np.abs(column - centre)
        # min_periods=1 on the spread: the centre has already decided which points
        # are judgeable, and requiring a full window of *deviations* as well would
        # blank a whole window at each end instead of half of one.
        spread = MAD_SCALE * rolling(
            deviation,
            self.window,
            "median",
            time=ts.time,
            center=self.center,
            min_periods=1,
        )
        scores = np.divide(
            deviation,
            spread,
            out=np.full(ts.n_rows, np.nan),
            where=spread > 0.0,
        )
        return ts.wrap(scores)


class HampelDetector(ScoreDetector):
    """Flag points too far from the local median to be part of the local noise.

    :class:`HampelScorer` with a fixed cut-off. The cut-off is fixed rather than
    learned because the score is already expressed in standard deviations of the
    local noise: ``factor=3.0`` means "three sigma away from where this stretch
    of the series sits", which is the Hampel filter's own rule and needs no
    reference to the distribution of the scores.

    Nothing is learned, so this detector can be used without :meth:`fit`.

    Parameters
    ----------
    window
        Observations or duration making up each point's neighbourhood.
    factor
        How many local standard deviations away is too far.
    center
        Centre the window on each point rather than trailing it.

    Raises
    ------
    ValueError
        The window is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([10.0, 11.0, 12.0, 11.0], 8)
    >>> values[17] = 40.0
    >>> time = np.arange("2024-01-01", "2024-02-02", dtype="datetime64[D]")
    >>> labels = HampelDetector().detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([17])
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self, window: Window = 7, factor: float = 3.0, center: bool = True
    ) -> None:
        self.window = window
        self.factor = factor
        self.center = center
        self._build()

    def _build(self) -> None:
        self.scorer = HampelScorer(window=self.window, center=self.center)
        self.threshold = FixedThreshold(high=self.factor)


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
