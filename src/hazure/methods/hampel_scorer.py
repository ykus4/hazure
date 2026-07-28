"""Distance from a rolling median, in rolling robust standard deviations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from hazure import BaseScorer, rolling
from hazure.thresholds import MAD_SCALE

if TYPE_CHECKING:
    from hazure import TimeSeries
    from hazure._core.window import Window

__all__ = [
    "HampelScorer",
]


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
