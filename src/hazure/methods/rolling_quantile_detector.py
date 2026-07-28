"""Flagging the points that leave a rolling quantile band by an unusual margin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection import ScoreDetector
from hazure.methods.rolling_quantile_scorer import RollingQuantileScorer
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from hazure._core.window import Window
    from hazure.thresholds import Factor


__all__ = [
    "RollingQuantileDetector",
]


class RollingQuantileDetector(ScoreDetector):
    """Flag the points that leave a normal range which follows the series.

    :class:`RollingQuantileScorer` paired with a one-sided inter-quartile-range
    rule on the excursion. What this buys over a global quantile rule is a range
    that moves: a series drifting upwards over a month spends the second half of
    it above any fixed upper quantile, so a global rule reports the drift, while a
    rolling band follows the level and reports only departures from it.

    Parameters
    ----------
    window
        Observations (``int``) or duration defining the band. Wide enough that the
        quantiles are estimable, narrow enough to track the drift.
    low
        Lower quantile of the band, in ``[0, 1]``.
    high
        Upper quantile of the band, in ``[0, 1]``.
    factor
        Inter-quartile-range factor deciding how large an excursion beyond the
        band is too large. One-sided: a point inside the band is never anomalous.

    Raises
    ------
    ValueError
        A quantile falls outside ``[0, 1]``, or ``low`` exceeds ``high``.

    Notes
    -----
    Why a fitted fence rather than "outside the band at all": the window trails
    and **includes the point being scored**, so on any series with a trend the
    newest observation is routinely the largest in its own window and sits at the
    band's own edge. Treating every non-zero excursion as an anomaly flags a
    sizeable fraction of a plainly ordinary series — 17 of the 40 points in the
    example below. The excursions are a distribution like any other, and what
    matters is an excursion out of proportion to the rest of them.

    Consequently the band's quantiles and ``factor`` do different jobs. The band
    decides what counts as an excursion at all; ``factor`` decides which
    excursions get reported. Widening the band shrinks every excursion toward
    zero, which eventually leaves the fence with nothing to separate — a window
    spanning most of the series flags nothing, correctly.

    An ``int`` window leaves the first ``window - 1`` observations NaN, since the
    band is not yet estimable. A duration window does not: it reports from the
    first observation, on however few observations the duration covers.

    Examples
    --------
    A series drifting upwards, with one point that breaks the drift:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> from hazure.methods import RollingQuantileDetector
    >>> values = np.arange(40.0) + np.tile([0.0, 0.5, -0.5, 0.2], 10)
    >>> values[26] += 12.0
    >>> time = np.arange("2024-01-01", "2024-02-10", dtype="datetime64[D]")
    >>> labels = RollingQuantileDetector(window=10).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([26])
    """

    def __init__(
        self,
        window: Window,
        low: float = 0.05,
        high: float = 0.95,
        factor: Factor = 3.0,
    ) -> None:
        self.window = window
        self.low = low
        self.high = high
        self.factor = factor
        self._build()

    def _build(self) -> None:
        """Rebuild the scorer and the threshold from the current parameters."""
        self.scorer = RollingQuantileScorer(
            window=self.window, low=self.low, high=self.high
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
