"""Detectors for a point unlike its own neighbourhood, when normal drifts.

A global range is the wrong yardstick for a series whose level wanders: the drift
itself would be reported. These judge each point against local order statistics
— a rolling median and spread, a rolling quantile band — that follow the series
and that the outliers being looked for cannot inflate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure._core import Detector
from hazure.detectors._common import upper_fence
from hazure.scorers import HampelScorer, RollingQuantileScorer
from hazure.thresholds import FixedThreshold

if TYPE_CHECKING:
    from hazure._core import Window
    from hazure.thresholds import Factor

__all__ = [
    "hampel",
    "rolling_quantile",
]


def hampel(window: Window = 7, factor: float = 3.0, center: bool = True) -> Detector:
    """Flag points too far from the local median to be part of the local noise.

    :class:`~hazure.scorers.HampelScorer` with a fixed cut-off. The cut-off is
    fixed rather than learned because the score is already expressed in standard
    deviations of the local noise: ``factor=3.0`` means "three sigma away from
    where this stretch of the series sits", which is the Hampel filter's own rule
    and needs no reference to the distribution of the scores.

    Parameters
    ----------
    window
        Observations or duration making up each point's neighbourhood.
    factor
        How many local standard deviations away is too far.
    center
        Centre the window on each point rather than trailing it.

    Returns
    -------
    Detector
        The Hampel score against ``FixedThreshold(high=factor)``. Nothing is
        learned, so it can be used without fitting.

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
    >>> labels = hampel().detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([17])
    """
    return Detector(
        HampelScorer(window=window, center=center), FixedThreshold(high=factor)
    )


def rolling_quantile(
    window: Window, low: float = 0.05, high: float = 0.95, factor: Factor = 3.0
) -> Detector:
    """Flag the points that leave a normal range which follows the series.

    :class:`~hazure.scorers.RollingQuantileScorer` paired with a one-sided
    inter-quartile-range rule on the excursion. What this buys over a global
    quantile rule is a range that moves: a series drifting upwards over a month
    spends the second half of it above any fixed upper quantile, so a global rule
    reports the drift, while a rolling band follows the level and reports only
    departures from it.

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

    Returns
    -------
    Detector
        The excursion beyond the band, fenced above.

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
    >>> values = np.arange(40.0) + np.tile([0.0, 0.5, -0.5, 0.2], 10)
    >>> values[26] += 12.0
    >>> time = np.arange("2024-01-01", "2024-02-10", dtype="datetime64[D]")
    >>> labels = rolling_quantile(window=10).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([26])
    """
    scorer = RollingQuantileScorer(window=window, low=low, high=high)
    return Detector(scorer, upper_fence(factor))
