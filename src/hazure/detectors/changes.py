"""Detectors for a change across a point: a spike, a level shift, a volatility shift.

All three compare the window before each point with the window from it onwards,
through :class:`~hazure.transformers.DoubleRollingAggregate`, and only the
settings differ:

* **spike** — a long left window and a right window of 1, so a single blip is
  measured against a stable notion of recent normal;
* **level shift** — two windows of equal length, long enough that both sides are
  stable, so a persistent change registers and a lone spike does not;
* **volatility shift** — the same symmetric windows with a dispersion statistic
  and a relative comparison, because a doubling of noise matters equally whether
  the series is quiet or loud.

The difference is signed, so each detector can report rises only or falls only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure._core import Detector
from hazure.detectors._common import CENTRE_AGGS, SPREAD_AGGS, check_agg, signed_fence
from hazure.scorers import AsScorer
from hazure.transformers import DoubleRollingAggregate

if TYPE_CHECKING:
    from hazure._core import Window
    from hazure.thresholds import Factor, Side

__all__ = [
    "level_shift",
    "spike",
    "volatility_shift",
]


def spike(
    window: Window = 1,
    factor: Factor = 3.0,
    side: Side = "both",
    min_periods: int | None = None,
    agg: str = "median",
) -> Detector:
    """Flag points that depart sharply from the values just before them.

    The window in front of each point is one observation wide and the window
    behind it is ``window`` wide: a short right window catches the blip while the
    long left window keeps a stable notion of recent normal, which is what makes
    this asymmetry the shape of spike detection. Because the comparison is local,
    a slow drift is invisible to it — which is the point.

    Parameters
    ----------
    window
        Size of the preceding window: observations (``int``) or a duration.
        The default of 1 compares each point with the one before it.
    factor
        Inter-quartile-range factor deciding how large a departure is too large.
    side
        ``"both"``, ``"positive"`` for jumps up only, ``"negative"`` for drops
        only.
    min_periods
        Minimum non-missing observations in the preceding window.
    agg
        How to summarise the preceding window: ``"median"`` or ``"mean"``. The
        median is unmoved by an earlier spike still inside the window.

    Returns
    -------
    Detector
        The signed difference between each point and the window before it,
        fenced on its magnitude.

    Raises
    ------
    ValueError
        ``side`` or ``agg`` is not one of the listed choices.

    Notes
    -----
    With the default ``window=1`` the preceding window is a single observation, so
    a one-point spike changes the score twice — once on the way up and once on the
    way back down — and ``side="both"`` flags both the spike and the point after
    it. ``side="positive"`` isolates the spike itself. A wider window has a median
    the spike cannot move, and then the spike alone scores.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.ones(20)
    >>> values[12] = 9.0
    >>> time = np.arange("2024-01-01", "2024-01-21", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> np.flatnonzero(spike().fit_detect(ts).values.ravel() == 1.0)
    array([12, 13])
    >>> labels = spike(side="positive").fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([12])
    """
    check_agg(agg, CENTRE_AGGS, "spike")
    scorer = AsScorer(
        DoubleRollingAggregate(
            window=(window, 1), agg=agg, diff="diff", min_periods=(min_periods, 1)
        )
    )
    return Detector(scorer, signed_fence(factor, side))


def level_shift(
    window: Window | tuple[Window, Window],
    factor: Factor = 6.0,
    side: Side = "both",
    min_periods: int | tuple[int | None, int | None] | None = None,
) -> Detector:
    """Flag the point at which the series settles at a new level.

    Two windows of equal length, one either side of each point, are summarised
    and compared. Both being long is what separates a level shift from a spike:
    a single odd value barely moves the median of a wide window, while a genuine
    step moves one window's median entirely away from the other's.

    Parameters
    ----------
    window
        Size of each window, or ``(left, right)``. Long enough that both sides
        are stable, short enough to place the change precisely.
    factor
        Inter-quartile-range factor deciding how large a shift is too large. Set
        higher than for spike detection by default, because the difference of two
        window medians is a much quieter signal than a single point's departure.
    side
        ``"both"``, ``"positive"`` for shifts up only, ``"negative"`` for shifts
        down only.
    min_periods
        Minimum non-missing observations per window, or ``(left, right)``.

    Returns
    -------
    Detector
        The signed difference between the medians of the two windows, fenced on
        its magnitude.

    Raises
    ------
    ValueError
        ``side`` is not one of the three directions.

    Notes
    -----
    The two windows overlap the shift for as long as it takes them to clear it,
    so the score stays high for a run of points around the change and the
    detector flags a short plateau rather than a single instant. Narrowing
    ``window`` narrows the plateau.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.concatenate([np.zeros(20), np.full(20, 10.0)])
    >>> time = np.arange("2024-01-01", "2024-02-10", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> labels = level_shift(window=3).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([19, 20, 21])
    """
    scorer = AsScorer(
        DoubleRollingAggregate(
            window=window, agg="median", diff="diff", min_periods=min_periods
        )
    )
    return Detector(scorer, signed_fence(factor, side))


def volatility_shift(
    window: Window | tuple[Window, Window],
    factor: Factor = 6.0,
    side: Side = "both",
    min_periods: int | tuple[int | None, int | None] | None = None,
    agg: str = "std",
) -> Detector:
    """Flag the point at which the series becomes more or less erratic.

    The same two symmetric windows as :func:`level_shift`, with two changes that
    matter. The statistic measures spread rather than position, so the level can
    stay put while the noise around it changes. And the comparison is *relative*
    — the change in spread divided by the earlier spread — because a doubling of
    noise is equally significant on a quiet series and a loud one, which an
    absolute difference would not capture.

    Parameters
    ----------
    window
        Size of each window, or ``(left, right)``. Wide enough that a spread can
        be estimated from each side.
    factor
        Inter-quartile-range factor deciding how large a relative change is too
        large.
    side
        ``"both"``, ``"positive"`` for increases in volatility only,
        ``"negative"`` for decreases only.
    min_periods
        Minimum non-missing observations per window, or ``(left, right)``.
    agg
        How to measure spread: ``"std"``, ``"var"``, ``"iqr"`` or ``"idr"``.

    Returns
    -------
    Detector
        The relative change in spread across each point, fenced on its
        magnitude.

    Raises
    ------
    ValueError
        ``side`` or ``agg`` is not one of the listed choices.

    Notes
    -----
    A spread is never negative, so the sign of the relative change is the sign of
    the change itself and ``side`` reads as expected. A window with no spread at
    all makes the relative change undefined, and those points score NaN.

    Two consequences of measuring spread over a window are worth knowing:

    * A relative *increase* is unbounded while a relative *decrease* cannot pass
      -1, so a fall in volatility produces a smaller score than the equivalent
      rise. Detecting ``side="negative"`` usually wants a smaller ``factor``.
    * A level shift falling inside a window inflates that window's spread, so a
      step registers here as well as in :func:`level_shift`. Running both is how
      the two are told apart.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> quiet, loud = rng.normal(scale=0.1, size=40), rng.normal(scale=5.0, size=40)
    >>> time = np.arange("2024-01-01", "2024-03-21", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, np.concatenate([quiet, loud]))
    >>> labels = volatility_shift(window=10).fit_detect(ts)
    >>> bool(labels.values.ravel()[40] == 1.0)
    True
    """
    check_agg(agg, SPREAD_AGGS, "volatility_shift")
    scorer = AsScorer(
        DoubleRollingAggregate(
            window=window, agg=agg, diff="rel_diff", min_periods=min_periods
        )
    )
    return Detector(scorer, signed_fence(factor, side))
