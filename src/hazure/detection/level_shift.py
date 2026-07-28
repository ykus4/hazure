"""Flagging the moment the series settles at a different level."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection.signed_score import SignedScoreDetector
from hazure.scoring import (
    DoubleRollingScorer,
)
from hazure.thresholds import (
    IqrThreshold,
)

if TYPE_CHECKING:
    from hazure._core.window import Window
    from hazure.detection.side import Side
    from hazure.thresholds.fence import Factor

__all__ = [
    "LevelShiftDetector",
]


class LevelShiftDetector(SignedScoreDetector):
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
    >>> labels = LevelShiftDetector(window=3).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([19, 20, 21])
    """

    def __init__(
        self,
        window: Window | tuple[Window, Window],
        factor: Factor = 6.0,
        side: Side = "both",
        min_periods: int | tuple[int | None, int | None] | None = None,
    ) -> None:
        self.window = window
        self.factor = factor
        self.side = side
        self.min_periods = min_periods
        self._build()

    def _build(self) -> None:
        super()._build()
        self.scorer = DoubleRollingScorer(
            window=self.window,
            agg="median",
            diff="diff",
            min_periods=self.min_periods,
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
