"""Flagging a point unlike the window just before it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection.aggregation import _CENTRE_AGGS, _check_agg
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
    "SpikeDetector",
]


# ---------------------------------------------------------------------------
# the value of a point, judged against its neighbourhood
# ---------------------------------------------------------------------------


class SpikeDetector(SignedScoreDetector):
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
    >>> np.flatnonzero(SpikeDetector().fit_detect(ts).values.ravel() == 1.0)
    array([12, 13])
    >>> labels = SpikeDetector(side="positive").fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([12])
    """

    def __init__(
        self,
        window: Window = 1,
        factor: Factor = 3.0,
        side: Side = "both",
        min_periods: int | None = None,
        agg: str = "median",
    ) -> None:
        self.window = window
        self.factor = factor
        self.side = side
        self.min_periods = min_periods
        self.agg = agg
        self._build()

    def _build(self) -> None:
        super()._build()
        _check_agg(self.agg, _CENTRE_AGGS, "SpikeDetector")
        self.scorer = DoubleRollingScorer(
            window=(self.window, 1),
            agg=self.agg,
            diff="diff",
            min_periods=(self.min_periods, 1),
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
