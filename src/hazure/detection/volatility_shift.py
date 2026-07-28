"""Flagging the moment the series starts varying by a different amount."""

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
    "VolatilityShiftDetector",
]


from hazure.detection.aggregation import _SPREAD_AGGS, _check_agg


class VolatilityShiftDetector(SignedScoreDetector):
    """Flag the point at which the series becomes more or less erratic.

    The same two symmetric windows as level-shift detection, with two changes
    that matter. The statistic measures spread rather than position, so the level
    can stay put while the noise around it changes. And the comparison is
    *relative* — the change in spread divided by the earlier spread — because a
    doubling of noise is equally significant on a quiet series and a loud one,
    which an absolute difference would not capture.

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
      step registers here as well as in :class:`LevelShiftDetector`. Running both
      is how the two are told apart.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> quiet, loud = rng.normal(scale=0.1, size=40), rng.normal(scale=5.0, size=40)
    >>> time = np.arange("2024-01-01", "2024-03-21", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, np.concatenate([quiet, loud]))
    >>> labels = VolatilityShiftDetector(window=10).fit_detect(ts)
    >>> bool(labels.values.ravel()[40] == 1.0)
    True
    """

    def __init__(
        self,
        window: Window | tuple[Window, Window],
        factor: Factor = 6.0,
        side: Side = "both",
        min_periods: int | tuple[int | None, int | None] | None = None,
        agg: str = "std",
    ) -> None:
        self.window = window
        self.factor = factor
        self.side = side
        self.min_periods = min_periods
        self.agg = agg
        self._build()

    def _build(self) -> None:
        super()._build()
        _check_agg(self.agg, _SPREAD_AGGS, "VolatilityShiftDetector")
        self.scorer = DoubleRollingScorer(
            window=self.window,
            agg=self.agg,
            diff="rel_diff",
            min_periods=self.min_periods,
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
