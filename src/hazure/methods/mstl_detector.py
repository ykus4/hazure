"""Flagging what an MSTL decomposition cannot explain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection import ScoreDetector
from hazure.methods.mstl_residual_scorer import MstlResidualScorer
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hazure.thresholds import Factor


__all__ = [
    "MstlDetector",
]


class MstlDetector(ScoreDetector):
    """Flag points that break none of several rhythms but still do not fit.

    :class:`MstlResidualScorer` paired with an inter-quartile-range rule. Use this
    rather than :class:`StlDetector` whenever the series has more than one
    rhythm: with a second cycle left in the residual, the residual's spread is set
    by that cycle rather than by the noise, and the threshold ends up asking how
    unusual a point is compared with a systematic pattern instead of compared with
    chance.

    Parameters
    ----------
    periods
        Cycle lengths in observations: one integer, or several.
    robust
        Reweight the loess fits to discount outliers.
    windows
        Seasonal smoother length per period.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.

    Raises
    ------
    ValueError
        The time axis is irregular, or a period is unusable.
    ImportError
        ``statsmodels`` is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> hours = np.arange(24 * 28)
    >>> values = (
    ...     10.0
    ...     + 3.0 * np.sin(hours * 2 * np.pi / 24)
    ...     + 5.0 * np.sin(hours * 2 * np.pi / 168)
    ... )
    >>> values[300] += 20.0
    >>> time = hours * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> labels = MstlDetector(periods=(24, 168), factor=25.0).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([300])
    """

    def __init__(
        self,
        periods: int | Sequence[int],
        robust: bool = True,
        windows: int | Sequence[int] | None = None,
        factor: Factor = 3.0,
    ) -> None:
        self.periods = periods
        self.robust = robust
        self.windows = windows
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = MstlResidualScorer(
            periods=self.periods, robust=self.robust, windows=self.windows
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
