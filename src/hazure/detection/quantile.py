"""Flagging values outside a percentile range of the fitted data."""

from __future__ import annotations

from hazure.detection.score import ScoreDetector
from hazure.thresholds import (
    QuantileThreshold,
)

__all__ = [
    "QuantileDetector",
]


class QuantileDetector(ScoreDetector):
    """Flag values in the tails of the training distribution.

    Makes no assumption about the shape of that distribution, only about how much
    of it is acceptable: ``high=0.99`` means "the top one per cent of what we have
    seen is worth a look".

    Parameters
    ----------
    low
        Lower quantile in ``[0, 1]``. None leaves the lower side unbounded.
    high
        Upper quantile in ``[0, 1]``. None leaves the upper side unbounded.

    Raises
    ------
    ValueError
        Both quantiles are None, or one falls outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 90])
    >>> time = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> QuantileDetector(high=0.9).fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    """

    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        self.low = low
        self.high = high
        self._build()

    def _build(self) -> None:
        self.scorer = None
        self.threshold = QuantileThreshold(low=self.low, high=self.high)
