"""Flagging values outside an inter-quartile fence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection.score import ScoreDetector
from hazure.thresholds import (
    IqrThreshold,
)

if TYPE_CHECKING:
    from hazure.thresholds.fence import FactorSpec

__all__ = [
    "IqrDetector",
]


class IqrDetector(ScoreDetector):
    """Flag values far outside the training inter-quartile range.

    The box-plot rule, and a sound default when nothing is known about the
    distribution: because quartiles ignore the tails, the outliers being looked
    for do not widen the range that is supposed to exclude them.

    Parameters
    ----------
    factor
        One factor for both tails, or ``(low, high)``. ``None`` on a side leaves
        that side unbounded.

    Raises
    ------
    ValueError
        A factor is negative, or the pair is not of length two.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.array([10.0, 11, 12, 11, 10, 12, 11, 10, 11, 60])
    >>> time = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> IqrDetector().fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    """

    def __init__(self, factor: FactorSpec = 3.0) -> None:
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = None
        self.threshold = IqrThreshold(factor=self.factor)
