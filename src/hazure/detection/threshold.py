"""Flagging values outside limits you supply."""

from __future__ import annotations

from typing import ClassVar

from hazure.detection.score import ScoreDetector
from hazure.thresholds import (
    FixedThreshold,
)

__all__ = [
    "ThresholdDetector",
]


# ---------------------------------------------------------------------------
# the value of a point, judged on its own
# ---------------------------------------------------------------------------


class ThresholdDetector(ScoreDetector):
    """Flag values outside a range the caller supplies.

    The simplest possible detector, and the only one that learns nothing: use it
    when the acceptable range is known in advance. There is no scorer, because a
    value is already the quantity being judged.

    Parameters
    ----------
    low
        Values below this are anomalous. None leaves the lower side unbounded.
    high
        Values above this are anomalous. None leaves the upper side unbounded.

    Raises
    ------
    ValueError
        Both bounds are None, which would make the detector inert.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [20.0, 21.0, 45.0, 19.0, -5.0])
    >>> ThresholdDetector(low=0.0, high=40.0).detect(ts).values.ravel()
    array([0., 0., 1., 0., 1.])
    """

    trainable: ClassVar[bool] = False

    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        self.low = low
        self.high = high
        self._build()

    def _build(self) -> None:
        self.scorer = None
        self.threshold = FixedThreshold(low=self.low, high=self.high)
