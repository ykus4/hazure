"""Flagging values a significance test calls too extreme."""

from __future__ import annotations

from hazure.detection.score import ScoreDetector
from hazure.thresholds import (
    EsdThreshold,
)

__all__ = [
    "EsdDetector",
]


class EsdDetector(ScoreDetector):
    """Flag values by the generalised extreme Studentized deviate test.

    Sets the line by a significance level rather than by a factor, which is
    useful when a false-positive rate is easier to justify than a multiple of a
    spread. Assumes the values are approximately normal; where that is doubtful,
    :class:`IqrDetector` asks less of the data.

    Parameters
    ----------
    alpha
        Significance level, in ``(0, 1)``.

    Raises
    ------
    ValueError
        ``alpha`` is not in ``(0, 1)``.
    ImportError
        SciPy is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(1)
    >>> values = rng.normal(loc=20.0, size=60)
    >>> values[42] = 30.0
    >>> time = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")
    >>> labels = EsdDetector().fit_detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([42])
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._build()

    def _build(self) -> None:
        self.scorer = None
        self.threshold = EsdThreshold(alpha=self.alpha)
