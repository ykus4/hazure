"""Flagging points an outlier model of yours rejects."""

from __future__ import annotations

from typing import Any

from hazure.detection.min_cluster import _MEMBERSHIP_CUTOFF
from hazure.detection.multivariate_score import MultivariateScoreDetector
from hazure.scoring import (
    OutlierScorer,
)
from hazure.thresholds import FixedThreshold

__all__ = [
    "OutlierDetector",
]


class OutlierDetector(MultivariateScoreDetector):
    """Flag points a general-purpose outlier model rejects.

    Treats each observation as a point in as many dimensions as there are columns
    and ignores the time axis entirely, which is the right trade when the anomaly
    is a combination of readings rather than a moment in a sequence. Any outlier
    model marking outliers with ``-1`` can be used.

    Parameters
    ----------
    model
        An outlier model with either ``fit(X)`` and ``predict(X)``, or
        ``fit_predict(X)`` alone. A model that only offers ``fit_predict`` can
        judge a batch only against itself, so it is re-run on every series rather
        than carrying a learned notion of normal.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> class FarFromCentre:
    ...     def fit(self, X):
    ...         self.centre_ = np.median(X, axis=0)
    ...         return self
    ...     def predict(self, X):
    ...         return np.where(np.abs(X - self.centre_).sum(axis=1) > 5.0, -1, 1)
    >>> pairs = np.column_stack([[0.0, 1, 0, 1, 20], [0.0, 1, 1, 0, 20]])
    >>> time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, pairs, ["a", "b"])
    >>> OutlierDetector(FarFromCentre()).fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 1.])
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._build()

    def _build(self) -> None:
        self.scorer = OutlierScorer(model=self.model)
        self.threshold = FixedThreshold(high=_MEMBERSHIP_CUTOFF)
