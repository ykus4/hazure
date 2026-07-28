"""Flagging points a clustering model of yours puts in the smallest group."""

from __future__ import annotations

from typing import Any, Final

from hazure.detection.multivariate_score import MultivariateScoreDetector
from hazure.scoring import (
    MinClusterScorer,
)
from hazure.thresholds import FixedThreshold

__all__ = [
    "MinClusterDetector",
]


#: Cut-off for the 1.0 / 0.0 membership scores the model adapters produce. Any
#: value between the two works; a half keeps the intent obvious.
_MEMBERSHIP_CUTOFF: Final = 0.5


class MinClusterDetector(MultivariateScoreDetector):
    """Flag points that fall in the rarest cluster.

    Clusters the observations, treating each as a point in as many dimensions as
    there are columns, and calls the smallest group anomalous. Nothing needs to be
    said about what anomalous looks like: the shape of the data decides, which
    makes this the detector to reach for when the failure mode is unknown but
    known to be rare.

    Parameters
    ----------
    model
        A clustering model with ``fit_predict(X)`` returning one integer label per
        row, and ``predict(X)`` to place new rows in the clusters it found.

    Raises
    ------
    ValueError
        The model has no ``predict`` method, so its clusters could never be
        applied to another series.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> class NearestOfTwo:
    ...     def fit_predict(self, X):
    ...         self.split_ = X.mean()
    ...         return self.predict(X)
    ...     def predict(self, X):
    ...         return (X.mean(axis=1) > self.split_).astype(int)
    >>> pairs = np.column_stack([[1.0, 1, 1, 1, 1, 9], [2.0, 2, 2, 2, 2, 9]])
    >>> time = np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, pairs, ["a", "b"])
    >>> MinClusterDetector(NearestOfTwo()).fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 1.])
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self._build()

    def _build(self) -> None:
        self.scorer = MinClusterScorer(model=self.model)
        # The scorer already answers 1.0 or 0.0, so the threshold only has to
        # separate the two.
        self.threshold = FixedThreshold(high=_MEMBERSHIP_CUTOFF)
