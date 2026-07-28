"""Membership of the smallest cluster a model of yours found."""

from __future__ import annotations

import copy
from collections import Counter
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hazure import BaseScorer
from hazure._core.series import complete_rows

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "MinClusterScorer",
]


class MinClusterScorer(BaseScorer):
    """Score each point by whether it falls in the rarest cluster.

    Clustering the observations and calling the smallest group anomalous needs no
    labels and no notion of a threshold — the shape of the data decides. The
    score is 1.0 for membership of the smallest cluster and 0.0 otherwise, so
    pairing it with ``FixedThreshold(high=0.5)`` turns it into labels.

    Parameters
    ----------
    model
        A clustering model with ``fit_predict(X)`` returning one integer label
        per row, and ``predict(X)`` to assign new rows to the clusters it found.

    Attributes
    ----------
    model_ : object
        The fitted clustering model.
    smallest_cluster_ : int or None
        Label of the cluster judged anomalous, or None when the training data
        formed a single cluster and so has no rare minority.

    Raises
    ------
    ValueError
        The model has no ``predict`` method, so the clusters it finds during
        fitting could never be assigned to another series.

    Notes
    -----
    Many clustering algorithms are transductive: they label the points they were
    given and cannot place a new one. Those cannot be used here, and the check
    happens at :meth:`fit`, where the fix — a model that generalises, such as
    k-means or a Gaussian mixture — is still actionable.

    Ties are broken towards the lowest cluster label, so the choice is
    deterministic when two clusters are equally small.

    A clustering that finds only one group has no smallest group, and every point
    scores 0. Calling the single cluster anomalous would flag the entire series.

    The model is deep-copied at :meth:`fit` time, leaving the caller's object
    untouched.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> class TwoMeans:
    ...     def fit_predict(self, X):
    ...         self.split_ = X[:, 0].mean()
    ...         return self.predict(X)
    ...     def predict(self, X):
    ...         return (X[:, 0] > self.split_).astype(int)
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]"),
    ...     np.column_stack([[0.0, 0, 0, 0, 0, 0, 9], [1.0, 1, 1, 1, 1, 1, 9]]),
    ...     ["a", "b"],
    ... )
    >>> MinClusterScorer(TwoMeans()).fit_score(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 1.])
    """

    multivariate: ClassVar[bool] = True

    model_: Any
    smallest_cluster_: int | None
    _trained: bool = True

    def __init__(self, model: Any) -> None:
        self.model = model

    def _learn(self, ts: TimeSeries) -> None:
        if not hasattr(self.model, "predict"):
            msg = (
                f"{type(self.model).__name__} has no predict() method, so the "
                f"clusters it finds cannot be assigned to any other series. Use "
                f"a clustering model that generalises to new points, such as "
                f"KMeans or GaussianMixture."
            )
            raise ValueError(msg)

        self.model_ = copy.deepcopy(self.model)
        self.smallest_cluster_ = None
        usable = complete_rows(ts.values)
        self._trained = bool(usable.any())
        if not self._trained:
            return

        labels = np.asarray(self.model_.fit_predict(ts.values[usable]))
        counts = Counter(int(label) for label in labels.ravel())
        if len(counts) > 1:
            # Ordering by (count, label) makes the tie-break deterministic.
            self.smallest_cluster_ = min(
                counts, key=lambda label: (counts[label], label)
            )

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores = np.full(ts.n_rows, np.nan, dtype=np.float64)
        usable = complete_rows(ts.values)
        if self._trained and usable.any():
            if self.smallest_cluster_ is None:
                scores[usable] = 0.0
            else:
                assigned = np.asarray(self.model_.predict(ts.values[usable])).ravel()
                scores[usable] = (assigned == self.smallest_cluster_).astype(np.float64)
        return ts.wrap(scores, ["min_cluster"])
