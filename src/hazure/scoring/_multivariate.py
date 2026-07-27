"""Scorers that need every column at once.

These look for a point that is unremarkable in each column taken alone and
impossible taken together: a flow that no longer matches its pressure, a set of
readings that has left the subspace it has always lived in, a moment that falls
outside every cluster the data has ever formed. Because the anomaly is in the
relationship, no per-column scorer can see it.

Rows with any missing value are excluded from fitting and score NaN: a row-wise
model has nothing to say about a point it cannot fully observe.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hazure import BaseScorer
from hazure.features import PcaReconstructionError, RegressionResidual
from hazure.scoring._adapter import TransformerScorer, complete_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import BaseTransformer, TimeSeries
    from hazure.features import Regressor

__all__ = [
    "MinClusterScorer",
    "OutlierScorer",
    "PcaReconstructionErrorScorer",
    "RegressionResidualScorer",
]


class RegressionResidualScorer(TransformerScorer):
    """Score each point by what the other columns fail to predict about one.

    Where columns are physically linked — a valve position and the flow it
    causes, a request rate and the CPU it burns — a regression captures the link
    and the signed residual is what the link cannot explain. A large residual
    means the relationship itself broke, which no single column would reveal.

    Parameters
    ----------
    target
        Name of the column to predict. Every other column is a feature.
    regressor
        Any object with ``fit(X, y)`` and ``predict(X)`` taking numpy arrays.
        Defaults to ordinary least squares.

    Attributes
    ----------
    transformer_ : hazure.features.RegressionResidual
        The fitted regression stage, whose ``regressor_`` is the fitted model.

    Notes
    -----
    The regressor is deep-copied at :meth:`fit` time, so the caller's object is
    left unfitted and reusable.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> drive = np.arange(8.0)
    >>> follow = 2.0 * drive + 1.0
    >>> follow[5] += 10.0
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]"),
    ...     np.column_stack([drive, follow]),
    ...     ["drive", "follow"],
    ... )
    >>> scores = RegressionResidualScorer(target="follow").fit_score(ts)
    >>> int(np.argmax(np.abs(scores.values)))
    5
    """

    multivariate: ClassVar[bool] = True

    def __init__(self, target: str, regressor: Regressor | None = None) -> None:
        self.target = target
        self.regressor = regressor

    def _new_transformer(self) -> BaseTransformer:
        return RegressionResidual(
            target=self.target, regressor=copy.deepcopy(self.regressor)
        )


class PcaReconstructionErrorScorer(TransformerScorer):
    """Score each point by how far it lies from the principal subspace.

    Correlated columns confine every observation to a low-dimensional subspace of
    the space they nominally span. Principal component analysis finds that
    subspace, and the squared distance from a point to it says how far the
    columns have stopped agreeing with one another — which is exactly the kind of
    anomaly that hides from every column individually.

    Parameters
    ----------
    k
        Number of principal components to keep. The score is the squared
        Euclidean distance to the best rank-``k`` reconstruction.

    Attributes
    ----------
    mean_ : numpy.ndarray
        Column means of the training data.
    components_ : numpy.ndarray
        The ``k`` leading components, shape ``(k, n_columns)``.

    Examples
    --------
    Two columns on a line lie in a one-dimensional subspace, so a point knocked
    off that line stands out even though neither of its coordinates is extreme:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> base = np.arange(8.0)
    >>> partner = 2.0 * base + 1.0
    >>> partner[3] = 2.0 * base[3] - 4.0
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-09", dtype="datetime64[D]"),
    ...     np.column_stack([base, partner]),
    ...     ["a", "b"],
    ... )
    >>> scores = PcaReconstructionErrorScorer(k=1).fit_score(ts)
    >>> int(np.argmax(scores.values))
    3
    """

    multivariate: ClassVar[bool] = True

    def __init__(self, k: int = 1) -> None:
        self.k = k

    def _new_transformer(self) -> BaseTransformer:
        return PcaReconstructionError(k=self.k)

    @property
    def mean_(self) -> NDArray[np.float64]:
        """Column means of the training data."""
        pca: Any = self.transformer_
        centre: NDArray[np.float64] = pca.mean_
        return centre

    @property
    def components_(self) -> NDArray[np.float64]:
        """The leading principal components, one per row."""
        pca: Any = self.transformer_
        basis: NDArray[np.float64] = pca.components_
        return basis


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


class OutlierScorer(BaseScorer):
    """Score each point with a time-independent outlier detection model.

    Treats every observation as a point in as many dimensions as there are
    columns and asks a general-purpose outlier model about it, ignoring the time
    axis entirely. The score is 1.0 for an outlier and 0.0 otherwise, so
    ``FixedThreshold(high=0.5)`` turns it into labels.

    Parameters
    ----------
    model
        An outlier model marking outliers with ``-1``. Two shapes are supported:

        * ``fit(X)`` plus ``predict(X)`` — the model learns the notion of normal
          once and then judges any later series against it;
        * ``fit_predict(X)`` alone — the model can only label the batch it is
          given, so it re-runs on each series and judges every batch against
          itself. Several well-known outlier models, including local outlier
          factor in its default mode, work only this way.

    Attributes
    ----------
    model_ : object
        The fitted outlier model.

    Notes
    -----
    The model is deep-copied at :meth:`fit` time, leaving the caller's object
    untouched.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> class FarFromCentre:
    ...     def fit(self, X):
    ...         self.centre_ = np.median(X, axis=0)
    ...         return self
    ...     def predict(self, X):
    ...         far = np.abs(X - self.centre_).sum(axis=1) > 5.0
    ...         return np.where(far, -1, 1)
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]"),
    ...     np.column_stack([[0.0, 1, 0, 1, 20], [0.0, 1, 1, 0, 20]]),
    ...     ["a", "b"],
    ... )
    >>> OutlierScorer(FarFromCentre()).fit_score(ts).values.ravel()
    array([0., 0., 0., 0., 1.])
    """

    multivariate: ClassVar[bool] = True

    model_: Any

    def __init__(self, model: Any) -> None:
        self.model = model

    def _learn(self, ts: TimeSeries) -> None:
        self.model_ = copy.deepcopy(self.model)
        usable = complete_rows(ts.values)
        if hasattr(self.model_, "fit") and usable.any():
            self.model_.fit(ts.values[usable])

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores = np.full(ts.n_rows, np.nan, dtype=np.float64)
        usable = complete_rows(ts.values)
        if usable.any():
            rows = ts.values[usable]
            judge = (
                self.model_.predict
                if hasattr(self.model_, "predict")
                else self.model_.fit_predict
            )
            scores[usable] = (np.asarray(judge(rows)).ravel() == -1).astype(np.float64)
        return ts.wrap(scores, ["outlier"])
