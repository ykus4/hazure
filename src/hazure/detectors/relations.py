"""Detectors for a frame whose columns stop agreeing with each other.

Every column can be in its usual range and the combination still be impossible:
a valve closed while flow continues, a request rate that fell while CPU rose.
These need every column at once, and emit a single verdict for the whole frame,
named ``anomaly``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from hazure._core import Detector
from hazure.detectors._common import signed_fence, upper_fence
from hazure.scorers import AsScorer, MinClusterScorer, OutlierScorer
from hazure.thresholds import FixedThreshold
from hazure.transformers import PcaReconstructionError, RegressionResidual

if TYPE_CHECKING:
    from hazure.thresholds import Factor, Side
    from hazure.transformers import Regressor

__all__ = [
    "min_cluster",
    "outlier",
    "pca",
    "regression",
]

#: Membership scores are 0 or 1, so anything over one half is a member.
_MEMBERSHIP: Final = 0.5


def regression(
    target: str,
    regressor: Regressor | None = None,
    factor: Factor = 3.0,
    side: Side = "both",
) -> Detector:
    """Flag points where one column stops matching the others.

    Predicts the target column from the rest and judges the signed residual.
    Where the columns are physically linked — a valve and the flow it causes, a
    request rate and the CPU it burns — the regression captures the link, and a
    large residual means the link itself has broken.

    Parameters
    ----------
    target
        Name of the column to predict. Every other column is a feature.
    regressor
        Any object with ``fit(X, y)`` and ``predict(X)`` taking numpy arrays.
        Defaults to ordinary least squares. It is copied at fit time, so the
        object passed stays unfitted.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.
    side
        ``"both"``, ``"positive"`` for the target running above its prediction
        only, ``"negative"`` for below only.

    Returns
    -------
    Detector
        The regression residual, fenced on its magnitude.

    Raises
    ------
    ValueError
        ``side`` is invalid, or the target column is absent at fit time.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> drive = np.tile([1.0, 2.0, 3.0, 4.0], 5)
    >>> follow = 3.0 * drive - 2.0
    >>> follow[11] += 20.0
    >>> time = np.arange("2024-01-01", "2024-01-21", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(
    ...     time, np.column_stack([drive, follow]), ["drive", "follow"]
    ... )
    >>> labels = regression(target="follow").fit_detect(ts)
    >>> list(labels.columns)
    ['anomaly']
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([11])
    """
    scorer = AsScorer(RegressionResidual(target=target, regressor=regressor))
    return Detector(scorer, signed_fence(factor, side))


def pca(k: int = 1, factor: Factor = 5.0) -> Detector:
    """Flag points that have left the subspace the data lives in.

    Correlated columns confine every observation to a low-dimensional subspace of
    the space they nominally span. Principal component analysis finds that
    subspace from the training data, and the squared distance from a point to it
    measures how far the columns have stopped agreeing with each other. Unlike
    :func:`regression` this singles out no column: any one of them, or several
    together, can be the one that drifted.

    Parameters
    ----------
    k
        Number of principal components to keep — how many directions of genuine
        variation the data has. Everything else counts as error.
    factor
        Inter-quartile-range factor deciding how large a reconstruction error is
        too large.

    Returns
    -------
    Detector
        The reconstruction error, fenced above. The fitted components are
        ``detector.scorer.transformer.components_``.

    Raises
    ------
    ValueError
        ``k`` is less than 1, exceeds the number of columns, or exceeds the number
        of complete training rows.

    Notes
    -----
    A reconstruction error is a squared distance and so never negative, which
    makes its lower tail meaningless. Only the upper tail is tested, so a very
    tightly clustered training set cannot produce a positive lower cut-off that
    would flag the best-reconstructed points as anomalies.

    To see which columns a flagged point's error is made of, run
    :class:`~hazure.transformers.PcaColumnError` alongside.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> base = np.arange(20.0)
    >>> partner = 2.0 * base + 1.0
    >>> partner[6] += 15.0
    >>> time = np.arange("2024-01-01", "2024-01-21", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(
    ...     time, np.column_stack([base, partner]), ["a", "b"]
    ... )
    >>> labels = pca(k=1).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([6])
    """
    return Detector(AsScorer(PcaReconstructionError(k=k)), upper_fence(factor))


def outlier(model: Any) -> Detector:
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

    Returns
    -------
    Detector
        The model's verdict as a 0/1 score, against a cut-off of one half.

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
    >>> outlier(FarFromCentre()).fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 1.])
    """
    return Detector(OutlierScorer(model=model), FixedThreshold(high=_MEMBERSHIP))


def min_cluster(model: Any) -> Detector:
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

    Returns
    -------
    Detector
        Membership of the smallest cluster as a 0/1 score, against a cut-off of
        one half.

    Raises
    ------
    ValueError
        The model has no ``predict`` method, so its clusters could never be
        applied to another series. Raised when fitting.

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
    >>> min_cluster(NearestOfTwo()).fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 1.])
    """
    return Detector(MinClusterScorer(model=model), FixedThreshold(high=_MEMBERSHIP))
