"""Ready-made detectors that need every column at once.

These look for anomalies that live in the relationship between columns: a link
between two sensors that has broken, a set of readings that has left the subspace
it has always occupied, a moment unlike any the data has grouped itself into.
None of these is visible in a single column, so none of them fans out; each fits
on the whole frame and reports one label series named ``anomaly``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from hazure.detection._composition import (
    MultivariateScoreDetector,
    MultivariateSignedScoreDetector,
)
from hazure.scoring import (
    MinClusterScorer,
    OutlierScorer,
    PcaReconstructionErrorScorer,
    RegressionResidualScorer,
)
from hazure.thresholds import FixedThreshold, IqrThreshold

if TYPE_CHECKING:
    from hazure.detection._composition import Side
    from hazure.features import Regressor
    from hazure.thresholds import Factor

__all__ = [
    "MinClusterDetector",
    "OutlierDetector",
    "PcaDetector",
    "RegressionDetector",
]

#: Cut-off for the 1.0 / 0.0 membership scores the model adapters produce. Any
#: value between the two works; a half keeps the intent obvious.
_MEMBERSHIP_CUTOFF: Final = 0.5


class RegressionDetector(MultivariateSignedScoreDetector):
    """Flag points where one column stops matching the others.

    Predicts the target column from the rest and judges the signed residual.
    Where the columns are physically linked — a valve and the flow it causes, a
    request rate and the CPU it burns — the regression captures the link, and a
    large residual means the link itself has broken. Both columns can be in their
    usual range and still be impossible together.

    Parameters
    ----------
    target
        Name of the column to predict. Every other column is a feature.
    regressor
        Any object with ``fit(X, y)`` and ``predict(X)`` taking numpy arrays.
        Defaults to ordinary least squares.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.
    side
        ``"both"``, ``"positive"`` for the target running above its prediction
        only, ``"negative"`` for below only.

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
    >>> labels = RegressionDetector(target="follow").fit_detect(ts)
    >>> list(labels.columns)
    ['anomaly']
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([11])
    """

    def __init__(
        self,
        target: str,
        regressor: Regressor | None = None,
        factor: Factor = 3.0,
        side: Side = "both",
    ) -> None:
        self.target = target
        self.regressor = regressor
        self.factor = factor
        self.side = side
        self._build()

    def _build(self) -> None:
        super()._build()
        self.scorer = RegressionResidualScorer(
            target=self.target, regressor=self.regressor
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))


class PcaDetector(MultivariateScoreDetector):
    """Flag points that have left the subspace the data lives in.

    Correlated columns confine every observation to a low-dimensional subspace of
    the space they nominally span. Principal component analysis finds that
    subspace from the training data, and the squared distance from a point to it
    measures how far the columns have stopped agreeing with each other. Unlike
    :class:`RegressionDetector` this singles out no column: any one of them, or
    several together, can be the one that drifted.

    Parameters
    ----------
    k
        Number of principal components to keep — how many directions of genuine
        variation the data has. Everything else counts as error.
    factor
        Inter-quartile-range factor deciding how large a reconstruction error is
        too large.

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
    >>> labels = PcaDetector(k=1).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([6])
    """

    def __init__(self, k: int = 1, factor: Factor = 5.0) -> None:
        self.k = k
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = PcaReconstructionErrorScorer(k=self.k)
        self.threshold = IqrThreshold(factor=(None, self.factor))


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
