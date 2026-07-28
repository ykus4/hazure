"""Flagging points that have left the subspace the columns normally occupy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection.multivariate_score import MultivariateScoreDetector
from hazure.scoring import (
    PcaReconstructionErrorScorer,
)
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from hazure.thresholds.fence import Factor

__all__ = [
    "PcaDetector",
]


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
