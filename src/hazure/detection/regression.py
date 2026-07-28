"""Flagging points where one column stops matching the others."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection.multivariate_signed_score import MultivariateSignedScoreDetector
from hazure.scoring import (
    RegressionResidualScorer,
)
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from hazure.detection.side import Side
    from hazure.features.regressor import Regressor
    from hazure.thresholds.fence import Factor

__all__ = [
    "RegressionDetector",
]


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
