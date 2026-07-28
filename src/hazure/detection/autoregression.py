"""Flagging what the recent past fails to predict."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection.signed_score import SignedScoreDetector
from hazure.scoring import (
    AutoregressionResidualScorer,
)
from hazure.thresholds import (
    IqrThreshold,
)

if TYPE_CHECKING:
    from hazure.detection.side import Side
    from hazure.features.regressor import Regressor
    from hazure.thresholds.fence import Factor

__all__ = [
    "AutoregressionDetector",
]


class AutoregressionDetector(SignedScoreDetector):
    """Flag points their own recent past fails to predict.

    Fits the relationship between each value and the values a few steps before
    it, and judges the signed residual. This asks a sharper question than whether
    a value is unusual: whether it is unusual *given* where the series just was.
    A break in the dynamics is caught even at a perfectly ordinary level.

    Parameters
    ----------
    n_steps
        Number of past values to regress on.
    step_size
        Gap in observations between them. With ``n_steps=2, step_size=3``, the
        values at ``t-3`` and ``t-6`` predict the value at ``t``.
    regressor
        Any object with ``fit(X, y)`` and ``predict(X)`` taking numpy arrays.
        Defaults to ordinary least squares.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.
    side
        ``"both"``, ``"positive"`` for values above the prediction only,
        ``"negative"`` for values below it only.

    Raises
    ------
    ValueError
        ``side`` is invalid, or ``n_steps`` or ``step_size`` is less than 1.

    Notes
    -----
    The first ``n_steps * step_size`` points have an incomplete history and are
    labelled NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([1.0, 3.0, 5.0, 3.0], 8)
    >>> values[17] = 11.0
    >>> time = np.arange("2024-01-01", "2024-02-02", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> labels = AutoregressionDetector(n_steps=3).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([17])
    """

    def __init__(
        self,
        n_steps: int = 1,
        step_size: int = 1,
        regressor: Regressor | None = None,
        factor: Factor = 3.0,
        side: Side = "both",
    ) -> None:
        self.n_steps = n_steps
        self.step_size = step_size
        self.regressor = regressor
        self.factor = factor
        self.side = side
        self._build()

    def _build(self) -> None:
        super()._build()
        self.scorer = AutoregressionResidualScorer(
            n_steps=self.n_steps, step_size=self.step_size, regressor=self.regressor
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
