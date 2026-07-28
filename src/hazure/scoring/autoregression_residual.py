"""What the recent past fails to predict.

Asks whether a value is unusual *given* where the series just was, which
catches a break in the dynamics at a perfectly ordinary level.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseScorer
from hazure._core.series import complete_rows
from hazure.features import (
    RegressionResidual,
    Retrospect,
)

if TYPE_CHECKING:
    from hazure import BaseTransformer, TimeSeries
    from hazure.features.regressor import Regressor

__all__ = [
    "AutoregressionResidualScorer",
]


class AutoregressionResidualScorer(BaseScorer):
    """Score each point by what its own recent past fails to predict.

    Many series are largely predictable from where they just were. Fitting that
    relationship and scoring the signed residual asks a sharper question than
    "is this value unusual": it asks whether the value is unusual *given* what
    came immediately before, which catches a break in the dynamics even at a
    perfectly ordinary level.

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

    Attributes
    ----------
    transformer_ : hazure.features.RegressionResidual
        The fitted regression stage, whose ``regressor_`` is the fitted model.

    Raises
    ------
    ValueError
        ``n_steps`` or ``step_size`` is less than 1.

    Notes
    -----
    The regressor is deep-copied at :meth:`fit` time. A scorer handed a frame
    fans out into one copy per column, and without the copy every column would
    fit the same model object and only the last would survive. It also leaves the
    caller's object untouched; the fitted one is reachable through
    ``transformer_.regressor_``.

    The first ``n_steps * step_size`` points have an incomplete history and so
    score NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([1.0, 2.0, 3.0], 6)
    >>> values[10] = 9.0
    >>> time = np.arange("2024-01-01", "2024-01-19", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> scores = AutoregressionResidualScorer(n_steps=3).fit_score(ts).values.ravel()
    >>> int(np.nanargmax(np.abs(scores)))
    10
    """

    lagger_: BaseTransformer
    transformer_: BaseTransformer
    _trained: bool = True

    def __init__(
        self,
        n_steps: int = 1,
        step_size: int = 1,
        regressor: Regressor | None = None,
    ) -> None:
        _check_positive(n_steps, "n_steps")
        _check_positive(step_size, "step_size")
        self.n_steps = n_steps
        self.step_size = step_size
        self.regressor = regressor

    def _learn(self, ts: TimeSeries) -> None:
        _check_positive(self.n_steps, "n_steps")
        _check_positive(self.step_size, "step_size")
        # One extra lag, at zero, so the point being predicted travels through
        # the design matrix as its target column.
        self.lagger_ = Retrospect(
            n_steps=self.n_steps + 1, step_size=self.step_size, till=0
        )
        self.transformer_ = RegressionResidual(
            target="t-0", regressor=copy.deepcopy(self.regressor)
        )
        lagged = self.lagger_.run(ts)
        # A series too short or too gappy to yield one complete row of history
        # supports no model, and every score is then unknown.
        self._trained = bool(complete_rows(lagged.values).any())
        if self._trained:
            self.transformer_.fit(lagged)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        if not self._trained:
            return ts.wrap(np.full(ts.n_rows, np.nan), ts.columns)
        residual = self.transformer_.run(self.lagger_.run(ts))
        # The regression stage names its output "residual"; carry the caller's
        # column name instead, as every other univariate scorer does.
        return ts.wrap(residual.values, ts.columns)


def _check_positive(value: int, name: str) -> None:
    """Reject a count that cannot describe a lag."""
    if value < 1:
        msg = f"{name} must be at least 1, got {value}."
        raise ValueError(msg)
