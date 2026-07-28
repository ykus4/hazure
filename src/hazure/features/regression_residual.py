"""What the other columns fail to predict about one of them."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from hazure import BaseTransformer
from hazure._core.series import complete_rows
from hazure.features.ordinary_least_squares import OrdinaryLeastSquares

if TYPE_CHECKING:
    from hazure import TimeSeries
    from hazure.features.regressor import Regressor


__all__ = [
    "RegressionResidual",
]


class RegressionResidual(BaseTransformer):
    """Predict one column from the others and return the residual.

    Where the columns are physically linked — a flow and the pressure it causes,
    a control and its outcome — the regression captures the link and the
    residual is what the link fails to explain. A large residual means the
    relationship broke, which no single column would show.

    Rows with any missing value are dropped from training and yield a missing
    residual. The output column is named ``residual``.

    Parameters
    ----------
    target
        Name of the column to predict. Every other column is a feature.
    regressor
        Any object with ``fit(X, y)`` and ``predict(X)`` taking and returning
        numpy arrays. Defaults to :class:`OrdinaryLeastSquares`. The object is
        fitted in place, so pass a fresh one per transformer.

    Attributes
    ----------
    regressor_ : Regressor
        The fitted regressor, which is ``regressor`` itself when one was given.
    features_ : tuple of str
        Feature column names, in the order the regressor was fitted with.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> drive = np.arange(6.0)
    >>> follow = 2.0 * drive + 1.0
    >>> follow[4] += 10.0
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]"),
    ...     np.column_stack([drive, follow]),
    ...     ["drive", "follow"],
    ... )
    >>> residual = RegressionResidual(target="follow").fit_transform(ts)
    >>> int(np.argmax(np.abs(residual.values)))
    4
    """

    multivariate: ClassVar[bool] = True

    def __init__(self, target: str, regressor: Regressor | None = None) -> None:
        self.target = target
        self.regressor = regressor

    def _learn(self, ts: TimeSeries) -> None:
        if self.target not in ts.columns:
            msg = (
                f"target={self.target!r} is not a column of the training data; "
                f"available: {list(ts.columns)}."
            )
            raise ValueError(msg)
        features = tuple(name for name in ts.columns if name != self.target)
        if not features:
            msg = (
                f"RegressionResidual needs at least one column besides "
                f"target={self.target!r} to predict it from."
            )
            raise ValueError(msg)

        complete = complete_rows(ts.values)
        if not bool(complete.any()):
            msg = (
                "Every row of the training data has a missing value, so there "
                "is nothing to regress on. Fill or drop the gaps first."
            )
            raise ValueError(msg)

        self.features_ = features
        self.regressor_ = (
            OrdinaryLeastSquares() if self.regressor is None else self.regressor
        )
        self.regressor_.fit(
            ts.select(features).values[complete],
            ts.column_values(self.target)[complete],
        )

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        missing = [name for name in self.features_ if name not in ts.columns]
        if missing or self.target not in ts.columns:
            msg = (
                f"Trained on target {self.target!r} with features "
                f"{list(self.features_)}, but the input is missing "
                f"{missing or [self.target]}."
            )
            raise ValueError(msg)

        features = ts.select(self.features_).values
        complete = complete_rows(features) & ~np.isnan(ts.column_values(self.target))
        residual = np.full(ts.n_rows, np.nan, dtype=np.float64)
        if bool(complete.any()):
            predicted = np.asarray(
                self.regressor_.predict(features[complete]), dtype=np.float64
            ).ravel()
            residual[complete] = ts.column_values(self.target)[complete] - predicted
        return ts.wrap(residual, ["residual"])
