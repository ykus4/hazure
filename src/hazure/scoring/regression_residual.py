"""What the other columns fail to predict about one of them."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, ClassVar

from hazure.features import RegressionResidual
from hazure.scoring.transformer_scorer import TransformerScorer

if TYPE_CHECKING:
    from hazure import BaseTransformer
    from hazure.features.regressor import Regressor

__all__ = [
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
