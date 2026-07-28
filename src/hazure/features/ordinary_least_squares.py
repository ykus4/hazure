"""Least-squares linear regression with an intercept."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "OrdinaryLeastSquares",
]


class OrdinaryLeastSquares:
    """Least-squares linear regression with an intercept.

    The default regressor for :class:`RegressionResidual`. Solved with
    :func:`numpy.linalg.lstsq`, which handles a rank-deficient design matrix by
    returning the minimum-norm solution rather than failing.

    Attributes
    ----------
    intercept_ : float
        Fitted constant term.
    coefficients_ : numpy.ndarray
        One fitted slope per feature column.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0]])
    >>> model = OrdinaryLeastSquares().fit(X, np.array([1.0, 3.0, 5.0]))
    >>> round(model.intercept_, 12), model.coefficients_.round(12)
    (1.0, array([2.]))
    """

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.float64], /
    ) -> OrdinaryLeastSquares:
        """Fit the model and return self, for chaining.

        Parameters
        ----------
        X
            Design matrix of shape ``(n_rows, n_features)``.
        y
            Target of shape ``(n_rows,)``.

        Returns
        -------
        OrdinaryLeastSquares
            This model.
        """
        design = np.column_stack([np.ones(X.shape[0], dtype=np.float64), X])
        solution = np.linalg.lstsq(design, y, rcond=None)[0]
        self.intercept_ = float(solution[0])
        self.coefficients_ = np.asarray(solution[1:], dtype=np.float64)
        return self

    def predict(self, X: NDArray[np.float64], /) -> NDArray[np.float64]:
        """Predict the target for each row of ``X``.

        Parameters
        ----------
        X
            Design matrix of shape ``(n_rows, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(n_rows,)``.
        """
        prediction: NDArray[np.float64] = self.intercept_ + X @ self.coefficients_
        return prediction
