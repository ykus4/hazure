"""Multivariate transformers: sums, regression residuals and PCA.

These reduce or re-express a whole frame at once, which is what makes them
useful for anomaly detection: a point can sit inside the normal range of every
individual series and still be impossible, because the series disagree with each
other. Projecting onto the subspace the columns normally occupy, or regressing
one column on the rest, turns that disagreement into a single number.

The linear algebra is numpy's: :func:`numpy.linalg.svd` for the principal
components and :func:`numpy.linalg.lstsq` for the default regression, so nothing
here adds a dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np

from hazure import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "CustomizedTransformer",
    "OrdinaryLeastSquares",
    "PcaProjection",
    "PcaReconstruction",
    "PcaReconstructionError",
    "RegressionResidual",
    "Regressor",
    "SumAll",
]


class Regressor(Protocol):
    """The interface :class:`RegressionResidual` needs from a regressor.

    Deliberately the same two methods scikit-learn estimators expose, so one can
    be passed straight in, but structural: any object with a matching ``fit`` and
    ``predict`` works and no import is implied.
    """

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], /) -> Any:
        """Learn from a design matrix and a target vector."""

    def predict(self, X: NDArray[np.float64], /) -> NDArray[np.float64]:
        """Predict the target for each row of a design matrix."""


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


class SumAll(BaseTransformer):
    """Sum every column into one series, propagating missing values.

    A row is only summable if every column contributed to it, so a single
    missing observation makes the total missing rather than quietly smaller.
    The output column is named ``sum``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]"),
    ...     np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]),
    ...     ["a", "b"],
    ... )
    >>> SumAll().run(ts).values.ravel()
    array([ 3., nan, 11.])
    """

    multivariate: ClassVar[bool] = True
    trainable: ClassVar[bool] = False

    def __init__(self) -> None:
        # Declared explicitly, with no parameters, so that get_params() and
        # clone() have a signature to read.
        pass

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(ts.values.sum(axis=1), ["sum"])


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

        complete = _complete_rows(ts.values)
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
        complete = _complete_rows(features) & ~np.isnan(ts.column_values(self.target))
        residual = np.full(ts.n_rows, np.nan, dtype=np.float64)
        if bool(complete.any()):
            predicted = np.asarray(
                self.regressor_.predict(features[complete]), dtype=np.float64
            ).ravel()
            residual[complete] = ts.column_values(self.target)[complete] - predicted
        return ts.wrap(residual, ["residual"])


# ---------------------------------------------------------------------------
# principal component analysis
# ---------------------------------------------------------------------------


class _PcaBase(BaseTransformer):
    """Shared fitting for the three PCA views of a frame.

    Every time point is a point in as many dimensions as there are columns. PCA
    finds the directions those points actually vary along, which for correlated
    sensors is far fewer directions than columns.

    Parameters
    ----------
    k
        Number of principal components to keep.

    Attributes
    ----------
    mean_ : numpy.ndarray
        Column means of the training data, one per column.
    components_ : numpy.ndarray
        The ``k`` leading right singular vectors, shape ``(k, n_columns)``,
        ordered by decreasing variance explained.
    """

    multivariate: ClassVar[bool] = True

    def __init__(self, k: int = 1) -> None:
        self.k = k

    def _learn(self, ts: TimeSeries) -> None:
        k = int(self.k)
        if k < 1:
            msg = f"k must be at least 1, got {k}."
            raise ValueError(msg)
        if k > ts.n_columns:
            msg = (
                f"k={k} exceeds the {ts.n_columns} column(s) available, so "
                f"there are not that many components to find."
            )
            raise ValueError(msg)

        complete = _complete_rows(ts.values)
        matrix = ts.values[complete]
        if matrix.shape[0] < k:
            msg = (
                f"Finding {k} component(s) needs at least {k} rows with no "
                f"missing values, but the training data has "
                f"{matrix.shape[0]}."
            )
            raise ValueError(msg)

        self.mean_ = matrix.mean(axis=0)
        directions = np.linalg.svd(matrix - self.mean_, full_matrices=False)[2]
        # Singular vector signs are arbitrary; fixing the largest entry of each
        # to be positive makes the projection reproducible across platforms.
        leading = np.argmax(np.abs(directions), axis=1)
        signs = np.sign(directions[np.arange(directions.shape[0]), leading])
        oriented = directions * np.where(signs == 0.0, 1.0, signs)[:, None]
        self.components_ = np.ascontiguousarray(oriented[:k], dtype=np.float64)

    def _project(self, ts: TimeSeries) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Return each complete row's component scores, and which rows those are."""
        if int(self.k) != self.components_.shape[0]:
            msg = (
                f"k is now {self.k} but {self.components_.shape[0]} "
                f"component(s) were fitted. Call fit() again."
            )
            raise ValueError(msg)
        complete = _complete_rows(ts.values)
        scores = np.full((ts.n_rows, self.components_.shape[0]), np.nan)
        scores[complete] = (ts.values[complete] - self.mean_) @ self.components_.T
        return scores, complete


class PcaProjection(_PcaBase):
    """Project each point onto the leading principal components.

    Emits ``pc0`` .. ``pc{k-1}``, the coordinates of each time point in the
    subspace the training data occupies. Rows with any missing value are
    missing throughout.

    Parameters
    ----------
    k
        Number of principal components to keep.

    Examples
    --------
    Two perfectly correlated columns vary along one direction only:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> base = np.array([0.0, 1.0, 2.0, 3.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.column_stack([base, 2.0 * base]),
    ...     ["a", "b"],
    ... )
    >>> PcaProjection(k=1).fit_transform(ts).values.ravel().round(6)
    array([-3.354102, -1.118034,  1.118034,  3.354102])
    """

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores, _ = self._project(ts)
        return ts.wrap(scores, [f"pc{i}" for i in range(scores.shape[1])])


class PcaReconstruction(_PcaBase):
    """Rebuild each point from the leading principal components only.

    The output keeps the input's column names: it is the input as the model
    believes it should have looked, with everything outside the ``k``-dimensional
    subspace discarded. Rows with any missing value are missing throughout.

    Parameters
    ----------
    k
        Number of principal components to keep.
    """

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores, complete = self._project(ts)
        rebuilt = np.full_like(ts.values, np.nan)
        rebuilt[complete] = self.mean_ + scores[complete] @ self.components_
        return ts.wrap(rebuilt, ts.columns)


class PcaReconstructionError(_PcaBase):
    """Measure how far each point lies from the principal subspace.

    The output, named ``error``, is the **squared** Euclidean distance between a
    point and its reconstruction — the sum over columns of the squared residual,
    not its square root. Squaring keeps it a sum of per-column contributions and
    avoids a square root that no threshold needs.

    A point can be unremarkable in every column and still have a large error,
    because the error measures whether the columns agree with each other.

    Parameters
    ----------
    k
        Number of principal components to keep.

    Examples
    --------
    Data lying exactly on a line is reconstructed exactly:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> base = np.array([0.0, 1.0, 2.0, 3.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.column_stack([base, 2.0 * base + 1.0]),
    ...     ["a", "b"],
    ... )
    >>> PcaReconstructionError(k=1).fit_transform(ts).values.ravel().round(12)
    array([0., 0., 0., 0.])
    """

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores, complete = self._project(ts)
        error = np.full(ts.n_rows, np.nan, dtype=np.float64)
        rebuilt = self.mean_ + scores[complete] @ self.components_
        error[complete] = ((ts.values[complete] - rebuilt) ** 2).sum(axis=1)
        return ts.wrap(error, ["error"])


# ---------------------------------------------------------------------------
# user-supplied transformations
# ---------------------------------------------------------------------------


class CustomizedTransformer(BaseTransformer):
    """Wrap user functions into a transformer.

    Lets a one-off calculation join a pipeline, be cloned, and be
    grid-searched, without subclassing anything.

    The contract is numpy in, numpy out:

    * ``transform_func(values, **params)`` receives the value matrix as a
      ``float64`` array of shape ``(n_rows, n_columns)``, with missing
      observations as ``NaN`` and columns in the order training saw them. It
      must return an array of shape ``(n_rows,)`` or ``(n_rows, m)``; the row
      count may not change, because the time axis is reused. ``params`` is
      ``transform_func_params`` merged over whatever ``fit_func`` returned, so
      an explicit parameter wins over a learned one of the same name.
    * ``fit_func(values, **fit_func_params)`` receives the same kind of array
      and must return a ``dict`` of keyword arguments for ``transform_func``.
      When it is None there is nothing to learn and the transformer may be used
      without calling :meth:`fit`.

    Output columns take the input's names when the width is unchanged, and are
    named ``value`` or ``value_0``, ``value_1``, ... otherwise.

    Parameters
    ----------
    transform_func
        The transformation, as described above.
    transform_func_params
        Extra keyword arguments for ``transform_func``.
    fit_func
        Optional training step, as described above.
    fit_func_params
        Extra keyword arguments for ``fit_func``.

    Attributes
    ----------
    learned_params_ : dict
        What ``fit_func`` returned, empty when there is no ``fit_func``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]"),
    ...     np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    ...     ["a", "b"],
    ... )
    >>> spread = CustomizedTransformer(
    ...     transform_func=lambda values: values.max(axis=1) - values.min(axis=1)
    ... )
    >>> spread.transform(ts).values.ravel()
    array([1., 1., 1.])
    """

    multivariate: ClassVar[bool] = True

    def __init__(
        self,
        transform_func: Callable[..., Any],
        transform_func_params: dict[str, Any] | None = None,
        fit_func: Callable[..., Any] | None = None,
        fit_func_params: dict[str, Any] | None = None,
    ) -> None:
        self.transform_func = transform_func
        self.transform_func_params = transform_func_params
        self.fit_func = fit_func
        self.fit_func_params = fit_func_params
        self.learned_params_: dict[str, Any] = {}
        if fit_func is None:
            # Nothing can be learned, so requiring fit() would be ceremony.
            self._fitted = True

    def _learn(self, ts: TimeSeries) -> None:
        if self.fit_func is None:
            self.learned_params_ = {}
            return
        learned = self.fit_func(ts.values, **(self.fit_func_params or {}))
        if not isinstance(learned, dict):
            msg = (
                f"fit_func must return a dict of keyword arguments for "
                f"transform_func, got {type(learned).__name__}."
            )
            raise TypeError(msg)
        self.learned_params_ = dict(learned)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        params = {**self.learned_params_, **(self.transform_func_params or {})}
        result = np.asarray(self.transform_func(ts.values, **params), dtype=np.float64)
        if result.ndim == 1:
            result = result[:, None]
        if result.ndim != 2 or result.shape[0] != ts.n_rows:
            msg = (
                f"transform_func returned an array of shape {result.shape}, but "
                f"a result on the same time axis must have shape "
                f"({ts.n_rows},) or ({ts.n_rows}, m)."
            )
            raise ValueError(msg)
        return ts.wrap(result)


def _complete_rows(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Flag the rows with no missing value in any column."""
    complete: NDArray[np.bool_] = np.asarray(
        ~np.isnan(values).any(axis=1), dtype=np.bool_
    )
    return complete
