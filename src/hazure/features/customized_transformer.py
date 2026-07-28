"""The escape hatch: a transformer built from a function of yours."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hazure import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from hazure import TimeSeries

__all__ = [
    "CustomizedTransformer",
]


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
