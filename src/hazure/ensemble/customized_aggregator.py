"""The escape hatch: an aggregator built from a function of yours."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from hazure import BaseAggregator

if TYPE_CHECKING:
    from collections.abc import Callable

    from hazure import TimeSeries

__all__ = [
    "CustomizedAggregator",
]


class CustomizedAggregator(BaseAggregator):
    """Wrap a user function into an aggregator.

    The contract is numpy in, numpy out: ``aggregate_func(labels, **params)``
    receives a ``float64`` array of shape ``(n_rows, n_inputs)`` whose columns
    are the inputs in the order they were passed, with unknown labels as
    ``NaN``, and must return an array of shape ``(n_rows,)`` — or
    ``(n_rows, 1)``, which is flattened. Returning ``NaN`` for a row means the
    combination is unknown there.

    Parameters
    ----------
    aggregate_func
        The combining function, as described above.
    aggregate_func_params
        Extra keyword arguments for ``aggregate_func``.

    Raises
    ------
    ValueError
        The function returned something other than one value per row.

    Examples
    --------
    A weighted vote, trusting the first input twice as much as the second:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-04", dtype="datetime64[D]"),
    ...     np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    ...     ["a", "b"],
    ... )
    >>> weighted = CustomizedAggregator(
    ...     aggregate_func=lambda labels, w: (labels @ w >= 0.5).astype(float),
    ...     aggregate_func_params={"w": np.array([0.67, 0.33])},
    ... )
    >>> weighted.aggregate(ts)["anomaly"].tolist()
    [1.0, 0.0, 0.0]
    """

    def __init__(
        self,
        aggregate_func: Callable[..., Any],
        aggregate_func_params: dict[str, Any] | None = None,
    ) -> None:
        self.aggregate_func = aggregate_func
        self.aggregate_func_params = aggregate_func_params

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        params = self.aggregate_func_params or {}
        result = np.asarray(self.aggregate_func(ts.values, **params), dtype=np.float64)
        if result.ndim == 2 and result.shape[1] == 1:
            result = result[:, 0]
        if result.shape != (ts.n_rows,):
            msg = (
                f"aggregate_func returned an array of shape {result.shape}, but "
                f"combining {ts.n_columns} inputs over {ts.n_rows} rows must "
                f"give shape ({ts.n_rows},)."
            )
            raise ValueError(msg)
        return ts.wrap(result, ["anomaly"])
