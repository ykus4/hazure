"""Adding the columns of a frame together."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from hazure import BaseTransformer

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "SumAll",
]


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
