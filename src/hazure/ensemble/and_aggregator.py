"""Anomalous only if every input says so."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseAggregator

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "AndAggregator",
]


from hazure.ensemble.states import _states


class AndAggregator(BaseAggregator):
    """Flag a point only where every input agrees.

    The intersection of the inputs: use it to demand corroboration, so that a
    point counts only when several independent tests object to it. One definite
    "normal" settles the row; short of that, an unknown input leaves the answer
    unknown.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.array([[1.0, 1.0], [1.0, 0.0], [0.0, np.nan], [np.nan, 1.0]]),
    ...     ["a", "b"],
    ... )
    >>> AndAggregator().aggregate(ts)["anomaly"].tolist()
    [1.0, 0.0, 0.0, nan]
    """

    def __init__(self) -> None:
        # Declared explicitly, with no parameters, so that get_params() and
        # clone() have a signature to read.
        pass

    def _combine(self, ts: TimeSeries) -> TimeSeries:
        anomalous, unknown = _states(ts.values)
        normal = ~anomalous & ~unknown
        combined = np.where(
            normal.any(axis=1), 0.0, np.where(unknown.any(axis=1), np.nan, 1.0)
        )
        return ts.wrap(combined, ["anomaly"])
