"""Anomalous if any input says so."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hazure._core import Aggregator
from hazure.ensemble.states import _states

if TYPE_CHECKING:
    from hazure._core import TimeSeries


__all__ = [
    "OrAggregator",
]


class OrAggregator(Aggregator):
    """Flag a point that any input flags.

    The union of the inputs: use it to collect several kinds of anomaly — a
    spike detector, a level-shift detector, a range check — into one label
    series. Unknown inputs only matter when nothing else has already flagged the
    point.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.array([[1.0, 1.0], [1.0, 0.0], [0.0, np.nan], [np.nan, 1.0]]),
    ...     ["a", "b"],
    ... )
    >>> OrAggregator().aggregate(ts)["anomaly"].tolist()
    [1.0, 1.0, nan, 1.0]
    """

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        anomalous, unknown = _states(ts.values)
        combined = np.where(
            anomalous.any(axis=1), 1.0, np.where(unknown.any(axis=1), np.nan, 0.0)
        )
        return ts.wrap(combined, ["anomaly"])
