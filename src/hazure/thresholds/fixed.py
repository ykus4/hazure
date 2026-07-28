"""A fence you supply rather than one learned from the scores."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar

from hazure import BaseThreshold
from hazure.thresholds.fence import _label, _require_a_bound

if TYPE_CHECKING:
    from hazure import TimeSeries


__all__ = [
    "FixedThreshold",
]


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


class FixedThreshold(BaseThreshold):
    """Flag scores outside a range the caller supplies.

    Nothing is learned, so this is usable without :meth:`fit`. It is the right
    choice when the acceptable range comes from domain knowledge — a pressure
    limit, a service-level objective — rather than from history.

    Parameters
    ----------
    low
        Scores below this are flagged. None leaves the lower side unbounded.
    high
        Scores above this are flagged. None leaves the upper side unbounded.

    Raises
    ------
    ValueError
        Both bounds are None, which would make the threshold inert.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [0.0, 9.0, 1.0, np.nan, -9.0])
    >>> FixedThreshold(low=-5.0, high=5.0).run(ts).values.ravel()
    array([ 0.,  1.,  0., nan,  1.])
    """

    trainable: ClassVar[bool] = False

    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        _require_a_bound(low, high, "FixedThreshold")
        self.low = low
        self.high = high

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        _require_a_bound(self.low, self.high, "FixedThreshold")
        low = -math.inf if self.low is None else float(self.low)
        high = math.inf if self.high is None else float(self.high)
        return ts.wrap(_label(ts.values[:, 0], low, high))
