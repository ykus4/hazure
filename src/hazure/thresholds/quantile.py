"""A fence at a percentile of the fitted scores."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseThreshold
from hazure.thresholds.fence import _label, _require_a_bound, _valid

if TYPE_CHECKING:
    from hazure import TimeSeries


__all__ = [
    "QuantileThreshold",
]


class QuantileThreshold(BaseThreshold):
    """Flag scores beyond quantiles of the training scores.

    The quantiles are turned into absolute cut-offs at :meth:`fit` time, so the
    line is drawn by history and then held fixed: a later series is judged
    against what used to be normal, not against itself.

    Parameters
    ----------
    low
        Lower quantile in ``[0, 1]``. None leaves the lower side unbounded.
    high
        Upper quantile in ``[0, 1]``. None leaves the upper side unbounded.

    Attributes
    ----------
    low_ : float
        Absolute lower cut-off learned from the training scores.
    high_ : float
        Absolute upper cut-off learned from the training scores.

    Raises
    ------
    ValueError
        Both quantiles are None, or one falls outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [1.0, 2.0, 3.0, 4.0, 100.0])
    >>> threshold = QuantileThreshold(high=0.9).fit(ts)
    >>> round(threshold.high_, 1)
    61.6
    >>> threshold.run(ts).values.ravel()
    array([0., 0., 0., 0., 1.])
    """

    low_: float
    high_: float

    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        _require_a_bound(low, high, "QuantileThreshold")
        for name, value in (("low", low), ("high", high)):
            if value is not None and not 0.0 <= value <= 1.0:
                msg = (
                    f"QuantileThreshold {name}={value} is not a quantile; it must "
                    f"lie in [0, 1]. For an absolute cut-off use FixedThreshold."
                )
                raise ValueError(msg)
        self.low = low
        self.high = high

    def _learn(self, ts: TimeSeries) -> None:
        _require_a_bound(self.low, self.high, "QuantileThreshold")
        observed = _valid(ts)
        if observed.size == 0:
            self.low_ = self.high_ = math.nan
            return
        self.low_ = (
            -math.inf if self.low is None else float(np.quantile(observed, self.low))
        )
        self.high_ = (
            math.inf if self.high is None else float(np.quantile(observed, self.high))
        )

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(_label(ts.values[:, 0], self.low_, self.high_))
