"""Tukey's box-plot fence: quartiles plus a multiple of their range."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseThreshold

if TYPE_CHECKING:
    from hazure import TimeSeries

__all__ = [
    "IqrThreshold",
]


from hazure.thresholds.fence import FactorSpec, _bound, _factors, _label, _valid


class IqrThreshold(BaseThreshold):
    """Flag scores far outside the training inter-quartile range.

    The cut-offs are ``Q1 - factor_low * IQR`` and ``Q3 + factor_high * IQR``,
    the rule behind a box plot's whiskers. Quartiles ignore the tails, so the
    very outliers being looked for cannot drag the line out to meet them, which
    is why this is the test the compound detectors end with.

    Parameters
    ----------
    factor
        One factor for both tails, or ``(low, high)``. ``None`` on a side leaves
        that side unbounded, which is how a one-sided test on a magnitude is
        expressed: ``factor=(None, 3.0)``.

    Attributes
    ----------
    low_ : float
        Absolute lower cut-off learned from the training scores.
    high_ : float
        Absolute upper cut-off learned from the training scores.

    Raises
    ------
    ValueError
        A factor is negative, or the pair is not of length two.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-10", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [1.0, 2, 3, 4, 5, 6, 7, 8, 99])
    >>> threshold = IqrThreshold(factor=1.5).fit(ts)
    >>> (threshold.low_, threshold.high_)
    (-3.0, 13.0)
    >>> threshold.run(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
    """

    low_: float
    high_: float

    def __init__(self, factor: FactorSpec = 3.0) -> None:
        _factors(factor, "factor")
        self.factor = factor

    def _learn(self, ts: TimeSeries) -> None:
        low_factor, high_factor = _factors(self.factor, "factor")
        observed = _valid(ts)
        if observed.size == 0:
            self.low_ = self.high_ = math.nan
            return
        q1, q3 = np.quantile(observed, [0.25, 0.75])
        spread = float(q3 - q1)
        self.low_ = _bound(float(q1), spread, low_factor, upper=False)
        self.high_ = _bound(float(q3), spread, high_factor, upper=True)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(_label(ts.values[:, 0], self.low_, self.high_))
