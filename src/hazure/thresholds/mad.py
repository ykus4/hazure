"""A fence at the median plus a multiple of the median absolute deviation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseThreshold
from hazure.thresholds.fence import (
    MAD_SCALE,
    FactorSpec,
    _bound,
    _factors,
    _label,
    _valid,
)

if TYPE_CHECKING:
    from hazure import TimeSeries


__all__ = [
    "MadThreshold",
]


class MadThreshold(BaseThreshold):
    """Flag scores far from the training median, in units of the MAD.

    The median absolute deviation is the median of ``|x - median(x)|``. Scaled by
    :data:`MAD_SCALE` it estimates the standard deviation of a normal sample, so
    ``factor`` reads on the familiar "number of sigmas" scale while staying
    immune to the outliers a real standard deviation would absorb.

    Parameters
    ----------
    factor
        One factor for both tails, or ``(low, high)``. ``None`` on a side leaves
        that side unbounded.

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
    >>> time = np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [4.0, 5.0, 6.0, 5.0, 4.0, 6.0, 40.0])
    >>> MadThreshold().fit(ts).run(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 1.])
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
        centre = float(np.median(observed))
        spread = MAD_SCALE * float(np.median(np.abs(observed - centre)))
        self.low_ = _bound(centre, spread, low_factor, upper=False)
        self.high_ = _bound(centre, spread, high_factor, upper=True)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(_label(ts.values[:, 0], self.low_, self.high_))
