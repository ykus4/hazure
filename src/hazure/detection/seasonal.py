"""Flagging what a learned seasonal shape fails to explain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection.signed_score import SignedScoreDetector
from hazure.scoring import (
    SeasonalResidualScorer,
)
from hazure.thresholds import (
    IqrThreshold,
)

if TYPE_CHECKING:
    from hazure.detection.side import Side
    from hazure.thresholds.fence import Factor

__all__ = [
    "SeasonalDetector",
]


# ---------------------------------------------------------------------------
# the value of a point, judged against a model of the series
# ---------------------------------------------------------------------------


class SeasonalDetector(SignedScoreDetector):
    """Flag points that break a repeating pattern.

    A daily or weekly cycle is normal behaviour, so it is subtracted before
    anything is judged: what remains is the part of the series the pattern does
    not explain, and it is that remainder the threshold is applied to. A value
    perfectly ordinary for a Tuesday is therefore anomalous on a Sunday.

    Parameters
    ----------
    period
        Length of a cycle in observations. When None it is detected from the
        autocorrelation of the training series.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.
    side
        ``"both"``, ``"positive"`` for values above the pattern only,
        ``"negative"`` for values below it only.
    trend
        Remove a moving-average trend as well as the seasonal profile. Costs a
        NaN margin of half a period at each end, where the centred average has no
        window.

    Raises
    ------
    ValueError
        ``side`` is invalid, the training time axis is irregular, or no period was
        given and none could be detected.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([1.0, 5.0, 3.0, 2.0], 8)
    >>> values[13] = 12.0
    >>> time = np.arange("2024-01-01", "2024-02-02", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> labels = SeasonalDetector(period=4).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([13])
    """

    def __init__(
        self,
        period: int | None = None,
        factor: Factor = 3.0,
        side: Side = "both",
        trend: bool = False,
    ) -> None:
        self.period = period
        self.factor = factor
        self.side = side
        self.trend = trend
        self._build()

    def _build(self) -> None:
        super()._build()
        self.scorer = SeasonalResidualScorer(period=self.period, trend=self.trend)
        self.threshold = IqrThreshold(factor=(None, self.factor))
