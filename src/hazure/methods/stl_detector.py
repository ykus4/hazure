"""Flagging what an STL decomposition cannot explain."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection import ScoreDetector
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from hazure.thresholds import Factor

__all__ = [
    "StlDetector",
]


from hazure.methods.stl_residual_scorer import StlResidualScorer


class StlDetector(ScoreDetector):
    """Flag points an STL decomposition cannot account for.

    :class:`StlResidualScorer` paired with an inter-quartile-range rule on the
    residual magnitudes. The rule is learned from the residuals rather than fixed,
    because how large a residual is large depends entirely on how well the
    decomposition fits the series in the first place.

    Parameters
    ----------
    period
        Length of the cycle, in observations. None derives it from the sampling
        interval.
    robust
        Reweight the loess fits to discount outliers.
    seasonal
        Length of the seasonal smoother, an odd number of at least 7.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.

    Raises
    ------
    ValueError
        The time axis is irregular, or the period is unusable.
    ImportError
        ``statsmodels`` is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> hours = np.arange(240)
    >>> values = 10.0 + 3.0 * np.sin(hours * 2 * np.pi / 24) + rng.normal(size=240)
    >>> values[100] += 12.0
    >>> time = hours * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> labels = StlDetector(factor=6.0).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([100])
    """

    def __init__(
        self,
        period: int | None = None,
        robust: bool = True,
        seasonal: int | None = None,
        factor: Factor = 3.0,
    ) -> None:
        self.period = period
        self.robust = robust
        self.seasonal = seasonal
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = StlResidualScorer(
            period=self.period, robust=self.robust, seasonal=self.seasonal
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
