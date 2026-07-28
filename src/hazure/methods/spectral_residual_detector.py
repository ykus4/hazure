"""Flagging the points a spectral-residual saliency map makes stand out."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection import ScoreDetector
from hazure.methods.spectral_residual_scorer import SpectralResidualScorer
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from hazure.thresholds import Factor


__all__ = [
    "SpectralResidualDetector",
]


class SpectralResidualDetector(ScoreDetector):
    """Flag points whose saliency stands far above its neighbourhood.

    :class:`SpectralResidualScorer` paired with an inter-quartile-range rule on
    the score. Useful when the series has structure nobody has characterised: no
    period, trend or distribution has to be named, and a point is reported when
    its saliency is out of proportion to the saliency around it.

    Parameters
    ----------
    window
        Width, in frequency bins, of the moving average over the log amplitude
        spectrum.
    series_window
        Trailing observations the right-edge extrapolation is estimated from.
    score_window
        Trailing saliency points each point is compared against.
    factor
        Inter-quartile-range factor deciding how high a relative saliency is too
        high. One-sided: a *low* saliency is never interesting.

    Raises
    ------
    ValueError
        A window is out of range, ``factor`` is negative, or the time axis is
        irregular.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = 10.0 + np.sin(np.arange(200) * np.pi / 12)
    >>> values[137] = 30.0
    >>> time = np.arange(200) * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> labels = SpectralResidualDetector(factor=12.0).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([137])
    """

    def __init__(
        self,
        window: int = 3,
        series_window: int = 21,
        score_window: int = 21,
        factor: Factor = 3.0,
    ) -> None:
        self.window = window
        self.series_window = series_window
        self.score_window = score_window
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = SpectralResidualScorer(
            window=self.window,
            series_window=self.series_window,
            score_window=self.score_window,
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
