"""The Hampel filter as a detector, with a fixed cut-off in local sigmas."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from hazure.detection import ScoreDetector
from hazure.methods.hampel_scorer import HampelScorer
from hazure.thresholds import FixedThreshold

if TYPE_CHECKING:
    from hazure._core.window import Window


__all__ = [
    "HampelDetector",
]


class HampelDetector(ScoreDetector):
    """Flag points too far from the local median to be part of the local noise.

    :class:`HampelScorer` with a fixed cut-off. The cut-off is fixed rather than
    learned because the score is already expressed in standard deviations of the
    local noise: ``factor=3.0`` means "three sigma away from where this stretch
    of the series sits", which is the Hampel filter's own rule and needs no
    reference to the distribution of the scores.

    Nothing is learned, so this detector can be used without :meth:`fit`.

    Parameters
    ----------
    window
        Observations or duration making up each point's neighbourhood.
    factor
        How many local standard deviations away is too far.
    center
        Centre the window on each point rather than trailing it.

    Raises
    ------
    ValueError
        The window is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([10.0, 11.0, 12.0, 11.0], 8)
    >>> values[17] = 40.0
    >>> time = np.arange("2024-01-01", "2024-02-02", dtype="datetime64[D]")
    >>> labels = HampelDetector().detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([17])
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self, window: Window = 7, factor: float = 3.0, center: bool = True
    ) -> None:
        self.window = window
        self.factor = factor
        self.center = center
        self._build()

    def _build(self) -> None:
        self.scorer = HampelScorer(window=self.window, center=self.center)
        self.threshold = FixedThreshold(high=self.factor)
