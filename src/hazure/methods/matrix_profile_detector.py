"""Flagging the subsequences unlike anything else in the series."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection import ScoreDetector
from hazure.methods.matrix_profile_scorer import MatrixProfileScorer
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from hazure.thresholds import Factor


__all__ = [
    "MatrixProfileDetector",
]


class MatrixProfileDetector(ScoreDetector):
    """Flag the stretches of the series least like anything else in it.

    :class:`MatrixProfileScorer` paired with an inter-quartile-range rule. The
    threshold is what turns "which shape is strangest" into "which shapes are
    strange enough to report": the profile always has a maximum, even in a series
    with nothing wrong in it, so the largest distance is only interesting when it
    is out of proportion to the rest of the profile.

    Parameters
    ----------
    window
        Subsequence length, in observations.
    factor
        Inter-quartile-range factor deciding how far from its neighbours a
        subsequence has to be. One-sided: a shape that matches the series well is
        never anomalous.

    Raises
    ------
    TypeError
        ``window`` is not an integer count of observations.
    ValueError
        ``window`` is below 3, or the series is too short for it.
    ImportError
        ``stumpy`` is not installed.

    Examples
    --------
    >>> from hazure.methods import MatrixProfileDetector
    >>> MatrixProfileDetector(window=24)  # doctest: +SKIP
    MatrixProfileDetector(window=24, factor=3.0)
    """

    def __init__(self, window: int, factor: Factor = 3.0) -> None:
        self.window = window
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = MatrixProfileScorer(window=self.window)
        self.threshold = IqrThreshold(factor=(None, self.factor))
