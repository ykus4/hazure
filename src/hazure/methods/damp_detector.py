"""Flagging the shapes that had never been seen when they happened."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure.detection import ScoreDetector
from hazure.methods.damp_scorer import DampScorer
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from hazure.thresholds import Factor


__all__ = [
    "DampDetector",
]


class DampDetector(ScoreDetector):
    """Flag the stretches of the series unlike anything that came before them.

    :class:`DampScorer` paired with an inter-quartile-range rule, the same pairing
    :class:`MatrixProfileDetector` uses and for the same reason: every series has
    a least-matched subsequence, so a distance is only interesting when it is out
    of proportion to the rest of the distances.

    The difference from :class:`MatrixProfileDetector` is which distances those
    are. Here a subsequence is compared only with subsequences that started
    earlier, so a shape that recurs later in the series cannot explain away its
    own first appearance. That makes this the one to reach for when an anomaly
    might happen twice, and the one to reach for when the question is "was this
    novel at the time" rather than "is this unique in the record".

    Parameters
    ----------
    window
        Subsequence length, in observations.
    factor
        Inter-quartile-range factor deciding how far from its nearest earlier
        neighbour a subsequence has to be. One-sided: a shape with a close match
        in the past is never anomalous.
    normalize
        Compare shapes (z-normalised) rather than raw amplitudes.

    Raises
    ------
    TypeError
        ``window`` is not an integer count of observations.
    ValueError
        ``window`` is below 3, or the series is too short for it.
    ImportError
        ``stumpy`` is not installed.

    Notes
    -----
    The first two windows score NaN — see :class:`DampScorer` — and the fence is
    fitted on the distances that remain. A series barely longer than the warm-up
    therefore learns its fence from very little, and the fence will move as more
    data arrives.

    Examples
    --------
    >>> from hazure.methods import DampDetector
    >>> DampDetector(window=24)
    DampDetector(window=24)
    """

    def __init__(
        self, window: int, factor: Factor = 3.0, normalize: bool = True
    ) -> None:
        self.window = window
        self.factor = factor
        self.normalize = normalize
        self._build()

    def _build(self) -> None:
        """Rebuild the scorer and the threshold from the current parameters."""
        self.scorer = DampScorer(window=self.window, normalize=self.normalize)
        self.threshold = IqrThreshold(factor=(None, self.factor))
