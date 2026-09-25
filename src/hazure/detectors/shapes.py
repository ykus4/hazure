"""Detectors for a stretch of the series shaped unlike the rest of it.

These score a subsequence rather than a point, by its distance to the nearest
other subsequence, so they find anomalies made entirely of ordinary values. Every
series has a least-matched subsequence, so a distance is only interesting when it
is out of proportion to the rest of the distances — which is what the fence on
the score decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure._core import Detector
from hazure.detectors._common import upper_fence
from hazure.scorers import DampScorer, MatrixProfileScorer

if TYPE_CHECKING:
    from hazure.thresholds import Factor

__all__ = [
    "damp",
    "matrix_profile",
]


def matrix_profile(
    window: int, factor: Factor = 3.0, normalize: bool = True
) -> Detector:
    """Flag the stretches of the series least like anything else in it.

    :class:`~hazure.scorers.MatrixProfileScorer` paired with an
    inter-quartile-range rule. The threshold is what turns "which shape is
    strangest" into "which shapes are strange enough to report": the profile
    always has a maximum, even in a series with nothing wrong in it.

    Parameters
    ----------
    window
        Subsequence length, in observations.
    factor
        Inter-quartile-range factor deciding how far from its neighbours a
        subsequence has to be. One-sided: a shape that matches the series well is
        never anomalous.
    normalize
        Compare shapes (z-normalised) rather than raw amplitudes. Leave it on to
        match a pattern wherever it sits and however large it is; turn it off when
        the level and the amplitude are part of what makes the shape itself.

    Returns
    -------
    Detector
        The matrix profile, fenced above.

    Raises
    ------
    TypeError
        ``window`` is not an integer count of observations.
    ValueError
        ``window`` is below 3, or the series is too short for it.
    ImportError
        ``stumpy`` is not installed, when fitting.

    Examples
    --------
    >>> matrix_profile(window=24).scorer
    MatrixProfileScorer(window=24)
    """
    scorer = MatrixProfileScorer(window=window, normalize=normalize)
    return Detector(scorer, upper_fence(factor))


def damp(window: int, factor: Factor = 3.0, normalize: bool = True) -> Detector:
    """Flag the stretches of the series unlike anything that came before them.

    :class:`~hazure.scorers.DampScorer` paired with an inter-quartile-range rule,
    the same pairing :func:`matrix_profile` uses and for the same reason.

    The difference is which distances those are. Here a subsequence is compared
    only with subsequences that started earlier, so a shape that recurs later in
    the series cannot explain away its own first appearance. That makes this the
    one to reach for when an anomaly might happen twice, and the one to reach for
    when the question is "was this novel at the time" rather than "is this unique
    in the record".

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

    Returns
    -------
    Detector
        The left matrix profile, fenced above.

    Raises
    ------
    TypeError
        ``window`` is not an integer count of observations.
    ValueError
        ``window`` is below 3, or the series is too short for it.
    ImportError
        ``stumpy`` is not installed, when fitting.

    Notes
    -----
    The first two windows score NaN — see :class:`~hazure.scorers.DampScorer` —
    and the fence is fitted on the distances that remain. A series barely longer
    than the warm-up therefore learns its fence from very little, and the fence
    will move as more data arrives.

    Examples
    --------
    >>> damp(window=24).scorer
    DampScorer(window=24)
    """
    scorer = DampScorer(window=window, normalize=normalize)
    return Detector(scorer, upper_fence(factor))
