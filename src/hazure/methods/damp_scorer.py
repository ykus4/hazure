"""Distance from each subsequence to the nearest one that came before it."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from hazure import BaseScorer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "DampScorer",
]


from hazure.methods.matrix_profile_scorer import (
    _broadcast,
    _check_window,
    _matrix_profile,
)

#: How many windows of history :class:`DampScorer` withholds judgement over.
_WARMUP_WINDOWS = 2


class DampScorer(BaseScorer):
    """Score each subsequence against its nearest neighbour in the *past* only.

    The same idea as :class:`MatrixProfileScorer` with one restriction: a
    subsequence is compared only against subsequences that started before it. The
    score then answers "has anything like this happened *yet*", which is the
    question a monitor asks, rather than "does anything like this happen anywhere
    in the series", which needs the future to answer.

    That restriction changes results in a way worth understanding. A two-off
    anomaly — an unusual shape that happens twice — is invisible to a full matrix
    profile, because each occurrence is the other's near neighbour and both score
    low. Scored against the past alone, the first occurrence has nothing to match
    and stands out.

    Parameters
    ----------
    window
        Subsequence length, in observations.
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
    ``stumpy`` reports the *index* of each subsequence's nearest left neighbour
    but not the distance to it, so the distances are computed here from those
    indices — one distance per subsequence, which costs a single pass and matches
    the profile's own metric (z-normalised Euclidean, or plain Euclidean when
    ``normalize`` is False). Should a future version expose a left profile
    directly, this becomes a lookup.

    The first two windows of the series are a **warm-up** and score NaN. They have
    to be: with almost no past to be compared against, the opening of any series
    is the least-matched part of it, and scoring it would report the shortage of
    history rather than an anomaly. Two windows is the minimum that leaves a
    subsequence something to match; a longer series is better judged from further
    in still. Subsequences whose neighbour is undefined — because one of the two is
    flat, and a flat shape does not z-normalise — also score NaN.

    Nothing is learned; :meth:`fit` is optional.

    Examples
    --------
    >>> from hazure.methods import DampScorer
    >>> DampScorer(window=24)  # doctest: +SKIP
    DampScorer(window=24, normalize=True)
    """

    trainable: ClassVar[bool] = False

    def __init__(self, window: int, normalize: bool = True) -> None:
        _check_window(window)
        self.window = window
        self.normalize = normalize

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        _check_window(self.window)
        column = ts.values[:, 0]
        profile = _matrix_profile(column, self.window, normalize=self.normalize)
        # Column 2 of a stumpy profile is the index of the nearest neighbour to
        # the left, or -1 where there is none.
        left = np.asarray(profile[:, 2], dtype=np.int64)
        distances = _pairwise(column, self.window, left, normalize=self.normalize)
        # The opening of the series has too little past to be judged against; see
        # the class's Notes.
        distances[: _WARMUP_WINDOWS * self.window] = np.nan
        return ts.wrap(_broadcast(distances, ts.n_rows, self.window))


def _pairwise(
    values: NDArray[np.float64],
    window: int,
    neighbour: NDArray[np.int64],
    *,
    normalize: bool,
) -> NDArray[np.float64]:
    """Distance from each subsequence to the one at ``neighbour``.

    Parameters
    ----------
    values
        1-D array.
    window
        Subsequence length.
    neighbour
        Index of the subsequence to compare each one against, or -1 for none.
    normalize
        Use the z-normalised metric.

    Returns
    -------
    numpy.ndarray
        One distance per subsequence, NaN where the comparison is undefined.
    """
    subsequences = np.lib.stride_tricks.sliding_window_view(values, window)
    block = np.asarray(subsequences, dtype=np.float64)
    if normalize:
        centre = block.mean(axis=1, keepdims=True)
        spread = block.std(axis=1, keepdims=True)
        # A flat subsequence has no shape to normalise, so it compares with
        # nothing rather than comparing with everything.
        block = np.divide(
            block - centre,
            spread,
            out=np.full(block.shape, np.nan),
            where=spread > 0.0,
        )

    partner = np.clip(neighbour, 0, block.shape[0] - 1)
    distances: NDArray[np.float64] = np.sqrt(
        ((block - block[partner]) ** 2).sum(axis=1)
    )
    distances[neighbour < 0] = np.nan
    return distances
