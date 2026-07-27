"""Discords: subsequences with no close match anywhere else in the series.

Every method elsewhere in hazure scores a *point*. These score a *shape*. The
matrix profile is the distance from each subsequence of length ``window`` to its
nearest neighbour among all the other subsequences of the series, and reading it
is a matter of asking which shapes repeat: a low value means "this pattern
happens elsewhere too", and the highest value marks the **discord**, the stretch
of the series least like anything else in it.

That question catches anomalies no per-point rule can see, because a shape can be
made entirely of unremarkable values. A machine that normally ramps up and then
down, and one day ramps down and then up, never leaves its usual range at any
instant; only the shape is wrong.

The profile is computed by ``stumpy``, which is imported lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hazure import BaseScorer, rolling
from hazure.detection import ScoreDetector
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries
    from hazure.thresholds import Factor

__all__ = ["DampScorer", "MatrixProfileDetector", "MatrixProfileScorer"]

#: Shortest subsequence a matrix profile is defined for. Below three points a
#: z-normalised shape carries no information beyond its slope.
_MIN_WINDOW = 3

#: How many windows of history :class:`DampScorer` withholds judgement over.
_WARMUP_WINDOWS = 2


class MatrixProfileScorer(BaseScorer):
    """Score each point by how unlike the rest of the series its shape is.

    For every subsequence of ``window`` consecutive observations, the matrix
    profile records the distance to the closest other subsequence in the series.
    A subsequence with no close match is a discord, and its distance is the score.

    Distances are z-normalised by default, which is what makes the comparison one
    of *shape*: a pattern is judged the same whether it happened at a high level
    or a low one, loudly or quietly. Set ``normalize=False`` to compare raw
    amplitudes instead, which is right when the level itself is part of the
    pattern.

    Parameters
    ----------
    window
        Subsequence length, in observations. This is the one parameter that
        matters: it declares how long the pattern of interest is. Roughly one
        period of the behaviour being described is the usual starting point.
    normalize
        Compare shapes (z-normalised) rather than raw amplitudes.

    Raises
    ------
    TypeError
        ``window`` is not an integer count of observations.
    ValueError
        ``window`` is below 3, or the series is shorter than two windows and so
        contains nothing for a subsequence to be compared against.
    ImportError
        ``stumpy`` is not installed.

    Notes
    -----
    A distance belongs to a subsequence, not to a point, so it is broadcast back
    over the ``window`` observations the subsequence covers, and each point takes
    the largest distance of any subsequence containing it. The maximum rather than
    the mean because a point that participates in one unmatched shape is
    implicated by it, however many ordinary shapes it also belongs to; the cost is
    that the flagged region is a window wide rather than a single point.

    Nothing is learned. The matrix profile is a self-join — the series is compared
    against itself — so it is a property of the series being scored, and
    :meth:`fit` is optional.

    Subsequences containing a missing observation have no comparable neighbour,
    and the points covered only by such subsequences score NaN.

    References
    ----------
    .. [1] C.-C. M. Yeh et al., "Matrix Profile I: All Pairs Similarity Joins for
       Time Series", IEEE ICDM 2016, pp. 1317-1322.

    Examples
    --------
    >>> from hazure.methods.motif import MatrixProfileScorer
    >>> MatrixProfileScorer(window=24)  # doctest: +SKIP
    MatrixProfileScorer(window=24, normalize=True)
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
        return ts.wrap(_broadcast(profile[:, 0], ts.n_rows, self.window))


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
    >>> from hazure.methods.motif import MatrixProfileDetector
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
    >>> from hazure.methods.motif import DampScorer
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


# ---------------------------------------------------------------------------
# the profile
# ---------------------------------------------------------------------------


def _matrix_profile(
    values: NDArray[np.float64], window: int, *, normalize: bool
) -> NDArray[Any]:
    """Compute a matrix profile.

    Parameters
    ----------
    values
        1-D array.
    window
        Subsequence length.
    normalize
        Use the z-normalised metric.

    Returns
    -------
    numpy.ndarray
        ``(n - window + 1, 4)``: distance, nearest-neighbour index, nearest left
        index, nearest right index.

    Raises
    ------
    ValueError
        The series is shorter than two windows.
    ImportError
        ``stumpy`` is not installed.
    """
    if values.shape[0] < 2 * window:
        msg = (
            f"A matrix profile with window={window} needs at least "
            f"{2 * window} observations so that a subsequence has something "
            f"other than itself to match; got {values.shape[0]}. Shorten the "
            f"window or lengthen the series."
        )
        raise ValueError(msg)

    try:
        import stumpy
    except ImportError as exc:  # pragma: no cover - exercised by the extras job
        msg = (
            "Matrix profile scoring needs the stumpy package. Install it with "
            "`pip install hazure[mp]`."
        )
        raise ImportError(msg) from exc

    compute = stumpy.stump if normalize else stumpy.aamp
    profile: NDArray[Any] = compute(values, m=window)
    return profile


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


def _broadcast(
    distances: NDArray[Any], n_rows: int, window: int
) -> NDArray[np.float64]:
    """Spread per-subsequence distances over the points each one covers.

    Parameters
    ----------
    distances
        One value per subsequence, in start order.
    n_rows
        Length of the series.
    window
        Subsequence length.

    Returns
    -------
    numpy.ndarray
        One value per observation: the largest distance of any subsequence
        covering it.
    """
    padded = np.full(n_rows, np.nan, dtype=np.float64)
    values = np.asarray(distances, dtype=np.float64)
    padded[: values.shape[0]] = values
    # Point i is covered by the subsequences starting in [i - window + 1, i],
    # which is exactly a trailing rolling window of width `window` over the
    # distances laid out at their start positions.
    covering = rolling(padded, window, "max", min_periods=1)
    # stumpy reports an unmatchable subsequence as infinitely far away; "no
    # comparison was possible" is unknown, not extreme.
    return np.where(np.isfinite(covering), covering, np.nan)


def _check_window(window: int) -> None:
    """Reject a subsequence length no profile is defined for.

    Parameters
    ----------
    window
        Subsequence length.

    Raises
    ------
    TypeError
        ``window`` is not an integer count of observations.
    ValueError
        ``window`` is below three observations.
    """
    if isinstance(window, bool) or not isinstance(window, int):
        msg = (
            f"window must be a number of observations, not {window!r}; a matrix "
            f"profile is defined on subsequences of a fixed length, so a "
            f"duration is not accepted."
        )
        raise TypeError(msg)
    if window < _MIN_WINDOW:
        msg = (
            f"window must be at least {_MIN_WINDOW} observations for a "
            f"subsequence to have a shape, got {window}."
        )
        raise ValueError(msg)
