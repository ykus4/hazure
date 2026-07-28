"""Distance from each subsequence to the nearest one like it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hazure import BaseScorer, rolling

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "MatrixProfileScorer",
]


#: Shortest subsequence a matrix profile is defined for. Below three points a
#: z-normalised shape carries no information beyond its slope.
_MIN_WINDOW = 3


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
    >>> from hazure.methods import MatrixProfileScorer
    >>> MatrixProfileScorer(window=24)
    MatrixProfileScorer(window=24)
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
