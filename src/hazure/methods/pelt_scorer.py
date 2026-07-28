"""Exact penalised segmentation by pruned dynamic programming."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

from hazure.thresholds import MAD_SCALE

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "PeltScorer",
]


from hazure.methods.breakpoint_scorer import Cost, _BreakpointScorer


class PeltScorer(_BreakpointScorer):
    """Segment the series exactly, by pruned exact linear time search.

    PELT [1]_ minimises the penalised segmentation cost by dynamic programming
    over the series::

        F(t) = min over s < t of  F(s) + cost(s..t) + penalty

    where ``F(t)`` is the best total cost of segmenting the first ``t``
    observations. Evaluated naively that is quadratic. The pruning rule is what
    makes it near-linear: once a candidate start ``s`` satisfies

        F(s) + cost(s..t) > F(t)

    it can never be part of an optimal segmentation of any longer prefix either,
    because extending the segment only adds cost — so ``s`` is discarded for good
    rather than reconsidered at every later step. The answer is nevertheless the
    exact optimum, not an approximation of it: nothing is pruned that could have
    won.

    Parameters
    ----------
    penalty
        Cost of admitting one more segment. None derives a value from the data;
        see Notes. Larger means fewer breakpoints.
    cost
        ``"l2"`` for the sum of squared deviations from the segment mean — changes
        in level, cheap, and the usual choice. ``"l1"`` for the sum of absolute
        deviations from the segment median, which is unmoved by outliers inside a
        segment but costs more to compute.
    min_size
        Shortest segment allowed, in observations. Also the resolution of the
        answer: no two breakpoints will be closer than this.
    jump
        Consider only breakpoints at multiples of this many observations. 1
        considers every position; larger values trade resolution for speed.

    Attributes
    ----------
    breakpoints_ : numpy.ndarray
        Positions at which a new segment starts.
    penalty_ : float
        The penalty used, whether given or derived.

    Raises
    ------
    ValueError
        ``cost`` is not ``"l1"`` or ``"l2"``, or ``min_size``, ``jump`` or
        ``penalty`` is not positive.

    Notes
    -----
    **The default penalty.** With no penalty supplied, a BIC-style value is used:
    ``2 * sigma**2 * log(n)`` of squared error is what an extra segment has to
    buy to be worth having. The information criterion charges ``log(n)`` per free
    parameter, and an extra segment introduces two — where the change happened,
    and the level after it — which is where the factor of two comes from. Charging
    for only the level admits visibly spurious segments: on pure noise, that
    weaker penalty found five breakpoints in 200 observations where this one finds
    none.

    ``sigma`` is estimated from the median absolute *difference* of consecutive
    observations, scaled by ``1.4826 / sqrt(2)``. Differencing is what makes that
    estimate usable here: it removes any level, so the changes being searched for
    do not inflate the noise estimate that decides how many of them are real,
    which the standard deviation of the raw series certainly would. The median
    absolute difference is in turn immune to the individual jumps at the
    breakpoints themselves; ``1.4826`` puts a median absolute deviation on a
    standard-deviation scale, and ``1 / sqrt(2)`` undoes the doubling of variance
    that differencing introduces. For ``cost="l1"`` the penalty is
    ``2 * sigma * log(n)`` instead, because absolute deviations are measured in
    units of the series and squared ones in units of its square: the same
    expression would otherwise mean something different for each cost.

    **Missing observations** are skipped. They contribute nothing to a segment's
    cost, so a gap neither creates nor hides a breakpoint.

    **Cost.** The ``l2`` cost of any segment is O(1) from cumulative sums of the
    values and their squares, which is what makes the whole search fast. The
    ``l1`` cost has no such summary — a median cannot be assembled from prefix
    statistics — so each step sorts its surviving candidates' segments, and a long
    series with many breakpoints will notice.

    The dynamic programme is a genuine sequential recurrence: each step's answer
    depends on the previous ones, so the loop over observations cannot be
    vectorised away. The work *inside* each step is vectorised across all
    surviving candidates.

    References
    ----------
    .. [1] R. Killick, P. Fearnhead and I. A. Eckley, "Optimal Detection of
       Changepoints With a Linear Computational Cost", Journal of the American
       Statistical Association 107(500), 2012, pp. 1590-1598.

    Examples
    --------
    A series that steps from 0 to 10 half way through:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> values = np.concatenate([rng.normal(size=60), rng.normal(loc=10.0, size=60)])
    >>> time = np.arange("2024-01-01", "2024-04-30", dtype="datetime64[D]")
    >>> scorer = PeltScorer().fit(TimeSeries.from_arrays(time, values))
    >>> scorer.breakpoints_
    array([60])
    >>> scores = scorer.score(TimeSeries.from_arrays(time, values)).values.ravel()
    >>> round(float(scores[60]), 1)
    10.0
    """

    def __init__(
        self,
        penalty: float | None = None,
        cost: Cost = "l2",
        min_size: int = 2,
        jump: int = 1,
    ) -> None:
        _check_search(penalty, cost, min_size, jump)
        self.penalty = penalty
        self.cost = cost
        self.min_size = min_size
        self.jump = jump

    def _segment(self, values: NDArray[np.float64]) -> NDArray[np.int64]:
        _check_search(self.penalty, self.cost, self.min_size, self.jump)
        # The check above leaves exactly two possibilities; naming them narrows the
        # adapter-facing `str` back to the pair the search understands.
        cost: Cost = "l1" if self.cost == "l1" else "l2"
        self.penalty_ = (
            _default_penalty(values, cost)
            if self.penalty is None
            else float(self.penalty)
        )
        return _pelt(values, cost, self.penalty_, self.min_size, self.jump)


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


def _pelt(
    values: NDArray[np.float64],
    cost: Cost,
    penalty: float,
    min_size: int,
    jump: int,
) -> NDArray[np.int64]:
    """Find the optimal penalised segmentation of a series.

    Parameters
    ----------
    values
        1-D array, possibly with missing observations.
    cost
        ``"l1"`` or ``"l2"``.
    penalty
        Cost of admitting one more segment.
    min_size
        Shortest segment allowed.
    jump
        Grid spacing for candidate breakpoints.

    Returns
    -------
    numpy.ndarray
        Positions at which a new segment starts, in increasing order.
    """
    n = values.shape[0]
    if n < 2 * min_size:
        # No partition into two admissible segments exists, so the series is one
        # regime by construction.
        return np.zeros(0, dtype=np.int64)

    prefix = _prefixes(values)
    # F(0) = -penalty so that F(t) = min(F(s) + cost + penalty) charges exactly
    # one penalty per segment, including the first.
    best = np.full(n + 1, np.inf, dtype=np.float64)
    best[0] = -penalty
    previous = np.zeros(n + 1, dtype=np.int64)

    ends = [t for t in range(min_size, n) if t % jump == 0]
    ends.append(n)
    candidates: list[int] = []
    next_start = 0

    for end in ends:
        # A start becomes admissible once a segment of at least min_size can run
        # from it to the current end.
        while next_start <= end - min_size:
            if next_start % jump == 0:
                candidates.append(next_start)
            next_start += 1
        if not candidates:  # pragma: no cover - min_size <= n keeps 0 admissible
            continue

        starts = np.asarray(candidates, dtype=np.int64)
        totals = best[starts] + _segment_costs(values, prefix, starts, end, cost)
        winner = int(np.argmin(totals))
        best[end] = totals[winner] + penalty
        previous[end] = starts[winner]
        # The pruning rule. A candidate whose own best-cost plus the cost of
        # reaching here already exceeds the optimum for here cannot win later,
        # because a longer segment costs at least as much.
        candidates = [int(s) for s in starts[totals <= best[end]]]

    if not math.isfinite(best[n]):  # pragma: no cover - reachable only if no end
        return np.zeros(0, dtype=np.int64)

    found: list[int] = []
    position = n
    while position > 0:
        start = int(previous[position])
        if start > 0:
            found.append(start)
        position = start
    return np.asarray(found[::-1], dtype=np.int64)


def _prefixes(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Cumulative count, sum and sum of squares, with a leading zero.

    Parameters
    ----------
    values
        1-D array, possibly with missing observations.

    Returns
    -------
    tuple of numpy.ndarray
        ``(count, sum, sum_of_squares)``, each of length ``len(values) + 1``, so
        that any segment's statistics are one subtraction apart. Missing
        observations contribute to none of the three.
    """
    present = (~np.isnan(values)).astype(np.float64)
    observed = np.where(present > 0.0, values, 0.0)
    zero = np.zeros(1, dtype=np.float64)
    return (
        np.concatenate([zero, np.cumsum(present)]),
        np.concatenate([zero, np.cumsum(observed)]),
        np.concatenate([zero, np.cumsum(observed**2)]),
    )


def _segment_costs(
    values: NDArray[np.float64],
    prefix: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    starts: NDArray[np.int64],
    stop: int,
    cost: Cost,
) -> NDArray[np.float64]:
    """Cost of every segment ``[start, stop)`` at once.

    Parameters
    ----------
    values
        The series.
    prefix
        Output of :func:`_prefixes`.
    starts
        Candidate segment starts.
    stop
        Common segment end, exclusive.
    cost
        ``"l1"`` or ``"l2"``.

    Returns
    -------
    numpy.ndarray
        One cost per start.
    """
    if cost == "l2":
        count, total, squares = prefix
        n = count[stop] - count[starts]
        summed = total[stop] - total[starts]
        squared = squares[stop] - squares[starts]
        # sum (x - mean)^2 = sum x^2 - (sum x)^2 / n, so a segment costs one
        # subtraction rather than a pass over its observations. Clipped at zero
        # because floating-point cancellation can make an exactly-constant
        # segment come out very slightly negative.
        deviation = np.divide(summed**2, n, out=np.zeros_like(summed), where=n > 0.0)
        squares_cost: NDArray[np.float64] = np.maximum(squared - deviation, 0.0)
        return squares_cost

    # Absolute deviations need each segment's median, which no prefix statistic
    # provides, so the segments are laid out as a padded matrix and reduced.
    width = stop - int(starts[0])
    index = starts[:, None] + np.arange(width, dtype=np.int64)[None, :]
    block = np.where(
        index < stop, values[np.minimum(index, values.shape[0] - 1)], np.nan
    )
    with warnings.catch_warnings():
        # An entirely missing segment has no median; its cost is zero either way.
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(block, axis=1)
        absolute: NDArray[np.float64] = np.nansum(
            np.abs(block - centre[:, None]), axis=1
        )
    return absolute


def _default_penalty(values: NDArray[np.float64], cost: Cost) -> float:
    """Derive a BIC-style penalty from the roughness of the series.

    Parameters
    ----------
    values
        The series.
    cost
        Which cost the penalty has to be commensurate with.

    Returns
    -------
    float
        The penalty. Zero for a series with no observed variation, which then
        yields no breakpoints because no segment can reduce a cost of zero.
    """
    observed = values[~np.isnan(values)]
    n = observed.shape[0]
    if n < 3:
        return 0.0
    differences = np.abs(np.diff(observed))
    # Differencing removes the level, so the changes being looked for cannot
    # inflate the noise estimate that decides how many of them are real; the
    # median makes the estimate immune to the jumps themselves; and the two
    # constants put a median absolute deviation of a difference onto the scale of
    # a standard deviation of the series.
    sigma = MAD_SCALE * float(np.median(differences)) / math.sqrt(2.0)
    if sigma <= 0.0:
        return 0.0
    # Two parameters per extra segment — where the change happened and the level
    # after it — hence the factor of two on the per-parameter log(n).
    span = 2.0 * math.log(n)
    return sigma * sigma * span if cost == "l2" else sigma * span


# ---------------------------------------------------------------------------
# validation and lazy loading
# ---------------------------------------------------------------------------


def _check_search(penalty: float | None, cost: str, min_size: int, jump: int) -> None:
    """Reject search parameters no segmentation can be run with.

    Parameters
    ----------
    penalty
        Cost of one more segment, or None.
    cost
        Cost function name.
    min_size
        Shortest segment allowed.
    jump
        Grid spacing for candidate breakpoints.

    Raises
    ------
    ValueError
        ``cost`` is unknown, a size is below 1, or the penalty is negative.
    """
    if cost not in ("l1", "l2"):
        msg = (
            f"cost={cost!r} is not one of ['l1', 'l2']. For other cost models "
            f"use RupturesScorer."
        )
        raise ValueError(msg)
    for name, value in (("min_size", min_size), ("jump", jump)):
        if value < 1:
            msg = f"{name} must be at least 1 observation, got {value}."
            raise ValueError(msg)
    if penalty is not None and penalty < 0:
        msg = (
            f"penalty={penalty} is negative, which would reward extra segments "
            f"instead of charging for them. Use 0 for no penalty at all."
        )
        raise ValueError(msg)
