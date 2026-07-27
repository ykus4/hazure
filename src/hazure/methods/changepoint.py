"""Segmentation: finding the points at which the series became a different series.

A change point is not an outlier. An outlier is one observation that does not
belong; a change point is the instant after which *every* observation belongs to
a different regime. Asking the second question needs a different formulation:
instead of scoring points against a model of normal, the series is partitioned,
and the partition is chosen to minimise

    sum over segments of cost(segment)  +  penalty * (number of segments)

The penalty is what stops the answer being "a segment per observation", and it is
the only real parameter: raise it for fewer, larger regimes and lower it for more.

:class:`PeltScorer` solves that problem exactly, in numpy, with no dependencies.
:class:`RupturesScorer` adapts the search strategies of the ``ruptures`` package
for the cases where a different one is wanted — most usefully a fixed number of
breakpoints rather than a penalty.

Both present themselves as scorers, so segmentation composes with a threshold
like everything else in hazure: the score is the size of the change at each
detected breakpoint and zero elsewhere. The partition itself is the more useful
answer, and it is available as ``breakpoints_``.
"""

from __future__ import annotations

import itertools
import math
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from hazure import BaseScorer
from hazure.detection import ScoreDetector
from hazure.thresholds import MAD_SCALE, FixedThreshold

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = ["Cost", "PeltDetector", "PeltScorer", "RupturesScorer"]

#: Cost functions the built-in segmentation understands.
Cost = Literal["l1", "l2"]

#: Search strategies :class:`RupturesScorer` can dispatch to.
_RUPTURES_MODELS: Final = ("binseg", "window", "dynp", "bottomup")


class _BreakpointScorer(BaseScorer):
    """Shared machinery for scorers whose model of the series is a partition.

    Subclasses supply :meth:`_segment`, which returns the positions where a new
    segment starts. Everything else — the change magnitudes, and how a partition
    learned from one series is expressed as a score on another — is here.

    Attributes
    ----------
    breakpoints_ : numpy.ndarray
        Positions in the training series at which a new segment starts, in
        increasing order. Empty when the series is one regime throughout. The
        first observation is not a breakpoint: every series starts a segment.
    penalty_ : float
        The penalty actually used, whether given or derived. NaN when the search
        was told how many breakpoints to find instead.
    """

    breakpoints_: NDArray[np.int64]
    penalty_: float

    #: Cost function name, which decides what a segment's level means. Set by
    #: every subclass's constructor.
    cost: str

    #: Timestamps of the breakpoints, so a learned partition can be located in a
    #: series that is not the training one.
    _change_time: NDArray[np.int64]
    #: Size of the level change at each breakpoint.
    _change_size: NDArray[np.float64]

    @abstractmethod
    def _segment(self, values: NDArray[np.float64]) -> NDArray[np.int64]:
        """Partition a series.

        Parameters
        ----------
        values
            1-D array, possibly with missing observations.

        Returns
        -------
        numpy.ndarray
            Positions at which a new segment starts, in increasing order.
        """

    @property
    def _level(self) -> str:
        """The statistic summarising where a segment sits.

        Returns
        -------
        str
            ``"median"`` when the cost function is absolute deviation, otherwise
            ``"mean"`` — in both cases the value the segment's cost is measured
            from, so the reported change is the change the search acted on.
        """
        return "median" if self.cost == "l1" else "mean"

    def _learn(self, ts: TimeSeries) -> None:
        values = ts.values[:, 0]
        self.breakpoints_ = self._segment(values)
        self._change_time = ts.time[self.breakpoints_]
        self._change_size = _change_sizes(values, self.breakpoints_, self._level)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        scores = np.zeros(ts.n_rows, dtype=np.float64)
        if ts.n_rows == 0 or self._change_time.size == 0:
            return ts.wrap(scores)
        # The partition was found on the training series, so the breakpoints are
        # located by *when* they happened rather than by position: a series on the
        # same time axis scores identically, and one on a different axis gets the
        # learned changes wherever those instants fall in it.
        position = np.clip(
            np.searchsorted(ts.time, self._change_time), 0, ts.n_rows - 1
        )
        matched = ts.time[position] == self._change_time
        scores[position[matched]] = self._change_size[matched]
        return ts.wrap(scores)


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


class PeltDetector(ScoreDetector):
    """Flag the points at which the series changed regime.

    :class:`PeltScorer` with a threshold that passes its non-zero scores through.
    There is deliberately no factor to tune: the penalty has already decided which
    changes are large enough to be worth a segment, and second-guessing that with
    a rule on the score would be answering the same question twice with less
    information. To report fewer changes, raise ``penalty``.

    Parameters
    ----------
    penalty
        Cost of admitting one more segment. None derives a BIC-style value from
        the data.
    cost
        ``"l2"`` for squared deviations from the segment mean, ``"l1"`` for
        absolute deviations from its median.
    min_size
        Shortest segment allowed, in observations.
    jump
        Consider only breakpoints at multiples of this many observations.

    Raises
    ------
    ValueError
        ``cost`` is unknown, or a size is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> values = np.concatenate([rng.normal(size=60), rng.normal(loc=10.0, size=60)])
    >>> time = np.arange("2024-01-01", "2024-04-30", dtype="datetime64[D]")
    >>> labels = PeltDetector().fit_detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([60])
    """

    def __init__(
        self,
        penalty: float | None = None,
        cost: Cost = "l2",
        min_size: int = 2,
        jump: int = 1,
    ) -> None:
        self.penalty = penalty
        self.cost = cost
        self.min_size = min_size
        self.jump = jump
        self._build()

    def _build(self) -> None:
        self.scorer = PeltScorer(
            penalty=self.penalty,
            cost=self.cost,
            min_size=self.min_size,
            jump=self.jump,
        )
        # Every change the segmentation kept is a change worth reporting, so the
        # line goes at zero.
        self.threshold = FixedThreshold(high=0.0)


class RupturesScorer(_BreakpointScorer):
    """Segment the series with one of the ``ruptures`` search strategies.

    An adapter, for the searches the built-in :class:`PeltScorer` does not
    provide. The one that earns its keep is ``n_bkps``: binary segmentation and
    dynamic programming can both be asked for *exactly* k breakpoints, which is
    the natural way to state the problem when the number of regimes is known and
    a penalty would have to be tuned backwards into it.

    Parameters
    ----------
    model
        Search strategy: ``"binseg"`` (greedy binary segmentation, fast),
        ``"window"`` (sliding two-window discrepancy), ``"dynp"`` (exhaustive
        dynamic programming, exact but expensive and requiring ``n_bkps``), or
        ``"bottomup"`` (greedy merging of an over-fine partition).
    cost
        A cost model name ``ruptures`` understands, such as ``"l1"``, ``"l2"``,
        ``"rbf"`` or ``"normal"``. Unlike :class:`PeltScorer`, this is not limited
        to the two costs hazure implements itself.
    penalty
        Cost of admitting one more segment. Ignored when ``n_bkps`` is given.
        None derives the same BIC-style value :class:`PeltScorer` uses.
    n_bkps
        Ask for exactly this many breakpoints instead of penalising their number.

    Attributes
    ----------
    breakpoints_ : numpy.ndarray
        Positions at which a new segment starts.
    penalty_ : float
        The penalty used, or NaN when ``n_bkps`` was given.

    Raises
    ------
    ValueError
        ``model`` is not one of the four strategies, or ``n_bkps`` is not
        positive.
    ImportError
        ``ruptures`` is not installed.

    Notes
    -----
    ``ruptures`` requires Python below 3.14, so this adapter is unavailable on
    newer interpreters. :class:`PeltScorer` is the portable option: it needs
    nothing beyond numpy, runs everywhere hazure runs, and solves the penalised
    problem exactly.

    Missing observations are filled by linear interpolation before the search,
    because the cost models take a dense matrix. Where that is unacceptable, use
    :class:`PeltScorer`, which skips them.

    Examples
    --------
    >>> from hazure.methods.changepoint import RupturesScorer
    >>> RupturesScorer(model="dynp", n_bkps=1)  # doctest: +SKIP
    RupturesScorer(model='dynp', cost='l2', penalty=None, n_bkps=1)
    """

    def __init__(
        self,
        model: str = "binseg",
        cost: str = "l2",
        penalty: float | None = None,
        n_bkps: int | None = None,
    ) -> None:
        _check_ruptures(model, penalty, n_bkps)
        self.model = model
        self.cost = cost
        self.penalty = penalty
        self.n_bkps = n_bkps

    def _segment(self, values: NDArray[np.float64]) -> NDArray[np.int64]:
        _check_ruptures(self.model, self.penalty, self.n_bkps)
        algorithm = _ruptures_algorithm(self.model, self.cost)

        missing = np.isnan(values)
        dense = values
        if bool(missing.any()):
            if bool(missing.all()):
                self.penalty_ = math.nan
                return np.zeros(0, dtype=np.int64)
            positions = np.arange(values.shape[0], dtype=np.float64)
            dense = values.copy()
            dense[missing] = np.interp(
                positions[missing], positions[~missing], values[~missing]
            )

        fitted = algorithm.fit(dense.reshape(-1, 1))
        if self.n_bkps is not None:
            self.penalty_ = math.nan
            found = fitted.predict(n_bkps=int(self.n_bkps))
        else:
            self.penalty_ = (
                _default_penalty(values, "l1" if self.cost == "l1" else "l2")
                if self.penalty is None
                else float(self.penalty)
            )
            found = fitted.predict(pen=self.penalty_)
        # ruptures reports the end of each segment, so the last entry is the
        # length of the series and the rest are the starts of the segments after
        # the first — which is hazure's convention already.
        return np.asarray(found[:-1], dtype=np.int64)


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


def _change_sizes(
    values: NDArray[np.float64], breakpoints: NDArray[np.int64], level: str
) -> NDArray[np.float64]:
    """Measure the level change across each breakpoint.

    Parameters
    ----------
    values
        The series.
    breakpoints
        Positions at which a new segment starts.
    level
        ``"mean"`` or ``"median"``, matching the cost function.

    Returns
    -------
    numpy.ndarray
        One non-negative magnitude per breakpoint. Zero where a segment held no
        observations at all and no change could be measured.
    """
    if breakpoints.size == 0:
        return np.zeros(0, dtype=np.float64)
    edges = [0, *breakpoints.tolist(), values.shape[0]]
    with warnings.catch_warnings():
        # An entirely missing segment has no level; the change across it is
        # reported as zero below rather than as NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        levels = np.asarray(
            [
                np.nanmedian(values[a:b])
                if level == "median"
                else np.nanmean(values[a:b])
                for a, b in itertools.pairwise(edges)
            ],
            dtype=np.float64,
        )
    change = np.abs(np.diff(levels))
    return np.nan_to_num(change, nan=0.0, posinf=0.0, neginf=0.0)


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


def _check_ruptures(model: str, penalty: float | None, n_bkps: int | None) -> None:
    """Reject adapter parameters before anything is imported.

    Parameters
    ----------
    model
        Search strategy name.
    penalty
        Cost of one more segment, or None.
    n_bkps
        Requested number of breakpoints, or None.

    Raises
    ------
    ValueError
        The model is unknown, the penalty is negative, or ``n_bkps`` is below 1.
    """
    if model not in _RUPTURES_MODELS:
        msg = f"model={model!r} is not one of {list(_RUPTURES_MODELS)}."
        raise ValueError(msg)
    if penalty is not None and penalty < 0:
        msg = f"penalty={penalty} must not be negative."
        raise ValueError(msg)
    if n_bkps is not None and n_bkps < 1:
        msg = f"n_bkps must be at least 1 when given, got {n_bkps}."
        raise ValueError(msg)


def _ruptures_algorithm(model: str, cost: str) -> Any:
    """Build an unfitted ``ruptures`` estimator.

    Parameters
    ----------
    model
        One of :data:`_RUPTURES_MODELS`.
    cost
        Cost model name to hand to ``ruptures``.

    Returns
    -------
    Any
        The estimator, ready for ``fit``.

    Raises
    ------
    ImportError
        ``ruptures`` is not installed, or the interpreter is too new for it.
    """
    try:
        import ruptures
    except ImportError as exc:  # pragma: no cover - exercised by the extras job
        msg = (
            "RupturesScorer needs the ruptures package. Install it with "
            "`pip install hazure[cpd]`, which requires Python below 3.14; on "
            "newer interpreters use PeltScorer, which needs only numpy."
        )
        raise ImportError(msg) from exc

    builders = {
        "binseg": ruptures.Binseg,
        "window": ruptures.Window,
        "dynp": ruptures.Dynp,
        "bottomup": ruptures.BottomUp,
    }
    return builders[model](model=cost)
