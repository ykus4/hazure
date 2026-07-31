"""A fence placed by extreme value theory, at a chosen false-alarm probability.

Every other threshold in this package is parameterised by something about the
sample: a quantile of it, a multiple of its spread, a critical value for its size.
This one is parameterised by the answer instead. You say how often you are
willing to be wrong — one alert in ten thousand samples — and the fence goes
wherever the tail of the fitted distribution puts that probability, which is
usually well beyond the largest score ever observed.

That extrapolation is the point, and it is what the Pickands-Balkema-de Haan
theorem licenses: for almost every distribution, the excesses over a high enough
threshold converge to a generalised Pareto distribution as that threshold rises.
So the tail gets a two-parameter model fitted to the few hundred largest scores,
and the model is asked a question the empirical quantile cannot answer — where
would a score be that we should expect to see once in ten thousand samples?
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseThreshold
from hazure.thresholds.fence import _label, _require_a_bound, _valid

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import NDArray

    from hazure import TimeSeries


__all__ = [
    "PotThreshold",
]


#: Fewest excesses a tail may be fitted from. Below this the two parameters of
#: the generalised Pareto are not identified in any useful sense — the fit will
#: succeed and the fence it implies will be governed by whichever three or four
#: points happened to be largest. Rather than extrapolate from that, the side is
#: left unknown and every label comes back NaN.
_MIN_PEAKS = 10

#: Points sampled per stretch of the scan for roots of the likelihood equation.
#: A sign change is bisected to convergence once found, so this only has to be
#: fine enough not to step over one — not fine enough to locate it.
_SCAN_POINTS = 12

#: Relative width at which bisection stops. The root is a nuisance parameter —
#: what it feeds is a shape and scale that then get compared by likelihood — so
#: there is nothing to gain from resolving it beyond this.
_TOLERANCE = 1e-10


class PotThreshold(BaseThreshold):
    """Flag scores beyond a generalised Pareto fit to the tail of the training scores.

    Peaks-over-threshold: the training scores above their own ``level`` quantile
    are kept as excesses over it, a generalised Pareto distribution is fitted to
    those excesses by maximum likelihood, and the fence is placed where that
    distribution says an exceedance has probability ``high``.

    What this buys over :class:`~hazure.QuantileThreshold` is extrapolation. A
    quantile of the training scores can never be placed beyond the largest one
    observed, so asking for a one-in-a-million fence from ten thousand samples
    quietly gives you the maximum instead. The fitted tail is a model, and can be
    asked about probabilities no sample of that size could resolve.

    What it costs is an assumption. The theorem behind it is asymptotic in
    ``level``, and the fit needs enough excesses to identify two parameters, so
    the two parameters pull against each other: raise ``level`` and the model is
    better justified but fitted from less data. The default keeps 2% of the
    training scores, which wants a few thousand of them to work with.

    Parameters
    ----------
    low
        Target exceedance probability for the lower tail, in ``(0, 1)``. None
        leaves the lower side unbounded.
    high
        Target exceedance probability for the upper tail, in ``(0, 1)``. None
        leaves the upper side unbounded. This is the false-alarm rate you are
        asking for: ``1e-4`` means one flagged sample in ten thousand, if the
        series goes on behaving as it did during training.
    level
        Quantile of the training scores where the tail is taken to start, in
        ``(0, 1)``. Excesses over it are what the generalised Pareto is fitted
        to.

    Attributes
    ----------
    low_ : float
        Absolute lower cut-off, ``-inf`` when the side is unbounded and ``nan``
        when the tail could not be fitted.
    high_ : float
        Absolute upper cut-off, ``inf`` when the side is unbounded and ``nan``
        when the tail could not be fitted.
    tail_ : dict of dict
        The fitted tail for each bounded side that had enough excesses, keyed
        ``"low"`` and ``"high"``. Each holds ``"start"`` (the ``level`` quantile,
        in the original units and sign), ``"shape"`` and ``"scale"`` (the
        generalised Pareto parameters, where a shape above zero means a
        heavy tail with no upper limit and a shape below zero means the tail ends
        at a finite point), and ``"peaks"`` (how many excesses the fit used).

    Raises
    ------
    ValueError
        Both probabilities are None, one falls outside ``(0, 1)``, ``level`` is
        outside ``(0, 1)``, or — at :meth:`~hazure.Component.fit` time — a
        requested probability is larger than the tail it would have to be found
        in.

    See Also
    --------
    hazure.QuantileThreshold : A quantile of the training scores, no model and no
        extrapolation.
    update : Drive the same fence online, letting it move as scores arrive.

    Notes
    -----
    Write ``t`` for the ``level`` quantile of the ``n`` training scores, ``N_t``
    for how many exceeded it, and ``gamma`` and ``sigma`` for the fitted shape and
    scale. The fence for a target probability ``q`` is then::

        p   = q * n / N_t
        z_q = t + (sigma / gamma) * (p ** -gamma - 1)
        z_q = t - sigma * log(p)                        # the gamma -> 0 limit

    That ``p`` is ``q`` rescaled from a probability over the whole distribution to
    one conditional on having cleared ``t``, and it has to be below 1 for the
    question to be about the tail at all — which is why ``high`` must be smaller
    than ``1 - level``. Fitting says so rather than silently answering a different
    question. The second line is the exponential tail, and is used exactly when
    the fit chose a shape of zero rather than approached one.

    The maximum likelihood fit follows Grimshaw [2]_: substituting
    ``theta = gamma / sigma`` reduces two parameters to one, because the
    likelihood can then be maximised over ``gamma`` in closed form for any fixed
    ``theta``. What is left is a scalar equation, solved here by scanning for sign
    changes and bisecting each. The exponential limit is always evaluated as a
    candidate too, and whichever candidate has the highest likelihood wins — so a
    tail that really is exponential is fitted as one rather than through a shape
    parameter driven to zero.

    The search covers shapes down to about ``-1`` and no further, which is the
    regularity condition rather than a limitation of the solver: below ``-1`` the
    likelihood is unbounded, and it is maximised by putting the tail's endpoint at
    the largest excess observed and letting the density diverge there. There is no
    estimate to find in that region, so a sample light-tailed enough to ask for one
    is fitted as exponential instead — which places the fence higher than the
    degenerate solution would, and so errs towards silence rather than towards
    false alarms.

    The lower tail is fitted by negating the scores and running the identical
    upper-tail machinery, so both sides come from the same code and the same
    tests.

    References
    ----------
    .. [1] A. Siffer, P.-A. Fouque, A. Termier and C. Largouet, "Anomaly
       Detection in Streams with Extreme Value Theory", KDD 2017, pp. 1067-1075.
    .. [2] S. D. Grimshaw, "Computing Maximum Likelihood Estimates for the
       Generalized Pareto Distribution", Technometrics 35(2), 1993, pp. 185-191.

    Examples
    --------
    Two thousand scores from a standard exponential, and a fence asked to sit
    where exceedance has probability one in ten thousand. It lands beyond the
    largest score ever observed, which is the extrapolation a quantile of the same
    sample cannot perform at all:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> scores = rng.exponential(size=2000)
    >>> time = np.arange(2000, dtype=np.int64) * 60_000_000_000
    >>> ts = TimeSeries.from_arrays(time, scores)
    >>> threshold = PotThreshold(high=1e-4).fit(ts)
    >>> round(threshold.high_, 2), round(float(scores.max()), 2)
    (8.79, 8.15)

    The true one-in-ten-thousand point of a standard exponential is
    ``-log(1e-4) = 9.21``, so 40 excesses got within 5% of a quantile the sample
    itself could never have located:

    >>> int(threshold.tail_["high"]["peaks"])
    40
    >>> round(threshold.tail_["high"]["start"], 2)
    4.01

    Nothing in the training scores is beyond the fence, which is what asking for
    one in ten thousand from two thousand samples should mean:

    >>> float(threshold.run(ts).values.sum())
    0.0

    Asking for a probability that is not in the tail is refused rather than
    answered from the body of the distribution:

    >>> PotThreshold(high=0.1, level=0.98).fit(ts)
    Traceback (most recent call last):
        ...
    ValueError: PotThreshold cannot place the high fence at high=0.1: the tail...
    """

    low_: float
    high_: float
    tail_: dict[str, dict[str, float]]

    # Retained only so that `update` can extend the fit; `run` needs none of it.
    _peaks: dict[str, NDArray[np.float64]]
    _seen: int

    def __init__(
        self,
        low: float | None = None,
        high: float | None = 1e-4,
        level: float = 0.98,
    ) -> None:
        _require_a_bound(low, high, "PotThreshold")
        for name, value in (("low", low), ("high", high)):
            if value is not None and not 0.0 < value < 1.0:
                msg = (
                    f"PotThreshold {name}={value} is a probability of exceeding "
                    f"the fence, so it must lie strictly between 0 and 1. For an "
                    f"absolute cut-off use FixedThreshold."
                )
                raise ValueError(msg)
        if not 0.0 < level < 1.0:
            msg = (
                f"PotThreshold level={level} must lie strictly between 0 and 1; "
                f"it is the quantile of the training scores where the tail is "
                f"taken to start."
            )
            raise ValueError(msg)
        self.low = low
        self.high = high
        self.level = level

    # -- fitting ------------------------------------------------------------

    def _learn(self, ts: TimeSeries) -> None:
        _require_a_bound(self.low, self.high, "PotThreshold")
        observed = _valid(ts)
        self.tail_ = {}
        self._peaks = {}
        self._seen = int(observed.size)
        # Each side is fitted through the upper-tail path, the lower one on
        # negated scores, so the sign only has to be undone once on the way out.
        self.high_ = (
            math.inf
            if self.high is None
            else self._fit_side("high", observed, self.high)
        )
        self.low_ = (
            -math.inf
            if self.low is None
            else -self._fit_side("low", -observed, self.low)
        )

    def _fit_side(self, side: str, sample: NDArray[np.float64], target: float) -> float:
        """Fit the upper tail of ``sample`` and return the fence it implies.

        Parameters
        ----------
        side
            ``"low"`` or ``"high"``, naming the tail in the caller's own
            orientation. ``sample`` has already been negated for ``"low"``.
        sample
            Valid training scores, oriented so the tail of interest is the upper
            one.
        target
            Exceedance probability asked of this side.

        Returns
        -------
        float
            The cut-off in ``sample``'s orientation, or ``nan`` when there were
            too few excesses to fit a tail.

        Raises
        ------
        ValueError
            The requested probability lies inside the body of the distribution
            rather than in the tail the fit is based on.
        """
        sign = -1.0 if side == "low" else 1.0

        if sample.size == 0:
            return math.nan
        start = float(np.quantile(sample, self.level))
        peaks = sample[sample > start] - start
        if peaks.size < _MIN_PEAKS:
            return math.nan

        # q is a probability over the whole distribution; the tail model only
        # knows about the fraction of it that cleared `start`.
        conditional = target * sample.size / peaks.size
        if conditional >= 1.0:
            msg = (
                f"PotThreshold cannot place the {side} fence at {side}={target}: "
                f"the tail starts at the {self.level:.4g} quantile of the "
                f"training scores, which leaves {peaks.size} of {sample.size} "
                f"above it, and a target probability of {target} is larger than "
                f"the {peaks.size / sample.size:.3g} of the distribution that "
                f"tail covers. Lower {side} below {peaks.size / sample.size:.3g}, "
                f"or lower level so the tail reaches further in."
            )
            raise ValueError(msg)

        shape, scale = _fit_gpd(peaks)
        self.tail_[side] = {
            "start": sign * start,
            "shape": shape,
            "scale": scale,
            "peaks": float(peaks.size),
        }
        self._peaks[side] = peaks
        return _fence(start, shape, scale, conditional)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(_label(ts.values[:, 0], self.low_, self.high_))

    # -- streaming ----------------------------------------------------------

    def update(self, value: float) -> float:
        """Absorb one new score, moving the fence, and return its label.

        This is the streaming half of the method — SPOT, in the terms of [1]_.
        The fence is not a fixed verdict on a series you already hold but a
        running one, refitted as the tail fills in, so a monitor can be started
        from a week of history and then left to run.

        Three things can happen to a score, and which one is the whole algorithm.
        A score beyond the fence is flagged and then *discarded* — not only from
        the excesses but from the count of observations too: an anomaly is not
        evidence about how the normal tail behaves, and letting it into either
        would move the fence on the strength of a point just declared not to
        belong. A score inside the fence but above the tail start joins the
        excesses and the fence is refitted from them. Anything else only increases
        the count, which lowers the fence slightly, since the same excesses now
        represent a rarer event.

        Parameters
        ----------
        value
            One score, on the same scale as the scores this was fitted on.
            ``NaN`` is returned unchanged and leaves the fit untouched, the same
            way :meth:`~hazure.BaseThreshold.apply` treats it.

        Returns
        -------
        float
            ``1.0`` if the score is beyond the fence as it stood when the score
            arrived, ``0.0`` if not, ``NaN`` if the score was ``NaN`` or the fence
            is unknown.

        Raises
        ------
        RuntimeError
            The threshold has not been fitted, so there is no tail to extend; or it
            was fitted on a frame, so it holds one fence per column and there is no
            single tail to extend.

        See Also
        --------
        update_many : The same, over a batch of scores.

        Notes
        -----
        The label is decided against the fence as it stood *before* the score was
        absorbed, which is the only causal way round: a monitor cannot use a
        point to move the line it is about to judge that point against.

        Refitting happens on every new excess rather than on a schedule. That is
        one generalised Pareto fit per excess, so for a default ``level`` of 0.98
        roughly one fit per fifty observations — cheap next to the interval
        between samples in anything this is meant for, and it keeps the fence a
        function of the data rather than of when the last refit happened.

        Examples
        --------
        A thousand scores from the distribution it was fitted on, streamed past a
        fence asking for one alert in a thousand:

        >>> import numpy as np
        >>> from hazure import TimeSeries
        >>> rng = np.random.default_rng(1)
        >>> time = np.arange(4000, dtype=np.int64) * 60_000_000_000
        >>> history = TimeSeries.from_arrays(time, rng.exponential(size=4000))
        >>> threshold = PotThreshold(high=1e-3).fit(history)
        >>> labels = threshold.update_many(rng.exponential(size=1000))
        >>> int(labels.sum())
        2

        A genuine excursion is caught, and does not drag the fence up after it:

        >>> before = threshold.high_
        >>> threshold.update(60.0)
        1.0
        >>> threshold.high_ == before
        True
        """
        if not self.fitted:
            msg = (
                "PotThreshold must be fitted before update() can extend it. "
                "Call fit() on a period you are willing to call normal."
            )
            raise RuntimeError(msg)
        if self._column_models is not None:
            msg = (
                f"This PotThreshold was fitted on {list(self._feature_names or ())} "
                f"and holds one independent fence per column, so update() has no "
                f"single tail to extend. Stream each column through its own "
                f"threshold, fitted on that column."
            )
            raise RuntimeError(msg)
        if math.isnan(value):
            return math.nan
        if math.isnan(self.low_) or math.isnan(self.high_):
            return math.nan

        if value > self.high_ or value < self.low_:
            # Neither the count nor the excesses move. They describe the sample the
            # tail was estimated from, and this score is being declared not to
            # belong to it — letting it raise the count would lower the fence,
            # which is how a burst of anomalies talks an adaptive threshold into
            # accepting the next one.
            return 1.0

        self._seen += 1
        for side, target in (("high", self.high), ("low", self.low)):
            if target is not None and side in self._peaks:
                self._extend(side, -value if side == "low" else value, target)
        return 0.0

    def _extend(self, side: str, value: float, target: float) -> None:
        """Take one accepted score into a side's tail, refitting if it lands there.

        Parameters
        ----------
        side
            ``"low"`` or ``"high"``.
        value
            The score, oriented so this side's tail is the upper one.
        target
            Exceedance probability asked of this side.
        """
        fitted = self.tail_[side]
        sign = -1.0 if side == "low" else 1.0
        start = sign * fitted["start"]

        if value > start:
            self._peaks[side] = np.append(self._peaks[side], value - start)
            peaks = self._peaks[side]
            shape, scale = _fit_gpd(peaks)
            fitted["shape"], fitted["scale"] = shape, scale
            fitted["peaks"] = float(peaks.size)
        else:
            peaks = self._peaks[side]
            shape, scale = fitted["shape"], fitted["scale"]

        conditional = target * self._seen / peaks.size
        # An accumulating count can push the target back out of the tail, at
        # which point the previous fence is the last honest answer available.
        if conditional < 1.0:
            fence = sign * _fence(start, shape, scale, conditional)
            if side == "high":
                self.high_ = fence
            else:
                self.low_ = fence

    def update_many(
        self, values: Sequence[float] | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Absorb a batch of scores in order, returning one label each.

        Parameters
        ----------
        values
            Scores, in the order they were observed.

        Returns
        -------
        numpy.ndarray
            One label per score, decided against the fence as it stood when that
            score arrived.

        Raises
        ------
        RuntimeError
            The threshold has not been fitted, or was fitted on a frame.

        See Also
        --------
        update : The same, one score at a time.
        """
        column = np.asarray(values, dtype=np.float64).ravel()
        return np.array([self.update(float(item)) for item in column], dtype=np.float64)


# ---------------------------------------------------------------------------
# the tail model
# ---------------------------------------------------------------------------


def _fence(start: float, shape: float, scale: float, conditional: float) -> float:
    """Invert the fitted tail to find the score exceeded with a given probability.

    Parameters
    ----------
    start
        Where the tail was taken to begin.
    shape
        Generalised Pareto shape.
    scale
        Generalised Pareto scale, positive.
    conditional
        Target probability, already rescaled to be conditional on having cleared
        ``start``. Must be in ``(0, 1)``.

    Returns
    -------
    float
        The cut-off.
    """
    if shape == 0.0:
        return start - scale * math.log(conditional)
    return start + (scale / shape) * (math.pow(conditional, -shape) - 1.0)


def _fit_gpd(peaks: NDArray[np.float64]) -> tuple[float, float]:
    """Maximum-likelihood shape and scale of a generalised Pareto, by Grimshaw.

    Parameters
    ----------
    peaks
        Strictly positive excesses over the tail start.

    Returns
    -------
    tuple of float
        ``(shape, scale)``. The scale is always positive; a shape of exactly zero
        means the exponential limit fitted better than any root did.

    Notes
    -----
    Writing ``mean_i`` for an average over the excesses, the substitution
    ``theta = gamma / sigma`` gives the profile maximum a closed form,
    ``gamma(theta) = mean_i log(1 + theta * Y_i)``, so the two-parameter fit
    collapses to finding the ``theta`` where::

        mean_i 1 / (1 + theta * Y_i)
            * (1 + mean_i log(1 + theta * Y_i)) = 1

    ``theta = 0`` solves that for every sample, and is the exponential case — so it
    is evaluated directly rather than searched for, and the search covers the two
    sides of it separately.
    """
    largest = float(peaks.max())
    mean = float(peaks.mean())

    # The exponential limit, which is both a real candidate and the fallback if
    # no root survives: its scale is always positive and its likelihood finite.
    best_shape, best_scale = 0.0, mean
    best = _log_likelihood(peaks, 0.0, mean)

    for theta in _stationary_points(peaks, largest, mean):
        shape = float(np.mean(np.log1p(theta * peaks)))
        scale = shape / theta
        if not (scale > 0.0 and math.isfinite(scale) and math.isfinite(shape)):
            continue
        likelihood = _log_likelihood(peaks, shape, scale)
        if likelihood > best:
            best_shape, best_scale, best = shape, scale, likelihood

    return best_shape, best_scale


def _stationary_points(
    peaks: NDArray[np.float64], largest: float, mean: float
) -> Iterator[float]:
    """Yield the non-zero roots of Grimshaw's likelihood equation.

    Parameters
    ----------
    peaks
        Strictly positive excesses.
    largest, mean
        Their maximum and mean, already computed by the caller.

    Yields
    ------
    float
        One root per sign change found on the scan.

    Notes
    -----
    The two sides of zero need scanning very differently, because the equation
    behaves very differently on them.

    Below zero the domain stops at ``-1 / max(peaks)``, where the log term
    diverges, and that edge is close to where the estimate ends up whenever the
    likelihood is maximised on the boundary — a light tail whose fitted endpoint
    coincides with the largest excess observed. So the negative grid is placed as
    a fraction of the distance to that edge and packed towards both ends of it,
    since an evenly spaced grid resolves neither the root that sits against the
    edge nor the one that creeps towards zero.

    Above zero the domain is unbounded, and the equation falls monotonically
    towards ``-1`` as theta grows, crossing at most once. A logarithmic grid
    spanning twenty-four decades of ``theta`` therefore cannot step over the
    crossing, and its far end lies where the implied shape is larger than any
    tail worth fitting.
    """

    def equation(theta: float) -> float:
        terms = 1.0 + theta * peaks
        return float(np.mean(1.0 / terms) * (1.0 + np.mean(np.log(terms))) - 1.0)

    ends = np.logspace(-12.0, -1.0, _SCAN_POINTS)
    fractions = np.unique(
        np.concatenate([ends, np.linspace(0.1, 0.9, _SCAN_POINTS), 1.0 - ends])
    )
    # Ascending in theta: the largest fraction of the distance to the edge is the
    # most negative, so the fractions run backwards.
    yield from _roots(equation, (-1.0 / largest) * fractions[::-1])
    yield from _roots(equation, np.logspace(-12.0, 12.0, 4 * _SCAN_POINTS) / mean)


def _roots(
    equation: Callable[[float], float], grid: NDArray[np.float64]
) -> Iterator[float]:
    """Bisect every sign change of ``equation`` across an ascending scan.

    Parameters
    ----------
    equation
        The function to solve for zero.
    grid
        Points to evaluate, strictly increasing.

    Yields
    ------
    float
        One root per sign change.
    """
    values = [equation(float(point)) for point in grid]
    for index in range(len(values) - 1):
        left, right = values[index], values[index + 1]
        if not (math.isfinite(left) and math.isfinite(right)):
            continue
        if left == 0.0:
            yield float(grid[index])
            continue
        if (left < 0.0) == (right < 0.0):
            continue
        yield _bisect(equation, float(grid[index]), float(grid[index + 1]), left)


def _bisect(
    equation: Callable[[float], float], low: float, high: float, at_low: float
) -> float:
    """Halve a bracket known to change sign until it is narrower than the tolerance.

    Parameters
    ----------
    equation
        The function to solve for zero.
    low, high
        A bracket over which ``equation`` changes sign.
    at_low
        ``equation(low)``, already computed by the caller.

    Returns
    -------
    float
        The midpoint of the final bracket.

    Notes
    -----
    The stopping rule has to allow for the bracket running out of floats before it
    runs out of width. The scan that produced these brackets packs points hard
    against the edge of the domain, where two neighbours can be a part in a
    trillion apart while sitting on numbers of order one — and a relative
    tolerance alone would then ask for a width no float64 between them can express,
    leaving the midpoint equal to one of the ends and the loop turning forever. So
    the bracket collapsing is a stopping condition in its own right.
    """
    negative_at_low = at_low < 0.0
    span = high - low
    middle = 0.5 * (low + high)
    while high - low > _TOLERANCE * span:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        value = equation(middle)
        if value == 0.0:
            return middle
        if (value < 0.0) == negative_at_low:
            low = middle
        else:
            high = middle
    return middle


def _log_likelihood(peaks: NDArray[np.float64], shape: float, scale: float) -> float:
    """Log-likelihood of a generalised Pareto at one parameter pair.

    Parameters
    ----------
    peaks
        Strictly positive excesses.
    shape
        Generalised Pareto shape. Zero is read as the exponential limit rather
        than approached through it.
    scale
        Generalised Pareto scale.

    Returns
    -------
    float
        The log-likelihood, or ``-inf`` where the parameters put any excess
        outside the distribution's support.
    """
    if scale <= 0.0:
        return -math.inf
    n = peaks.size
    if shape == 0.0:
        return -n * math.log(scale) - float(peaks.sum()) / scale
    support = 1.0 + shape * peaks / scale
    if bool(np.any(support <= 0.0)):
        return -math.inf
    return -n * math.log(scale) - (1.0 + 1.0 / shape) * float(np.sum(np.log(support)))
