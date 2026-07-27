"""Turning a continuous score into 1.0 / 0.0 / NaN labels.

Every rule here answers the same question — *where do we draw the line?* — and
none of them knows or cares which scorer produced the numbers. Keeping the line
separate from the score is what lets one policy be reused across every scorer, a
score be inspected on its own for ranking, and a test be swapped without
touching the thing being tested.

Two conventions hold throughout:

* A score of NaN means "unknown", and an unknown score cannot be normal, so its
  label is NaN too. Labels are never invented for points nobody measured.
* Fitting on scores that are *entirely* missing leaves the cut-offs unknown, and
  every label is then NaN. An all-missing column part-way through a pipeline is
  answered with "no idea" rather than an exception, which is both the honest
  answer and the one that composes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np

from hazure import BaseThreshold

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "MAD_SCALE",
    "EsdThreshold",
    "Factor",
    "FactorSpec",
    "FixedThreshold",
    "IqrThreshold",
    "MadThreshold",
    "QuantileThreshold",
]

#: Scale factor that turns a median absolute deviation into an estimate of the
#: standard deviation of a normal sample. For X ~ N(mu, sigma) the median of
#: |X - mu| is sigma * Phi^-1(0.75) = 0.6745 * sigma, so dividing the MAD by that
#: constant — equivalently multiplying by 1 / 0.6745 = 1.4826 — puts the MAD on
#: the same scale as a standard deviation. Without it, ``factor=3`` would mean
#: three MADs, which is only two standard deviations.
MAD_SCALE: Final = 1.482602218505602

#: A tail factor: a number, or None to leave that tail unbounded.
Factor = float | None
#: One factor for both tails, or ``(low, high)``.
FactorSpec = Factor | tuple[Factor, Factor]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _label(values: NDArray[np.float64], low: float, high: float) -> NDArray[np.float64]:
    """Flag values outside ``[low, high]``, leaving missing scores unknown."""
    if math.isnan(low) or math.isnan(high):
        # The cut-offs were never learned from valid data, so no label can be
        # justified for any point.
        return np.full(values.shape, np.nan, dtype=np.float64)
    labels = ((values > high) | (values < low)).astype(np.float64)
    labels[np.isnan(values)] = np.nan
    return labels


def _valid(ts: TimeSeries) -> NDArray[np.float64]:
    """Return the non-missing values of a univariate series."""
    column = ts.values[:, 0]
    return column[~np.isnan(column)]


def _factors(spec: FactorSpec, name: str) -> tuple[Factor, Factor]:
    """Expand a scalar-or-pair factor into ``(low, high)``.

    Parameters
    ----------
    spec
        One factor for both tails, or ``(low, high)``. ``None`` on a side means
        that side is unbounded.
    name
        Parameter name, for error messages.

    Returns
    -------
    tuple
        The low-side and high-side factors.

    Raises
    ------
    ValueError
        The pair is not of length two, or a factor is negative.
    """
    if isinstance(spec, tuple):
        pair: tuple[Factor, ...] = spec
        if len(pair) != 2:
            msg = (
                f"{name} must be a number, None, or a (low, high) pair; got "
                f"{len(pair)} items."
            )
            raise ValueError(msg)
        low, high = pair
    else:
        low, high = spec, spec

    for side, value in (("low", low), ("high", high)):
        if value is not None and value < 0:
            msg = (
                f"The {side} side of {name}={spec!r} is negative, which would "
                f"invert the normal range. Use a non-negative factor, or None "
                f"to leave that side unbounded."
            )
            raise ValueError(msg)
    return low, high


def _bound(centre: float, spread: float, factor: Factor, *, upper: bool) -> float:
    """Offset ``centre`` by ``factor * spread``, or go unbounded for ``None``."""
    if factor is None:
        return math.inf if upper else -math.inf
    return centre + factor * spread if upper else centre - factor * spread


def _require_a_bound(low: object, high: object, name: str) -> None:
    """Reject a threshold that could never flag anything."""
    if low is None and high is None:
        msg = (
            f"{name} needs at least one of low=... or high=...; with both unset "
            f"it can never flag a point."
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


class FixedThreshold(BaseThreshold):
    """Flag scores outside a range the caller supplies.

    Nothing is learned, so this is usable without :meth:`fit`. It is the right
    choice when the acceptable range comes from domain knowledge — a pressure
    limit, a service-level objective — rather than from history.

    Parameters
    ----------
    low
        Scores below this are flagged. None leaves the lower side unbounded.
    high
        Scores above this are flagged. None leaves the upper side unbounded.

    Raises
    ------
    ValueError
        Both bounds are None, which would make the threshold inert.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [0.0, 9.0, 1.0, np.nan, -9.0])
    >>> FixedThreshold(low=-5.0, high=5.0).run(ts).values.ravel()
    array([ 0.,  1.,  0., nan,  1.])
    """

    trainable: ClassVar[bool] = False

    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        _require_a_bound(low, high, "FixedThreshold")
        self.low = low
        self.high = high

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        _require_a_bound(self.low, self.high, "FixedThreshold")
        low = -math.inf if self.low is None else float(self.low)
        high = math.inf if self.high is None else float(self.high)
        return ts.wrap(_label(ts.values[:, 0], low, high))


class QuantileThreshold(BaseThreshold):
    """Flag scores beyond quantiles of the training scores.

    The quantiles are turned into absolute cut-offs at :meth:`fit` time, so the
    line is drawn by history and then held fixed: a later series is judged
    against what used to be normal, not against itself.

    Parameters
    ----------
    low
        Lower quantile in ``[0, 1]``. None leaves the lower side unbounded.
    high
        Upper quantile in ``[0, 1]``. None leaves the upper side unbounded.

    Attributes
    ----------
    low_ : float
        Absolute lower cut-off learned from the training scores.
    high_ : float
        Absolute upper cut-off learned from the training scores.

    Raises
    ------
    ValueError
        Both quantiles are None, or one falls outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [1.0, 2.0, 3.0, 4.0, 100.0])
    >>> threshold = QuantileThreshold(high=0.9).fit(ts)
    >>> round(threshold.high_, 1)
    61.6
    >>> threshold.run(ts).values.ravel()
    array([0., 0., 0., 0., 1.])
    """

    low_: float
    high_: float

    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        _require_a_bound(low, high, "QuantileThreshold")
        for name, value in (("low", low), ("high", high)):
            if value is not None and not 0.0 <= value <= 1.0:
                msg = (
                    f"QuantileThreshold {name}={value} is not a quantile; it must "
                    f"lie in [0, 1]. For an absolute cut-off use FixedThreshold."
                )
                raise ValueError(msg)
        self.low = low
        self.high = high

    def _learn(self, ts: TimeSeries) -> None:
        _require_a_bound(self.low, self.high, "QuantileThreshold")
        observed = _valid(ts)
        if observed.size == 0:
            self.low_ = self.high_ = math.nan
            return
        self.low_ = (
            -math.inf if self.low is None else float(np.quantile(observed, self.low))
        )
        self.high_ = (
            math.inf if self.high is None else float(np.quantile(observed, self.high))
        )

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(_label(ts.values[:, 0], self.low_, self.high_))


class IqrThreshold(BaseThreshold):
    """Flag scores far outside the training inter-quartile range.

    The cut-offs are ``Q1 - factor_low * IQR`` and ``Q3 + factor_high * IQR``,
    the rule behind a box plot's whiskers. Quartiles ignore the tails, so the
    very outliers being looked for cannot drag the line out to meet them, which
    is why this is the test the compound detectors end with.

    Parameters
    ----------
    factor
        One factor for both tails, or ``(low, high)``. ``None`` on a side leaves
        that side unbounded, which is how a one-sided test on a magnitude is
        expressed: ``factor=(None, 3.0)``.

    Attributes
    ----------
    low_ : float
        Absolute lower cut-off learned from the training scores.
    high_ : float
        Absolute upper cut-off learned from the training scores.

    Raises
    ------
    ValueError
        A factor is negative, or the pair is not of length two.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-10", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [1.0, 2, 3, 4, 5, 6, 7, 8, 99])
    >>> threshold = IqrThreshold(factor=1.5).fit(ts)
    >>> (threshold.low_, threshold.high_)
    (-3.0, 13.0)
    >>> threshold.run(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 1.])
    """

    low_: float
    high_: float

    def __init__(self, factor: FactorSpec = 3.0) -> None:
        _factors(factor, "factor")
        self.factor = factor

    def _learn(self, ts: TimeSeries) -> None:
        low_factor, high_factor = _factors(self.factor, "factor")
        observed = _valid(ts)
        if observed.size == 0:
            self.low_ = self.high_ = math.nan
            return
        q1, q3 = np.quantile(observed, [0.25, 0.75])
        spread = float(q3 - q1)
        self.low_ = _bound(float(q1), spread, low_factor, upper=False)
        self.high_ = _bound(float(q3), spread, high_factor, upper=True)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(_label(ts.values[:, 0], self.low_, self.high_))


class MadThreshold(BaseThreshold):
    """Flag scores far from the training median, in units of the MAD.

    The median absolute deviation is the median of ``|x - median(x)|``. Scaled by
    :data:`MAD_SCALE` it estimates the standard deviation of a normal sample, so
    ``factor`` reads on the familiar "number of sigmas" scale while staying
    immune to the outliers a real standard deviation would absorb.

    Parameters
    ----------
    factor
        One factor for both tails, or ``(low, high)``. ``None`` on a side leaves
        that side unbounded.

    Attributes
    ----------
    low_ : float
        Absolute lower cut-off learned from the training scores.
    high_ : float
        Absolute upper cut-off learned from the training scores.

    Raises
    ------
    ValueError
        A factor is negative, or the pair is not of length two.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [4.0, 5.0, 6.0, 5.0, 4.0, 6.0, 40.0])
    >>> MadThreshold().fit(ts).run(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 1.])
    """

    low_: float
    high_: float

    def __init__(self, factor: FactorSpec = 3.0) -> None:
        _factors(factor, "factor")
        self.factor = factor

    def _learn(self, ts: TimeSeries) -> None:
        low_factor, high_factor = _factors(self.factor, "factor")
        observed = _valid(ts)
        if observed.size == 0:
            self.low_ = self.high_ = math.nan
            return
        centre = float(np.median(observed))
        spread = MAD_SCALE * float(np.median(np.abs(observed - centre)))
        self.low_ = _bound(centre, spread, low_factor, upper=False)
        self.high_ = _bound(centre, spread, high_factor, upper=True)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        return ts.wrap(_label(ts.values[:, 0], self.low_, self.high_))


class EsdThreshold(BaseThreshold):
    """Flag scores by the generalised extreme Studentized deviate test.

    The test [1]_ repeatedly removes the observation furthest from the mean and
    compares its Studentized deviate against a critical value derived from the t
    distribution, stopping at the first observation the test accepts. Fitting
    therefore splits the training scores into outliers and a normal set, of which
    only three sufficient statistics need keeping: count, sum and sum of squares.
    Prediction adds one candidate point back to that set and runs a single step
    of the test, which is pure arithmetic and so vectorises over the whole series
    at once.

    The test assumes approximately normal scores. Use it only where that holds;
    :class:`IqrThreshold` makes no distributional assumption.

    Parameters
    ----------
    alpha
        Significance level, in ``(0, 1)``.

    Attributes
    ----------
    count_ : int
        Number of training scores judged normal.
    sum_ : float
        Their sum.
    sum_squares_ : float
        Their sum of squares.
    critical_value_ : float
        Critical value for the one-step test used at prediction time.

    Raises
    ------
    ValueError
        ``alpha`` is not in ``(0, 1)``.
    ImportError
        SciPy is not installed.

    Notes
    -----
    Fitting is a sequential loop, because each removal moves the mean that
    decides the next removal. Sorting the scores first makes every step O(1) —
    the point furthest from the mean is always one of the two extremes of what
    remains — so the whole fit costs O(n log n).

    Fewer than two valid training scores leave the statistics unknown and every
    label NaN, since a spread cannot be estimated from a single point.

    References
    ----------
    .. [1] B. Rosner, "Percentage Points for a Generalized ESD Many-Outlier
       Procedure", Technometrics 25(2), 1983, pp. 165-172.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> values = rng.normal(size=60)
    >>> values[17] = 12.0
    >>> time = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> labels = EsdThreshold().fit(ts).run(ts).values.ravel()
    >>> np.flatnonzero(labels == 1.0)
    array([17])
    """

    count_: int
    sum_: float
    sum_squares_: float
    critical_value_: float

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            msg = f"EsdThreshold alpha={alpha} must lie strictly between 0 and 1."
            raise ValueError(msg)
        self.alpha = alpha

    def _learn(self, ts: TimeSeries) -> None:
        observed = np.sort(_valid(ts))
        n = int(observed.size)
        if n < 2:
            self.count_ = 0
            self.sum_ = self.sum_squares_ = self.critical_value_ = math.nan
            return

        # One vectorised call covers every step of the loop below, so SciPy is
        # touched once per fit rather than once per removal.
        criticals = _critical_values(n, np.arange(1, max(n - 1, 2)), self.alpha)

        total = float(observed.sum())
        squares = float((observed**2).sum())
        count = n
        low, high = 0, n - 1
        for critical in criticals:
            mean = total / count
            take_high = observed[high] - mean >= mean - observed[low]
            index = high if take_high else low
            extreme = float(observed[index])
            variance = max((squares - total * total / count) / (count - 1), 0.0)
            spread = math.sqrt(variance)
            statistic = abs(extreme - mean) / spread if spread > 0.0 else 0.0
            if statistic <= critical:
                break
            total -= extreme
            squares -= extreme * extreme
            count -= 1
            if take_high:
                high -= 1
            else:
                low += 1

        self.count_ = count
        self.sum_ = total
        self.sum_squares_ = squares
        self.critical_value_ = float(
            _critical_values(count + 1, np.array([1]), self.alpha)[0]
        )

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        column = ts.values[:, 0]
        if self.count_ < 2:
            return ts.wrap(np.full(column.shape, np.nan))

        # Add each candidate point back to the normal set and re-test it. The
        # mean and standard deviation of that (count_ + 1)-point set follow from
        # the stored sums, so none of the training data needs keeping.
        n = self.count_ + 1
        total = self.sum_ + column
        mean = total / n
        squares = self.sum_squares_ + column**2
        variance = np.maximum((squares - total * total / n) / (n - 1), 0.0)
        spread = np.sqrt(variance)
        statistic = np.divide(
            np.abs(column - mean),
            spread,
            out=np.zeros_like(column),
            where=spread > 0.0,
        )
        labels = (statistic > self.critical_value_).astype(np.float64)
        labels[np.isnan(column)] = np.nan
        return ts.wrap(labels)


def _critical_values(
    n: int, steps: NDArray[np.int_], alpha: float
) -> NDArray[np.float64]:
    """Critical values of the generalised ESD test.

    Parameters
    ----------
    n
        Size of the set under test.
    steps
        Iteration indices, each in ``1 .. n - 2`` so that the t distribution
        keeps at least one degree of freedom.
    alpha
        Significance level.

    Returns
    -------
    numpy.ndarray
        One critical value per step.

    Raises
    ------
    ImportError
        SciPy is not installed.
    """
    try:
        from scipy.stats import t as student_t  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised by the extras job
        msg = (
            "EsdThreshold needs SciPy for the t distribution. Install it with "
            "`pip install hazure[stats]`."
        )
        raise ImportError(msg) from exc

    remaining = n - steps
    probability = 1.0 - alpha / (2.0 * (remaining + 1))
    degrees = remaining - 1
    quantile: NDArray[np.float64] = student_t.ppf(probability, degrees)
    denominator = np.sqrt((degrees + quantile**2) * (remaining + 1))
    critical: NDArray[np.float64] = remaining * quantile / denominator
    return critical
