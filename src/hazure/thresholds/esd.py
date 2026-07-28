"""The generalized extreme Studentized deviate test.

Rather than picking a cut-off and counting what falls outside it, this
asks whether the most extreme point is too extreme to belong to a normal
sample of this size, removes it if so, and repeats.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from hazure import BaseThreshold
from hazure.thresholds.fence import _valid

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries


__all__ = [
    "EsdThreshold",
]


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
