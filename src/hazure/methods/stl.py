"""Anomalies as what a seasonal-trend decomposition cannot explain.

A series with a daily rhythm and a slow upward trend is not anomalous for being
high on a Monday afternoon in December. Decomposition takes that reasoning
literally: split the series into a trend, one or more seasonal components, and a
remainder, then judge only the remainder. Whatever the model explains is by
definition normal.

STL — seasonal-trend decomposition using loess — earns its place over a plain
average-by-phase decomposition on two counts. Its seasonal shape is allowed to
*evolve* across the series rather than being one fixed profile repeated, which is
what real daily patterns do; and in its robust mode the loess fits are reweighted
to discount outliers, so a single extreme value does not bend the trend towards
itself and hide in the fit. MSTL then runs STL once per period, so a daily and a
weekly cycle can both be removed — something no single-period decomposition can
do, and the common case for any series sampled more often than daily.

Both are provided by ``statsmodels``, imported lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np

from hazure import BaseScorer
from hazure.detection import ScoreDetector
from hazure.thresholds import IqrThreshold

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from hazure import TimeSeries
    from hazure.thresholds import Factor

__all__ = [
    "MstlDetector",
    "MstlResidualScorer",
    "StlDetector",
    "StlResidualScorer",
]

#: Nanoseconds in a day, the cycle a sub-daily series is assumed to carry when no
#: period is given.
_NS_PER_DAY: Final = 86_400_000_000_000

#: Days in a week, the cycle a daily series is assumed to carry.
_DAYS_PER_WEEK: Final = 7


class StlResidualScorer(BaseScorer):
    """Score each point by the size of its STL residual.

    The series is decomposed into a trend, a seasonal component of one period, and
    a remainder; the score is the magnitude of that remainder. A point scores high
    when it is far from what the rhythm and the drift of the series together
    predicted for its position — which is a different and usually sharper question
    than whether its value is unusual outright.

    Parameters
    ----------
    period
        Length of the cycle, in observations. None derives it from the sampling
        interval: one day's worth of observations for anything sampled more often
        than daily, and one week for daily data. Pass it explicitly whenever the
        cycle is neither.
    robust
        Reweight the loess fits to discount outliers. On by default, and the
        reason to keep it on is circular in a useful way: the anomalies are
        exactly what must not be allowed to shape the decomposition they are being
        measured against. Turning it off is faster and appropriate for clean data.
    seasonal
        Length of the smoother applied to the seasonal component, in observations;
        must be an odd number of at least 7. Larger means a seasonal shape that
        changes more slowly over the series. None leaves it to ``statsmodels``.

    Raises
    ------
    ValueError
        The time axis is irregular, ``period`` is below 2, or no period was given
        and the sampling interval implies none.
    ImportError
        ``statsmodels`` is not installed.

    Notes
    -----
    Requires a regular time axis: a period expressed in observations only means
    something if observations are evenly spaced. Missing observations are filled
    by linear interpolation, because the loess fits need a dense series, and their
    scores are set back to NaN afterwards.

    The score is a magnitude, so it is one-sided by construction. Where the
    *direction* of the departure matters, take the residual's sign from the
    decomposition itself.

    Nothing is learned: STL is a smoother, not a model that predicts, so it has to
    see the series it is decomposing and :meth:`fit` is optional. A consequence
    worth knowing is that the decomposition is retrospective — every point's score
    depends on the whole series, including what came after it.

    References
    ----------
    .. [1] R. B. Cleveland, W. S. Cleveland, J. E. McRae and I. Terpenning,
       "STL: A Seasonal-Trend Decomposition Procedure Based on Loess", Journal of
       Official Statistics 6(1), 1990, pp. 3-73.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> hours = np.arange(240)
    >>> values = 10.0 + 3.0 * np.sin(hours * 2 * np.pi / 24)
    >>> values[100] += 9.0
    >>> time = hours * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> scorer = StlResidualScorer()
    >>> scores = scorer.score(TimeSeries.from_arrays(time, values))
    >>> int(np.nanargmax(scores.values.ravel()))
    100
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        period: int | None = None,
        robust: bool = True,
        seasonal: int | None = None,
    ) -> None:
        self.period = period
        self.robust = robust
        self.seasonal = seasonal

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        period = _resolve_period(ts, self.period, "StlResidualScorer")
        options: dict[str, Any] = {"period": period, "robust": self.robust}
        if self.seasonal is not None:
            options["seasonal"] = int(self.seasonal)
        return _residual_score(ts, "STL", options)


class StlDetector(ScoreDetector):
    """Flag points an STL decomposition cannot account for.

    :class:`StlResidualScorer` paired with an inter-quartile-range rule on the
    residual magnitudes. The rule is learned from the residuals rather than fixed,
    because how large a residual is large depends entirely on how well the
    decomposition fits the series in the first place.

    Parameters
    ----------
    period
        Length of the cycle, in observations. None derives it from the sampling
        interval.
    robust
        Reweight the loess fits to discount outliers.
    seasonal
        Length of the seasonal smoother, an odd number of at least 7.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.

    Raises
    ------
    ValueError
        The time axis is irregular, or the period is unusable.
    ImportError
        ``statsmodels`` is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> hours = np.arange(240)
    >>> values = 10.0 + 3.0 * np.sin(hours * 2 * np.pi / 24) + rng.normal(size=240)
    >>> values[100] += 12.0
    >>> time = hours * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> labels = StlDetector(factor=6.0).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([100])
    """

    def __init__(
        self,
        period: int | None = None,
        robust: bool = True,
        seasonal: int | None = None,
        factor: Factor = 3.0,
    ) -> None:
        self.period = period
        self.robust = robust
        self.seasonal = seasonal
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = StlResidualScorer(
            period=self.period, robust=self.robust, seasonal=self.seasonal
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))


class MstlResidualScorer(BaseScorer):
    """Score each point by the size of its residual after removing several cycles.

    MSTL applies STL once per period, longest cycle last, subtracting each
    seasonal component before fitting the next. A series sampled hourly usually
    has two rhythms — the hour of the day and the day of the week — and a
    single-period decomposition has to choose one and leave the other in the
    residual, where it dominates and drowns everything else. Removing both leaves
    a remainder that is genuinely just the remainder.

    Parameters
    ----------
    periods
        Cycle lengths in observations: one integer, or several. For hourly data,
        ``(24, 168)`` is the daily-and-weekly pair.
    robust
        Reweight the loess fits to discount outliers, in every STL pass.
    windows
        Seasonal smoother length per period, as an integer or one per period.
        None leaves them to ``statsmodels``.

    Raises
    ------
    ValueError
        The time axis is irregular, ``periods`` is empty, or a period is below 2.
    ImportError
        ``statsmodels`` is not installed.

    Notes
    -----
    Requires a regular time axis. Missing observations are interpolated before the
    decomposition and their scores set back to NaN, as for
    :class:`StlResidualScorer`.

    Each period needs two full cycles of data to be estimable, so the longest
    period governs how much history the scorer needs.

    Nothing is learned; :meth:`fit` is optional.

    References
    ----------
    .. [1] K. Bandara, R. J. Hyndman and C. Bergmeir, "MSTL: A Seasonal-Trend
       Decomposition Algorithm for Time Series with Multiple Seasonal Patterns",
       International Journal of Operational Research, 2021.

    Examples
    --------
    A daily and a weekly rhythm together, with one point breaking both:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> hours = np.arange(24 * 28)
    >>> values = (
    ...     10.0
    ...     + 3.0 * np.sin(hours * 2 * np.pi / 24)
    ...     + 5.0 * np.sin(hours * 2 * np.pi / 168)
    ... )
    >>> values[300] += 20.0
    >>> time = hours * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> scorer = MstlResidualScorer(periods=(24, 168))
    >>> scores = scorer.score(TimeSeries.from_arrays(time, values))
    >>> int(np.nanargmax(scores.values.ravel()))
    300
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        periods: int | Sequence[int],
        robust: bool = True,
        windows: int | Sequence[int] | None = None,
    ) -> None:
        _check_periods(periods)
        self.periods = periods
        self.robust = robust
        self.windows = windows

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        _check_periods(self.periods)
        if ts.freq is None:
            msg = _IRREGULAR.format(name="MstlResidualScorer")
            raise ValueError(msg)
        options: dict[str, Any] = {
            "periods": self.periods,
            "stl_kwargs": {"robust": self.robust},
        }
        if self.windows is not None:
            options["windows"] = self.windows
        return _residual_score(ts, "MSTL", options)


class MstlDetector(ScoreDetector):
    """Flag points that break none of several rhythms but still do not fit.

    :class:`MstlResidualScorer` paired with an inter-quartile-range rule. Use this
    rather than :class:`StlDetector` whenever the series has more than one
    rhythm: with a second cycle left in the residual, the residual's spread is set
    by that cycle rather than by the noise, and the threshold ends up asking how
    unusual a point is compared with a systematic pattern instead of compared with
    chance.

    Parameters
    ----------
    periods
        Cycle lengths in observations: one integer, or several.
    robust
        Reweight the loess fits to discount outliers.
    windows
        Seasonal smoother length per period.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.

    Raises
    ------
    ValueError
        The time axis is irregular, or a period is unusable.
    ImportError
        ``statsmodels`` is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> hours = np.arange(24 * 28)
    >>> values = (
    ...     10.0
    ...     + 3.0 * np.sin(hours * 2 * np.pi / 24)
    ...     + 5.0 * np.sin(hours * 2 * np.pi / 168)
    ... )
    >>> values[300] += 20.0
    >>> time = hours * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> labels = MstlDetector(periods=(24, 168), factor=25.0).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([300])
    """

    def __init__(
        self,
        periods: int | Sequence[int],
        robust: bool = True,
        windows: int | Sequence[int] | None = None,
        factor: Factor = 3.0,
    ) -> None:
        self.periods = periods
        self.robust = robust
        self.windows = windows
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = MstlResidualScorer(
            periods=self.periods, robust=self.robust, windows=self.windows
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------

_IRREGULAR: Final = (
    "{name} needs a regular time axis, because a period counted in "
    "observations only lines up with a cycle when observations are evenly "
    "spaced; this series has an irregular or unknown sampling interval. "
    "Resample it first."
)


def _residual_score(ts: TimeSeries, model: str, options: dict[str, Any]) -> TimeSeries:
    """Decompose a series and return the magnitude of the remainder.

    Parameters
    ----------
    ts
        Univariate input.
    model
        ``"STL"`` or ``"MSTL"``, the ``statsmodels`` class to use.
    options
        Keyword arguments for that class.

    Returns
    -------
    TimeSeries
        Absolute residuals, NaN where the input was missing.

    Raises
    ------
    ImportError
        ``statsmodels`` is not installed.
    """
    column = ts.values[:, 0]
    missing = np.isnan(column)
    if bool(missing.all()) or ts.n_rows == 0:
        # Nothing to decompose, and no residual to report.
        return ts.wrap(np.full(ts.n_rows, np.nan))

    decompose = _statsmodels_class(model)
    dense = _fill_gaps(column, missing)
    residual = np.abs(np.asarray(decompose(dense, **options).fit().resid, dtype=float))
    residual[missing] = np.nan
    return ts.wrap(residual)


def _statsmodels_class(model: str) -> Any:
    """Import a ``statsmodels`` decomposition class lazily.

    Parameters
    ----------
    model
        ``"STL"`` or ``"MSTL"``.

    Returns
    -------
    Any
        The class.

    Raises
    ------
    ImportError
        ``statsmodels`` is not installed.
    """
    try:
        from statsmodels.tsa.seasonal import MSTL, STL
    except ImportError as exc:  # pragma: no cover - exercised by the extras job
        msg = (
            f"{model} decomposition needs statsmodels. Install it with "
            f"`pip install hazure[stats]`."
        )
        raise ImportError(msg) from exc
    return STL if model == "STL" else MSTL


def _resolve_period(ts: TimeSeries, period: int | None, name: str) -> int:
    """Settle on a cycle length for a series.

    Parameters
    ----------
    ts
        The series, whose sampling interval is consulted when ``period`` is None.
    period
        The requested period, or None.
    name
        Component name, for error messages.

    Returns
    -------
    int
        The cycle length in observations.

    Raises
    ------
    ValueError
        The time axis is irregular, the given period is below 2, or none was
        given and the sampling interval implies none.
    """
    if ts.freq is None:
        msg = _IRREGULAR.format(name=name)
        raise ValueError(msg)
    if period is not None:
        if period < 2:
            msg = f"period must be at least 2 observations, got {period}."
            raise ValueError(msg)
        return int(period)

    # The conventional cycle for a series sampled this often: a day for anything
    # finer than daily, a week for daily data. Anything else is too ambiguous to
    # guess at, and guessing wrong is worse than asking.
    if _NS_PER_DAY % ts.freq == 0 and ts.freq < _NS_PER_DAY:
        return int(_NS_PER_DAY // ts.freq)
    if ts.freq == _NS_PER_DAY:
        return _DAYS_PER_WEEK
    msg = (
        f"{name} could not infer a period from a sampling interval of "
        f"{ts.freq} ns: a daily cycle would not be a whole number of "
        f"observations. Pass period=... explicitly."
    )
    raise ValueError(msg)


def _check_periods(periods: int | Sequence[int]) -> None:
    """Reject cycle lengths a multi-seasonal decomposition cannot use.

    Parameters
    ----------
    periods
        One period or several.

    Raises
    ------
    ValueError
        No periods were given, or one is below 2 observations.
    """
    candidates = [periods] if isinstance(periods, int) else list(periods)
    if not candidates:
        msg = (
            "periods is empty; pass at least one cycle length, e.g. "
            "periods=24 for a daily cycle in hourly data."
        )
        raise ValueError(msg)
    for value in candidates:
        if value < 2:
            msg = f"Every period must be at least 2 observations, got {value}."
            raise ValueError(msg)


def _fill_gaps(
    column: NDArray[np.float64], missing: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Replace missing observations by linear interpolation between neighbours.

    Parameters
    ----------
    column
        1-D array, not entirely missing.
    missing
        Where ``column`` is NaN.

    Returns
    -------
    numpy.ndarray
        A copy with no missing values.
    """
    if not bool(missing.any()):
        return column
    positions = np.arange(column.shape[0], dtype=np.float64)
    filled = column.copy()
    filled[missing] = np.interp(
        positions[missing], positions[~missing], column[~missing]
    )
    return filled
