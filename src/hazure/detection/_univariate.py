"""Ready-made detectors for one series at a time.

Each is a named pairing of a scorer with a threshold, chosen so that the common
cases need no assembly. The parts stay reachable as ``.scorer`` and
``.threshold``, so a detector doubles as a worked example of how to build one.

Handed a frame, every detector here fans out into one independently fitted copy
per column, and the labels come back under the input column names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

from hazure.detection._composition import ScoreDetector, SignedScoreDetector
from hazure.scoring import (
    AutoregressionResidualScorer,
    DoubleRollingScorer,
    SeasonalResidualScorer,
)
from hazure.thresholds import (
    EsdThreshold,
    FixedThreshold,
    IqrThreshold,
    QuantileThreshold,
)

if TYPE_CHECKING:
    from hazure._core.window import Window
    from hazure.detection._composition import Side
    from hazure.features import Regressor
    from hazure.thresholds import Factor, FactorSpec

__all__ = [
    "AutoregressionDetector",
    "EsdDetector",
    "IqrDetector",
    "LevelShiftDetector",
    "QuantileDetector",
    "SeasonalDetector",
    "SpikeDetector",
    "ThresholdDetector",
    "VolatilityShiftDetector",
]

#: Statistics that summarise where a window sits, for :class:`SpikeDetector`.
_CENTRE_AGGS: Final = ("median", "mean")
#: Statistics that summarise how much a window varies, for
#: :class:`VolatilityShiftDetector`.
_SPREAD_AGGS: Final = ("std", "var", "iqr", "idr")


def _check_agg(agg: object, allowed: tuple[str, ...], detector: str) -> None:
    """Reject an aggregation that does not measure what the detector needs."""
    if agg not in allowed:
        msg = f"{detector} agg={agg!r} must be one of {list(allowed)}."
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# the value of a point, judged on its own
# ---------------------------------------------------------------------------


class ThresholdDetector(ScoreDetector):
    """Flag values outside a range the caller supplies.

    The simplest possible detector, and the only one that learns nothing: use it
    when the acceptable range is known in advance. There is no scorer, because a
    value is already the quantity being judged.

    Parameters
    ----------
    low
        Values below this are anomalous. None leaves the lower side unbounded.
    high
        Values above this are anomalous. None leaves the upper side unbounded.

    Raises
    ------
    ValueError
        Both bounds are None, which would make the detector inert.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [20.0, 21.0, 45.0, 19.0, -5.0])
    >>> ThresholdDetector(low=0.0, high=40.0).detect(ts).values.ravel()
    array([0., 0., 1., 0., 1.])
    """

    trainable: ClassVar[bool] = False

    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        self.low = low
        self.high = high
        self._build()

    def _build(self) -> None:
        self.scorer = None
        self.threshold = FixedThreshold(low=self.low, high=self.high)


class QuantileDetector(ScoreDetector):
    """Flag values in the tails of the training distribution.

    Makes no assumption about the shape of that distribution, only about how much
    of it is acceptable: ``high=0.99`` means "the top one per cent of what we have
    seen is worth a look".

    Parameters
    ----------
    low
        Lower quantile in ``[0, 1]``. None leaves the lower side unbounded.
    high
        Upper quantile in ``[0, 1]``. None leaves the upper side unbounded.

    Raises
    ------
    ValueError
        Both quantiles are None, or one falls outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 90])
    >>> time = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> QuantileDetector(high=0.9).fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    """

    def __init__(self, low: float | None = None, high: float | None = None) -> None:
        self.low = low
        self.high = high
        self._build()

    def _build(self) -> None:
        self.scorer = None
        self.threshold = QuantileThreshold(low=self.low, high=self.high)


class IqrDetector(ScoreDetector):
    """Flag values far outside the training inter-quartile range.

    The box-plot rule, and a sound default when nothing is known about the
    distribution: because quartiles ignore the tails, the outliers being looked
    for do not widen the range that is supposed to exclude them.

    Parameters
    ----------
    factor
        One factor for both tails, or ``(low, high)``. ``None`` on a side leaves
        that side unbounded.

    Raises
    ------
    ValueError
        A factor is negative, or the pair is not of length two.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.array([10.0, 11, 12, 11, 10, 12, 11, 10, 11, 60])
    >>> time = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> IqrDetector().fit_detect(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
    """

    def __init__(self, factor: FactorSpec = 3.0) -> None:
        self.factor = factor
        self._build()

    def _build(self) -> None:
        self.scorer = None
        self.threshold = IqrThreshold(factor=self.factor)


class EsdDetector(ScoreDetector):
    """Flag values by the generalised extreme Studentized deviate test.

    Sets the line by a significance level rather than by a factor, which is
    useful when a false-positive rate is easier to justify than a multiple of a
    spread. Assumes the values are approximately normal; where that is doubtful,
    :class:`IqrDetector` asks less of the data.

    Parameters
    ----------
    alpha
        Significance level, in ``(0, 1)``.

    Raises
    ------
    ValueError
        ``alpha`` is not in ``(0, 1)``.
    ImportError
        SciPy is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(1)
    >>> values = rng.normal(loc=20.0, size=60)
    >>> values[42] = 30.0
    >>> time = np.arange("2024-01-01", "2024-03-01", dtype="datetime64[D]")
    >>> labels = EsdDetector().fit_detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([42])
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._build()

    def _build(self) -> None:
        self.scorer = None
        self.threshold = EsdThreshold(alpha=self.alpha)


# ---------------------------------------------------------------------------
# the value of a point, judged against its neighbourhood
# ---------------------------------------------------------------------------


class SpikeDetector(SignedScoreDetector):
    """Flag points that depart sharply from the values just before them.

    The window in front of each point is one observation wide and the window
    behind it is ``window`` wide: a short right window catches the blip while the
    long left window keeps a stable notion of recent normal, which is what makes
    this asymmetry the shape of spike detection. Because the comparison is local,
    a slow drift is invisible to it — which is the point.

    Parameters
    ----------
    window
        Size of the preceding window: observations (``int``) or a duration.
        The default of 1 compares each point with the one before it.
    factor
        Inter-quartile-range factor deciding how large a departure is too large.
    side
        ``"both"``, ``"positive"`` for jumps up only, ``"negative"`` for drops
        only.
    min_periods
        Minimum non-missing observations in the preceding window.
    agg
        How to summarise the preceding window: ``"median"`` or ``"mean"``. The
        median is unmoved by an earlier spike still inside the window.

    Raises
    ------
    ValueError
        ``side`` or ``agg`` is not one of the listed choices.

    Notes
    -----
    With the default ``window=1`` the preceding window is a single observation, so
    a one-point spike changes the score twice — once on the way up and once on the
    way back down — and ``side="both"`` flags both the spike and the point after
    it. ``side="positive"`` isolates the spike itself. A wider window has a median
    the spike cannot move, and then the spike alone scores.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.ones(20)
    >>> values[12] = 9.0
    >>> time = np.arange("2024-01-01", "2024-01-21", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> np.flatnonzero(SpikeDetector().fit_detect(ts).values.ravel() == 1.0)
    array([12, 13])
    >>> labels = SpikeDetector(side="positive").fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([12])
    """

    def __init__(
        self,
        window: Window = 1,
        factor: Factor = 3.0,
        side: Side = "both",
        min_periods: int | None = None,
        agg: str = "median",
    ) -> None:
        self.window = window
        self.factor = factor
        self.side = side
        self.min_periods = min_periods
        self.agg = agg
        self._build()

    def _build(self) -> None:
        super()._build()
        _check_agg(self.agg, _CENTRE_AGGS, "SpikeDetector")
        self.scorer = DoubleRollingScorer(
            window=(self.window, 1),
            agg=self.agg,
            diff="diff",
            min_periods=(self.min_periods, 1),
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))


class LevelShiftDetector(SignedScoreDetector):
    """Flag the point at which the series settles at a new level.

    Two windows of equal length, one either side of each point, are summarised
    and compared. Both being long is what separates a level shift from a spike:
    a single odd value barely moves the median of a wide window, while a genuine
    step moves one window's median entirely away from the other's.

    Parameters
    ----------
    window
        Size of each window, or ``(left, right)``. Long enough that both sides
        are stable, short enough to place the change precisely.
    factor
        Inter-quartile-range factor deciding how large a shift is too large. Set
        higher than for spike detection by default, because the difference of two
        window medians is a much quieter signal than a single point's departure.
    side
        ``"both"``, ``"positive"`` for shifts up only, ``"negative"`` for shifts
        down only.
    min_periods
        Minimum non-missing observations per window, or ``(left, right)``.

    Raises
    ------
    ValueError
        ``side`` is not one of the three directions.

    Notes
    -----
    The two windows overlap the shift for as long as it takes them to clear it,
    so the score stays high for a run of points around the change and the
    detector flags a short plateau rather than a single instant. Narrowing
    ``window`` narrows the plateau.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.concatenate([np.zeros(20), np.full(20, 10.0)])
    >>> time = np.arange("2024-01-01", "2024-02-10", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> labels = LevelShiftDetector(window=3).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([19, 20, 21])
    """

    def __init__(
        self,
        window: Window | tuple[Window, Window],
        factor: Factor = 6.0,
        side: Side = "both",
        min_periods: int | tuple[int | None, int | None] | None = None,
    ) -> None:
        self.window = window
        self.factor = factor
        self.side = side
        self.min_periods = min_periods
        self._build()

    def _build(self) -> None:
        super()._build()
        self.scorer = DoubleRollingScorer(
            window=self.window,
            agg="median",
            diff="diff",
            min_periods=self.min_periods,
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))


class VolatilityShiftDetector(SignedScoreDetector):
    """Flag the point at which the series becomes more or less erratic.

    The same two symmetric windows as level-shift detection, with two changes
    that matter. The statistic measures spread rather than position, so the level
    can stay put while the noise around it changes. And the comparison is
    *relative* — the change in spread divided by the earlier spread — because a
    doubling of noise is equally significant on a quiet series and a loud one,
    which an absolute difference would not capture.

    Parameters
    ----------
    window
        Size of each window, or ``(left, right)``. Wide enough that a spread can
        be estimated from each side.
    factor
        Inter-quartile-range factor deciding how large a relative change is too
        large.
    side
        ``"both"``, ``"positive"`` for increases in volatility only,
        ``"negative"`` for decreases only.
    min_periods
        Minimum non-missing observations per window, or ``(left, right)``.
    agg
        How to measure spread: ``"std"``, ``"var"``, ``"iqr"`` or ``"idr"``.

    Raises
    ------
    ValueError
        ``side`` or ``agg`` is not one of the listed choices.

    Notes
    -----
    A spread is never negative, so the sign of the relative change is the sign of
    the change itself and ``side`` reads as expected. A window with no spread at
    all makes the relative change undefined, and those points score NaN.

    Two consequences of measuring spread over a window are worth knowing:

    * A relative *increase* is unbounded while a relative *decrease* cannot pass
      -1, so a fall in volatility produces a smaller score than the equivalent
      rise. Detecting ``side="negative"`` usually wants a smaller ``factor``.
    * A level shift falling inside a window inflates that window's spread, so a
      step registers here as well as in :class:`LevelShiftDetector`. Running both
      is how the two are told apart.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> rng = np.random.default_rng(0)
    >>> quiet, loud = rng.normal(scale=0.1, size=40), rng.normal(scale=5.0, size=40)
    >>> time = np.arange("2024-01-01", "2024-03-21", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, np.concatenate([quiet, loud]))
    >>> labels = VolatilityShiftDetector(window=10).fit_detect(ts)
    >>> bool(labels.values.ravel()[40] == 1.0)
    True
    """

    def __init__(
        self,
        window: Window | tuple[Window, Window],
        factor: Factor = 6.0,
        side: Side = "both",
        min_periods: int | tuple[int | None, int | None] | None = None,
        agg: str = "std",
    ) -> None:
        self.window = window
        self.factor = factor
        self.side = side
        self.min_periods = min_periods
        self.agg = agg
        self._build()

    def _build(self) -> None:
        super()._build()
        _check_agg(self.agg, _SPREAD_AGGS, "VolatilityShiftDetector")
        self.scorer = DoubleRollingScorer(
            window=self.window,
            agg=self.agg,
            diff="rel_diff",
            min_periods=self.min_periods,
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))


# ---------------------------------------------------------------------------
# the value of a point, judged against a model of the series
# ---------------------------------------------------------------------------


class SeasonalDetector(SignedScoreDetector):
    """Flag points that break a repeating pattern.

    A daily or weekly cycle is normal behaviour, so it is subtracted before
    anything is judged: what remains is the part of the series the pattern does
    not explain, and it is that remainder the threshold is applied to. A value
    perfectly ordinary for a Tuesday is therefore anomalous on a Sunday.

    Parameters
    ----------
    period
        Length of a cycle in observations. When None it is detected from the
        autocorrelation of the training series.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.
    side
        ``"both"``, ``"positive"`` for values above the pattern only,
        ``"negative"`` for values below it only.
    trend
        Remove a moving-average trend as well as the seasonal profile. Costs a
        NaN margin of half a period at each end, where the centred average has no
        window.

    Raises
    ------
    ValueError
        ``side`` is invalid, the training time axis is irregular, or no period was
        given and none could be detected.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([1.0, 5.0, 3.0, 2.0], 8)
    >>> values[13] = 12.0
    >>> time = np.arange("2024-01-01", "2024-02-02", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> labels = SeasonalDetector(period=4).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([13])
    """

    def __init__(
        self,
        period: int | None = None,
        factor: Factor = 3.0,
        side: Side = "both",
        trend: bool = False,
    ) -> None:
        self.period = period
        self.factor = factor
        self.side = side
        self.trend = trend
        self._build()

    def _build(self) -> None:
        super()._build()
        self.scorer = SeasonalResidualScorer(period=self.period, trend=self.trend)
        self.threshold = IqrThreshold(factor=(None, self.factor))


class AutoregressionDetector(SignedScoreDetector):
    """Flag points their own recent past fails to predict.

    Fits the relationship between each value and the values a few steps before
    it, and judges the signed residual. This asks a sharper question than whether
    a value is unusual: whether it is unusual *given* where the series just was.
    A break in the dynamics is caught even at a perfectly ordinary level.

    Parameters
    ----------
    n_steps
        Number of past values to regress on.
    step_size
        Gap in observations between them. With ``n_steps=2, step_size=3``, the
        values at ``t-3`` and ``t-6`` predict the value at ``t``.
    regressor
        Any object with ``fit(X, y)`` and ``predict(X)`` taking numpy arrays.
        Defaults to ordinary least squares.
    factor
        Inter-quartile-range factor deciding how large a residual is too large.
    side
        ``"both"``, ``"positive"`` for values above the prediction only,
        ``"negative"`` for values below it only.

    Raises
    ------
    ValueError
        ``side`` is invalid, or ``n_steps`` or ``step_size`` is less than 1.

    Notes
    -----
    The first ``n_steps * step_size`` points have an incomplete history and are
    labelled NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([1.0, 3.0, 5.0, 3.0], 8)
    >>> values[17] = 11.0
    >>> time = np.arange("2024-01-01", "2024-02-02", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> labels = AutoregressionDetector(n_steps=3).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([17])
    """

    def __init__(
        self,
        n_steps: int = 1,
        step_size: int = 1,
        regressor: Regressor | None = None,
        factor: Factor = 3.0,
        side: Side = "both",
    ) -> None:
        self.n_steps = n_steps
        self.step_size = step_size
        self.regressor = regressor
        self.factor = factor
        self.side = side
        self._build()

    def _build(self) -> None:
        super()._build()
        self.scorer = AutoregressionResidualScorer(
            n_steps=self.n_steps, step_size=self.step_size, regressor=self.regressor
        )
        self.threshold = IqrThreshold(factor=(None, self.factor))
