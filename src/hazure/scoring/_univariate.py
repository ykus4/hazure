"""Scorers for one series at a time.

Each of these turns a series into a continuous measure of how unusual each point
is. Higher magnitude means more unusual. Several are signed, and the sign is
meaningful — it says which direction the anomaly went — so a detector can act on
increases only, or decreases only, while the magnitude decides how big is too
big.

Handed a frame, every scorer here fans out into one independently fitted copy per
column, so each column learns its own idea of normal.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

from hazure import BaseScorer
from hazure.features import (
    DoubleRollingAggregate,
    RegressionResidual,
    Retrospect,
    RollingAggregate,
    SeasonalDecomposition,
)
from hazure.scoring._adapter import TransformerScorer, complete_rows

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import BaseTransformer, TimeSeries
    from hazure._core.window import Closed, Window
    from hazure.features import Regressor

__all__ = [
    "AutoregressionResidualScorer",
    "DeviationScorer",
    "DoubleRollingScorer",
    "RollingAggregateScorer",
    "SeasonalResidualScorer",
]

#: How to compare the two windows of :class:`DoubleRollingScorer`.
Diff = Literal["l1", "l2", "diff", "rel_diff", "abs_rel_diff"]
#: Where :class:`DeviationScorer` puts the centre of normal.
Centre = Literal["median", "mean"]
#: What :class:`DeviationScorer` measures the deviation in units of.
Scale = Literal["iqr", "idr", "mad", "std"]

#: Scales a median absolute deviation into a standard-deviation estimate; see
#: :data:`hazure.thresholds.MAD_SCALE`.
_MAD_SCALE = 1.482602218505602


class RollingAggregateScorer(TransformerScorer):
    """Score each point by a statistic of the window ending at it.

    A feature-style score: not anomalous or not on its own, but a summary the
    threshold can be pointed at. Counting non-zero values in a rolling day, for
    instance, turns "the pump idled" into a number a quantile rule can judge.

    Parameters
    ----------
    window
        Observations (``int``) or duration (``"7d"``, ``timedelta``).
    agg
        A name from :data:`hazure.AGGREGATIONS`.
    center
        Centre the window on each point instead of trailing it.
    min_periods
        Minimum non-missing observations for a result. Defaults to the full
        window for integer windows and to 1 for duration windows.
    closed
        Which window endpoints to include; defaults to ``"right"``.
    q
        Quantile in ``[0, 1]``, required when ``agg="quantile"``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [1.0, 1.0, 1.0, 9.0, 1.0, 1.0])
    >>> RollingAggregateScorer(window=2, agg="max").score(ts).values.ravel()
    array([nan,  1.,  1.,  9.,  9.,  1.])
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        window: Window,
        agg: str = "mean",
        center: bool = False,
        min_periods: int | None = None,
        closed: Closed | None = None,
        q: float | None = None,
    ) -> None:
        self.window = window
        self.agg = agg
        self.center = center
        self.min_periods = min_periods
        self.closed = closed
        self.q = q

    def _new_transformer(self) -> BaseTransformer:
        return RollingAggregate(
            window=self.window,
            agg=self.agg,
            agg_params=None if self.q is None else {"q": self.q},
            center=self.center,
            min_periods=self.min_periods,
            closed=self.closed,
        )


class DoubleRollingScorer(TransformerScorer):
    """Score each point by how much the series changes across it.

    The window before each point is summarised, the window from it onwards is
    summarised, and the two are compared. One scorer covers three phenomena, and
    only the settings differ:

    * **spike** — a long left window, a right window of 1, so a single blip is
      measured against a stable notion of recent normal;
    * **level shift** — two windows of equal length, long enough that both sides
      are stable, so a persistent change registers and a lone spike does not;
    * **volatility shift** — the same symmetric windows with a dispersion
      statistic (``agg="std"``) and a relative comparison
      (``diff="rel_diff"``), because a doubling of noise matters equally whether
      the series is quiet or loud.

    Parameters
    ----------
    window
        One spec for both sides, or ``(left, right)``.
    agg
        One statistic for both sides, or ``(left, right)``. The median resists
        the very outliers being detected.
    diff
        How to compare the two sides: ``"l1"`` or ``"l2"`` for the unsigned
        magnitude, ``"diff"`` for the signed ``right - left``, ``"rel_diff"`` for
        that divided by the left value, ``"abs_rel_diff"`` for its magnitude.
    min_periods
        Minimum non-missing observations per side, or ``(left, right)``.
    q
        Quantile in ``[0, 1]``, required when ``agg="quantile"``.

    Examples
    --------
    A step of 5 registers at the step, and the score is unavailable within a
    window of each end:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    >>> DoubleRollingScorer(window=2, diff="diff").score(ts).values.ravel()
    array([nan, nan, 2.5, 5. , 2.5, nan])
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        window: Window | tuple[Window, Window],
        agg: str | tuple[str, str] = "median",
        diff: Diff = "l1",
        min_periods: int | tuple[int | None, int | None] | None = None,
        q: float | None = None,
    ) -> None:
        self.window = window
        self.agg = agg
        self.diff = diff
        self.min_periods = min_periods
        self.q = q

    def _new_transformer(self) -> BaseTransformer:
        return DoubleRollingAggregate(
            window=self.window,
            agg=self.agg,
            agg_params=None if self.q is None else {"q": self.q},
            min_periods=self.min_periods,
            diff=self.diff,
        )


class DeviationScorer(BaseScorer):
    """Score each point by its signed distance from a learned centre.

    A robust z-score: ``(x - centre) / scale``, where both are learned once from
    the training series. With a median centre and an inter-quartile-range scale,
    neither estimate is moved by the outliers being looked for, which a mean and
    a standard deviation both are — a single value a thousand times too large
    inflates the scale enough to hide itself.

    The score keeps its sign, so a detector can act on excursions in one
    direction only.

    Parameters
    ----------
    center
        Where normal sits: ``"median"`` or ``"mean"``.
    scale
        What one unit of deviation is: ``"iqr"`` (quartile spread), ``"idr"``
        (10th-to-90th-percentile spread), ``"mad"`` (median absolute deviation,
        scaled to estimate a standard deviation) or ``"std"``.

    Attributes
    ----------
    center_ : float
        The learned centre.
    scale_ : float
        The learned scale. Zero for a constant training series.

    Raises
    ------
    ValueError
        ``center`` or ``scale`` is not one of the listed choices.

    Notes
    -----
    A constant training series has no spread, so every point equal to the centre
    scores 0 and anything else scores infinity. That is the honest reading: with
    no observed variation, any change at all is unprecedented.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> time = np.arange("2024-01-01", "2024-01-08", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, [10.0, 12.0, 11.0, 13.0, 12.0, 10.0, 40.0])
    >>> scorer = DeviationScorer().fit(ts)
    >>> (scorer.center_, scorer.scale_)
    (12.0, 2.0)
    >>> scorer.score(ts).values.ravel()
    array([-1. ,  0. , -0.5,  0.5,  0. , -1. , 14. ])
    """

    center_: float
    scale_: float

    def __init__(self, center: Centre = "median", scale: Scale = "iqr") -> None:
        _check_choice(center, ("median", "mean"), "center")
        _check_choice(scale, ("iqr", "idr", "mad", "std"), "scale")
        self.center = center
        self.scale = scale

    def _learn(self, ts: TimeSeries) -> None:
        _check_choice(self.center, ("median", "mean"), "center")
        _check_choice(self.scale, ("iqr", "idr", "mad", "std"), "scale")

        column = ts.values[:, 0]
        observed = column[~np.isnan(column)]
        if observed.size == 0:
            self.center_ = self.scale_ = math.nan
            return

        self.center_ = float(
            np.median(observed) if self.center == "median" else observed.mean()
        )
        self.scale_ = _spread(observed, self.scale, self.center_)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        deviation = ts.values[:, 0] - self.center_
        if self.scale_ == 0.0:
            # No observed variation: a point on the centre is unremarkable and
            # anything else is unprecedented. Chosen rather than divided by zero,
            # which would turn a point on the centre into 0/0.
            scores = np.where(deviation > 0.0, np.inf, -np.inf)
            scores = np.where(deviation == 0.0, 0.0, scores)
            scores[np.isnan(deviation)] = np.nan
            return ts.wrap(scores)
        return ts.wrap(deviation / self.scale_)


def _spread(observed: NDArray[np.float64], scale: Scale, centre: float) -> float:
    """Measure the dispersion of ``observed`` the way ``scale`` asks for."""
    if scale == "std":
        # ddof=1, and a single observation therefore has no spread rather than a
        # spread of zero; treat that as zero so scoring stays defined.
        return 0.0 if observed.size < 2 else float(observed.std(ddof=1))
    if scale == "mad":
        return _MAD_SCALE * float(np.median(np.abs(observed - np.median(observed))))
    edges = (0.25, 0.75) if scale == "iqr" else (0.1, 0.9)
    low, high = np.quantile(observed, edges)
    return float(high - low)


def _check_choice(value: object, allowed: tuple[str, ...], name: str) -> None:
    """Reject a parameter that is not one of a small set of names."""
    if value not in allowed:
        msg = f"{name}={value!r} is not one of {list(allowed)}."
        raise ValueError(msg)


class SeasonalResidualScorer(TransformerScorer):
    """Score each point by what a repeating pattern fails to explain.

    A daily or weekly cycle is normal behaviour, so it belongs in the model
    rather than in the anomalies. Classic additive decomposition removes it: the
    seasonal profile is the average shape of one cycle, optionally on top of a
    moving-average trend, and the residual is the signed remainder. The profile
    is learned once, so a later series is judged against the pattern that used
    to hold rather than against its own.

    Requires a regular time axis at :meth:`fit`, since a cycle length in
    observations is only meaningful if observations are evenly spaced. Later
    series may have gaps: the phase of each timestamp is recovered arithmetically
    from the training anchor.

    Parameters
    ----------
    period
        Length of a cycle in observations. When None it is detected from the
        autocorrelation of the training series.
    trend
        Estimate and remove a moving-average trend as well. Adds a NaN margin of
        half a period at each end, where the centred average has no window.

    Attributes
    ----------
    period_ : int
        Cycle length used, whether given or detected.
    seasonal_ : numpy.ndarray
        The learned profile, of length ``period_``, phase 0 being the first
        training observation.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([0.0, 1.0, 0.0, -1.0], 4)
    >>> values[9] = 6.0
    >>> time = np.arange("2024-01-01", "2024-01-17", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> scorer = SeasonalResidualScorer(period=4).fit(ts)
    >>> scorer.period_
    4
    >>> int(np.argmax(scorer.score(ts).values))
    9
    """

    def __init__(self, period: int | None = None, trend: bool = False) -> None:
        self.period = period
        self.trend = trend

    def _new_transformer(self) -> BaseTransformer:
        return SeasonalDecomposition(
            period=self.period, trend=self.trend, component="residual"
        )

    @property
    def period_(self) -> int:
        """Cycle length used, whether given or detected."""
        decomposition: Any = self.transformer_
        return int(decomposition.period_)

    @property
    def seasonal_(self) -> NDArray[np.float64]:
        """The learned seasonal profile, one value per phase."""
        decomposition: Any = self.transformer_
        profile: NDArray[np.float64] = decomposition.seasonal_
        return profile


class AutoregressionResidualScorer(BaseScorer):
    """Score each point by what its own recent past fails to predict.

    Many series are largely predictable from where they just were. Fitting that
    relationship and scoring the signed residual asks a sharper question than
    "is this value unusual": it asks whether the value is unusual *given* what
    came immediately before, which catches a break in the dynamics even at a
    perfectly ordinary level.

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

    Attributes
    ----------
    transformer_ : hazure.features.RegressionResidual
        The fitted regression stage, whose ``regressor_`` is the fitted model.

    Raises
    ------
    ValueError
        ``n_steps`` or ``step_size`` is less than 1.

    Notes
    -----
    The regressor is deep-copied at :meth:`fit` time. A scorer handed a frame
    fans out into one copy per column, and without the copy every column would
    fit the same model object and only the last would survive. It also leaves the
    caller's object untouched; the fitted one is reachable through
    ``transformer_.regressor_``.

    The first ``n_steps * step_size`` points have an incomplete history and so
    score NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = np.tile([1.0, 2.0, 3.0], 6)
    >>> values[10] = 9.0
    >>> time = np.arange("2024-01-01", "2024-01-19", dtype="datetime64[D]")
    >>> ts = TimeSeries.from_arrays(time, values)
    >>> scores = AutoregressionResidualScorer(n_steps=3).fit_score(ts).values.ravel()
    >>> int(np.nanargmax(np.abs(scores)))
    10
    """

    lagger_: BaseTransformer
    transformer_: BaseTransformer
    _trained: bool = True

    def __init__(
        self,
        n_steps: int = 1,
        step_size: int = 1,
        regressor: Regressor | None = None,
    ) -> None:
        _check_positive(n_steps, "n_steps")
        _check_positive(step_size, "step_size")
        self.n_steps = n_steps
        self.step_size = step_size
        self.regressor = regressor

    def _learn(self, ts: TimeSeries) -> None:
        _check_positive(self.n_steps, "n_steps")
        _check_positive(self.step_size, "step_size")
        # One extra lag, at zero, so the point being predicted travels through
        # the design matrix as its target column.
        self.lagger_ = Retrospect(
            n_steps=self.n_steps + 1, step_size=self.step_size, till=0
        )
        self.transformer_ = RegressionResidual(
            target="t-0", regressor=copy.deepcopy(self.regressor)
        )
        lagged = self.lagger_.run(ts)
        # A series too short or too gappy to yield one complete row of history
        # supports no model, and every score is then unknown.
        self._trained = bool(complete_rows(lagged.values).any())
        if self._trained:
            self.transformer_.fit(lagged)

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        if not self._trained:
            return ts.wrap(np.full(ts.n_rows, np.nan), ts.columns)
        residual = self.transformer_.run(self.lagger_.run(ts))
        # The regression stage names its output "residual"; carry the caller's
        # column name instead, as every other univariate scorer does.
        return ts.wrap(residual.values, ts.columns)


def _check_positive(value: int, name: str) -> None:
    """Reject a count that cannot describe a lag."""
    if value < 1:
        msg = f"{name} must be at least 1, got {value}."
        raise ValueError(msg)
