"""Univariate transformers: rolling windows, lags, scaling, seasonality.

Each class here turns one series into one or more derived series. They sit
upstream of scorers in a pipeline, so the numbers a scorer sees are already the
feature the algorithm cares about — a rolling median, a lag matrix, a seasonal
residual — rather than the raw observation.

A univariate transformer handed a multi-column frame fans out automatically, one
independently fitted copy per column; :class:`hazure.Component` arranges that, so
nothing here has to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar

import numpy as np

from hazure import BaseTransformer, double_rolling, parse_duration, rolling
from hazure._core.window import window_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from hazure import TimeSeries
    from hazure._core.window import Closed, Window

__all__ = [
    "DoubleRollingAggregate",
    "Retrospect",
    "RollingAggregate",
    "SeasonalDecomposition",
    "StandardScale",
]

Diff = Literal["l1", "l2", "diff", "rel_diff", "abs_rel_diff"]
SeasonalComponent = Literal["residual", "seasonal", "trend"]

_T = TypeVar("_T")

#: Smallest normalised autocorrelation a peak must reach to be believed as a
#: seasonal period. Below this the "cycle" is indistinguishable from noise.
_AUTOCORRELATION_FLOOR = 0.3

_COMPONENTS = ("residual", "seasonal", "trend")


# ---------------------------------------------------------------------------
# rolling aggregates
# ---------------------------------------------------------------------------


class RollingAggregate(BaseTransformer):
    """Aggregate a sliding window into a new series.

    A rolling statistic is the workhorse feature of time series anomaly
    detection: comparing a point with a summary of its neighbourhood is what
    turns "3.4 volts" into "3.4 volts, where the last hour averaged 1.1".

    Most aggregations return one number per window and therefore one column.
    Two return a vector and so widen the series: ``"quantile"`` with a list of
    quantiles, and ``"hist"``.

    Parameters
    ----------
    window
        Observations (``int``) or duration (``"7d"``, ``timedelta``).
    agg
        A name from :data:`hazure.AGGREGATIONS`, or ``"hist"``.
    agg_params
        Extra arguments for the aggregation:

        * ``agg="quantile"`` requires ``{"q": 0.9}`` for one column, or
          ``{"q": [0.1, 0.5, 0.9]}`` for one column per quantile, named
          ``q0.1``, ``q0.5``, ``q0.9``.
        * ``agg="hist"`` requires ``{"bins": [...]}`` — ``n`` edges defining
          ``n - 1`` bins ``[b0, b1), [b1, b2), ..., [b{n-2}, b{n-1}]``, the last
          closed at both ends — or ``{"bins": 8}`` for that many equal-width
          bins spanning the series. Columns are named after their interval,
          e.g. ``[0, 1)``.
    center
        Centre the window on each row instead of trailing it.
    min_periods
        Minimum non-missing observations for a result. Defaults to the full
        window for integer windows and to 1 for duration windows.
    closed
        Which window endpoints to include: ``"right"`` (the default),
        ``"left"``, ``"both"`` or ``"neither"``.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-06", dtype="datetime64[D]"),
    ...     np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ... )
    >>> RollingAggregate(window=2).run(ts).values.ravel()
    array([nan, 1.5, 2.5, 3.5, 4.5])

    A list of quantiles fans out into one column per quantile:

    >>> RollingAggregate(
    ...     window=3, agg="quantile", agg_params={"q": [0.0, 1.0]}
    ... ).run(ts).columns
    ('q0.0', 'q1.0')
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        window: Window,
        agg: str = "mean",
        agg_params: dict[str, Any] | None = None,
        center: bool = False,
        min_periods: int | None = None,
        closed: Closed | None = None,
    ) -> None:
        self.window = window
        self.agg = agg
        self.agg_params = agg_params
        self.center = center
        self.min_periods = min_periods
        self.closed = closed

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        if self.agg == "hist":
            return self._histogram(ts)
        if self.agg == "quantile":
            return self._quantiles(ts)
        return ts.wrap(self._roll(ts, self.agg))

    def _roll(self, ts: TimeSeries, agg: str, q: float | None = None) -> Any:
        """Apply one scalar aggregation over this transformer's window."""
        return rolling(
            ts.values[:, 0],
            self.window,
            agg,
            time=ts.time,
            center=self.center,
            min_periods=self.min_periods,
            closed=self.closed,
            q=q,
        )

    def _quantiles(self, ts: TimeSeries) -> TimeSeries:
        """Roll one or several quantiles, widening the series for a list."""
        requested = (self.agg_params or {}).get("q")
        if requested is None:
            msg = (
                "agg='quantile' needs a quantile: pass "
                "agg_params={'q': 0.9}, or a list for one column each."
            )
            raise ValueError(msg)
        if not _is_sequence(requested):
            return ts.wrap(self._roll(ts, "quantile", float(requested)))

        quantiles = [float(value) for value in requested]
        if not quantiles:
            msg = (
                "agg_params={'q': []} asks for no columns; give at least one quantile."
            )
            raise ValueError(msg)
        stacked = np.column_stack(
            [self._roll(ts, "quantile", value) for value in quantiles]
        )
        return ts.wrap(stacked, [f"q{value}" for value in quantiles])

    def _histogram(self, ts: TimeSeries) -> TimeSeries:
        """Count each window's observations into bins, one column per bin."""
        bins = (self.agg_params or {}).get("bins")
        if bins is None:
            msg = (
                "agg='hist' needs bins: pass agg_params={'bins': [0, 1, 2]} "
                "for explicit edges, or agg_params={'bins': 8} for that many "
                "equal-width bins."
            )
            raise ValueError(msg)

        values = ts.values[:, 0]
        edges = _bin_edges(bins, values)
        start, stop = window_bounds(
            ts.time, self.window, ts.n_rows, center=self.center, closed=self.closed
        )
        counts = _windowed_counts(values, edges, start, stop)
        # A window that fails min_periods has no answer at all, so blank the
        # whole row rather than reporting counts over too little data.
        short = np.isnan(self._roll(ts, "count"))
        counts[short, :] = np.nan
        return ts.wrap(counts, _bin_names(edges))


class DoubleRollingAggregate(BaseTransformer):
    """Compare the window before each point with the window after it.

    This is the primitive behind spike, level-shift and volatility-shift
    detection: summarise the recent past and the near future separately, then
    measure the gap. A large gap means the series changed character there.

    ``window``, ``agg``, ``agg_params`` and ``min_periods`` each accept a
    2-tuple to configure the two sides independently. Asymmetric settings are
    what separate the use cases — a long left window characterises "normal",
    a short right window catches a blip.

    Parameters
    ----------
    window
        One spec for both sides, or ``(left, right)``.
    agg
        One statistic for both sides, or ``(left, right)``. ``"std"``, ``"iqr"``
        or ``"idr"`` turn this into volatility-shift detection.
    agg_params
        Extra arguments for the aggregation, or ``(left, right)``. Only
        ``{"q": ...}`` for ``agg="quantile"`` is meaningful, and the two sides
        must ask for the same quantile.
    center
        When True (the default) the row sits on the boundary between the two
        windows, so a change is reported at the observation where it happens.
        When False the result is reported at the *last* observation of the right
        window, which is the earliest point at which a trailing-only detector
        could have known about it.
    min_periods
        Minimum non-missing observations per side, or ``(left, right)``.
    diff
        How to compare the two sides: ``"l1"`` or ``"l2"`` for the unsigned
        magnitude, ``"diff"`` for the signed ``right - left``, ``"rel_diff"``
        for that divided by the left value, ``"abs_rel_diff"`` for its
        magnitude.

    Examples
    --------
    A step change shows up as a peak at the step:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-07", dtype="datetime64[D]"),
    ...     np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0]),
    ... )
    >>> DoubleRollingAggregate(window=2, diff="diff").run(ts).values.ravel()
    array([nan, nan, 2.5, 5. , 2.5, nan])
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        window: Window | tuple[Window, Window],
        agg: str | tuple[str, str] = "mean",
        agg_params: dict[str, Any] | tuple[Any, Any] | None = None,
        center: bool = True,
        min_periods: int | tuple[int | None, int | None] | None = None,
        diff: Diff = "l1",
    ) -> None:
        self.window = window
        self.agg = agg
        self.agg_params = agg_params
        self.center = center
        self.min_periods = min_periods
        self.diff = diff

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        boundary = double_rolling(
            ts.values[:, 0],
            self.window,
            self.agg,
            time=ts.time,
            diff=self.diff,
            min_periods=self.min_periods,
            q=self._quantile(),
        )
        if self.center:
            return ts.wrap(boundary)
        windows: tuple[Window, Window] = _pair(self.window)
        return ts.wrap(_report_at_window_end(boundary, ts.time, windows[1]))

    def _quantile(self) -> float | None:
        """Resolve the single quantile both windows use, if either needs one."""
        aggs: tuple[str, str] = _pair(self.agg)
        params: tuple[dict[str, Any] | None, dict[str, Any] | None] = _pair(
            self.agg_params
        )
        wanted: list[float] = []
        for agg, side_params, side in zip(aggs, params, ("left", "right"), strict=True):
            if agg == "hist":
                msg = (
                    "agg='hist' returns a vector per window, which "
                    "DoubleRollingAggregate cannot difference. Use a scalar "
                    "aggregation such as 'mean', 'median' or 'std'."
                )
                raise ValueError(msg)
            if agg != "quantile":
                continue
            q = (side_params or {}).get("q")
            if q is None:
                msg = (
                    f"agg='quantile' on the {side} window needs a quantile: "
                    f"pass agg_params={{'q': 0.9}}."
                )
                raise ValueError(msg)
            if _is_sequence(q):
                msg = (
                    f"The {side} window was given several quantiles {q!r}. "
                    f"DoubleRollingAggregate differences one number per side; "
                    f"pass a single float."
                )
                raise ValueError(msg)
            wanted.append(float(q))

        if not wanted:
            return None
        if len(wanted) == 2 and wanted[0] != wanted[1]:
            msg = (
                f"The two windows ask for different quantiles "
                f"({wanted[0]} and {wanted[1]}), but one quantile is shared by "
                f"both. Use the same q on each side."
            )
            raise ValueError(msg)
        return wanted[0]


def _report_at_window_end(
    values: NDArray[np.float64], time: NDArray[np.int64], window: Window
) -> NDArray[np.float64]:
    """Move each boundary result to the last observation of its right window.

    The right window at row ``j`` runs from ``j`` forward, so its last member is
    the row a trailing-only view would first be able to report at.
    """
    n_rows = time.shape[0]
    if n_rows == 0:
        return values
    if isinstance(window, int):
        target = np.arange(n_rows) + window - 1
    else:
        edge = time + parse_duration(window)
        target = np.searchsorted(time, edge, side="left") - 1
    shifted = np.full(n_rows, np.nan, dtype=np.float64)
    # A boundary whose right window runs off the end of the series has no row to
    # be reported at, and is dropped rather than piled onto the last one. Where a
    # gap does make two windows end on the same row, the later one wins.
    reportable = target < n_rows
    shifted[target[reportable]] = values[reportable]
    return shifted


# ---------------------------------------------------------------------------
# lagging and scaling
# ---------------------------------------------------------------------------


class Retrospect(BaseTransformer):
    """Emit lagged copies of the series as columns.

    The result is the design matrix for autoregression: row ``t`` holds the
    values at ``t - till``, ``t - till - step_size``, and so on, which is what a
    model needs to learn how a control's effect is delayed and how long it
    lasts. Columns are named ``t-0``, ``t-1``, ...; a negative lag looks ahead
    and is named ``t+1``, ``t+2``, ...

    Parameters
    ----------
    n_steps
        Number of lagged columns.
    step_size
        Gap in observations between consecutive columns.
    till
        Nearest lag, in observations. 0 is the current point.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]"),
    ...     np.array([0.0, 1.0, 2.0, 3.0]),
    ... )
    >>> lagged = Retrospect(n_steps=2, step_size=2, till=1).run(ts)
    >>> lagged.columns
    ('t-1', 't-3')
    >>> lagged.values
    array([[nan, nan],
           [ 0., nan],
           [ 1., nan],
           [ 2.,  0.]])
    """

    trainable: ClassVar[bool] = False

    def __init__(self, n_steps: int = 1, step_size: int = 1, till: int = 0) -> None:
        self.n_steps = n_steps
        self.step_size = step_size
        self.till = till

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        if self.n_steps < 1:
            msg = f"n_steps must be at least 1, got {self.n_steps}."
            raise ValueError(msg)
        if self.step_size < 1:
            msg = (
                f"step_size must be at least 1, got {self.step_size}; a step of "
                f"0 would emit the same lag several times."
            )
            raise ValueError(msg)
        if ts.freq is None:
            msg = (
                "Retrospect needs a regular time axis: shifting by a number of "
                "observations means nothing when the sampling interval varies. "
                "Resample the series first, or use RollingAggregate with a "
                "duration window."
            )
            raise ValueError(msg)

        values = ts.values[:, 0]
        lags = self.till + np.arange(self.n_steps) * self.step_size
        lagged = np.full((ts.n_rows, self.n_steps), np.nan, dtype=np.float64)
        # One pass per column, which is n_steps iterations rather than n_rows.
        for position, lag in enumerate(int(value) for value in lags):
            if abs(lag) >= ts.n_rows:
                continue
            if lag >= 0:
                lagged[lag:, position] = values[: ts.n_rows - lag]
            else:
                lagged[: ts.n_rows + lag, position] = values[-lag:]
        return ts.wrap(lagged, [_lag_name(int(lag)) for lag in lags])


def _lag_name(lag: int) -> str:
    """Name a column after its lag, keeping the sign readable."""
    return f"t-{lag}" if lag >= 0 else f"t+{-lag}"


class StandardScale(BaseTransformer):
    """Centre and scale a series by its own mean and standard deviation.

    Scaling is per-series and computed from the series being transformed, so a
    frame of columns in different units becomes comparable without training.
    The standard deviation is the sample one (``ddof=1``); a constant series has
    none, and is centred but left unscaled rather than divided by zero.
    """

    trainable: ClassVar[bool] = False

    def __init__(self) -> None:
        # Declared explicitly, with no parameters, so that get_params() and
        # clone() have a signature to read.
        pass

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        values = ts.values[:, 0]
        present = values[~np.isnan(values)]
        if present.size == 0:
            return ts.wrap(values)
        spread = float(present.std(ddof=1)) if present.size > 1 else 0.0
        if not np.isfinite(spread) or spread == 0.0:
            spread = 1.0
        return ts.wrap((values - float(present.mean())) / spread)


# ---------------------------------------------------------------------------
# seasonality
# ---------------------------------------------------------------------------


class SeasonalDecomposition(BaseTransformer):
    """Split a series into trend, a repeating seasonal profile, and residual.

    Additive classic decomposition, in numpy alone: the trend is a centred
    moving average one period wide, and the seasonal profile is the mean of each
    phase of the detrended series. The residual is what neither explains, which
    is usually the series an anomaly detector should look at.

    :meth:`fit` learns the period (unless given) and the seasonal profile;
    :meth:`transform` re-estimates the trend of the series it is given but
    reuses the trained profile, which assumes seasonality is stable over time.

    The profile is anchored to the first training timestamp, and a later
    series' phase is recovered arithmetically from its timestamps, so a test
    window arbitrarily far from training costs the same as an adjacent one.

    Parameters
    ----------
    period
        Length of a cycle in observations. When None it is detected from the
        autocorrelation of the training series.
    trend
        Estimate and remove a moving-average trend. When False the series is
        taken to be a seasonal profile plus residual, and the profile carries
        the series' level.
    component
        Which part to return: ``"residual"``, ``"seasonal"`` or ``"trend"``.
        ``"trend"`` requires ``trend=True``.

    Attributes
    ----------
    period_ : int
        Cycle length used, whether given or detected.
    seasonal_ : numpy.ndarray
        The seasonal profile, of length ``period_``, phase 0 being the first
        training observation. Centred on zero when ``trend`` is set, since the
        trend then carries the level.

    Raises
    ------
    ValueError
        The time axis is irregular, the period is unusable, no seasonality
        could be detected, ``component`` is unknown, or the series to transform
        is out of phase with training.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> profile = np.array([0.0, 1.0, 0.0, -1.0])
    >>> ts = TimeSeries.from_arrays(
    ...     np.arange("2024-01-01", "2024-01-13", dtype="datetime64[D]"),
    ...     np.tile(profile, 3),
    ... )
    >>> model = SeasonalDecomposition(period=4).fit(ts)
    >>> model.seasonal_
    array([ 0.,  1.,  0., -1.])
    >>> model.run(ts).values.ravel()
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    """

    def __init__(
        self,
        period: int | None = None,
        trend: bool = False,
        component: SeasonalComponent = "residual",
    ) -> None:
        self.period = period
        self.trend = trend
        self.component = component

    # -- training -----------------------------------------------------------

    def _learn(self, ts: TimeSeries) -> None:
        self._check_component()
        if ts.freq is None:
            msg = (
                "SeasonalDecomposition needs a regular time axis to line up "
                "cycles; this series has an irregular or unknown sampling "
                "interval. Resample it first."
            )
            raise ValueError(msg)

        values = ts.values[:, 0]
        period = _detect_period(values) if self.period is None else int(self.period)
        if period < 2:
            msg = f"period must be at least 2 observations, got {period}."
            raise ValueError(msg)
        if ts.n_rows < 2 * period:
            msg = (
                f"Learning a profile of {period} phases needs at least two full "
                f"cycles ({2 * period} observations), but the training series "
                f"has {ts.n_rows}."
            )
            raise ValueError(msg)

        self.period_ = period
        detrended = (
            values - _centred_moving_average(values, period) if self.trend else values
        )
        profile = _phase_means(detrended, period)
        if bool(np.isnan(profile).any()):
            blank = [int(i) for i in np.flatnonzero(np.isnan(profile))]
            msg = (
                f"Phases {blank} of a {period}-observation cycle have no "
                f"training observations, so their seasonal level is unknown. "
                f"Provide a longer or less gappy training series."
            )
            raise ValueError(msg)
        # With a trend the level belongs to the trend, so the profile is a pure
        # deviation; without one it has nowhere else to live.
        self.seasonal_ = profile - profile.mean() if self.trend else profile
        self._datum = int(ts.time[0])
        self._step = int(ts.freq)

    # -- application --------------------------------------------------------

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        self._check_component()
        seasonal = self.seasonal_[self._phase(ts)]
        if self.component == "seasonal":
            return ts.wrap(seasonal)

        values = ts.values[:, 0]
        trend = (
            _centred_moving_average(values, self.period_)
            if self.trend
            else np.zeros(ts.n_rows, dtype=np.float64)
        )
        if self.component == "trend":
            return ts.wrap(trend)
        return ts.wrap(values - trend - seasonal)

    def _phase(self, ts: TimeSeries) -> NDArray[np.int64]:
        """Locate each row in the cycle, counting from the training datum."""
        if ts.freq is None:
            msg = (
                "SeasonalDecomposition needs a regular time axis to line up "
                "cycles; this series has an irregular or unknown sampling "
                "interval. Resample it first."
            )
            raise ValueError(msg)
        if ts.freq != self._step:
            msg = (
                f"Trained on a series sampled every {self._step} ns but asked "
                f"to transform one sampled every {ts.freq} ns. Resample to the "
                f"training interval, or fit again."
            )
            raise ValueError(msg)

        offset = ts.time - self._datum
        if bool(np.any(offset % self._step != 0)):
            msg = (
                "This series is out of phase with training: its timestamps are "
                "not a whole number of sampling intervals from the first "
                "training observation, so no phase can be assigned. Align the "
                "series to the training grid, or fit again."
            )
            raise ValueError(msg)
        # Floor division and modulo are both non-negative for a positive step,
        # so a series that starts before the datum needs no special case.
        phase: NDArray[np.int64] = (offset // self._step) % self.period_
        return phase

    def _check_component(self) -> None:
        """Reject a component this configuration cannot produce."""
        if self.component not in _COMPONENTS:
            msg = (
                f"Unknown component={self.component!r}; expected one of "
                f"{list(_COMPONENTS)}."
            )
            raise ValueError(msg)
        if self.component == "trend" and not self.trend:
            msg = (
                "component='trend' has nothing to return while trend=False, "
                "which assumes the series has no trend. Pass trend=True."
            )
            raise ValueError(msg)


def _centred_moving_average(
    values: NDArray[np.float64], period: int
) -> NDArray[np.float64]:
    """Average one cycle centred on each row, half-weighting an even period.

    An even-length cycle has no single central observation, so the window spans
    ``period + 1`` points with the two ends at half weight. That keeps the
    average symmetric about the row and makes it cancel the cycle exactly.
    """
    if period % 2 == 0:
        weights = np.concatenate(([0.5], np.ones(period - 1), [0.5])) / period
    else:
        weights = np.ones(period) / period

    span = weights.size
    trend = np.full(values.shape[0], np.nan, dtype=np.float64)
    if values.shape[0] < span:
        return trend
    interior = np.convolve(values, weights, mode="valid")
    lead = (span - 1) // 2
    trend[lead : lead + interior.shape[0]] = interior
    return trend


def _phase_means(values: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Average the observations sharing each phase of the cycle.

    ``bincount`` sums by phase in one pass and, unlike a masked mean, says
    nothing about phases it never saw instead of warning about them.
    """
    phase = np.arange(values.shape[0]) % period
    present = ~np.isnan(values)
    totals = np.bincount(phase[present], weights=values[present], minlength=period)
    counts = np.bincount(phase[present], minlength=period)
    means: NDArray[np.float64] = np.where(
        counts > 0, totals / np.maximum(counts, 1), np.nan
    )
    return means


def _detect_period(values: NDArray[np.float64]) -> int:
    """Find the cycle length from the strongest autocorrelation peak.

    The autocorrelation is obtained through the frequency domain: the inverse
    transform of the power spectrum is the autocovariance, which costs one FFT
    pair instead of one dot product per lag. Zero-padding to twice the length
    keeps the wrap-around of a circular transform out of the result.

    Parameters
    ----------
    values
        The training series. Missing observations are treated as sitting on the
        mean, which neither invents nor hides a cycle.

    Returns
    -------
    int
        Lag of the strongest local peak, in observations.

    Raises
    ------
    ValueError
        The series is too short, constant, or has no peak worth believing.
    """
    present = ~np.isnan(values)
    n_rows = values.shape[0]
    if present.sum() < 4 or n_rows < 4:
        msg = (
            f"Detecting a period needs at least 4 observations, got "
            f"{int(present.sum())}. Pass period=... instead."
        )
        raise ValueError(msg)

    centred = np.where(present, values - values[present].mean(), 0.0)
    padded = 1 << int(np.ceil(np.log2(2 * n_rows)))
    spectrum = np.fft.rfft(centred, n=padded)
    autocovariance = np.fft.irfft(np.abs(spectrum) ** 2, n=padded)[:n_rows]
    if autocovariance[0] <= 0.0:
        msg = (
            "The training series is constant, so it has no seasonality to "
            "detect. Pass period=... explicitly."
        )
        raise ValueError(msg)
    autocorrelation = autocovariance / autocovariance[0]

    # Beyond half the series a "cycle" is supported by too few overlapping
    # observations to trust, and a peak needs a lag on either side of it.
    lags = np.arange(1, n_rows // 2 + 1)
    is_peak = (
        (autocorrelation[lags] > autocorrelation[lags - 1])
        & (autocorrelation[lags] > autocorrelation[lags + 1])
        & (autocorrelation[lags] >= _AUTOCORRELATION_FLOOR)
    )
    if not bool(is_peak.any()):
        msg = (
            f"No autocorrelation peak reached {_AUTOCORRELATION_FLOOR}, so no "
            f"seasonality was found. Pass period=... if you know the cycle "
            f"length, or use a detector that does not assume one."
        )
        raise ValueError(msg)
    candidates = lags[is_peak]
    return int(candidates[np.argmax(autocorrelation[candidates])])


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _pair(spec: _T | tuple[_T, _T]) -> tuple[_T, _T]:
    """Expand a single value into a pair, or pass a 2-tuple through."""
    if isinstance(spec, tuple):
        if len(spec) != 2:
            msg = (
                f"Expected one value or a (left, right) 2-tuple, got {len(spec)} items."
            )
            raise ValueError(msg)
        return spec
    return spec, spec


def _is_sequence(value: Any) -> bool:
    """Report whether a parameter asks for several values rather than one."""
    return isinstance(value, (list, tuple, np.ndarray))


def _bin_edges(
    bins: int | Sequence[float] | NDArray[np.float64], values: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Resolve a bin specification into a strictly increasing array of edges."""
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 1:
            msg = f"bins must be at least 1, got {bins}."
            raise ValueError(msg)
        present = values[~np.isnan(values)]
        if present.size == 0:
            msg = (
                "Cannot choose bins from a series with no observations. Pass "
                "explicit edges, e.g. agg_params={'bins': [0, 1, 2]}."
            )
            raise ValueError(msg)
        return np.asarray(
            np.histogram_bin_edges(present, bins=int(bins)), dtype=np.float64
        )

    edges = np.asarray(bins, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        msg = f"bins needs at least 2 edges to define a bin, got {edges.size} value(s)."
        raise ValueError(msg)
    if not bool(np.all(np.diff(edges) > 0)):
        msg = f"bin edges must be strictly increasing, got {edges.tolist()}."
        raise ValueError(msg)
    return edges


def _bin_names(edges: NDArray[np.float64]) -> list[str]:
    """Name each bin after its interval, closing the last one on the right."""
    return [
        f"[{edges[i]:g}, {edges[i + 1]:g}{')' if i < edges.size - 2 else ']'}"
        for i in range(edges.size - 1)
    ]


def _windowed_counts(
    values: NDArray[np.float64],
    edges: NDArray[np.float64],
    start: NDArray[np.int64],
    stop: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Count each window's observations per bin, from the window bounds.

    Each observation belongs to at most one bin, so a running total per bin
    turns every window into one subtraction: no per-window pass over the data.
    """
    n_bins = edges.size - 1
    # searchsorted places NaN and anything above the last edge past the end,
    # which the `inside` mask then drops.
    index = np.searchsorted(edges, values, side="right") - 1
    index[values == edges[-1]] = n_bins - 1
    inside = (index >= 0) & (index < n_bins)

    occupancy = np.zeros((values.shape[0] + 1, n_bins), dtype=np.float64)
    occupancy[np.flatnonzero(inside) + 1, index[inside]] = 1.0
    running = np.cumsum(occupancy, axis=0)
    counts: NDArray[np.float64] = running[stop] - running[start]
    return counts
