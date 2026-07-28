"""Classical seasonal decomposition into profile, trend and remainder.

Where a cycle is normal behaviour it belongs in the model rather than in
the anomalies, and the residual is what the cycle fails to explain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from hazure import BaseTransformer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "SeasonalDecomposition",
]


SeasonalComponent = Literal["residual", "seasonal", "trend"]


#: Smallest normalised autocorrelation a peak must reach to be believed as a
#: seasonal period. Below this the "cycle" is indistinguishable from noise.
_AUTOCORRELATION_FLOOR = 0.3


_COMPONENTS = ("residual", "seasonal", "trend")


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
