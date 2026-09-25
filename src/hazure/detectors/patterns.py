"""Detectors for a break in a pattern the series normally follows.

Each models what the series usually does — a repeating seasonal shape, its own
short-term dynamics, several overlapping rhythms, or simply its spectrum — and
reports the points that model fails to explain. A value perfectly ordinary in
isolation can be anomalous here, because the question is whether it was expected
*there*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hazure._core import Detector
from hazure.detectors._common import signed_fence, upper_fence
from hazure.scorers import (
    AsScorer,
    AutoregressionResidualScorer,
    MstlResidualScorer,
    SpectralResidualScorer,
    StlResidualScorer,
)
from hazure.transformers import SeasonalDecomposition

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hazure.thresholds import Factor, Side
    from hazure.transformers import Regressor

__all__ = [
    "autoregression",
    "mstl",
    "seasonal",
    "spectral_residual",
    "stl",
]


def seasonal(
    period: int | None = None,
    factor: Factor = 3.0,
    side: Side = "both",
    trend: bool = False,
) -> Detector:
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

    Returns
    -------
    Detector
        The residual of a classical seasonal decomposition, fenced on its
        magnitude. The learned profile is ``detector.scorer.transformer.seasonal_``.

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
    >>> labels = seasonal(period=4).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([13])
    """
    scorer = AsScorer(SeasonalDecomposition(period=period, trend=trend))
    return Detector(scorer, signed_fence(factor, side))


def autoregression(
    n_steps: int = 1,
    step_size: int = 1,
    regressor: Regressor | None = None,
    factor: Factor = 3.0,
    side: Side = "both",
) -> Detector:
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

    Returns
    -------
    Detector
        The autoregression residual, fenced on its magnitude.

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
    >>> labels = autoregression(n_steps=3).fit_detect(ts)
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([17])
    """
    scorer = AutoregressionResidualScorer(
        n_steps=n_steps, step_size=step_size, regressor=regressor
    )
    return Detector(scorer, signed_fence(factor, side))


def stl(
    period: int | None = None,
    robust: bool = True,
    seasonal: int | None = None,
    factor: Factor = 3.0,
) -> Detector:
    """Flag points an STL decomposition cannot account for.

    :class:`~hazure.scorers.StlResidualScorer` paired with an inter-quartile-range
    rule on the residual magnitudes. The rule is learned from the residuals rather
    than fixed, because how large a residual is large depends entirely on how well
    the decomposition fits the series in the first place.

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

    Returns
    -------
    Detector
        The STL residual magnitude, fenced above.

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
    >>> labels = stl(factor=6.0).fit_detect(TimeSeries.from_arrays(time, values))
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([100])
    """
    scorer = StlResidualScorer(period=period, robust=robust, seasonal=seasonal)
    return Detector(scorer, upper_fence(factor))


def mstl(
    periods: int | Sequence[int],
    robust: bool = True,
    windows: int | Sequence[int] | None = None,
    factor: Factor = 3.0,
) -> Detector:
    """Flag points that break none of several rhythms but still do not fit.

    :class:`~hazure.scorers.MstlResidualScorer` paired with an
    inter-quartile-range rule. Use this rather than :func:`stl` whenever the
    series has more than one rhythm: with a second cycle left in the residual, the
    residual's spread is set by that cycle rather than by the noise, and the
    threshold ends up asking how unusual a point is compared with a systematic
    pattern instead of compared with chance.

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

    Returns
    -------
    Detector
        The MSTL residual magnitude, fenced above.

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
    >>> labels = mstl(periods=(24, 168), factor=25.0).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([300])
    """
    scorer = MstlResidualScorer(periods=periods, robust=robust, windows=windows)
    return Detector(scorer, upper_fence(factor))


def spectral_residual(
    window: int = 3,
    series_window: int = 21,
    score_window: int = 21,
    factor: Factor = 3.0,
) -> Detector:
    """Flag points whose saliency stands far above its neighbourhood.

    :class:`~hazure.scorers.SpectralResidualScorer` paired with an
    inter-quartile-range rule on the score. Useful when the series has structure
    nobody has characterised: no period, trend or distribution has to be named,
    and a point is reported when its saliency is out of proportion to the saliency
    around it.

    Parameters
    ----------
    window
        Width, in frequency bins, of the moving average over the log amplitude
        spectrum.
    series_window
        Trailing observations the right-edge extrapolation is estimated from.
    score_window
        Trailing saliency points each point is compared against.
    factor
        Inter-quartile-range factor deciding how high a relative saliency is too
        high. One-sided: a *low* saliency is never interesting.

    Returns
    -------
    Detector
        The relative saliency, fenced above.

    Raises
    ------
    ValueError
        A window is out of range, ``factor`` is negative, or the time axis is
        irregular.

    Examples
    --------
    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = 10.0 + np.sin(np.arange(200) * np.pi / 12)
    >>> values[137] = 30.0
    >>> time = np.arange(200) * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> labels = spectral_residual(factor=12.0).fit_detect(
    ...     TimeSeries.from_arrays(time, values)
    ... )
    >>> np.flatnonzero(labels.values.ravel() == 1.0)
    array([137])
    """
    scorer = SpectralResidualScorer(
        window=window, series_window=series_window, score_window=score_window
    )
    return Detector(scorer, upper_fence(factor))
