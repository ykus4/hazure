"""Saliency from the spectral residual of the Fourier amplitude spectrum."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

import numpy as np

from hazure import BaseScorer, rolling

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from hazure import TimeSeries

__all__ = [
    "SpectralResidualScorer",
]


#: How many points are extrapolated past the end of the series before the
#: transform. Five is enough to move the edge artefact of a discrete transform
#: off the last real observation, and few enough that the linear extrapolation
#: they are built from stays credible.
_EXTENSION: Final = 5


#: Added before taking a logarithm so that a spectral component of exactly zero —
#: which a constant or perfectly periodic series produces — gives a finite
#: residual instead of -inf.
_EPS: Final = 1e-12


class SpectralResidualScorer(BaseScorer):
    """Score each point by how far its saliency exceeds its neighbourhood.

    Three steps, all in the frequency domain:

    1. Take the discrete Fourier transform of the series and split it into a log
       amplitude spectrum and a phase.
    2. Subtract a moving average (``window`` wide) from the log amplitude. What
       an average of neighbouring frequencies predicts is the *expected* spectrum
       of a well-behaved signal; the difference — the **spectral residual** — is
       what the signal does that its own smooth spectral envelope does not
       explain.
    3. Invert the transform from the residual amplitude and the original phase.
       The magnitude of the result is the saliency map: it is near zero wherever
       the series is either flat or regularly structured, and rises where the
       series does something unaccounted for.

    Saliency is then judged locally, by dividing each point by the average
    saliency of the ``score_window`` points up to and including it, so the score
    is dimensionless and a busy series does not read as anomalous throughout.

    Parameters
    ----------
    window
        Width, in frequency bins, of the moving average subtracted from the log
        amplitude spectrum. Small: a wide average would smooth away the very
        structure it is supposed to predict.
    series_window
        How many trailing observations the extrapolation of the right edge is
        estimated from. See Notes.
    score_window
        How many trailing points of the saliency map each point is compared
        against.

    Raises
    ------
    ValueError
        A window is less than 1, ``series_window`` is less than 2, or the series
        has an irregular or unknown sampling interval.

    Notes
    -----
    A discrete transform assumes the series repeats, so it behaves badly at the
    right edge: the last few points are the ones whose saliency is most distorted
    by the jump between the end of the series and its wrapped-around beginning.
    That edge is exactly the part anyone monitoring a live series cares about —
    the most recent point is usually the point in question. So before the
    transform, ``_EXTENSION`` points are appended, each one step further along the
    average gradient of the last ``series_window`` observations, and after the
    saliency map is built the extrapolated tail is discarded. The edge artefact
    then falls on the invented points rather than on the real ones, which is what
    makes the score usable on the most recent observation.

    Missing observations are filled by linear interpolation before the transform,
    since a transform has no notion of a gap, and their scores are set back to
    NaN afterwards.

    Because the score is a *ratio*, it is never zero: the first point's
    neighbourhood is itself, so its score is exactly 1 by construction, and a
    series with no anomalies scores around 1 throughout rather than around 0. It
    is the relative height that carries the meaning, which is what makes one
    threshold factor usable across series of wildly different scales.

    Nothing is learned: the saliency map is a property of the series being
    scored, so :meth:`fit` is optional and each call recomputes it.

    Examples
    --------
    A single spike in an otherwise smooth series is the most salient point:

    >>> import numpy as np
    >>> from hazure import TimeSeries
    >>> values = 10.0 + np.sin(np.arange(200) * np.pi / 12)
    >>> values[137] = 30.0
    >>> time = np.arange(200) * np.timedelta64(1, "h") + np.datetime64("2024-01-01")
    >>> scores = SpectralResidualScorer().score(TimeSeries.from_arrays(time, values))
    >>> int(np.nanargmax(scores.values.ravel()))
    137
    """

    trainable: ClassVar[bool] = False

    def __init__(
        self,
        window: int = 3,
        series_window: int = 21,
        score_window: int = 21,
    ) -> None:
        _check_windows(window, series_window, score_window)
        self.window = window
        self.series_window = series_window
        self.score_window = score_window

    def _compute(self, ts: TimeSeries) -> TimeSeries:
        _check_windows(self.window, self.series_window, self.score_window)
        if ts.freq is None:
            msg = (
                "SpectralResidualScorer needs a regular time axis, because a "
                "discrete Fourier transform is defined on evenly spaced samples; "
                "this series has an irregular or unknown sampling interval. "
                "Resample it first."
            )
            raise ValueError(msg)

        column = ts.values[:, 0]
        missing = np.isnan(column)
        if bool(missing.all()):
            return ts.wrap(np.full(ts.n_rows, np.nan))

        filled = _fill_gaps(column, missing)
        extended = np.concatenate(
            [filled, _extrapolate(filled, self.series_window, _EXTENSION)]
        )
        saliency = _saliency(extended, self.window)[: ts.n_rows]

        neighbourhood = rolling(saliency, self.score_window, "mean", min_periods=1)
        # A neighbourhood with no saliency at all offers nothing to stand out
        # from, so the honest reading is "not unusual" rather than 0/0.
        scores = np.divide(
            saliency,
            neighbourhood,
            out=np.zeros_like(saliency),
            where=neighbourhood > 0.0,
        )
        scores[missing] = np.nan
        return ts.wrap(scores)


# ---------------------------------------------------------------------------
# the transform itself
# ---------------------------------------------------------------------------


def _saliency(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Build the saliency map of a series from its spectral residual.

    Parameters
    ----------
    values
        1-D array with no missing values.
    window
        Width of the moving average over the log amplitude spectrum.

    Returns
    -------
    numpy.ndarray
        Non-negative saliency, the same length as ``values``.
    """
    spectrum = np.fft.fft(values)
    log_amplitude = np.log(np.abs(spectrum) + _EPS)
    # A trailing average with min_periods=1 expands over the first few bins
    # rather than blanking them, which matters because bin 0 carries the mean of
    # the series and blanking it would lose the whole scale of the result.
    expected = rolling(log_amplitude, window, "mean", min_periods=1)
    residual = log_amplitude - expected
    # Amplitude comes from the residual, phase from the original transform:
    # keeping the phase is what puts the surviving energy back at the *place* in
    # the series it came from.
    restored = np.fft.ifft(np.exp(residual + 1j * np.angle(spectrum)))
    magnitude: NDArray[np.float64] = np.abs(restored)
    return magnitude


def _extrapolate(
    values: NDArray[np.float64], series_window: int, count: int
) -> NDArray[np.float64]:
    """Continue a series past its end along its recent average gradient.

    Parameters
    ----------
    values
        1-D array with no missing values.
    series_window
        How many trailing observations the gradient is averaged over.
    count
        How many points to produce.

    Returns
    -------
    numpy.ndarray
        ``count`` extrapolated values.
    """
    tail = values[-max(min(series_window, values.shape[0]), 2) :]
    # The mean of the differences, which is the least-squares gradient of a
    # straight line through the tail's endpoints; a single trailing difference
    # would let one final blip set the direction of the whole extension.
    gradient = float(np.diff(tail).mean()) if tail.shape[0] > 1 else 0.0
    steps: NDArray[np.float64] = np.arange(1, count + 1, dtype=np.float64)
    extended: NDArray[np.float64] = values[-1] + gradient * steps
    return extended


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
        A copy with no missing values. Gaps at either end are held flat at the
        nearest observation, which is what ``numpy.interp`` does outside its
        range.
    """
    if not bool(missing.any()):
        return column
    positions = np.arange(column.shape[0], dtype=np.float64)
    filled = column.copy()
    filled[missing] = np.interp(
        positions[missing], positions[~missing], column[~missing]
    )
    return filled


def _check_windows(window: int, series_window: int, score_window: int) -> None:
    """Reject window sizes the algorithm cannot work with.

    Parameters
    ----------
    window
        Spectral averaging width.
    series_window
        Extrapolation window.
    score_window
        Local comparison window.

    Raises
    ------
    ValueError
        ``window`` or ``score_window`` is below 1, or ``series_window`` is below
        2 and so spans no gradient.
    """
    for name, value in (("window", window), ("score_window", score_window)):
        if value < 1:
            msg = f"{name} must be at least 1 observation, got {value}."
            raise ValueError(msg)
    if series_window < 2:
        msg = (
            f"series_window must be at least 2 observations, since a gradient "
            f"needs two points to be estimated from; got {series_window}."
        )
        raise ValueError(msg)
