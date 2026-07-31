"""Series with anomalies you put there yourself.

The awkward thing about unsupervised detection is that trying anything out
requires data you already know the answer for, and real data with trustworthy
labels is rare enough that most of it is a benchmark somebody is scoring against.
So: a generator. Five shapes of anomaly, one knob for how obvious each is, and
ground truth that comes back with the series rather than in a separate file whose
conventions have to be reconciled with the library's.

The shapes are the ones the detectors are organised around, and they are genuinely
different problems rather than the same problem at different sizes. A spike is
visible in the value; a level shift is invisible in the value and obvious in the
rolling mean; a volatility shift is invisible in both and obvious in the rolling
spread. A detector that finds all three is unusual, and a benchmark that only
contains spikes will not tell you that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hazure._core import TimeSeries, parse_duration
from hazure.datasets.dataset import Dataset
from hazure.events import to_events

if TYPE_CHECKING:
    from datetime import timedelta

    from numpy.typing import NDArray

    from hazure.events import Events


__all__ = [
    "KINDS",
    "make_series",
]


#: The anomaly shapes :func:`make_series` can plant, and what each one does to a
#: stretch of the series.
KINDS: dict[str, str] = {
    "spike": "the level jumps up for a moment and comes straight back",
    "dip": "the level drops for a moment and comes straight back",
    "level_shift": "the level moves to a new plateau and stays there a while",
    "volatility_shift": "the level is unchanged and the noise around it grows",
    "seasonal_break": "the daily rhythm flattens out while the level holds",
}


#: Default length of a planted anomaly, in samples, per kind. A spike is a moment
#: by definition; the other three are only visible over a stretch, and a stretch
#: shorter than the window a detector uses to see them is not a fair test.
_WIDTHS: dict[str, int] = {
    "spike": 1,
    "dip": 1,
    "level_shift": 24,
    "volatility_shift": 24,
    "seasonal_break": 24,
}


def make_series(
    kind: str = "spike",
    *,
    n: int = 2000,
    n_anomalies: int = 3,
    width: int | None = None,
    strength: float = 8.0,
    noise: float = 1.0,
    level: float = 100.0,
    period: int | None = None,
    amplitude: float = 20.0,
    freq: str | timedelta | np.timedelta64 = "5min",
    start: str = "2024-01-01",
    seed: int = 0,
    backend: str | None = None,
) -> Dataset:
    """Generate a series with anomalies of one shape planted in it.

    Parameters
    ----------
    kind
        Which shape to plant. One of the keys of :data:`KINDS`.
    n
        Length of the series in samples.
    n_anomalies
        How many to plant. They are spaced evenly across the middle 70% of the
        series, so none of them straddles an end where a rolling window has
        nothing to compare against — a detector missing an anomaly in its first
        window is a fact about the window, not about the detector.
    width
        Length of each anomaly in samples. Defaults to something sensible for the
        kind: 1 for a spike or dip, 24 for the three that are only visible over a
        stretch.
    strength
        How obvious each anomaly is, in multiples of ``noise``, and it means the
        same thing for every kind: the series departs from what it should have been
        by ``strength * noise``. A ``volatility_shift`` multiplies the standard
        deviation of the noise by it rather than offsetting the level, and a
        ``seasonal_break`` flattens as much of the rhythm as that departure amounts
        to — capped at all of it, since there is no deeper break than no rhythm at
        all. Turn it down until detection starts failing; that number is the
        interesting one.
    noise
        Standard deviation of the Gaussian noise on the normal stretches.
    level
        The baseline the series sits at.
    period
        Samples per seasonal cycle, or None for no seasonality. Required for
        ``kind="seasonal_break"``, which has nothing to break otherwise.
    amplitude
        Peak deviation of the seasonal component, when ``period`` is set.
    freq
        Sampling interval: a duration string such as ``"5min"``, a
        :class:`~datetime.timedelta`, or a :class:`numpy.timedelta64`.
    start
        Timestamp of the first sample, as an ISO 8601 string.
    seed
        Seed for the noise, so a given call always produces the same series.
    backend
        Emit the series into this dataframe backend — ``"pandas"``, ``"polars"``,
        ``"pyarrow"``. The default returns a :class:`~hazure.TimeSeries`, which
        every part of hazure accepts and which needs no dataframe library
        installed.

    Returns
    -------
    Dataset
        The series, the intervals the anomalies occupy, and a description of what
        was planted.

    Raises
    ------
    KeyError
        ``kind`` is not one of :data:`KINDS`.
    ValueError
        An argument is non-positive, ``period`` is missing for a seasonal break,
        or the anomalies asked for will not fit in the series without touching.

    See Also
    --------
    hazure.datasets.load_nab : Real series, with labels somebody else argued over.

    Examples
    --------
    Three spikes in a fortnight of five-minute samples:

    >>> from hazure.datasets import make_series
    >>> dataset = make_series("spike", n=4032, n_anomalies=3)
    >>> dataset.events.n_events
    3
    >>> dataset
    Dataset('spike', 4032 rows, 3 events)

    The ground truth is intervals, and comes back as labels on the series' own
    time axis when that is what a metric wants:

    >>> float(dataset.labels.values.sum())
    3.0

    A level shift, which is invisible in the value distribution — the shifted
    samples are all well inside the normal range — and obvious in the rolling
    mean:

    >>> shifted = make_series("level_shift", n=2000, strength=4.0)
    >>> from hazure import IqrDetector, LevelShiftDetector
    >>> from hazure.evaluation import recall
    >>> recall(shifted.events, to_events(IqrDetector().fit_detect(shifted.data)))
    0.0
    >>> detector = LevelShiftDetector(window=24)
    >>> recall(shifted.events, to_events(detector.fit_detect(shifted.data)))
    1.0
    """
    if kind not in KINDS:
        msg = f"{kind!r} is not a kind make_series knows; it plants {sorted(KINDS)}."
        raise KeyError(msg)
    span = _WIDTHS[kind] if width is None else width
    _check(n, n_anomalies, span, noise, strength, kind, period)

    step = parse_duration(freq)
    time = np.datetime64(start, "ns").astype(np.int64) + step * np.arange(
        n, dtype=np.int64
    )

    stretches = [
        slice(begin, begin + span) for begin in _positions(n, n_anomalies, span)
    ]
    labels = np.zeros(n, dtype=np.float64)
    for stretch in stretches:
        labels[stretch] = 1.0

    # A volatility shift is the one kind that changes how the noise is *drawn*
    # rather than what is added to it, so it has to be settled before the noise
    # exists. Scaling the draw is what makes `strength` the multiplier it claims
    # to be: adding a second independent draw instead would give a spread of
    # noise * sqrt(1 + (strength - 1) ** 2), which is neither the documented
    # number nor a memorable one.
    spread = np.full(n, float(noise), dtype=np.float64)
    if kind == "volatility_shift":
        for stretch in stretches:
            spread[stretch] = noise * strength

    rng = np.random.default_rng(seed)
    values = np.full(n, float(level), dtype=np.float64)
    if period is not None:
        values += amplitude * np.sin(2.0 * np.pi * np.arange(n) / period)
    values += rng.standard_normal(n) * spread

    for stretch in stretches:
        _plant(kind, values, stretch, strength, noise, period, amplitude, n)

    series = TimeSeries.from_arrays(time, values, ["value"])
    marks = to_events(series.wrap(labels))
    return Dataset(
        name=kind,
        data=series.to_native(backend=backend),
        events=_one_events(marks),
        description=(
            f"{n_anomalies} x {kind} ({KINDS[kind]}), {span} sample(s) each at "
            f"strength {strength:g}, over {n} samples of {freq} data with "
            f"noise {noise:g}"
            + ("" if period is None else f" and a {period}-sample cycle")
        ),
    )


def _check(
    n: int,
    n_anomalies: int,
    width: int,
    noise: float,
    strength: float,
    kind: str,
    period: int | None,
) -> None:
    """Reject arguments that cannot produce the series they describe.

    Parameters
    ----------
    n, n_anomalies, width, noise, strength, kind, period
        As passed to :func:`make_series`.

    Raises
    ------
    ValueError
        Something is non-positive, a seasonal break has no season to break, or the
        anomalies will not fit without running into each other.
    """
    for name, value in (("n", n), ("n_anomalies", n_anomalies), ("width", width)):
        if value < 1:
            msg = f"{name} must be at least 1, got {value}."
            raise ValueError(msg)
    for name, amount in (("noise", noise), ("strength", strength)):
        if amount <= 0.0:
            msg = f"{name} must be positive, got {amount}."
            raise ValueError(msg)
    if kind == "volatility_shift" and strength <= 1.0:
        msg = (
            f"kind='volatility_shift' needs strength above 1, got {strength}: the "
            f"strength multiplies the noise, so 1 would leave it unchanged."
        )
        raise ValueError(msg)
    if kind == "seasonal_break" and period is None:
        msg = (
            "kind='seasonal_break' needs period=... — there is no rhythm to break "
            "in a series that has none."
        )
        raise ValueError(msg)
    if period is not None and period < 2:
        msg = f"period must be at least 2 samples, got {period}."
        raise ValueError(msg)

    # The anomalies live in the middle 70%, and two that touch would merge into
    # one event, quietly making n_anomalies a lie.
    usable = int(0.7 * n)
    if n_anomalies * (width + 1) > usable:
        msg = (
            f"{n_anomalies} anomalies of {width} sample(s) will not fit in the "
            f"{usable} samples make_series plants into, and any that touch would "
            f"merge into one event. Raise n, or lower n_anomalies or width."
        )
        raise ValueError(msg)


def _positions(n: int, n_anomalies: int, width: int) -> list[int]:
    """Space the anomalies evenly across the middle of the series.

    Parameters
    ----------
    n
        Length of the series.
    n_anomalies
        How many to place.
    width
        Length of each.

    Returns
    -------
    list of int
        Starting index of each anomaly, in order.
    """
    first = int(0.15 * n)
    last = int(0.85 * n) - width
    if n_anomalies == 1:
        return [(first + last) // 2]
    stride = (last - first) / (n_anomalies - 1)
    return [first + round(index * stride) for index in range(n_anomalies)]


def _plant(
    kind: str,
    values: NDArray[np.float64],
    stretch: slice,
    strength: float,
    noise: float,
    period: int | None,
    amplitude: float,
    n: int,
) -> None:
    """Deform one stretch of the series into the requested anomaly, in place.

    Parameters
    ----------
    kind
        Which shape to plant. ``"volatility_shift"`` is already done by the time
        this is called — it changed how the noise was drawn — so it does nothing
        here.
    values
        The series so far, modified in place.
    stretch
        The samples to deform.
    strength, noise, period, amplitude, n
        As passed to :func:`make_series`.
    """
    if kind in ("spike", "level_shift"):
        values[stretch] += strength * noise
    elif kind == "dip":
        values[stretch] -= strength * noise
    elif kind == "seasonal_break" and period is not None:
        # Flatten the rhythm by as much as `strength` multiples of the noise
        # amount to, so that strength means the same thing here as it does for a
        # spike — a departure of strength * noise from what the series should have
        # been. Removing the whole cycle is the most that can be asked, so a
        # strength beyond that saturates rather than inventing a deeper break.
        removed = min(1.0, strength * noise / amplitude)
        index = np.arange(n)[stretch]
        values[stretch] -= removed * amplitude * np.sin(2.0 * np.pi * index / period)


def _one_events(marks: Events | dict[str, Events]) -> Events:
    """Narrow :func:`~hazure.to_events` output, which is a dict for wide input."""
    if isinstance(marks, dict):  # pragma: no cover - the label series is univariate
        msg = "make_series builds one label column, so to_events returns one Events."
        raise TypeError(msg)
    return marks
