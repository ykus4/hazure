"""Moving between binary labels and :class:`Events`.

A detector emits one label per sample; a human reads intervals. Converting
between the two turns on a single question — does a labelled timestamp stand for
an *instant*, or for the *period it opens*? On hourly data a spike at 03:00 is
usually the whole 03:00 hour, and two labelled samples in a row are one anomaly
lasting two hours, not two anomalies.

Frequency is carried on the :class:`~hazure.TimeSeries`, so the answer is
decided once, in :func:`_resolve_period`, from one place:

* ``ts.freq is not None`` — a labelled timestamp ``t`` covers
  ``[t, t + freq - 1ns]``, so consecutive labelled samples merge into one
  continuous interval.
* ``ts.freq is None`` — the series is irregular and a labelled timestamp is an
  instant, ``[t, t]``.

Every function here takes ``as_periods`` to override that inference explicitly.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from hazure import TimeSeries, parse_duration
from hazure.events._events import Events

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["expand_events", "to_events", "to_labels", "validate_series"]

#: Column name given to a label series built from a single, unnamed event set.
_DEFAULT_LABEL = "anomaly"


def to_events(
    labels: Any,
    *,
    as_periods: bool | None = None,
) -> Events | dict[str, Events]:
    """Convert binary labels into anomalous intervals.

    Parameters
    ----------
    labels
        Anything :meth:`hazure.TimeSeries.from_any` accepts: a pandas Series or
        DataFrame with a ``DatetimeIndex``, a polars or pyarrow frame with a
        temporal column, or a ``TimeSeries``. Values are clipped to ``[0, 1]``
        and compared to 1, so booleans and 0/1 floats both work, and ``NaN`` is
        read as *not anomalous*.
    as_periods
        Whether a labelled timestamp stands for the period it opens rather than
        an instant. ``None`` (the default) decides from the series' inferred
        frequency: periods when it is regular, instants when it is not.

    Returns
    -------
    Events or dict of Events
        One ``Events`` for a single-column input; a dict keyed by column name
        when the input has more than one column.

    Raises
    ------
    ValueError
        ``as_periods=True`` but the series has no inferable frequency.

    Examples
    --------
    Two adjacent labels on hourly data are one two-hour event:

    >>> import pandas as pd
    >>> index = pd.date_range("2024-01-01", periods=4, freq="h")
    >>> to_events(pd.Series([0, 1, 1, 0], index=index))
    Events([2024-01-01T01:00:00..2024-01-01T02:59:59.999999999])

    Treat each labelled sample as an instant instead:

    >>> to_events(pd.Series([0, 1, 1, 0], index=index), as_periods=False)
    Events([2024-01-01T01:00:00, 2024-01-01T02:00:00])
    """
    ts = TimeSeries.from_any(labels)
    period = _resolve_period(ts.freq, as_periods, what="the label series")

    if ts.n_columns > 1:
        return {name: _column_events(ts, name, period) for name in ts.columns}
    return _column_events(ts, ts.columns[0], period)


def to_labels(
    events: Any,
    time: Any,
    *,
    as_periods: bool | None = None,
    backend: str | None = None,
) -> Any:
    """Convert anomalous intervals back into binary labels on a time axis.

    Inverse of :func:`to_events`: with matching ``as_periods``,
    ``to_events(to_labels(events, time)) == events`` for any ``events`` that
    :func:`to_events` could have produced on this axis. Bounds that fall between
    samples are snapped outwards to the samples they touch, since a label series
    can only say something about the samples it has.

    Parameters
    ----------
    events
        An ``Events``, anything :meth:`Events.from_any` accepts, or a dict of
        those. A dict produces one output column per key, in key order.
    time
        The time axis to label. May be a ``TimeSeries``, any native series or
        frame (whose time axis is used and whose values are ignored), or an
        ``int64`` array of UTC nanoseconds / a ``datetime64`` array.
    as_periods
        Whether a sample stands for the period it opens. ``None`` decides from
        the axis' inferred frequency. Under period semantics a sample is
        labelled when an event overlaps ``[t, t + freq - 1ns]``; under instant
        semantics, when an event contains ``t``.
    backend
        Emit into this backend instead of the one the time axis came from.

    Returns
    -------
    Any
        A native label series or frame of ``1.0`` / ``0.0``, on ``time``.

    Raises
    ------
    ValueError
        ``as_periods=True`` but the axis has no inferable frequency.

    Examples
    --------
    >>> import pandas as pd
    >>> index = pd.date_range("2024-01-01", periods=4, freq="h")
    >>> labels = pd.Series([0, 1, 1, 0], index=index)
    >>> to_labels(to_events(labels), labels).tolist()
    [0.0, 1.0, 1.0, 0.0]

    A dict of event sets becomes one column per key:

    >>> frame = to_labels({"spike": [index[0]], "shift": [index[3]]}, labels)
    >>> list(frame.columns)
    ['spike', 'shift']
    """
    ts = _time_axis(time)
    period = _resolve_period(ts.freq, as_periods, what="the time axis")

    if isinstance(events, dict):
        if not events:
            msg = "Cannot build labels from an empty dict; pass at least one key."
            raise ValueError(msg)
        names = list(events)
        matrix = np.column_stack(
            [_mark(Events.from_any(events[name]), ts.time, period) for name in names]
        )
    else:
        names = [_DEFAULT_LABEL]
        matrix = _mark(Events.from_any(events), ts.time, period)[:, None]

    return ts.wrap(matrix, names).to_native(backend=backend)


def expand_events(
    events: Any,
    *,
    before: int | str | timedelta | np.timedelta64 = 0,
    after: int | str | timedelta | np.timedelta64 = 0,
) -> Events | dict[str, Events]:
    """Widen every event by a margin, then re-merge what now overlaps.

    Useful for scoring tolerantly — an alert a minute early still counts — and
    for turning instantaneous detections into inspectable windows.

    .. warning::
       **A bare ``int`` is a number of nanoseconds**, not seconds and not
       samples. Pass a string such as ``"1h"`` or a ``timedelta`` whenever you
       mean a human-scale duration.

    Parameters
    ----------
    events
        An ``Events``, anything :meth:`Events.from_any` accepts, or a dict of
        those. Both margins apply uniformly to every key of a dict.
    before
        Margin added before each ``start``. A duration string (``"1h"``,
        ``"30min"``), a ``timedelta``, a ``numpy.timedelta64``, or an ``int`` of
        nanoseconds.
    after
        Margin added after each ``end``, same accepted types.

    Returns
    -------
    Events or dict of Events
        Expanded events, re-merged. A dict input gives a dict output with the
        same keys.

    Raises
    ------
    TypeError
        A margin is neither a duration nor an int of nanoseconds.
    ValueError
        A margin is negative, or a duration string is unparseable.

    Examples
    --------
    >>> events = Events.from_any([("2024-01-01T06:00", "2024-01-01T07:00")])
    >>> expand_events(events, before="1h")
    Events([2024-01-01T05:00:00..2024-01-01T07:00:00])

    Ten nanoseconds, not ten seconds:

    >>> expand_events(Events.from_bounds([[100, 200]]), after=10).bounds.tolist()
    [[100, 210]]
    """
    left = _as_margin(before, "before")
    right = _as_margin(after, "after")

    if isinstance(events, dict):
        return {name: _expand_one(value, left, right) for name, value in events.items()}
    return _expand_one(events, left, right)


def validate_series(
    data: Any,
    *,
    sort: bool = True,
    drop_duplicates: bool = True,
) -> Any:
    """Normalise a time series and hand it back in its own flavour.

    Every detector runs this normalisation internally. Calling it yourself makes
    the result inspectable, so a surprising detection can be traced back to a
    reordered axis or a dropped duplicate rather than guessed at.

    The normalisation is: read the time axis, sort it, keep the first
    observation at each timestamp, and cast the value columns to ``float64``.

    Parameters
    ----------
    data
        Anything :meth:`hazure.TimeSeries.from_any` accepts.
    sort
        Sort by time when the input is not already ordered.
    drop_duplicates
        Keep the first observation at each timestamp. When False, duplicated
        timestamps raise instead of being collapsed.

    Returns
    -------
    Any
        A native object of the same flavour as ``data``.

    Raises
    ------
    TypeError
        ``data`` is not a recognised dataframe, or its time axis is not
        temporal.
    ValueError
        Timestamps are missing, or duplicated with ``drop_duplicates=False``.

    Examples
    --------
    >>> import pandas as pd
    >>> index = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"])
    >>> validate_series(pd.Series([3.0, 1.0, 9.0], index=index)).tolist()
    [1.0, 3.0]
    """
    return TimeSeries.from_any(
        data, sort=sort, drop_duplicates=drop_duplicates
    ).to_native()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_period(
    freq: int | None, as_periods: bool | None, *, what: str
) -> int | None:
    """Return the period length in nanoseconds, or None for instant semantics."""
    if as_periods is None:
        return freq
    if not as_periods:
        return None
    if freq is None:
        msg = (
            f"as_periods=True needs a sampling frequency, but {what} is "
            f"irregular (or has fewer than 3 samples, which is too few to "
            f"infer one). Pass as_periods=False to treat each timestamp as an "
            f"instant, or resample onto a regular axis first."
        )
        raise ValueError(msg)
    return freq


def _column_events(ts: TimeSeries, name: str, period: int | None) -> Events:
    """Turn one label column into events under the resolved semantics."""
    values = ts.column_values(name)
    flagged = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0) == 1.0
    starts = ts.time[flagged]
    if starts.size == 0:
        return Events.empty(origin=ts.origin)

    # One interval per labelled sample; from_bounds merges the runs. Under
    # period semantics consecutive samples end up a nanosecond apart, which
    # counts as adjacent, so a run collapses into a single interval.
    ends = starts if period is None else starts + period - 1
    return Events.from_bounds(np.column_stack((starts, ends)), origin=ts.origin)


def _mark(
    events: Events, axis: NDArray[np.int64], period: int | None
) -> NDArray[np.float64]:
    """Return 1.0/0.0 per position of ``axis``, for the given event set."""
    labels = np.zeros(axis.shape[0], dtype=np.float64)
    if events.n_events == 0 or axis.size == 0:
        return labels

    # Under period semantics sample t owns [t, t + period - 1ns], so it is hit
    # by any event reaching past t - period + 1ns.
    reach = events.bounds[:, 0] if period is None else events.bounds[:, 0] - period + 1
    lo = np.searchsorted(axis, reach, side="left")
    hi = np.searchsorted(axis, events.bounds[:, 1], side="right")

    # Difference array rather than a loop over intervals: +1 where each covered
    # run opens, -1 just past where it closes, then a cumulative sum.
    deltas = np.zeros(axis.shape[0] + 1, dtype=np.int64)
    np.add.at(deltas, lo, 1)
    np.add.at(deltas, hi, -1)
    labels[np.cumsum(deltas[:-1]) > 0] = 1.0
    return labels


def _time_axis(time: Any) -> TimeSeries:
    """Normalise whatever describes a time axis into a ``TimeSeries``."""
    if isinstance(time, TimeSeries):
        return time
    if isinstance(time, np.ndarray):
        if time.dtype.kind not in "iuM":
            msg = (
                f"A time axis array must be int64 UTC nanoseconds or "
                f"datetime64, got dtype {time.dtype}."
            )
            raise TypeError(msg)
        # Values are irrelevant here; from_arrays is the shortest route to a
        # validated axis with its frequency inferred the usual way.
        return TimeSeries.from_arrays(time, np.zeros(time.shape[0], dtype=np.float64))
    return TimeSeries.from_any(time)


def _expand_one(events: Any, before: int, after: int) -> Events:
    """Widen one event set and re-merge."""
    original = Events.from_any(events)
    if original.n_events == 0:
        return original
    widened = original.bounds + np.array([-before, after], dtype=np.int64)
    return Events.from_bounds(widened, origin=original.origin)


def _as_margin(spec: Any, name: str) -> int:
    """Read an expansion margin as a non-negative number of nanoseconds."""
    if isinstance(spec, bool):
        msg = f"Margin {name}={spec!r} must be a duration, not a bool."
        raise TypeError(msg)
    # numpy.timedelta64 subclasses numpy.signedinteger, so it has to be routed
    # to parse_duration before the plain-integer branch claims it.
    if isinstance(spec, np.timedelta64 | timedelta | str):
        margin = parse_duration(spec)
    elif isinstance(spec, int | np.integer):
        margin = int(spec)
    else:
        msg = (
            f"Margin {name}={spec!r} must be a duration string, a timedelta, "
            f"or an int of nanoseconds, not {type(spec).__name__}."
        )
        raise TypeError(msg)
    if margin < 0:
        msg = (
            f"Margin {name}={spec!r} is negative; expand_events only widens "
            f"events. Use a non-negative duration."
        )
        raise ValueError(msg)
    return margin
