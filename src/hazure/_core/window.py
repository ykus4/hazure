"""Rolling window statistics on numpy arrays.

hazure computes these itself rather than delegating, for three reasons. narwhals,
the dataframe-agnostic layer at the I/O boundary, offers only
``rolling_mean``/``sum``/``std``/``var`` over integer windows — no median, no
quantile, no duration windows — and those omissions are exactly what robust
detection leans on. Owning the engine keeps the core free of heavier
dependencies. And two adjacent windows, the primitive behind spike and
level-shift detection, fall out symmetrically from explicit bounds, where a
dataframe API needs the series reversed and its index mirrored to look forwards.

Semantics follow ``pandas.Series.rolling`` so results are directly comparable,
including the ``closed`` boundary rules and the fact that a centred duration
window ignores ``closed``. The one deliberate divergence is ``count``: pandas
exempts it from ``min_periods``, hazure does not. See :func:`rolling`.

Everything routes through :func:`window_bounds`, which turns a window spec into
per-row ``[start, stop)`` positions, and :func:`aggregate_windows`, which
applies a statistic to those bounds. Aggregations are computed on a padded
matrix of windows, evaluated in row chunks so peak memory stays bounded
regardless of window width.
"""

from __future__ import annotations

import re
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, Final, Literal, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_T = TypeVar("_T")

__all__ = [
    "AGGREGATIONS",
    "Closed",
    "Window",
    "aggregate_windows",
    "double_rolling",
    "parse_duration",
    "rolling",
    "window_bounds",
]

Closed = Literal["right", "left", "both", "neither"]
Window = int | str | np.timedelta64 | timedelta

#: Statistics :func:`rolling` understands.
AGGREGATIONS: Final = frozenset(
    {
        "count",
        "nnz",
        "nunique",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        "skew",
        "kurt",
        "quantile",
        "iqr",
        "idr",
    }
)

# Cap on the padded window matrix, so a wide window costs time rather than
# memory. 8M float64 is 64 MiB.
_MAX_BLOCK_ELEMENTS: Final = 8_000_000

_DURATION_PATTERN: Final = re.compile(
    r"^\s*(?P<count>\d+(?:\.\d+)?)?\s*(?P<unit>[A-Za-zµ]+)\s*$"
)

# Fixed-length units only. Calendar units are deliberately absent: a month is
# not a duration, so a "1M" window would silently mean something different in
# February. Keys are lowercased before lookup, except where case disambiguates.
_DURATION_UNITS: Final[dict[str, int]] = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": 1_000,
    "µs": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "secs": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "min": 60 * 1_000_000_000,
    "mins": 60 * 1_000_000_000,
    "minute": 60 * 1_000_000_000,
    "minutes": 60 * 1_000_000_000,
    "h": 3_600 * 1_000_000_000,
    "hr": 3_600 * 1_000_000_000,
    "hour": 3_600 * 1_000_000_000,
    "hours": 3_600 * 1_000_000_000,
    "d": 86_400 * 1_000_000_000,
    "day": 86_400 * 1_000_000_000,
    "days": 86_400 * 1_000_000_000,
    "w": 7 * 86_400 * 1_000_000_000,
    "week": 7 * 86_400 * 1_000_000_000,
    "weeks": 7 * 86_400 * 1_000_000_000,
}

_CALENDAR_UNITS: Final = frozenset(
    {"month", "months", "mo", "y", "year", "years", "q", "quarter", "quarters", "b"}
)


# ---------------------------------------------------------------------------
# window specification
# ---------------------------------------------------------------------------


def parse_duration(spec: str | np.timedelta64 | timedelta) -> int:
    """Convert a duration to nanoseconds.

    Parameters
    ----------
    spec
        A string such as ``"7d"``, ``"30min"``, ``"1h30min"`` is *not*
        supported — use a single unit — or a ``timedelta`` /
        ``numpy.timedelta64``.

    Returns
    -------
    int
        Length of the duration in nanoseconds.

    Raises
    ------
    ValueError
        The string is unparseable, names a calendar unit, or is not positive.

    Examples
    --------
    >>> parse_duration("2h")
    7200000000000
    >>> parse_duration("500ms")
    500000000
    """
    if isinstance(spec, np.timedelta64):
        return int(spec.astype("timedelta64[ns]").astype(np.int64))
    if isinstance(spec, timedelta):
        return int(spec / timedelta(microseconds=1)) * 1_000

    match = _DURATION_PATTERN.match(spec)
    if match is None:
        msg = (
            f"Cannot read {spec!r} as a duration. Use a number followed by a "
            f"unit, e.g. '7d', '30min', '500ms'."
        )
        raise ValueError(msg)

    unit = match["unit"]
    # Case matters for exactly one unit: "M" is a calendar month, "m" is
    # minutes. Rule that out before folding case, so "5M" cannot quietly become
    # five minutes.
    if unit == "M":
        msg = (
            f"Unit 'M' in {spec!r} is a calendar unit (month), not a fixed "
            f"duration. Use 'm' for minutes, or days/hours."
        )
        raise ValueError(msg)
    key = unit.lower()
    if key in _CALENDAR_UNITS:
        msg = (
            f"Unit {unit!r} in {spec!r} is a calendar unit, not a fixed "
            f"duration, so a window of it would vary in length. Use days or "
            f"hours instead."
        )
        raise ValueError(msg)
    if key not in _DURATION_UNITS:
        msg = (
            f"Unknown duration unit {unit!r} in {spec!r}. Known units: "
            f"ns, us, ms, s, m/min, h, d, w."
        )
        raise ValueError(msg)

    count = float(match["count"]) if match["count"] is not None else 1.0
    nanoseconds = round(count * _DURATION_UNITS[key])
    if nanoseconds <= 0:
        msg = f"Duration {spec!r} must be positive."
        raise ValueError(msg)
    return nanoseconds


def window_bounds(
    time: NDArray[np.int64] | None,
    window: Window,
    n_rows: int,
    *,
    center: bool = False,
    closed: Closed | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Resolve a window spec into per-row ``[start, stop)`` positions.

    Parameters
    ----------
    time
        UTC nanoseconds, strictly increasing. Required for duration windows and
        ignored for integer ones.
    window
        A positive integer count of observations, or a duration.
    n_rows
        Length of the series.
    center
        Centre the window on each row rather than trailing it. A centred
        duration window spans ``(t - w/2, t + w/2]`` and ignores ``closed``,
        matching pandas.
    closed
        Which endpoints to include. Defaults to ``"right"``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(start, stop)``, each of length ``n_rows``, with
        ``0 <= start <= stop <= n_rows``.

    Raises
    ------
    ValueError
        The window is not positive, ``closed`` is unknown, or a duration window
        was requested without a time axis.
    """
    if closed is None:
        closed = "right"
    if closed not in ("right", "left", "both", "neither"):
        msg = (
            f"Unknown closed={closed!r}; expected 'right', 'left', 'both' or 'neither'."
        )
        raise ValueError(msg)

    if isinstance(window, bool):
        # bool is an int subclass, so this would otherwise mean "a window of 1".
        msg = f"A window must be a count or a duration, not {window!r}."
        raise TypeError(msg)
    if isinstance(window, int):
        if window < 1:
            msg = f"An integer window must be at least 1, got {window}."
            raise ValueError(msg)
        return _bounds_by_position(n_rows, window, center=center, closed=closed)

    span = parse_duration(window)
    if time is None:
        msg = (
            f"A duration window ({window!r}) needs a time axis; pass one, or "
            f"use an integer number of observations."
        )
        raise ValueError(msg)
    if time.shape[0] != n_rows:
        msg = f"Time axis has {time.shape[0]} entries but n_rows is {n_rows}."
        raise ValueError(msg)
    return _bounds_by_time(time, span, center=center, closed=closed)


def _bounds_by_position(
    n_rows: int, window: int, *, center: bool, closed: Closed
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Positional bounds, matching pandas' fixed-window indexer."""
    # pandas puts the extra observation of an even centred window on the left.
    offset = (window - 1) // 2 if center else 0
    stop = np.arange(1 + offset, n_rows + 1 + offset, dtype=np.int64)
    start = stop - window
    if closed == "left":
        start -= 1
        stop -= 1
    elif closed == "both":
        start -= 1
    elif closed == "neither":
        stop -= 1
    np.clip(start, 0, n_rows, out=start)
    np.clip(stop, 0, n_rows, out=stop)
    return start, np.maximum(stop, start)


def _bounds_by_time(
    time: NDArray[np.int64], span: int, *, center: bool, closed: Closed
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Bounds from timestamps, so gaps in the data shrink the window.

    ``searchsorted`` turns each row's time bounds into positions in one
    vectorised pass, which is what makes irregular sampling cost nothing extra.
    """
    if center:
        # pandas centres a duration window symmetrically and disregards
        # `closed`; mirroring that keeps results comparable.
        half = span // 2
        start = np.searchsorted(time, time - half, side="right")
        stop = np.searchsorted(time, time + half, side="right")
        return start.astype(np.int64), stop.astype(np.int64)

    # Including the left endpoint means the first equal timestamp is inside the
    # window, which is what side="left" finds.
    if closed in ("left", "both"):
        start = np.searchsorted(time, time - span, side="left")
    else:
        start = np.searchsorted(time, time - span, side="right")
    if closed in ("right", "both"):
        stop = np.arange(1, time.shape[0] + 1, dtype=np.int64)
    else:
        stop = np.searchsorted(time, time, side="left").astype(np.int64)
    return start.astype(np.int64), np.maximum(stop, start.astype(np.int64))


def default_min_periods(window: Window, closed: Closed | None = None) -> int:
    """Return pandas' default ``min_periods`` for a window spec.

    An integer window defaults to requiring a full window; a duration window
    defaults to a single observation, because its width legitimately varies.

    Parameters
    ----------
    window
        The window spec.
    closed
        Boundary rule, which changes how many observations a full integer
        window holds.

    Returns
    -------
    int
        The default minimum number of observations.
    """
    if isinstance(window, int) and not isinstance(window, bool):
        if closed == "both":
            return window + 1
        if closed == "neither":
            return max(window - 1, 1)
        return window
    return 1


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def aggregate_windows(
    values: NDArray[np.float64],
    start: NDArray[np.int64],
    stop: NDArray[np.int64],
    agg: str,
    *,
    min_periods: int = 1,
    q: float | None = None,
) -> NDArray[np.float64]:
    """Apply a statistic to explicit window bounds.

    Missing values are skipped, as in pandas, and a row whose window holds
    fewer than ``min_periods`` observations yields NaN.

    Parameters
    ----------
    values
        1-D float array.
    start, stop
        Per-row window bounds, half-open. Same length as the result.
    agg
        A name from :data:`AGGREGATIONS`.
    min_periods
        Minimum number of non-missing observations for a result.
    q
        Quantile in ``[0, 1]``. Required when ``agg="quantile"``.

    Returns
    -------
    numpy.ndarray
        Float array of length ``len(start)``.

    Raises
    ------
    ValueError
        The aggregation is unknown, the bounds disagree in length, or ``q`` is
        missing or out of range.
    """
    if agg not in AGGREGATIONS:
        msg = f"Unknown aggregation {agg!r}. Available: {sorted(AGGREGATIONS)}."
        raise ValueError(msg)
    if start.shape != stop.shape:
        msg = f"start has shape {start.shape} but stop has {stop.shape}."
        raise ValueError(msg)
    if agg == "quantile":
        if q is None:
            msg = "agg='quantile' needs q=... in [0, 1]."
            raise ValueError(msg)
        if not 0.0 <= q <= 1.0:
            msg = f"q must lie in [0, 1], got {q}."
            raise ValueError(msg)

    n_rows = start.shape[0]
    out = np.full(n_rows, np.nan, dtype=np.float64)
    if n_rows == 0:
        return out

    widths = stop - start
    max_width = int(widths.max()) if widths.size else 0
    if max_width == 0:
        return out

    # Bound peak memory rather than window width: one chunk of the padded
    # matrix at a time.
    rows_per_chunk = max(1, _MAX_BLOCK_ELEMENTS // max_width)
    offsets = np.arange(max_width, dtype=np.int64)
    last = values.shape[0] - 1

    for lo in range(0, n_rows, rows_per_chunk):
        hi = min(lo + rows_per_chunk, n_rows)
        chunk_start = start[lo:hi]
        chunk_stop = stop[lo:hi]
        index = chunk_start[:, None] + offsets[None, :]
        inside = index < chunk_stop[:, None]
        # Clip before gathering so out-of-window slots read a valid address;
        # `inside` masks whatever they picked up.
        block = np.where(inside, values[np.minimum(index, last)], np.nan)
        out[lo:hi] = _reduce_block(block, agg, min_periods=min_periods, q=q)

    return out


def _reduce_block(
    block: NDArray[np.float64], agg: str, *, min_periods: int, q: float | None
) -> NDArray[np.float64]:
    """Reduce a padded ``(rows, width)`` window matrix to one value per row."""
    present = ~np.isnan(block)
    counts = present.sum(axis=1)
    enough = counts >= max(min_periods, 1)

    if agg == "count":
        # Deliberate divergence from pandas, which exempts count() from
        # min_periods; being consistent here is less surprising.
        return np.where(enough, counts.astype(np.float64), np.nan)
    if agg == "nnz":
        nonzero = (present & (block != 0.0)).sum(axis=1)
        return np.where(enough, nonzero.astype(np.float64), np.nan)
    if agg == "nunique":
        return np.where(enough, _nunique(block), np.nan)

    # numpy warns on all-NaN slices and on the degenerate arithmetic that
    # short windows produce. Both are expected: `enough` already decides which
    # rows are reportable.
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        result = _statistic(block, agg, counts=counts, q=q)

    return np.where(enough, result, np.nan)


def _statistic(
    block: NDArray[np.float64],
    agg: str,
    *,
    counts: NDArray[np.int_],
    q: float | None,
) -> NDArray[np.float64]:
    """Compute one statistic across the rows of a padded window matrix.

    The result is assigned to a typed local before returning because numpy's
    ``nan*`` reductions are annotated loosely enough to erase the dtype.
    """
    result: NDArray[np.float64]
    if agg == "sum":
        # A window of only NaN sums to 0 in numpy; `enough` discards those rows.
        result = np.nansum(block, axis=1)
    elif agg == "mean":
        result = np.nanmean(block, axis=1)
    elif agg == "median":
        result = np.nanmedian(block, axis=1)
    elif agg == "min":
        result = np.nanmin(block, axis=1)
    elif agg == "max":
        result = np.nanmax(block, axis=1)
    elif agg in ("var", "std"):
        # ddof=1 to match pandas; a single observation gives NaN, as it should.
        variance: NDArray[np.float64] = np.nanvar(block, axis=1, ddof=1)
        result = variance if agg == "var" else np.sqrt(variance)
    elif agg == "quantile":
        assert q is not None
        result = np.nanquantile(block, q, axis=1)
    elif agg in ("iqr", "idr"):
        edges = (0.25, 0.75) if agg == "iqr" else (0.1, 0.9)
        spread: NDArray[np.float64] = np.nanquantile(block, edges, axis=1)
        result = spread[1] - spread[0]
    else:
        result = _shape_statistic(block, agg, counts=counts)
    return result


def _shape_statistic(
    block: NDArray[np.float64], agg: str, *, counts: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Compute bias-corrected skewness or excess kurtosis, as pandas defines them."""
    n = counts.astype(np.float64)
    centre: NDArray[np.float64] = np.nanmean(block, axis=1, keepdims=True)
    deviation = block - centre
    m2: NDArray[np.float64] = np.nanmean(deviation**2, axis=1)
    # A window with no spread has no shape; report NaN rather than dividing by
    # zero, as pandas does.
    m2 = np.where(m2 > 0.0, m2, np.nan)

    if agg == "skew":
        m3: NDArray[np.float64] = np.nanmean(deviation**3, axis=1)
        skew = np.sqrt(n * (n - 1.0)) * m3 / ((n - 2.0) * m2**1.5)
        return np.where(n >= 3, skew, np.nan)

    m4: NDArray[np.float64] = np.nanmean(deviation**4, axis=1)
    numerator = (n * n - 1.0) * m4 / (m2 * m2) - 3.0 * (n - 1.0) ** 2
    kurt = numerator / ((n - 2.0) * (n - 3.0))
    return np.where(n >= 4, kurt, np.nan)


def _nunique(block: NDArray[np.float64]) -> NDArray[np.float64]:
    """Count distinct non-missing values per row.

    Sorting pushes NaN to the end of each row, so distinctness reduces to
    counting neighbours that differ within the valid prefix.
    """
    ordered = np.sort(block, axis=1)
    valid = ~np.isnan(ordered)
    fresh = np.empty_like(valid)
    fresh[:, 0] = valid[:, 0]
    if ordered.shape[1] > 1:
        fresh[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
    distinct: NDArray[np.float64] = fresh.sum(axis=1).astype(np.float64)
    return distinct


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def rolling(
    values: NDArray[np.float64],
    window: Window,
    agg: str = "mean",
    *,
    time: NDArray[np.int64] | None = None,
    center: bool = False,
    min_periods: int | None = None,
    closed: Closed | None = None,
    q: float | None = None,
) -> NDArray[np.float64]:
    """Roll a window over a series and aggregate it.

    Mirrors ``pandas.Series.rolling(...).agg()``, extended with statistics
    pandas exposes only via ``apply`` (``nnz``, ``nunique``, ``iqr``, ``idr``)
    and available for duration windows on irregular data.

    Parameters
    ----------
    values
        1-D float array. NaN marks a missing observation and is skipped.
    window
        Observations (``int``) or duration (``"7d"``, ``timedelta``).
    agg
        A name from :data:`AGGREGATIONS`.
    time
        UTC nanoseconds. Required for duration windows.
    center
        Centre the window instead of trailing it.
    min_periods
        Minimum non-missing observations for a result. Defaults to the full
        window for integer windows and to 1 for duration windows, as in pandas.
    closed
        Which endpoints to include; defaults to ``"right"``. Ignored when
        ``center`` is set on a duration window, matching pandas.
    q
        Quantile in ``[0, 1]``, required when ``agg="quantile"``.

    Returns
    -------
    numpy.ndarray
        Float array the same length as ``values``.

    Notes
    -----
    Unlike pandas, ``agg="count"`` respects ``min_periods``: pandas returns the
    observation count even when it falls below ``min_periods``, which makes
    ``count`` behave unlike every other statistic.

    Examples
    --------
    >>> import numpy as np
    >>> rolling(np.array([1.0, 2.0, 3.0, 4.0]), 2, "mean")
    array([nan, 1.5, 2.5, 3.5])
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        msg = f"rolling expects a 1-D array, got {series.ndim}-D."
        raise ValueError(msg)

    start, stop = window_bounds(
        time, window, series.shape[0], center=center, closed=closed
    )
    resolved = (
        default_min_periods(window, closed) if min_periods is None else min_periods
    )
    return aggregate_windows(series, start, stop, agg, min_periods=resolved, q=q)


def double_rolling(
    values: NDArray[np.float64],
    window: Window | tuple[Window, Window],
    agg: str | tuple[str, str] = "median",
    *,
    time: NDArray[np.int64] | None = None,
    diff: Literal["l1", "l2", "diff", "rel_diff", "abs_rel_diff"] = "l1",
    min_periods: int | tuple[int | None, int | None] | None = None,
    q: float | None = None,
) -> NDArray[np.float64]:
    """Compare the window before each point with the window after it.

    This is the primitive behind spike and level-shift detection: aggregate the
    recent past and the near future separately, then measure how far apart they
    are. A large gap means the series changed character at that point.

    The left window covers observations strictly before each row; the right
    window covers the row itself and what follows. Both are expressed directly
    as bounds, so unlike pandas there is no need to reverse the series or build
    a mirrored index to get a forward-looking window.

    Parameters
    ----------
    values
        1-D float array.
    window
        One spec for both sides, or ``(left, right)``. Asymmetric windows are
        how spike detection differs from level-shift detection: a long left
        window characterises "normal", a short right window catches a blip.
    agg
        One statistic for both sides, or ``(left, right)``. Median by default,
        for robustness against the very outliers being detected. ``"std"``,
        ``"iqr"`` or ``"idr"`` turn this into volatility-shift detection.
    time
        UTC nanoseconds. Required for duration windows.
    diff
        How to compare the two sides. ``"l1"`` and ``"l2"`` give the unsigned
        magnitude, ``"diff"`` the signed ``right - left``, ``"rel_diff"`` that
        divided by the left value, and ``"abs_rel_diff"`` its magnitude.
    min_periods
        One value for both sides, or ``(left, right)``.
    q
        Quantile for ``agg="quantile"``.

    Returns
    -------
    numpy.ndarray
        Float array the same length as ``values``. NaN where either side lacked
        enough observations.

    Raises
    ------
    ValueError
        ``diff`` is unknown.

    Examples
    --------
    A step change shows up as a spike in the output:

    >>> import numpy as np
    >>> series = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    >>> double_rolling(series, 2, "mean", diff="diff")
    array([nan, nan, 2.5, 5. , 2.5, nan])
    """
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        msg = f"double_rolling expects a 1-D array, got {series.ndim}-D."
        raise ValueError(msg)
    if diff not in ("l1", "l2", "diff", "rel_diff", "abs_rel_diff"):
        msg = (
            f"Unknown diff={diff!r}; expected 'l1', 'l2', 'diff', 'rel_diff' "
            f"or 'abs_rel_diff'."
        )
        raise ValueError(msg)

    windows: tuple[Window, Window] = _as_pair(window)
    left_window, right_window = windows
    aggs: tuple[str, str] = _as_pair(agg)
    left_agg, right_agg = aggs
    mins: tuple[int | None, int | None] = _as_pair(min_periods)
    left_min, right_min = mins
    n_rows = series.shape[0]
    del window, agg, min_periods  # only the expanded pairs are used below

    # The left window trails and stops short of the current row: closed="left"
    # gives exactly `[i - w, i)`. The right window leads and includes the row.
    # Reversing the series turns that leading window back into a trailing one,
    # which is the only place the symmetry needs help.
    left_start, left_stop = window_bounds(time, left_window, n_rows, closed="left")
    left = aggregate_windows(
        series,
        left_start,
        left_stop,
        left_agg,
        min_periods=_resolve_min(left_min, left_window, "left"),
        q=q,
    )

    reversed_time = None if time is None else -time[::-1]
    right_start, right_stop = window_bounds(
        reversed_time, right_window, n_rows, closed="right"
    )
    right = aggregate_windows(
        series[::-1],
        right_start,
        right_stop,
        right_agg,
        min_periods=_resolve_min(right_min, right_window, "right"),
        q=q,
    )[::-1]

    if diff == "l1":
        return np.abs(right - left)
    if diff == "l2":
        return np.sqrt((right - left) ** 2)
    if diff == "diff":
        return right - left
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = (right - left) / left
    return relative if diff == "rel_diff" else np.abs(relative)


def _as_pair(spec: _T | tuple[_T, _T]) -> tuple[_T, _T]:
    """Expand a single value into a pair, or pass a 2-tuple through unchanged."""
    if isinstance(spec, tuple):
        if len(spec) != 2:
            msg = f"Expected one value or a 2-tuple, got {len(spec)} items."
            raise ValueError(msg)
        return spec
    return spec, spec


def _resolve_min(min_periods: int | None, window: Window, closed: Closed) -> int:
    """Fall back to the pandas default when ``min_periods`` was not given."""
    if min_periods is None:
        return default_min_periods(window, closed)
    return int(min_periods)
