"""Aggregating a sliding window into a new series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from hazure import BaseTransformer, rolling
from hazure._core.window import window_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from hazure import TimeSeries
    from hazure._core.window import Closed, Window

__all__ = [
    "RollingAggregate",
]


from hazure.features.spec import _is_sequence

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
