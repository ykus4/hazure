"""Looking at a series, its anomalies and its scores.

Detection output is easiest to judge by eye, so this module offers one function,
:func:`plot`, which draws a series together with any number of verdicts about it.
Three things shaped it:

* **Verdicts have to be comparable.** ``anomaly`` accepts a mapping of names to
  label series or :class:`~hazure.events.Events`, and each one gets its own
  colour on the same time axis, so "did these two detectors agree" is a glance
  rather than a join.
* **Scores and data share no units.** A score is drawn on its own panel below the
  data rather than on top of it, so a score in the thousands cannot flatten the
  series it came from.
* **Nothing global changes.** matplotlib is imported inside the function, no
  style sheet is installed and ``rcParams`` is only ever read, so a caller's
  theme survives the call. Wrap the call in ``matplotlib.style.context(...)`` to
  draw under a different style.

matplotlib is an optional extra: ``pip install hazure[viz]``.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

from hazure._core.series import TimeSeries
from hazure.events import Events, to_events

__all__ = ["plot"]

# matplotlib ships type hints, but annotating against it would make
# type-checking hazure require an optional extra. These aliases keep the
# signature readable; the docstrings name the real classes.
Figure: TypeAlias = Any
Axes: TypeAlias = Any

#: Legend name given to an unnamed set of anomalies.
_DEFAULT_NAME = "anomaly"

_STYLES = ("span", "marker")
_PANEL_WIDTH = 10.0
_PANEL_HEIGHT = 2.2

# Matplotlib excludes artists whose label is this from the legend, which is how a
# repeated span contributes one entry rather than one per interval.
_HIDDEN = "_nolegend_"


def plot(
    series: Any = None,
    *,
    anomaly: Any = None,
    score: Any = None,
    layout: str | Sequence[str | Sequence[str]] = "each",
    axes: Sequence[Axes] | None = None,
    figsize: tuple[float, float] | None = None,
    style: str = "span",
    palette: Sequence[Any] | Mapping[str, Any] | None = None,
    alpha: float = 0.3,
    legend: bool = True,
    title: str | None = None,
) -> tuple[Figure, list[Axes]]:
    """Draw a time series with anomalies marked and scores panelled below it.

    Every argument is optional, and at least one of ``series``, ``anomaly`` and
    ``score`` must be given. Passing only ``anomaly`` draws the flagged intervals
    on a bare time axis, which is a useful way to compare several detectors'
    verdicts without the data getting in the way.

    Parameters
    ----------
    series
        The data to draw. Any pandas / polars / pyarrow object with a time axis,
        or a :class:`~hazure.TimeSeries`.
    anomaly
        Anomalies to mark. One of:

        - a label series or frame of ``1.0`` / ``0.0`` / ``NaN``, as a detector
          returns. A frame gives one named set per column;
        - an :class:`~hazure.events.Events`, or a list of timestamps and
          ``(start, end)`` pairs;
        - a mapping of any of those, keyed by the name to use in the legend.
          This is the form to reach for when comparing detectors.
    score
        Continuous scores, as a :class:`~hazure.scoring` component returns.
        Drawn on separate panels below the data, sharing its time axis, because
        a score and the series it describes have no units in common.
    layout
        How to distribute ``series`` columns over panels. ``"each"`` gives one
        panel per column, ``"all"`` puts every column on one panel, and an
        explicit sequence of column groups gives one panel per group — for
        instance ``[["cpu", "load"], ["latency"]]``. Score columns follow the
        same rule under ``"each"``, and otherwise share a single panel.
    axes
        Draw into these axes instead of creating a figure. Must hold at least as
        many axes as the layout needs; the surplus is left untouched.
    figsize
        Figure size in inches. Defaults to a width of 10 and 2.2 per panel.
        Ignored when ``axes`` is given.
    style
        ``"span"`` shades each anomalous interval across the full height of the
        panel; ``"marker"`` marks the flagged observations on the line itself.
        A panel with no line on it is always shaded, since a marker needs a
        value to sit on.
    palette
        Colours for the anomaly overlays, overriding the property cycle. Either
        a sequence, used in order and cycled, or a mapping from anomaly name to
        colour, which must cover every name.
    alpha
        Opacity of the shaded spans, in ``[0, 1]``. Markers are drawn opaque.
    legend
        Add a legend to each panel that has something to name.
    title
        Figure title. A single-panel figure gets it as an axes title instead, so
        it sits closer to the data.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn into, so it can be saved and closed by the caller.
    list of matplotlib.axes.Axes
        The panels used, data first and scores after, in drawing order.

    Raises
    ------
    ImportError
        matplotlib is not installed.
    ValueError
        Nothing was passed to draw, ``style`` or ``layout`` is not one of the
        accepted values, ``layout`` names a column that is absent, ``alpha`` is
        out of range, ``palette`` is empty or incomplete, or ``axes`` holds too
        few axes.

    Examples
    --------
    >>> import numpy as np, pandas as pd                       # doctest: +SKIP
    >>> from hazure.detection import SpikeDetector             # doctest: +SKIP
    >>> from hazure.plotting import plot                       # doctest: +SKIP
    >>> index = pd.date_range("2024-01-01", periods=500, freq="h")
    ... # doctest: +SKIP
    >>> values = pd.Series(np.sin(np.arange(500) / 12), index=index)
    ... # doctest: +SKIP
    >>> labels = SpikeDetector(window=24).fit_detect(values)   # doctest: +SKIP
    >>> fig, axs = plot(values, anomaly=labels)                # doctest: +SKIP

    Compare two detectors on one chart, and read the score that drove them:

    >>> from hazure.scoring import DeviationScorer             # doctest: +SKIP
    >>> fig, axs = plot(                                       # doctest: +SKIP
    ...     values,
    ...     anomaly={"spike": labels, "level": other_labels},
    ...     score=DeviationScorer().fit_score(values),
    ...     style="marker",
    ... )
    >>> fig.savefig("anomalies.png")                           # doctest: +SKIP
    """
    # Imported here, not at module scope, so that `import hazure` stays free of
    # matplotlib. Routed through importlib rather than an `import` statement so
    # that type-checking hazure does not require the viz extra to be installed.
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover - exercised by the extras job
        msg = (
            "hazure.plotting.plot needs matplotlib, which is an optional extra. "
            "Install it with `pip install hazure[viz]`."
        )
        raise ImportError(msg) from exc

    if style not in _STYLES:
        msg = f"style={style!r} must be one of {list(_STYLES)}."
        raise ValueError(msg)
    if not 0.0 <= alpha <= 1.0:
        msg = f"alpha={alpha!r} must lie in [0, 1]."
        raise ValueError(msg)
    if series is None and score is None and anomaly is None:
        msg = "plot() needs at least one of series=, anomaly= or score= to draw."
        raise ValueError(msg)

    data = None if series is None else TimeSeries.from_any(series)
    scores = None if score is None else TimeSeries.from_any(score)
    marks = _as_event_sets(anomaly)

    if data is None:
        data_groups: list[tuple[str, ...]] = []
        if not isinstance(layout, str):
            msg = (
                f"layout={list(layout)!r} names series columns, but no series "
                f'was passed. Use layout="each" or pass series=.'
            )
            raise ValueError(msg)
    else:
        data_groups = _resolve_groups(data.columns, layout)

    # Scores only get one panel each when the data does; any coarser layout is a
    # request for fewer panels, and the scores should honour that too.
    score_groups = (
        _resolve_groups(scores.columns, "each" if layout == "each" else "all")
        if scores is not None
        else []
    )

    # With neither data nor scores there is still an event timeline to show, so
    # keep one panel for it rather than returning an empty figure.
    panels = [*data_groups, *score_groups]
    n_panels = max(len(panels), 1)

    if axes is None:
        fig, grid = plt.subplots(
            n_panels,
            1,
            sharex=True,
            squeeze=False,
            figsize=figsize or (_PANEL_WIDTH, _PANEL_HEIGHT * n_panels),
        )
        panel_axes: list[Any] = list(grid[:, 0])
        created = True
    else:
        supplied = list(axes)
        if len(supplied) < n_panels:
            msg = (
                f"plot() needs {n_panels} axes for this layout but was given "
                f"{len(supplied)}."
            )
            raise ValueError(msg)
        panel_axes = supplied[:n_panels]
        fig = panel_axes[0].get_figure()
        created = False

    # Offset the anomaly colours past the widest panel so shading never lands on
    # the same colour as a line beneath it.
    widest = max((len(group) for group in panels), default=1)
    colours = _resolve_colours(
        list(marks), palette, _cycle_colours(plt.rcParams), offset=widest
    )

    for ax, group in zip(panel_axes, panels, strict=False):
        source = data if group in data_groups and data is not None else scores
        assert source is not None  # a group only exists for a series that exists
        _draw_lines(ax, source, group)
        _overlay(ax, source, group, marks, colours, style=style, alpha=alpha)
        ax.set_ylabel(group[0] if len(group) == 1 else ", ".join(group))
    if not panels:
        _overlay(panel_axes[0], None, (), marks, colours, style=style, alpha=alpha)

    if legend:
        for ax in panel_axes[:n_panels]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper left", fontsize="small")

    axis_name = data.origin.time_name if data is not None else "time"
    panel_axes[n_panels - 1].set_xlabel(axis_name)
    if title is not None:
        if n_panels == 1:
            panel_axes[0].set_title(title)
        else:
            fig.suptitle(title)
    if created:
        # Date ticks overlap at default sizes; rotating them touches this figure
        # only, unlike an rcParams change.
        fig.autofmt_xdate()

    return fig, panel_axes


# ---------------------------------------------------------------------------
# input normalisation
# ---------------------------------------------------------------------------


def _as_event_sets(anomaly: Any) -> dict[str, Events]:
    """Read the ``anomaly`` argument as named event sets.

    Everything is reduced to intervals, whichever form it arrived in, so the
    drawing code has one shape to handle and shading and marking agree about
    what is anomalous.
    """
    if anomaly is None:
        return {}

    if isinstance(anomaly, Mapping):
        named: dict[str, Events] = {}
        for key, value in anomaly.items():
            resolved = _one_event_set(value)
            if isinstance(resolved, dict):
                # A frame under a single key: qualify each column so two frames
                # sharing a column name stay distinguishable in the legend.
                named.update({f"{key}.{col}": ev for col, ev in resolved.items()})
            else:
                named[str(key)] = resolved
        return named

    resolved = _one_event_set(anomaly)
    return resolved if isinstance(resolved, dict) else {_DEFAULT_NAME: resolved}


def _one_event_set(value: Any) -> Events | dict[str, Events]:
    """Convert one anomaly argument to events, or to events per column."""
    if isinstance(value, Events):
        return value
    if isinstance(value, list | tuple | np.ndarray):
        # A bare sequence carries no time axis, so it can only be a list of
        # timestamps or (start, end) pairs.
        return Events.from_any(value)
    return to_events(value)


def _resolve_groups(
    columns: tuple[str, ...],
    layout: str | Sequence[str | Sequence[str]],
) -> list[tuple[str, ...]]:
    """Turn ``layout`` into the list of column groups, one per panel."""
    if layout == "each":
        return [(name,) for name in columns]
    if layout == "all":
        return [tuple(columns)]
    if isinstance(layout, str):
        msg = f'layout={layout!r} must be "each", "all", or a list of column groups.'
        raise ValueError(msg)

    groups: list[tuple[str, ...]] = []
    for entry in layout:
        names = (entry,) if isinstance(entry, str) else tuple(entry)
        unknown = [name for name in names if name not in columns]
        if unknown:
            msg = (
                f"layout names unknown column(s) {unknown}; available: {list(columns)}."
            )
            raise ValueError(msg)
        if not names:
            msg = "layout contains an empty column group; every panel needs a column."
            raise ValueError(msg)
        groups.append(names)
    if not groups:
        msg = "layout is empty; pass at least one column group."
        raise ValueError(msg)
    return groups


def _cycle_colours(rcparams: Any) -> list[Any]:
    """Read the active property cycle's colours, so a user's theme is respected."""
    cycle = rcparams["axes.prop_cycle"]
    colours = list(cycle.by_key().get("color", []))
    # A cycle can be built on markers alone, leaving no colours to borrow.
    return colours or ["C0"]


def _resolve_colours(
    names: list[str],
    palette: Sequence[Any] | Mapping[str, Any] | None,
    cycle: list[Any],
    *,
    offset: int,
) -> list[Any]:
    """Assign one colour per anomaly name."""
    if isinstance(palette, Mapping):
        missing = [name for name in names if name not in palette]
        if missing:
            msg = (
                f"palette has no colour for {missing}; give one per anomaly "
                f"name, or pass a sequence of colours instead."
            )
            raise ValueError(msg)
        return [palette[name] for name in names]

    pool = cycle if palette is None else list(palette)
    if not pool:
        msg = "palette is empty; pass at least one colour."
        raise ValueError(msg)
    start = offset if palette is None else 0
    return [pool[(start + i) % len(pool)] for i in range(len(names))]


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------


def _stamps(time_ns: Any) -> Any:
    """View an int64 nanosecond axis as ``datetime64``, which matplotlib reads."""
    return np.asarray(time_ns, dtype=np.int64).astype("datetime64[ns]")


def _draw_lines(ax: Any, ts: TimeSeries, group: tuple[str, ...]) -> None:
    """Draw one line per column of ``group``, taking colours from the cycle."""
    stamps = _stamps(ts.time)
    for name in group:
        ax.plot(stamps, ts.column_values(name), linewidth=1.2, label=name)


def _overlay(
    ax: Any,
    ts: TimeSeries | None,
    group: tuple[str, ...],
    marks: dict[str, Events],
    colours: list[Any],
    *,
    style: str,
    alpha: float,
) -> None:
    """Mark each named event set on one panel."""
    for (name, events), colour in zip(marks.items(), colours, strict=True):
        if style == "marker" and ts is not None and group:
            _draw_markers(ax, ts, group, events, name, colour)
        else:
            _draw_spans(ax, events, name, colour, alpha)


def _draw_markers(
    ax: Any,
    ts: TimeSeries,
    group: tuple[str, ...],
    events: Events,
    name: str,
    colour: Any,
) -> None:
    """Mark the flagged observations of every column on this panel."""
    covered = _covers(events, ts.time)
    if not covered.any():
        return
    stamps = _stamps(ts.time[covered])
    for position, column in enumerate(group):
        ax.plot(
            stamps,
            ts.column_values(column)[covered],
            linestyle="none",
            marker="o",
            markersize=5,
            color=colour,
            label=name if position == 0 else _HIDDEN,
        )


def _draw_spans(ax: Any, events: Events, name: str, colour: Any, alpha: float) -> None:
    """Shade each anomalous interval, or rule a line through instantaneous ones."""
    bounds = _stamps(events.bounds.reshape(-1)).reshape(events.bounds.shape)
    for position, (start, end) in enumerate(bounds):
        label = name if position == 0 else _HIDDEN
        if start == end:
            # A zero-width span draws nothing, so an instant needs a rule.
            ax.axvline(start, color=colour, alpha=alpha, linewidth=1.5, label=label)
        else:
            ax.axvspan(start, end, color=colour, alpha=alpha, linewidth=0, label=label)


def _covers(events: Events, time_ns: Any) -> Any:
    """Return a boolean mask of which timestamps fall inside ``events``.

    A difference array over the covered runs rather than a loop over intervals:
    ``+1`` where each run opens and ``-1`` just past where it closes, then one
    cumulative sum.
    """
    axis = np.asarray(time_ns, dtype=np.int64)
    if events.n_events == 0 or axis.size == 0:
        return np.zeros(axis.shape[0], dtype=bool)

    lo = np.searchsorted(axis, events.bounds[:, 0], side="left")
    hi = np.searchsorted(axis, events.bounds[:, 1], side="right")
    deltas = np.zeros(axis.shape[0] + 1, dtype=np.int64)
    np.add.at(deltas, lo, 1)
    np.add.at(deltas, hi, -1)
    return np.asarray(np.cumsum(deltas[:-1]) > 0)
