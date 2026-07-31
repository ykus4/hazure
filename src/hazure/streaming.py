"""Driving a fitted component one observation at a time.

Everything else in hazure takes a whole series and answers about all of it. That
is the right shape for looking back over a month of history, and the wrong shape
for the thing the library is aimed at: a metric that is still being produced, one
sample at a time, where the question is asked again every minute and the answer
has to arrive before the next sample does.

:class:`Stream` bridges the two without a second implementation of any algorithm.
It keeps the recent past in a buffer, and for each arriving observation runs the
component over that buffer and returns the verdict on the last row. So the online
answer is the batch answer, by construction — not a reimplementation that agrees
with it on the cases anyone thought to test.

What that costs is a buffer long enough for the component to see what it needs.
Get it wrong and a rolling window is computed over a window that was never full,
which is the kind of mistake that produces plausible numbers rather than an error.
And not every component can be streamed at all: one that reads *forward* — a
centred window, the right-hand window of a double rolling aggregate — needs
observations that have not happened yet.

:meth:`Stream.prime` refuses both. It computes the component's answers from the
full history you hand it, recomputes several of them from the buffer alone, and
says which of the two problems it found when they disagree.

The other half of the story is that the fit does not move. A component is fitted
once, on a period you are willing to call normal, and can be stored with
:meth:`~hazure.Component.to_dict` and reloaded next week — so what "normal" means
is a decision you made deliberately and can point at, rather than a property of
whatever window the monitor happens to be looking at. Where you *do* want the
fence to move as scores arrive, :meth:`hazure.PotThreshold.update` is the piece
that does it, and the two compose: stream the scorer, and hand each score to the
threshold.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from hazure._core import Component, Configurable, TimeSeries, parse_duration
from hazure.events.interval import _timestamp_to_ns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


__all__ = [
    "Stream",
]


class Stream(Configurable):
    """A fitted component, fed one observation at a time.

    Parameters
    ----------
    component
        Any fitted :class:`~hazure.Component` — a detector, a scorer, a
        transformer, a :class:`~hazure.Pipeline` or a :class:`~hazure.Graph`. It
        is used, never refitted; fitting it is a separate decision made on data
        you chose.
    history
        How much of the past to keep. An ``int`` retains that many of the most
        recent observations. A duration — ``"7d"``, ``"90min"``, a
        :class:`~datetime.timedelta` — retains every observation within that
        distance of the newest one, which is what an irregular series wants,
        since a count of samples there is not a length of time.

    Attributes
    ----------
    n_seen : int
        Observations pushed in so far, counting those that arrived through
        :meth:`prime`.
    columns : tuple of str or None
        Column names being tracked, or None before the first observation.

    Raises
    ------
    TypeError
        ``component`` is not a :class:`~hazure.Component`, or ``history`` is
        neither an integer nor a duration.
    ValueError
        ``history`` is an integer below 2, or a duration that is not positive.

    See Also
    --------
    prime : Fill the buffer from history before the first live observation.
    hazure.PotThreshold.update : A fence that moves as scores arrive, for the
        cases where holding it fixed is not what you want.

    Notes
    -----
    Each observation costs one run of the component over the whole buffer, so a
    buffer of *h* rows makes the per-observation cost *O(h)* rather than the
    *O(1)* an incremental reformulation of each algorithm could reach. That is a
    deliberate trade. The series this is for arrive at human timescales — a sample
    a minute, a sample every ten seconds — and at those rates the cost is
    irrelevant next to the interval between samples, while the guarantee it buys
    is not: there is exactly one implementation of every algorithm, so the online
    and batch answers cannot drift apart.

    A stream carries its buffer, so :meth:`~hazure.Component.to_dict` stores a
    running monitor whole — the fitted component and the recent past it was
    judging against — and :meth:`~hazure.Component.from_dict` resumes it where it
    left off. ``history`` has to be an ``int`` or a string for that, since a
    :class:`~datetime.timedelta` has no JSON representation.

    Sizing ``history`` is the one thing that needs care, and it is not simply the
    window a detector was configured with. A detector built on
    :class:`~hazure.DoubleRollingAggregate` looks back over two windows;
    :class:`~hazure.SeasonalDetector` needs at least a period, and reads better
    with several; a :class:`~hazure.Pipeline` needs the sum of what its steps
    consume, because each step's output starts later than its input did. Rather
    than reason about it, hand :meth:`prime` more history than you think you need
    and let it tell you.

    Examples
    --------
    Fit on a fortnight, then stream the next day past it. The detector is fitted
    once and never sees the live data as training material:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from hazure import SpikeDetector
    >>> index = pd.date_range("2024-03-01", periods=24 * 14, freq="h", name="time")
    >>> rng = np.random.default_rng(0)
    >>> history = pd.Series(100 + rng.normal(0, 2, len(index)), index=index, name="rps")
    >>> detector = SpikeDetector(window=24).fit(history)
    >>> stream = Stream(detector, history=48).prime(history)

    A sample in line with the fortnight is unremarkable, and one four times the
    normal level is not:

    >>> stream.update("2024-03-15T00:00", 101.0)
    0.0
    >>> stream.update("2024-03-15T01:00", 400.0)
    1.0

    Streaming a batch answers the same question repeatedly, which is how a
    streaming setup gets backtested before it is deployed:

    >>> later = pd.Series(
    ...     100 + rng.normal(0, 2, 12),
    ...     index=pd.date_range("2024-03-15T02:00", periods=12, freq="h", name="time"),
    ...     name="rps",
    ... )
    >>> labels = stream.update_many(later)
    >>> float(labels.sum())
    0.0
    >>> stream.n_seen
    350
    """

    _time: NDArray[np.int64]
    _values: NDArray[np.float64]
    _columns: tuple[str, ...] | None = None
    _seen: int = 0

    def __init__(self, component: Component, history: int | str | timedelta) -> None:
        _check_component(component)
        self.component = component
        self.history = _as_history(history)
        self._time = np.empty(0, dtype=np.int64)
        self._values = np.empty((0, 0), dtype=np.float64)

    # -- state --------------------------------------------------------------

    @property
    def n_seen(self) -> int:
        """Observations pushed in so far, including those from :meth:`prime`."""
        return self._seen

    @property
    def columns(self) -> tuple[str, ...] | None:
        """Column names being tracked, or None before the first observation."""
        return self._columns

    @property
    def buffer(self) -> TimeSeries:
        """The retained past, as the component sees it.

        Returns
        -------
        TimeSeries
            The buffered observations. This is the internal representation rather
            than a reconstruction of whatever backend the data arrived in, since
            the buffer is assembled from many separate observations and has no
            single origin to return to.
        """
        return TimeSeries.from_arrays(
            self._time, self._values, self._columns or ("value",)
        )

    # -- filling ------------------------------------------------------------

    def prime(self, data: Any, *, check: bool = True) -> Stream:
        """Fill the buffer from history, and check that it is long enough.

        Without this the first observations are judged against an almost empty
        buffer, and a component that needs a window to look back over has no
        choice but to answer ``NaN`` until one has accumulated — for a detector
        with a daily period, that is a day of blindness immediately after
        deployment.

        Parameters
        ----------
        data
            Historical observations, anything :meth:`hazure.TimeSeries.from_any`
            accepts. Only the part ``history`` covers is retained; passing more
            than that is not wasteful, it is what makes the check below possible.
        check
            Verify that the retained buffer is long enough for the component, by
            comparing the answer it gives for the final observation against the
            answer the full ``data`` gives for that same observation. Turn it off
            only when the cost of one extra pass over ``data`` matters.

        Returns
        -------
        Stream
            This stream, for chaining.

        Raises
        ------
        RuntimeError
            The component has not been fitted.
        ValueError
            The buffer is too short for what the component looks back over, or the
            component reads forward and so cannot be streamed at any buffer
            length. Also raised if ``data`` carries columns the component was not
            fitted on.

        Notes
        -----
        The check compares the newest observation and a handful of interior ones,
        because the two ways streaming can fail show up in different places. A
        buffer that does not reach far enough back is visible at the newest
        observation. A component that reads *forward* is not — at the newest
        observation there is no future in either the batch pass or the buffer, so
        both are equally blind and agree. Interior rows are where the batch pass
        can see what came after and a stream could not.

        It is exact rather than statistical, and still not a proof: a buffer one
        sample short of a rolling window can agree on a quiet stretch of history
        by luck. Prime on a stretch that contains something interesting where you
        can.

        Examples
        --------
        A buffer far too short for a 24-sample window says so, rather than
        quietly reporting a window's worth of ``NaN``:

        >>> import numpy as np
        >>> import pandas as pd
        >>> from hazure import SpikeDetector
        >>> index = pd.date_range("2024-01-01", periods=200, freq="h", name="time")
        >>> rng = np.random.default_rng(0)
        >>> values = pd.Series(rng.normal(size=200), index=index, name="x")
        >>> detector = SpikeDetector(window=24).fit(values)
        >>> Stream(detector, history=5).prime(values)
        Traceback (most recent call last):
            ...
        ValueError: Stream(history=5) is too short for SpikeDetector: ...
        """
        ts = TimeSeries.from_any(data)
        self._columns = _resolve_columns(self.component, ts.columns)
        ordered = ts.select(self._columns)

        keep = self._retained(ordered.time)
        self._time = np.ascontiguousarray(ordered.time[keep])
        self._values = np.ascontiguousarray(ordered.values[keep])
        self._seen = ordered.n_rows

        if check:
            self._check_the_buffer_reproduces(ordered)
        return self

    def _retained(self, time: NDArray[np.int64]) -> slice:
        """Return the slice of ``time`` that ``history`` covers."""
        if isinstance(self.history, int):
            start = max(time.shape[0] - self.history, 0)
        else:
            span = parse_duration(self.history)
            if time.shape[0] == 0:
                return slice(0, 0)
            # Inclusive of the oldest observation still within the span, so a
            # duration equal to k steps retains k + 1 samples.
            start = int(np.searchsorted(time, time[-1] - span, side="left"))
        return slice(start, time.shape[0])

    def _check_the_buffer_reproduces(self, full: TimeSeries) -> None:
        """Verify that streaming would reproduce the batch answer, and say why not.

        Parameters
        ----------
        full
            Everything :meth:`prime` was given, already narrowed to the
            component's columns.

        Raises
        ------
        ValueError
            The buffer is too short for what the component looks back over, or the
            component reads *forward* and cannot be streamed at all.

        Notes
        -----
        Two separate failures, and probing only the newest observation would find
        one of them. Comparing the last row catches a buffer that does not reach far
        enough back. It cannot catch a component that reads forward — a centred
        window, or the right-hand window of a
        :class:`~hazure.DoubleRollingAggregate` — because at the newest observation
        there is no future in *either* series, so both answers are equally blind and
        equally NaN, and they agree.

        So interior rows are probed too. For those the batch pass can see what came
        after and a stream could not, which is exactly the discrepancy a
        forward-looking component produces. The last row is checked first, because
        a buffer that is too short fails every probe and the shortest explanation is
        the right one.
        """
        batch = self.component.run(full).values
        probes = _probe_positions(full.n_rows, self._time.shape[0])
        for position in probes:
            streamed = self.component.run(self._up_to(full, position)).values[-1]
            if np.allclose(batch[position], streamed, equal_nan=True):
                continue
            raise ValueError(
                self._explain(full, position, batch[position], streamed)
                if position == full.n_rows - 1
                else self._explain_forward(full, position, batch[position], streamed)
            )

    def _up_to(self, full: TimeSeries, position: int) -> TimeSeries:
        """Return the buffer as it would stand with ``position`` the newest row."""
        time, values = full.time[: position + 1], full.values[: position + 1]
        keep = self._retained(time)
        return TimeSeries.from_arrays(time[keep], values[keep], full.columns)

    def _explain(
        self,
        full: TimeSeries,
        position: int,
        expected: NDArray[np.float64],
        got: NDArray[np.float64],
    ) -> str:
        """Describe a buffer that does not reach far enough back."""
        return (
            f"Stream(history={self.history!r}) is too short for "
            f"{type(self.component).__name__}: it retains {self._time.shape[0]} of "
            f"the {full.n_rows} observations primed, and over that buffer the last "
            f"observation comes out as {_render(got)} rather than the "
            f"{_render(expected)} the full history gives it. The component looks "
            f"further back than the buffer reaches, so every live answer would be "
            f"computed from a window that was never full. Raise history until this "
            f"passes."
        )

    def _explain_forward(
        self,
        full: TimeSeries,
        position: int,
        expected: NDArray[np.float64],
        got: NDArray[np.float64],
    ) -> str:
        """Describe a component that reads forward and so cannot be streamed."""
        return (
            f"{type(self.component).__name__} cannot be streamed: it reads "
            f"observations that come *after* the one it is judging. Row {position} "
            f"of the primed history comes out as {_render(expected)} when the whole "
            f"series is available and {_render(got)} when only the "
            f"{self._time.shape[0]} observations up to it are — and only the second "
            f"is knowable live. Raising history will not help; a centred window, or "
            f"the right-hand window of a double rolling aggregate, needs the future. "
            f"Use a component that looks only backwards, or accept the delay by "
            f"holding each observation back until its window has filled."
        )

    # -- streaming ----------------------------------------------------------

    def update(
        self,
        time: Any,
        values: float | Mapping[str, float] | Sequence[float] | NDArray[np.float64],
    ) -> float | dict[str, float]:
        """Push one observation and return the component's verdict on it.

        Parameters
        ----------
        time
            Its timestamp: a :class:`~datetime.datetime`, a
            :class:`numpy.datetime64`, a pandas ``Timestamp``, an ISO 8601 string,
            or an int of UTC nanoseconds. Must be later than the newest
            observation already buffered.
        values
            The observation. A single number for a one-column component, or a
            mapping from column name to number, or a sequence in the order the
            component was fitted on.

        Returns
        -------
        float or dict of float
            The component's output for this observation — a label for a detector,
            a score for a scorer — as one number, or one per output column when
            the component emits several. ``NaN`` where the component could not
            place the observation, most often because the buffer has not filled.

        Raises
        ------
        RuntimeError
            The component has not been fitted.
        TypeError
            ``time`` is not a timestamp, or ``values`` is not a number, mapping or
            sequence.
        ValueError
            ``time`` is not after the newest buffered observation, or ``values``
            does not match the columns being tracked.

        See Also
        --------
        update_many : The same, over a batch of observations.

        Notes
        -----
        The observation goes into the buffer *before* it is judged, which is what
        makes the answer identical to the batch one: every detector here decides
        about a point using that point, and a spike is only a spike relative to
        the neighbours it sits among. Nothing after it is used, because nothing
        after it exists yet — which is also why a detector whose batch answer
        depends on the future, as a centred window does, will not agree with
        itself here. :meth:`prime` catches that.
        """
        stamp = _timestamp_to_ns(time)
        if self._time.shape[0] and stamp <= int(self._time[-1]):
            msg = (
                f"Stream.update() went backwards: {_render_stamp(stamp)} is not "
                f"after the newest buffered observation "
                f"{_render_stamp(int(self._time[-1]))}. A stream has to be fed in "
                f"time order; buffer out-of-order arrivals and pass them with "
                f"update_many() once they are sorted."
            )
            raise ValueError(msg)

        row = self._row(values)
        self._time = np.append(self._time, stamp)
        self._values = np.vstack([self._values, row]) if self._values.size else row
        self._seen += 1

        keep = self._retained(self._time)
        self._time = self._time[keep]
        self._values = self._values[keep]

        result = self.component.run(self.buffer)
        last = result.values[-1]
        if result.n_columns == 1:
            return float(last[0])
        return {name: float(last[i]) for i, name in enumerate(result.columns)}

    def update_many(self, data: Any) -> Any:
        """Push a batch of observations in order, returning one verdict each.

        Every row is judged as if it had just arrived, so this is not the same as
        running the component over ``data`` directly: a row is placed against the
        buffer as it stood at that moment and never against what came after it.
        That makes this the way to backtest a streaming setup — the numbers it
        produces are the numbers the monitor would have produced.

        Parameters
        ----------
        data
            Observations in time order, anything
            :meth:`hazure.TimeSeries.from_any` accepts.

        Returns
        -------
        Any
            One verdict per row of ``data``, in the same flavour as ``data``.

        Raises
        ------
        RuntimeError
            The component has not been fitted.
        ValueError
            The batch is not entirely after the newest buffered observation, or it
            carries columns the component was not fitted on.

        See Also
        --------
        update : The same, one observation at a time.
        """
        ts = TimeSeries.from_any(data)
        if self._columns is None:
            self._columns = _resolve_columns(self.component, ts.columns)
        ordered = ts.select(self._columns)

        verdicts: list[list[float]] = []
        names: tuple[str, ...] = ()
        for position in range(ordered.n_rows):
            verdict = self.update(
                int(ordered.time[position]),
                {
                    name: float(ordered.values[position, i])
                    for i, name in enumerate(ordered.columns)
                },
            )
            if isinstance(verdict, dict):
                names = tuple(verdict)
                verdicts.append(list(verdict.values()))
            else:
                verdicts.append([verdict])

        if not verdicts:
            return ts.wrap(np.empty((0, 1), dtype=np.float64)).to_native()
        matrix = np.asarray(verdicts, dtype=np.float64)
        return ts.wrap(matrix, names or None).to_native()

    def _row(
        self,
        values: float | Mapping[str, float] | Sequence[float] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Read one observation as a ``(1, n_columns)`` row, fixing the columns.

        Parameters
        ----------
        values
            A number, a mapping from column name to number, or a sequence.

        Returns
        -------
        numpy.ndarray
            The observation, ordered to match :attr:`columns`.

        Raises
        ------
        TypeError
            ``values`` is none of the accepted shapes.
        ValueError
            It names or counts columns differently from what is being tracked.
        """
        if isinstance(values, float | int | np.number) and not isinstance(values, bool):
            self._columns = self._columns or self._invented_columns(1)
            if len(self._columns) != 1:
                msg = (
                    f"Stream.update() was given one number, but this component "
                    f"tracks {list(self._columns)}. Pass a mapping from column "
                    f"name to value."
                )
                raise ValueError(msg)
            return np.array([[float(values)]], dtype=np.float64)

        if isinstance(values, Mapping):
            mapping = {str(name): float(item) for name, item in values.items()}
            self._columns = self._columns or _resolve_columns(
                self.component, tuple(mapping)
            )
            missing = [name for name in self._columns if name not in mapping]
            if missing:
                msg = (
                    f"Stream.update() is missing {missing}; this component tracks "
                    f"{list(self._columns)}."
                )
                raise ValueError(msg)
            return np.array(
                [[mapping[name] for name in self._columns]], dtype=np.float64
            )

        try:
            row = np.asarray(values, dtype=np.float64).reshape(1, -1)
        except (TypeError, ValueError) as error:
            msg = (
                f"Stream.update() cannot read {values!r} as an observation. Pass "
                f"a number, a mapping from column name to value, or a sequence."
            )
            raise TypeError(msg) from error

        self._columns = self._columns or self._invented_columns(row.shape[1])
        if row.shape[1] != len(self._columns):
            msg = (
                f"Stream.update() was given {row.shape[1]} values, but this "
                f"component tracks {len(self._columns)}: {list(self._columns)}."
            )
            raise ValueError(msg)
        return row

    def _invented_columns(self, count: int) -> tuple[str, ...]:
        """Name columns for a caller who passed bare numbers rather than a mapping.

        Parameters
        ----------
        count
            How many values arrived.

        Returns
        -------
        tuple of str
            The names the component was fitted on when there are as many of those
            as values arrived, and placeholders otherwise. Preferring the fitted
            names matters: a caller streaming one number into a component fitted on
            a column called ``"rps"`` means that column, and inventing ``"value"``
            instead would fail a comparison the caller never made.
        """
        learned = self.component.feature_names
        if learned is not None and len(learned) == count:
            return learned
        if count == 1:
            return ("value",)
        return tuple(f"value_{index}" for index in range(count))

    def __repr__(self) -> str:
        return (
            f"Stream({self.component!r}, history={self.history!r}, "
            f"buffered={self._time.shape[0]}, n_seen={self._seen})"
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


#: How many interior rows :meth:`Stream.prime` probes on top of the newest one.
#: Each probe costs one run over the buffer, so this is cheap; what it buys is
#: coverage of how far *forward* a component reaches, which the newest row cannot
#: reveal.
_PROBES = 4


def _probe_positions(n_rows: int, retained: int) -> list[int]:
    """Choose the rows :meth:`Stream.prime` compares batch against streamed.

    Parameters
    ----------
    n_rows
        How many observations were primed.
    retained
        How many of them the buffer kept.

    Returns
    -------
    list of int
        Positions to probe, newest first. Empty when the primed history is too
        short for any probe to mean anything.

    Notes
    -----
    The newest row comes first, so a buffer that is simply too short is diagnosed
    as that rather than as a component reading forward — it fails every probe, and
    the newest one carries the more useful explanation.

    Interior probes are spread over the back half of the primed history. A probe
    needs the buffer behind it to be full, or the disagreement it finds is about
    the warm-up rather than about the component; and it needs rows after it, or it
    is the newest row again under another name.
    """
    if n_rows < 2:
        return []
    positions = [n_rows - 1]
    earliest = max(retained - 1, 1)
    for step in range(1, _PROBES + 1):
        # 0.95, 0.85, 0.75, 0.65 of the way through: far enough in to have a full
        # buffer behind, far enough from the end to have a future ahead.
        position = int(n_rows * (1.0 - 0.1 * step)) - 1
        if earliest <= position < n_rows - 1 and position not in positions:
            positions.append(position)
    return positions


def _check_component(component: Any) -> None:
    """Reject anything that has no series to stream.

    Parameters
    ----------
    component
        The value passed to :class:`Stream`.

    Raises
    ------
    TypeError
        It is not a :class:`~hazure.Component`.
    """
    if not isinstance(component, Component):
        msg = (
            f"Stream needs a hazure Component to drive, got "
            f"{type(component).__name__}. An Aggregator takes several label "
            f"series rather than one, and so has nothing to stream on its own; "
            f"put it in a Graph and stream that."
        )
        raise TypeError(msg)


def _as_history(history: Any) -> int | str | timedelta | np.timedelta64:
    """Validate a buffer length and normalise it to something storable.

    Parameters
    ----------
    history
        The value passed to :class:`Stream`.

    Returns
    -------
    int or str or datetime.timedelta or numpy.timedelta64
        The same length, with a numpy integer narrowed to a plain ``int`` — the
        retention slice and :meth:`~hazure.Component.to_dict` both want one.

    Raises
    ------
    TypeError
        It is neither an integer nor a duration.
    ValueError
        It is an integer below 2, or a duration that is not positive.
    """
    if isinstance(history, bool):
        msg = "Stream history must be a count of samples or a duration, not a bool."
        raise TypeError(msg)
    if isinstance(history, int | np.integer):
        if history < 2:
            msg = (
                f"Stream history={history} keeps too little to compute anything "
                f"from; a buffer needs at least 2 observations, and in practice "
                f"at least as many as the component looks back over."
            )
            raise ValueError(msg)
        return int(history)
    if not isinstance(history, str | timedelta | np.timedelta64):
        msg = (
            f"Stream history={history!r} is neither a count of samples nor a "
            f"duration. Pass an int, or a string such as '7d' or '90min'."
        )
        raise TypeError(msg)
    # A string that is the right type but not a duration, and a duration that is
    # not positive, both belong to parse_duration; its messages are better than
    # anything that could be said here.
    parse_duration(history)
    return history


def _resolve_columns(component: Component, offered: tuple[str, ...]) -> tuple[str, ...]:
    """Decide which columns a stream tracks, and in which order.

    Parameters
    ----------
    component
        The component being streamed.
    offered
        Column names the caller's data carries, or invented placeholders when the
        caller passed bare numbers.

    Returns
    -------
    tuple of str
        The columns to buffer, ordered as the component was fitted.

    Raises
    ------
    ValueError
        The component was fitted on columns the caller has not offered.
    """
    learned = component.feature_names
    if learned is None:
        return offered
    missing = [name for name in learned if name not in offered]
    if missing:
        msg = (
            f"{type(component).__name__} was fitted on {list(learned)}, and "
            f"{missing} is not among the {list(offered)} being streamed. Feed the "
            f"stream the same columns the component was fitted on."
        )
        raise ValueError(msg)
    return learned


def _render(row: NDArray[np.float64]) -> str:
    """Render one output row for an error message."""
    if row.shape[0] == 1:
        return f"{float(row[0]):.6g}"
    return "[" + ", ".join(f"{float(value):.6g}" for value in row) + "]"


def _render_stamp(nanoseconds: int) -> str:
    """Render one timestamp for an error message."""
    return str(np.datetime64(nanoseconds, "ns"))
