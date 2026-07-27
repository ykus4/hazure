"""The dataframe boundary.

Everything hazure computes happens on plain numpy arrays. ``TimeSeries`` is the
single place where a caller's pandas / polars / pyarrow object is turned into
those arrays, and the single place where results are turned back into whatever
the caller handed us.

Keeping that translation in one class buys three things that the design depends
on:

* **The time axis is data, not metadata.** Internally there is no index — just a
  sorted ``int64`` array of UTC nanoseconds. Combining two series is an explicit
  join on that array rather than an implicit pandas index alignment, so polars
  and pandas cannot silently disagree.
* **Sampling frequency is state on the object.** It is inferred once, on ingest,
  and then carried. Nothing downstream re-derives it from whatever frame it
  happens to be holding, which is what makes ``to_events`` and friends behave
  identically on every backend.
* **Round-tripping is total.** Time unit, time zone, and whether the timestamps
  arrived on an index or in a column are all remembered, so ``to_native``
  reproduces the caller's shape instead of an approximation of it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import narwhals.stable.v2 as nw
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray

__all__ = ["TimeSeries"]

# Nanoseconds per unit, for normalising every backend onto one integer scale.
_UNIT_TO_NS: dict[str, int] = {
    "s": 1_000_000_000,
    "ms": 1_000_000,
    "us": 1_000,
    "ns": 1,
}

_TimeUnit = Literal["s", "ms", "us", "ns"]

#: Backend recorded for a series built from raw arrays, where there is no native
#: object to reconstruct. numpy is the only dependency such a series has, and
#: emitting it must not silently acquire another.
NO_BACKEND = "arrays"


@dataclass(frozen=True, slots=True)
class Origin:
    """How the caller's data was shaped, so we can hand back the same shape.

    Attributes
    ----------
    backend
        Narwhals backend name the data came from, e.g. ``"pandas"``.
    container
        Whether the caller passed a 1-D series or a 2-D frame.
    time_on_index
        True when the timestamps arrived on a pandas-style index rather than in
        a column. Determines whether :meth:`TimeSeries.to_native` restores an
        index or emits a time column.
    time_name
        Name of the index or column that carried the timestamps.
    time_unit
        Resolution of the original timestamps. pandas 3 and polars both default
        to microseconds, so this is rarely ``"ns"``.
    time_zone
        IANA zone name of the original timestamps, or None if they were naive.
    index_name
        Name the pandas index had, which may be None. Tracked separately from
        ``time_name`` because resetting an unnamed index invents a placeholder
        name, and we want to hand back the caller's unnamed index verbatim.
    """

    backend: str
    container: Literal["series", "frame"]
    time_on_index: bool
    time_name: str
    time_unit: _TimeUnit
    time_zone: str | None
    index_name: str | None = None

    @classmethod
    def default(cls) -> Origin:
        """Return the origin used for series built directly from numpy arrays.

        Its backend is :data:`NO_BACKEND`, because there was no native object to
        come back to. Emitting such a series returns the ``TimeSeries`` itself
        rather than reaching for a dataframe library the caller may not have
        installed — see :meth:`TimeSeries.to_native`.
        """
        return cls(
            backend=NO_BACKEND,
            container="frame",
            time_on_index=True,
            time_name="time",
            time_unit="ns",
            time_zone=None,
            index_name="time",
        )


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """A time-indexed block of floating point data, backend-independent.

    Instances are immutable; every operation returns a new object. Construct
    them with :meth:`from_any` rather than calling ``TimeSeries(...)`` directly,
    unless you already hold validated arrays.

    Attributes
    ----------
    time
        UTC nanoseconds since the epoch, strictly increasing.
    values
        Shape ``(len(time), len(columns))``, always ``float64``. Missing
        observations are ``NaN``; boolean labels are stored as 0.0 / 1.0 / NaN.
    columns
        Column names, unique and in the same order as ``values``.
    freq
        Nanoseconds between consecutive samples, or None when the series is
        irregular or too short to tell. Calendar-based frequencies such as
        "month start" are irregular in nanoseconds and so report None.
    origin
        Provenance used to rebuild the caller's native type.
    """

    time: NDArray[np.int64]
    values: NDArray[np.float64]
    columns: tuple[str, ...]
    freq: int | None
    origin: Origin

    # -- construction -------------------------------------------------------

    @classmethod
    def from_any(
        cls,
        data: Any,
        *,
        time: str | None = None,
        sort: bool = True,
        drop_duplicates: bool = True,
    ) -> TimeSeries:
        """Build a ``TimeSeries`` from any supported native object.

        Parameters
        ----------
        data
            A pandas Series/DataFrame with a ``DatetimeIndex``, or a polars /
            pyarrow / modin / cuDF frame containing a temporal column. Passing
            an existing ``TimeSeries`` returns it unchanged.
        time
            Name of the column holding timestamps. Required only when a frame
            has more than one temporal column; otherwise it is detected. For
            pandas input this selects a column *instead of* the index.
        sort
            Sort by time when the input is not already ordered.
        drop_duplicates
            Keep the first observation at each timestamp. When False, duplicate
            timestamps raise instead.

        Returns
        -------
        TimeSeries
            Validated series with a strictly increasing time axis.

        Raises
        ------
        TypeError
            The object is not a recognised dataframe, or its time axis is not
            temporal.
        ValueError
            Column names are duplicated, timestamps are missing, or duplicate
            timestamps were found with ``drop_duplicates=False``.
        """
        if isinstance(data, TimeSeries):
            return data

        frame, origin, index = _ingest(data, time_name=time)
        if index is not None:
            time_ns, unit, tz = _time_from_index(index)
            value_names = tuple(frame.columns)
        else:
            time_ns, unit, tz = _time_from_column(frame, origin.time_name)
            value_names = tuple(c for c in frame.columns if c != origin.time_name)
        origin = replace(origin, time_unit=unit, time_zone=tz)

        if not value_names:
            msg = (
                f"{_describe(data)} has no value columns besides {origin.time_name!r}."
            )
            raise ValueError(msg)

        values = _to_float_matrix(frame, value_names)
        return cls._assemble(
            time_ns,
            values,
            value_names,
            origin,
            sort=sort,
            drop_duplicates=drop_duplicates,
        )

    @classmethod
    def from_arrays(
        cls,
        time: NDArray[Any],
        values: NDArray[Any],
        columns: Sequence[str] | None = None,
        *,
        origin: Origin | None = None,
        sort: bool = True,
        drop_duplicates: bool = True,
    ) -> TimeSeries:
        """Build a ``TimeSeries`` from raw numpy arrays.

        Parameters
        ----------
        time
            ``datetime64`` of any unit, or ``int64`` already in UTC nanoseconds.
        values
            1-D (one column) or 2-D ``(n_rows, n_columns)``. Cast to float64.
        columns
            Column names. Defaults to ``value``/``value_0``, ``value_1``, ...
        origin
            Provenance for :meth:`to_native`. Defaults to a pandas frame.
        sort
            Sort by time when not already ordered.
        drop_duplicates
            Keep the first observation at each timestamp.

        Returns
        -------
        TimeSeries
            Validated series.
        """
        time_ns = _as_epoch_ns(np.asarray(time))
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        elif matrix.ndim != 2:
            msg = f"values must be 1-D or 2-D, got {matrix.ndim}-D."
            raise ValueError(msg)

        if columns is None:
            names = (
                ("value",)
                if matrix.shape[1] == 1
                else tuple(f"value_{i}" for i in range(matrix.shape[1]))
            )
        else:
            names = tuple(columns)

        return cls._assemble(
            time_ns,
            matrix,
            names,
            origin or Origin.default(),
            sort=sort,
            drop_duplicates=drop_duplicates,
        )

    @classmethod
    def _assemble(
        cls,
        time_ns: NDArray[np.int64],
        values: NDArray[np.float64],
        columns: tuple[str, ...],
        origin: Origin,
        *,
        sort: bool,
        drop_duplicates: bool,
    ) -> TimeSeries:
        """Validate, order and de-duplicate, then freeze into a ``TimeSeries``."""
        if len(columns) != values.shape[1]:
            msg = (
                f"Got {len(columns)} column names for {values.shape[1]} "
                f"columns of data."
            )
            raise ValueError(msg)
        if len(set(columns)) != len(columns):
            duplicated = sorted({c for c in columns if columns.count(c) > 1})
            msg = f"Column names must be unique; duplicated: {duplicated}."
            raise ValueError(msg)
        if time_ns.shape[0] != values.shape[0]:
            msg = (
                f"Time axis has {time_ns.shape[0]} entries but values have "
                f"{values.shape[0]} rows."
            )
            raise ValueError(msg)

        # np.diff on an empty or single-element array is empty, so both the
        # ordering and duplicate checks below degrade to no-ops naturally.
        steps = np.diff(time_ns)
        if sort and steps.size and bool(np.any(steps < 0)):
            order = np.argsort(time_ns, kind="stable")
            time_ns = time_ns[order]
            values = values[order]
            steps = np.diff(time_ns)
        elif steps.size and bool(np.any(steps < 0)):
            msg = "Time axis is not sorted; pass sort=True to reorder it."
            raise ValueError(msg)

        if steps.size and bool(np.any(steps == 0)):
            if not drop_duplicates:
                n_dup = int(np.count_nonzero(steps == 0))
                msg = (
                    f"Time axis has {n_dup} duplicated timestamp(s); pass "
                    f"drop_duplicates=True to keep the first of each."
                )
                raise ValueError(msg)
            keep = np.ones(time_ns.shape[0], dtype=bool)
            keep[1:] = steps != 0
            time_ns = time_ns[keep]
            values = values[keep]

        return cls(
            time=np.ascontiguousarray(time_ns, dtype=np.int64),
            values=np.ascontiguousarray(values, dtype=np.float64),
            columns=columns,
            freq=_infer_freq(time_ns),
            origin=origin,
        )

    # -- shape --------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        """Number of observations."""
        return int(self.time.shape[0])

    @property
    def n_columns(self) -> int:
        """Number of value columns."""
        return len(self.columns)

    @property
    def is_univariate(self) -> bool:
        """True when the series carries exactly one column."""
        return len(self.columns) == 1

    def __len__(self) -> int:
        return self.n_rows

    def __repr__(self) -> str:
        freq = "irregular" if self.freq is None else _format_ns(self.freq)
        return (
            f"TimeSeries({self.n_rows} rows x {self.n_columns} cols, "
            f"freq={freq}, columns={list(self.columns)})"
        )

    # -- derivation ---------------------------------------------------------

    def wrap(
        self,
        values: NDArray[Any],
        columns: Sequence[str] | None = None,
    ) -> TimeSeries:
        """Return a new series on this time axis carrying ``values``.

        This is how scorers and transformers emit results: the time axis, freq
        and provenance are inherited, so only the numbers need supplying.

        Parameters
        ----------
        values
            1-D or 2-D array with ``n_rows`` rows.
        columns
            Names for the new columns. Defaults to this series' names when the
            width is unchanged, otherwise ``value_0``, ``value_1``, ...

        Returns
        -------
        TimeSeries
            Series sharing this time axis.
        """
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.shape[0] != self.n_rows:
            msg = (
                f"Cannot wrap {matrix.shape[0]} rows onto a time axis of {self.n_rows}."
            )
            raise ValueError(msg)

        if columns is not None:
            names = tuple(columns)
        elif matrix.shape[1] == self.n_columns:
            names = self.columns
        elif matrix.shape[1] == 1:
            names = ("value",)
        else:
            names = tuple(f"value_{i}" for i in range(matrix.shape[1]))

        if len(names) != matrix.shape[1]:
            msg = f"Got {len(names)} column names for {matrix.shape[1]} columns."
            raise ValueError(msg)
        if len(set(names)) != len(names):
            msg = f"Column names must be unique; got {list(names)}."
            raise ValueError(msg)

        # The time axis is already validated, so skip straight to the frozen
        # object rather than re-running the ordering checks.
        return TimeSeries(
            time=self.time,
            values=np.ascontiguousarray(matrix),
            columns=names,
            freq=self.freq,
            origin=self.origin,
        )

    def select(self, columns: Sequence[str] | str) -> TimeSeries:
        """Return a series restricted to ``columns``, in the order given.

        Provenance is carried through unchanged. Narrowing to one column does not
        turn a frame the caller passed into a series they did not: whether the
        result is emitted as a series depends on what arrived, not on how many
        columns happen to be left mid-computation.

        Parameters
        ----------
        columns
            One name, or a sequence of names.

        Returns
        -------
        TimeSeries
            Series carrying only the requested columns.

        Raises
        ------
        KeyError
            A requested name is not present.
        """
        names = (columns,) if isinstance(columns, str) else tuple(columns)
        missing = [c for c in names if c not in self.columns]
        if missing:
            msg = f"Unknown column(s) {missing}; available: {list(self.columns)}."
            raise KeyError(msg)
        index = [self.columns.index(c) for c in names]
        return TimeSeries(
            time=self.time,
            values=np.ascontiguousarray(self.values[:, index]),
            columns=names,
            freq=self.freq,
            origin=self.origin,
        )

    def iter_columns(self) -> Iterator[TimeSeries]:
        """Yield each column as its own univariate ``TimeSeries``.

        Yields
        ------
        TimeSeries
            One single-column series per column, in order.
        """
        for name in self.columns:
            yield self.select(name)

    def join(self, *others: TimeSeries) -> TimeSeries:
        """Outer-join other series onto this one along the time axis.

        This replaces pandas' implicit index alignment with something explicit
        and backend-independent: the union of all time axes, with NaN where a
        series had no observation.

        Parameters
        ----------
        *others
            Series to merge in. Their columns are appended in order.

        Returns
        -------
        TimeSeries
            Combined series over the union of every time axis.

        Raises
        ------
        ValueError
            Two inputs contribute the same column name.
        """
        if not others:
            return self

        parts = (self, *others)
        names: list[str] = []
        for part in parts:
            for name in part.columns:
                if name in names:
                    msg = (
                        f"Cannot join: column {name!r} appears in more than one series."
                    )
                    raise ValueError(msg)
                names.append(name)

        axis = parts[0].time
        for part in parts[1:]:
            if part.time.shape != axis.shape or not np.array_equal(part.time, axis):
                axis = np.union1d(axis, part.time)
                break
        else:
            # Every axis was identical, so the columns line up positionally.
            return TimeSeries(
                time=axis,
                values=np.hstack([p.values for p in parts]),
                columns=tuple(names),
                freq=self.freq,
                # Provenance is carried unchanged rather than forced to "frame":
                # joining is often a step on the way back down to one column, as
                # when an aggregator combines several label series, and the
                # caller who passed series should get a series back.
                origin=self.origin,
            )

        for part in parts[1:]:
            axis = np.union1d(axis, part.time)

        merged = np.full((axis.shape[0], len(names)), np.nan, dtype=np.float64)
        offset = 0
        for part in parts:
            rows = np.searchsorted(axis, part.time)
            merged[rows, offset : offset + part.n_columns] = part.values
            offset += part.n_columns

        return TimeSeries(
            time=axis,
            values=merged,
            columns=tuple(names),
            freq=_infer_freq(axis),
            origin=self.origin,
        )

    # -- egress -------------------------------------------------------------

    def to_native(self, *, backend: str | None = None) -> Any:
        """Rebuild the caller's native object.

        pandas input comes back with its ``DatetimeIndex`` restored, including
        time zone and resolution. polars and pyarrow input come back with the
        time column in its original position and name. A single-column series
        that arrived as a pandas Series leaves as a pandas Series.

        A series built by :meth:`from_arrays` has no native counterpart, so it
        returns *itself*. Building a dataframe would mean importing a library the
        caller never asked for — numpy is the only dependency such a series has,
        and emitting it should not add one. Pass ``backend`` to opt in.

        Parameters
        ----------
        backend
            Emit into this backend instead of the original one, e.g.
            ``"pandas"`` or ``"polars"``.

        Returns
        -------
        Any
            A native dataframe or series, or this ``TimeSeries`` when it was
            built from arrays and no ``backend`` was requested.
        """
        origin = self.origin
        target = backend or origin.backend
        if target == NO_BACKEND:
            return self

        stamps = _from_epoch_ns(self.time, origin.time_unit)
        payload: dict[str, NDArray[Any]] = {origin.time_name: stamps}
        for i, name in enumerate(self.columns):
            payload[name] = self.values[:, i]

        # Narwhals types `backend` as a literal union of the names it knows.
        # An unknown name raises from narwhals with a clearer message than any
        # check we could add here, so pass it straight through.
        frame = nw.from_dict(payload, backend=cast("Any", target))
        if origin.time_zone is not None:
            # The stored instants are UTC; declare that, then shift the display
            # zone back to whatever the caller was using.
            frame = frame.with_columns(
                nw.col(origin.time_name)
                .dt.replace_time_zone("UTC")
                .dt.convert_time_zone(origin.time_zone)
            )

        if origin.time_on_index:
            frame = nw.maybe_set_index(frame, column_names=[origin.time_name])

        native = frame.to_native()
        if origin.time_on_index and origin.index_name != origin.time_name:
            # Restoring an index that was originally unnamed: narwhals has no
            # index concept to express this through, and the object is one we
            # just built, so naming it directly is safe.
            native.index.name = origin.index_name
        if origin.container == "series" and self.n_columns == 1:
            return native[self.columns[0]]
        return native

    def to_numpy(self) -> NDArray[np.float64]:
        """Return the values as a 2-D float64 array."""
        return self.values

    def __getitem__(self, name: str) -> NDArray[np.float64]:
        """Return one column as a 1-D float64 array.

        Parameters
        ----------
        name
            Column name.

        Returns
        -------
        numpy.ndarray
            The column's values.
        """
        return self.column_values(name)

    def column_values(self, name: str) -> NDArray[np.float64]:
        """Return one column as a 1-D float64 array.

        Parameters
        ----------
        name
            Column name.

        Returns
        -------
        numpy.ndarray
            The column's values.

        Raises
        ------
        KeyError
            The name is not present.
        """
        if name not in self.columns:
            msg = f"Unknown column {name!r}; available: {list(self.columns)}."
            raise KeyError(msg)
        return self.values[:, self.columns.index(name)]


# ---------------------------------------------------------------------------
# ingest helpers
# ---------------------------------------------------------------------------


def _ingest(
    data: Any, *, time_name: str | None
) -> tuple[nw.DataFrame[Any], Origin, Any]:
    """Normalise a native object into a value frame, provenance, and time index.

    The third element is the pandas index carrying the timestamps, or None when
    the timestamps live in a column of the frame instead. Narwhals deliberately
    has no index concept — ``maybe_reset_index`` discards an index rather than
    promoting it to a column — so an index-borne time axis is read straight off
    the pandas object and never enters the frame.
    """
    series = nw.from_native(data, series_only=True, pass_through=True)
    if isinstance(series, nw.Series):
        index = nw.maybe_get_index(series)
        if index is None or not _is_temporal_index(index):
            msg = (
                f"{_describe(data)} carries no time axis. Give it a "
                f"DatetimeIndex, or pass a dataframe with a temporal column."
            )
            raise TypeError(msg)
        name = series.name if series.name else "value"
        return (
            series.rename(name).to_frame(),
            Origin(
                backend=series.implementation.name.lower(),
                container="series",
                time_on_index=True,
                time_name=index.name if index.name else "time",
                time_unit="ns",
                time_zone=None,
                index_name=index.name,
            ),
            index,
        )

    frame = nw.from_native(data, eager_only=True, pass_through=True)
    if not isinstance(frame, nw.DataFrame):
        msg = (
            f"{_describe(data)} is not a supported dataframe. Pass a pandas, "
            f"polars or pyarrow object, or use TimeSeries.from_arrays."
        )
        raise TypeError(msg)

    # A pandas frame keeps its time on the index unless the caller pointed at a
    # column explicitly; polars and pyarrow always carry it in a column.
    index = nw.maybe_get_index(frame)
    if (
        index is not None
        and _is_temporal_index(index)
        and (time_name is None or time_name not in frame.columns)
    ):
        return (
            frame,
            Origin(
                backend=frame.implementation.name.lower(),
                container="frame",
                time_on_index=True,
                time_name=index.name if index.name else "time",
                time_unit="ns",
                time_zone=None,
                index_name=index.name,
            ),
            index,
        )

    resolved = _resolve_time_column(frame, requested=time_name, source=_describe(data))
    return (
        frame,
        Origin(
            backend=frame.implementation.name.lower(),
            container="frame",
            time_on_index=False,
            time_name=resolved,
            time_unit="ns",
            time_zone=None,
            index_name=resolved,
        ),
        None,
    )


def _is_temporal_index(index: Any) -> bool:
    """Report whether a pandas index holds datetimes rather than labels."""
    return bool(getattr(index.dtype, "kind", None) == "M")


def _resolve_time_column(
    frame: nw.DataFrame[Any], *, requested: str | None, source: str
) -> str:
    """Pick the column holding timestamps, or explain why we cannot."""
    schema = frame.schema
    if requested is not None:
        if requested not in schema:
            msg = f"{source} has no column {requested!r}; available: {list(schema)}."
            raise ValueError(msg)
        if not schema[requested].is_temporal():
            msg = (
                f"Column {requested!r} of {source} is {schema[requested]}, "
                f"not a temporal type."
            )
            raise TypeError(msg)
        return requested

    temporal = [name for name, dtype in schema.items() if dtype.is_temporal()]
    if not temporal:
        msg = (
            f"{source} has no temporal column to use as a time axis; "
            f"columns are {list(schema)}."
        )
        raise TypeError(msg)
    if len(temporal) > 1:
        msg = (
            f"{source} has several temporal columns {temporal}; pass "
            f"time=... to choose one."
        )
        raise ValueError(msg)
    return temporal[0]


def _time_from_index(index: Any) -> tuple[NDArray[np.int64], _TimeUnit, str | None]:
    """Read a pandas DatetimeIndex as UTC nanoseconds, with its unit and zone.

    ``asi8`` is used rather than ``np.asarray`` because a time-zone-aware index
    converts to an *object* array of Timestamps, whereas ``asi8`` is always
    UTC integers at the index's own resolution.
    """
    dtype = index.dtype
    zone = getattr(dtype, "tz", None)
    unit = _unit_of(dtype)
    raw = np.asarray(index.asi8, dtype=np.int64)
    _reject_missing(raw, f"index {index.name!r}" if index.name else "the index")
    return raw * _UNIT_TO_NS[unit], unit, None if zone is None else str(zone)


def _time_from_column(
    frame: nw.DataFrame[Any], name: str
) -> tuple[NDArray[np.int64], _TimeUnit, str | None]:
    """Read a temporal column as UTC nanoseconds, with its unit and zone."""
    dtype = frame.schema[name]
    unit: str = getattr(dtype, "time_unit", None) or "ns"
    zone = getattr(dtype, "time_zone", None)
    if unit not in _UNIT_TO_NS:  # pragma: no cover - guards future units
        msg = f"Unsupported time unit {unit!r} on column {name!r}."
        raise TypeError(msg)

    time_ns = _as_epoch_ns(frame[name].to_numpy())
    _reject_missing(time_ns, f"column {name!r}")
    return time_ns, unit, None if zone is None else str(zone)  # type: ignore[return-value]


def _unit_of(dtype: Any) -> _TimeUnit:
    """Extract the resolution of a pandas datetime dtype.

    A time-zone-aware dtype exposes ``.unit`` directly; a naive one is a plain
    numpy dtype whose resolution is only visible in its repr.
    """
    unit = getattr(dtype, "unit", None)
    if unit is None:
        text = str(dtype)
        unit = text[text.index("[") + 1 : text.index("]")] if "[" in text else "ns"
    if unit not in _UNIT_TO_NS:  # pragma: no cover - guards future units
        msg = f"Unsupported time resolution {unit!r}."
        raise TypeError(msg)
    return unit  # type: ignore[return-value]


def _reject_missing(raw: NDArray[np.int64], where: str) -> None:
    """Fail loudly on NaT rather than letting int64 sentinels leak downstream."""
    if bool(np.any(raw == np.iinfo(np.int64).min)):
        msg = (
            f"Time axis in {where} contains missing timestamps; drop or fill "
            f"them before detection."
        )
        raise ValueError(msg)


def _to_float_matrix(
    frame: nw.DataFrame[Any], names: Sequence[str]
) -> NDArray[np.float64]:
    """Cast the named columns to float64 and stack them column-wise."""
    schema = frame.schema
    bad = [n for n in names if schema[n].is_temporal() or schema[n] == nw.String]
    if bad:
        msg = (
            f"Column(s) {bad} are not numeric. Encode them before detection "
            f"(e.g. one-hot for categoricals)."
        )
        raise TypeError(msg)
    casted = frame.select(nw.col(n).cast(nw.Float64) for n in names)
    return np.asarray(casted.to_numpy(), dtype=np.float64)


# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------


def _as_epoch_ns(values: NDArray[Any]) -> NDArray[np.int64]:
    """Convert a datetime64 or integer array to UTC epoch nanoseconds."""
    if values.dtype.kind == "M":
        return np.asarray(values.astype("datetime64[ns]").view(np.int64))
    if values.dtype.kind in "iu":
        return np.asarray(values, dtype=np.int64)
    if values.dtype.kind == "O":
        # pyarrow hands back objects for tz-aware columns on some versions.
        return np.asarray(
            np.array(values.tolist(), dtype="datetime64[ns]").view(np.int64)
        )
    msg = (
        f"Cannot read a time axis from dtype {values.dtype}; expected "
        f"datetime64 or integer nanoseconds."
    )
    raise TypeError(msg)


def _from_epoch_ns(time_ns: NDArray[np.int64], unit: _TimeUnit) -> NDArray[Any]:
    """Convert UTC epoch nanoseconds back to datetime64 at ``unit``."""
    return time_ns.view("datetime64[ns]").astype(f"datetime64[{unit}]")


def _infer_freq(time_ns: NDArray[np.int64]) -> int | None:
    """Return the constant step in nanoseconds, or None if there isn't one.

    Three points are the minimum needed to distinguish a real frequency from a
    single arbitrary gap, matching what pandas' ``inferred_freq`` requires.
    """
    if time_ns.shape[0] < 3:
        return None
    steps = np.diff(time_ns)
    first = int(steps[0])
    if first <= 0:
        return None
    return first if bool(np.all(steps == first)) else None


def _format_ns(nanoseconds: int) -> str:
    """Render a nanosecond step as a short human-readable duration."""
    for unit, size in (("d", 86_400), ("h", 3_600), ("m", 60), ("s", 1)):
        scale = size * 1_000_000_000
        if nanoseconds >= scale and nanoseconds % scale == 0:
            return f"{nanoseconds // scale}{unit}"
    return f"{nanoseconds}ns"


def _describe(data: Any) -> str:
    """Name an object well enough to make an error message actionable."""
    return f"{type(data).__module__}.{type(data).__qualname__}"
