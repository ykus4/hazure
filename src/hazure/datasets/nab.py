"""The Numenta Anomaly Benchmark, fetched on demand and cached.

Fifty-eight labelled series — AWS CloudWatch metrics, taxi demand, machine
temperature, request latencies, and a set of artificial ones — with intervals
that were labelled by hand and argued over in public [1]_. That last part is what
makes it worth having: the labels are not perfect, but they are *fixed*, and a
number quoted against them can be compared with a number somebody else quoted
against them.

Nothing is bundled. The files are fetched from the benchmark's own repository the
first time they are asked for and cached on disk after that, so the download
happens once, visibly, and only for the series you name.

References
----------
.. [1] A. Lavin and S. Ahmad, "Evaluating Real-time Anomaly Detection Algorithms
   - the Numenta Anomaly Benchmark", IEEE ICMLA 2015, pp. 38-44.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

from hazure._core import TimeSeries
from hazure.datasets.dataset import Dataset
from hazure.events import Events

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = [
    "load_nab",
    "nab_names",
]


#: Where the benchmark's files are read from. Pinned to a tag rather than to a
#: branch: a benchmark whose contents can change under a cached copy is not a
#: benchmark, and a relabelled file would silently move every score computed
#: against it.
_BASE = "https://raw.githubusercontent.com/numenta/NAB/v1.1/"

#: The label file, which also serves as the catalogue — it has one key per
#: labelled series, so listing what is available costs the same single fetch that
#: loading one series does.
_LABELS = "labels/combined_windows.json"

#: How long to wait on the network before giving up, in seconds.
_TIMEOUT = 30


def nab_names(
    *, cache_dir: str | Path | None = None, download: bool = True
) -> list[str]:
    """List the series the benchmark has labels for.

    Parameters
    ----------
    cache_dir
        Where to keep downloaded files. Defaults to ``$HAZURE_DATA_HOME/nab``, or
        ``~/.cache/hazure/nab`` when that is unset.
    download
        Fetch the catalogue if it is not cached. With ``False`` an absent cache
        raises instead of reaching the network, which is what a test suite or an
        offline machine wants.

    Returns
    -------
    list of str
        Names accepted by :func:`load_nab`, such as
        ``"realAWSCloudwatch/ec2_cpu_utilization_5f5533"``, in sorted order.

    Raises
    ------
    OSError
        The catalogue is neither cached nor reachable.

    See Also
    --------
    load_nab : Load one of these.
    """
    windows = _windows(cache_dir, download=download)
    return sorted(name.removesuffix(".csv") for name in windows)


def load_nab(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    download: bool = True,
    backend: str | None = None,
) -> Dataset:
    """Load one labelled series from the Numenta Anomaly Benchmark.

    Parameters
    ----------
    name
        Which series, as ``"category/file"`` — for instance
        ``"realAWSCloudwatch/ec2_cpu_utilization_5f5533"``. A trailing ``".csv"``
        is accepted and ignored. :func:`nab_names` lists them.
    cache_dir
        Where to keep downloaded files. Defaults to ``$HAZURE_DATA_HOME/nab``, or
        ``~/.cache/hazure/nab`` when that is unset. Two files are cached: the
        series, and the label file shared by all of them.
    download
        Fetch what is not cached. With ``False`` an absent cache raises rather
        than reaching the network.
    backend
        Emit the series into this dataframe backend — ``"pandas"``, ``"polars"``,
        ``"pyarrow"``. The default returns a :class:`~hazure.TimeSeries`, which
        needs no dataframe library installed.

    Returns
    -------
    Dataset
        The series, its labelled anomaly windows, and its name.

    Raises
    ------
    KeyError
        The benchmark has no series by that name.
    OSError
        A needed file is neither cached nor reachable.

    See Also
    --------
    nab_names : What names this accepts.
    hazure.datasets.make_series : Synthetic series, no network and a known answer.

    Notes
    -----
    A NAB label window is a stretch of wall-clock time centred on the anomaly and
    deliberately wider than it, because the benchmark scores early detection
    favourably. So the windows are what the benchmark's own scoring uses, and they
    are not a claim that every sample inside one is abnormal — an event-based
    :func:`~hazure.recall` against them is the metric they support, and a
    sample-based :func:`~hazure.precision` against them is not really answerable.

    The series are irregular in places: several have gaps of hours where
    collection stopped. Detectors configured with sample counts are unaffected;
    ones configured with durations will find the gaps, which is a property of the
    data worth knowing about rather than one to paper over.

    The files come from tag ``v1.1`` of the benchmark's repository, so a cached
    copy and a fresh download are the same bytes.

    Examples
    --------
    Needs the network the first time, and nothing after that::

        >>> from hazure.datasets import load_nab, nab_names
        >>> nab_names()[:2]                                     # doctest: +SKIP
        ['artificialNoAnomaly/art_daily_no_noise',
         'artificialNoAnomaly/art_daily_perfect_square_wave']
        >>> taxi = load_nab("realKnownCause/nyc_taxi")          # doctest: +SKIP
        >>> taxi                                                # doctest: +SKIP
        Dataset('realKnownCause/nyc_taxi', 10320 rows, 5 events)
    """
    key = f"{name.removesuffix('.csv')}.csv"
    windows = _windows(cache_dir, download=download)
    if key not in windows:
        stem = name.split("/")[-1].removesuffix(".csv")
        near = [
            candidate.removesuffix(".csv")
            for candidate in sorted(windows)
            if stem in candidate
        ]
        msg = (
            f"The benchmark has no series {name!r}."
            + (f" Did you mean one of {near[:3]}?" if near else "")
            + f" nab_names() lists all {len(windows)}."
        )
        raise KeyError(msg)

    time, values = _series(key, cache_dir, download=download)
    series = TimeSeries.from_arrays(time, values, ["value"])
    bounds = [[_parse_stamp(start), _parse_stamp(end)] for start, end in windows[key]]
    return Dataset(
        name=key.removesuffix(".csv"),
        data=series.to_native(backend=backend),
        events=Events.from_bounds(bounds or np.empty((0, 2), dtype=np.int64)),
        description=(
            f"Numenta Anomaly Benchmark {key}, {series.n_rows} samples with "
            f"{len(bounds)} labelled window(s)"
        ),
    )


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------


def _windows(
    cache_dir: str | Path | None, *, download: bool
) -> dict[str, list[list[str]]]:
    """Read the benchmark's label file, fetching it if needed.

    Parameters
    ----------
    cache_dir
        Cache location, or None for the default.
    download
        Whether the network may be used.

    Returns
    -------
    dict
        Series filename mapped to its list of ``[start, end]`` window strings.
    """
    raw = _fetch(_LABELS, cache_dir, download=download)
    parsed: dict[str, list[list[str]]] = json.loads(raw.decode("utf-8"))
    return parsed


def _series(
    key: str, cache_dir: str | Path | None, *, download: bool
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Read one benchmark CSV as a time axis and a value column.

    Parameters
    ----------
    key
        Path within the benchmark's ``data/`` directory, ending in ``.csv``.
    cache_dir
        Cache location, or None for the default.
    download
        Whether the network may be used.

    Returns
    -------
    tuple
        UTC nanoseconds, and the values as float64.

    Raises
    ------
    ValueError
        The file does not have the ``timestamp,value`` columns every NAB file has.
    """
    raw = _fetch(f"data/{key}", cache_dir, download=download)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    # The header, not the rows: a file with the wrong columns and no rows at all
    # would otherwise load as an empty series rather than say what was wrong.
    columns = reader.fieldnames or []
    if "timestamp" not in columns or "value" not in columns:
        msg = (
            f"{key} does not look like a NAB series: expected columns "
            f"'timestamp' and 'value', found {sorted(columns)}."
        )
        raise ValueError(msg)

    stamps: list[int] = []
    numbers: list[float] = []
    for row in reader:
        stamps.append(_parse_stamp(row["timestamp"]))
        numbers.append(float(row["value"]))
    return (
        np.asarray(stamps, dtype=np.int64),
        np.asarray(numbers, dtype=np.float64),
    )


def _fetch(path: str, cache_dir: str | Path | None, *, download: bool) -> bytes:
    """Return one benchmark file's bytes, from the cache or from the network.

    Parameters
    ----------
    path
        Path within the benchmark repository.
    cache_dir
        Cache location, or None for the default.
    download
        Whether the network may be used when the cache misses.

    Returns
    -------
    bytes
        The file's contents.

    Raises
    ------
    OSError
        The file is not cached and either ``download`` is False or the fetch
        failed.
    """
    cached = _cache_root(cache_dir) / path
    if cached.exists():
        return cached.read_bytes()

    if not download:
        msg = (
            f"{path} is not cached at {cached} and download=False. Run once with "
            f"download=True, or point cache_dir at a directory that has it."
        )
        raise OSError(msg)

    url = _BASE + path
    try:
        # The URL is this module's own constant with a path appended; nothing a
        # caller passes reaches the scheme.
        with urlopen(url, timeout=_TIMEOUT) as response:
            payload: bytes = response.read()
    except (URLError, TimeoutError, OSError) as error:
        msg = (
            f"Could not fetch {url}: {error}. The benchmark is downloaded on "
            f"demand rather than bundled; if this machine has no network, fetch "
            f"it elsewhere and copy the file to {cached}."
        )
        raise OSError(msg) from error

    cached.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename, so an interrupted download cannot leave
    # a truncated file that every later run reads out of the cache.
    partial = cached.with_suffix(cached.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(cached)
    return payload


def _cache_root(cache_dir: str | Path | None) -> Path:
    """Decide where downloads live.

    Parameters
    ----------
    cache_dir
        An explicit location, or None to take it from the environment.

    Returns
    -------
    pathlib.Path
        The cache root. Not created here; :func:`_fetch` creates it when it has
        something to put in it.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    # The benchmark gets its own subdirectory either way, so pointing
    # HAZURE_DATA_HOME at the default root reproduces the default layout rather
    # than a second one beside it.
    root = os.environ.get("HAZURE_DATA_HOME")
    return (Path(root) if root else Path.home() / ".cache" / "hazure") / "nab"


def _parse_stamp(text: str) -> int:
    """Read one benchmark timestamp as UTC nanoseconds.

    Parameters
    ----------
    text
        A timestamp as the benchmark writes them, ``"2014-02-14 14:25:00"`` or
        ``"2014-02-14 14:25:00.000000"``.

    Returns
    -------
    int
        UTC nanoseconds since the epoch.
    """
    # numpy wants the ISO 8601 'T', and the benchmark writes a space.
    return int(np.datetime64(text.strip().replace(" ", "T"), "ns").astype(np.int64))
