"""What a loader hands back: a series, its ground truth, and where both came from."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hazure.events import to_labels

if TYPE_CHECKING:
    from hazure.events import Events


__all__ = [
    "Dataset",
]


@dataclass(frozen=True, slots=True)
class Dataset:
    """A series and the intervals somebody says are anomalous in it.

    Ground truth is carried as :class:`~hazure.Events` rather than as a label
    column, because that is the form it arrives in and the form it means. An
    incident is a stretch of time; which samples fall inside it depends on the
    sampling interval, and a benchmark whose labels were snapped onto one axis
    cannot be resampled onto another without quietly changing what it claims.
    :attr:`labels` does the conversion when a metric wants it.

    Attributes
    ----------
    name : str
        What this series is, short enough to use as a row label in a table.
    data : Any
        The series itself, in whatever backend the loader was asked for.
    events : Events
        The anomalous intervals.
    description : str
        Where it came from, or what was planted in it and how strongly.

    See Also
    --------
    hazure.datasets.make_series : Generate one of these with a known answer.
    hazure.datasets.load_nab : Load one from the Numenta Anomaly Benchmark.

    Examples
    --------
    >>> from hazure.datasets import make_series
    >>> dataset = make_series("spike", n=1000, n_anomalies=2)
    >>> dataset.name, dataset.events.n_events
    ('spike', 2)
    >>> print(dataset.description)  # doctest: +ELLIPSIS
    2 x spike (the level jumps up for a moment...), 1 sample(s) each at strength 8...
    """

    name: str
    data: Any
    events: Events
    description: str

    @property
    def labels(self) -> Any:
        """The ground truth as a label series on :attr:`data`'s own time axis.

        Returns
        -------
        Any
            ``1.0`` inside an event and ``0.0`` outside, in the same flavour as
            :attr:`data`. Use it with the sample-based metrics; the event-based
            ones take :attr:`events` directly and lose nothing on the way.
        """
        return to_labels(self.events, self.data)

    def __repr__(self) -> str:
        rows = getattr(self.data, "__len__", None)
        size = f"{len(self.data)} rows, " if callable(rows) else ""
        return f"Dataset({self.name!r}, {size}{self.events.n_events} events)"
