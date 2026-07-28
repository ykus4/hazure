"""Working with labels and anomalous intervals.

Two jobs live here, both about the shape of the data rather than about
detection:

* :class:`Events` and the converters :func:`to_events` / :func:`to_labels`, which
  move between the one-label-per-sample and the anomalous-interval view of the
  same thing, plus :func:`expand_events`, which widens intervals by a margin.
* :func:`validate_series`, which applies the normalisation every detector does
  internally and hands the result back, so it can be inspected rather than
  guessed at.
"""

from __future__ import annotations

from hazure.events.convert import expand_events, to_events, to_labels, validate_series
from hazure.events.interval import Events

__all__ = [
    "Events",
    "expand_events",
    "to_events",
    "to_labels",
    "validate_series",
]
