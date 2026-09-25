"""Backend-independent primitives every other module is built on.

Modules inside hazure import from here, never from the top-level package, so
that no module's import depends on the order ``hazure/__init__.py`` happens to
list things in.
"""

from __future__ import annotations

from hazure._core.component import (
    Aggregator,
    Component,
    OutputKind,
    Scorer,
    Threshold,
    Transformer,
)
from hazure._core.detector import Detector
from hazure._core.params import Configurable
from hazure._core.persist import Persistent
from hazure._core.series import Origin, TimeSeries
from hazure._core.window import (
    AGGREGATIONS,
    Closed,
    Window,
    aggregate_windows,
    double_rolling,
    parse_duration,
    rolling,
    window_bounds,
)

__all__ = [
    "AGGREGATIONS",
    "Aggregator",
    "Closed",
    "Component",
    "Configurable",
    "Detector",
    "Origin",
    "OutputKind",
    "Persistent",
    "Scorer",
    "Threshold",
    "TimeSeries",
    "Transformer",
    "Window",
    "aggregate_windows",
    "double_rolling",
    "parse_duration",
    "rolling",
    "window_bounds",
]
