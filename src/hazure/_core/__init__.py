"""Backend-independent primitives every other module is built on."""

from __future__ import annotations

from hazure._core.component import (
    BaseAggregator,
    BaseDetector,
    BaseScorer,
    BaseThreshold,
    BaseTransformer,
    Component,
)
from hazure._core.config import Configurable
from hazure._core.series import Origin, TimeSeries
from hazure._core.window import (
    AGGREGATIONS,
    aggregate_windows,
    double_rolling,
    parse_duration,
    rolling,
    window_bounds,
)

__all__ = [
    "AGGREGATIONS",
    "BaseAggregator",
    "BaseDetector",
    "BaseScorer",
    "BaseThreshold",
    "BaseTransformer",
    "Component",
    "Configurable",
    "Origin",
    "TimeSeries",
    "aggregate_windows",
    "double_rolling",
    "parse_duration",
    "rolling",
    "window_bounds",
]
