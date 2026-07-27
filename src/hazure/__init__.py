"""hazure — rule-based and unsupervised anomaly detection for time series.

Asking "how unusual is this point" and asking "is that unusual enough to report"
are different questions, so hazure keeps them apart. Everything is built from
five composable pieces::

    Scorer      TimeSeries -> continuous score
    Threshold   continuous score -> binary labels
    Detector    a Scorer and Threshold paired, ready to use
    Aggregator  several binary label series -> one
    Transformer TimeSeries -> TimeSeries (feature engineering)

One threshold policy is therefore reusable across every scorer, a scorer can be
swapped without revisiting the policy, and a score is useful on its own for
ranking.

Any pandas, polars or pyarrow object with a time axis is accepted, and results
come back in the flavour they went in as. The runtime dependencies are narwhals
and numpy; SciPy, scikit-learn, statsmodels, matplotlib and the rest are extras,
loaded only by the algorithms that need them.

Everything public is importable straight from ``hazure``:

    >>> import hazure
    >>> hazure.__version__  # doctest: +ELLIPSIS
    '0.1...'

Algorithms are grouped by the phenomenon they look for, under
``hazure.methods``: value ranges in ``level``, abrupt changes in ``change``,
broken periodicity in ``seasonal``, and so on. Each of those modules holds the
scorer, the ready-made detector, and any related transformer together, so one
technique is one file.
"""

from __future__ import annotations

from hazure._core import (
    AGGREGATIONS,
    BaseAggregator,
    BaseDetector,
    BaseScorer,
    BaseThreshold,
    BaseTransformer,
    Component,
    TimeSeries,
    double_rolling,
    parse_duration,
    rolling,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "AGGREGATIONS",
    "BaseAggregator",
    "BaseDetector",
    "BaseScorer",
    "BaseThreshold",
    "BaseTransformer",
    "Component",
    "TimeSeries",
    "__version__",
    "double_rolling",
    "parse_duration",
    "rolling",
]
