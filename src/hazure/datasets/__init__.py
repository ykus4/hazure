"""Series to try things on, and a loop for comparing detectors over them.

Two sources, for two different questions.

:func:`make_series` generates one, with anomalies of a chosen shape planted at a
chosen strength. Nothing is downloaded, the answer is exact, and the strength is a
dial — which makes it the right tool for "at what point does this detector stop
seeing a level shift", a question no fixed benchmark can answer.

:func:`load_nab` fetches a series from the Numenta Anomaly Benchmark, labelled by
hand from real systems. Real data misbehaves in ways nobody thinks to generate:
gaps, drift, seasonality that is nearly but not quite daily, and anomalies that
are arguable. The labels are fixed and public, so a number quoted against them
means something to somebody else.

:func:`compare` runs several detectors over either kind and lays the metrics side
by side, alert counts included — because recall alone can be bought.
"""

from __future__ import annotations

from hazure.datasets.benchmark import compare
from hazure.datasets.dataset import Dataset
from hazure.datasets.nab import load_nab, nab_names
from hazure.datasets.synthetic import KINDS, make_series

__all__ = [
    "KINDS",
    "Dataset",
    "compare",
    "load_nab",
    "make_series",
    "nab_names",
]
