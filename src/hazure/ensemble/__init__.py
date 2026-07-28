"""Ensembling: several label series in, one out.

An aggregator reduces the outputs of several detectors to a single label series,
which is how multi-condition rules and detector ensembles are expressed. Every
aggregator emits one column named ``anomaly``.

Three-valued logic
------------------

A label is ``1.0`` anomalous, ``0.0`` normal or ``NaN`` unknown, all in a float
column. ``NaN`` is a real third state, not a missing 0: a rolling detector cannot
label the first few points of a series, and saying "unknown" there is different
from saying "normal". Combining labels is therefore three-valued logic, and a
result is only known once the inputs settle it:

===== ===== ===== =====
``a`` ``b`` OR    AND
===== ===== ===== =====
1     1     1     1
1     0     1     0
1     NaN   1     NaN
0     1     1     0
0     0     0     0
0     NaN   NaN   0
NaN   1     1     NaN
NaN   0     NaN   0
NaN   NaN   NaN   NaN
===== ===== ===== =====

Read the columns as: OR is 1 as soon as one input says 1, whatever the others
say; it is 0 only when every input is a definite 0; otherwise the unknown input
could have gone either way, so the answer is unknown. AND is the mirror image —
0 as soon as one input says 0, 1 only when every input is a definite 1.

:class:`VoteAggregator` generalises both by counting rather than quantifying:
unknown inputs leave the vote entirely, contributing to neither the numerator nor
the denominator, and a row where nothing is known is unknown.

Any non-zero label counts as anomalous, so a detector that emits counts or
confidences rather than a strict 0/1 still aggregates sensibly.
"""

from __future__ import annotations

from hazure.ensemble.and_aggregator import AndAggregator
from hazure.ensemble.customized_aggregator import CustomizedAggregator
from hazure.ensemble.or_aggregator import OrAggregator
from hazure.ensemble.vote_aggregator import VoteAggregator

__all__ = [
    "AndAggregator",
    "CustomizedAggregator",
    "OrAggregator",
    "VoteAggregator",
]
