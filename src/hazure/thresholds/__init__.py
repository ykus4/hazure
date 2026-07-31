"""Thresholding: continuous scores in, binary labels out.

A threshold is the decision "is that unusual enough to report", kept as its own
object so that one policy can be reused across every scorer and swapped without
touching the scorer. Labels are ``1.0`` anomalous, ``0.0`` normal and ``NaN``
unknown; a score of NaN yields a label of NaN, because a point nobody measured
cannot be declared normal.

:class:`FixedThreshold` draws the line where the caller says. The rest learn it
from history: a quantile of the training scores, a multiple of their
inter-quartile range or median absolute deviation, or a formal test of how
extreme a value can be before it stops looking like a sample from the same
distribution. :class:`PotThreshold` is parameterised the other way round: you
give it the false-alarm probability you are willing to accept, and it fits the
tail of the training scores well enough to place a fence there — including
beyond the largest score ever seen, which no quantile of a sample can reach.
"""

from __future__ import annotations

from hazure.thresholds.esd import EsdThreshold
from hazure.thresholds.fence import MAD_SCALE, Factor, FactorSpec
from hazure.thresholds.fixed import FixedThreshold
from hazure.thresholds.iqr import IqrThreshold
from hazure.thresholds.mad import MadThreshold
from hazure.thresholds.pot import PotThreshold
from hazure.thresholds.quantile import QuantileThreshold

__all__ = [
    "MAD_SCALE",
    "EsdThreshold",
    "Factor",
    "FactorSpec",
    "FixedThreshold",
    "IqrThreshold",
    "MadThreshold",
    "PotThreshold",
    "QuantileThreshold",
]
