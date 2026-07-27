"""Detection: series in, binary labels out.

A detector is a scorer and a threshold in one object, which is how anomaly
detection is usually wanted: ``fit_detect`` on a series, labels back. The parts
remain visible as ``.scorer`` and ``.threshold``, so a detector can be pulled
apart to inspect the raw score, to reuse the scorer under a different rule, or
simply to read how it was assembled.

:class:`ScoreDetector` pairs any scorer with any threshold. The named detectors
are the pairings worth having ready, one per phenomenon: a value outside its
usual range, a spike, a level shift, a change in volatility, a break in a
seasonal pattern, a break in the series' own dynamics, and — for several columns
at once — a broken relationship between them.

Labels are ``1.0`` anomalous, ``0.0`` normal and ``NaN`` unknown. NaN is common
and meaningful: a window-based detector cannot judge the first few observations,
and saying so is more useful than calling them normal.
"""

from __future__ import annotations

from hazure.detection._composition import (
    MultivariateScoreDetector,
    MultivariateSignedScoreDetector,
    ScoreDetector,
    Side,
    SignedScoreDetector,
)
from hazure.detection._multivariate import (
    MinClusterDetector,
    OutlierDetector,
    PcaDetector,
    RegressionDetector,
)
from hazure.detection._univariate import (
    AutoregressionDetector,
    EsdDetector,
    IqrDetector,
    LevelShiftDetector,
    QuantileDetector,
    SeasonalDetector,
    SpikeDetector,
    ThresholdDetector,
    VolatilityShiftDetector,
)

__all__ = [
    "AutoregressionDetector",
    "EsdDetector",
    "IqrDetector",
    "LevelShiftDetector",
    "MinClusterDetector",
    "MultivariateScoreDetector",
    "MultivariateSignedScoreDetector",
    "OutlierDetector",
    "PcaDetector",
    "QuantileDetector",
    "RegressionDetector",
    "ScoreDetector",
    "SeasonalDetector",
    "Side",
    "SignedScoreDetector",
    "SpikeDetector",
    "ThresholdDetector",
    "VolatilityShiftDetector",
]
