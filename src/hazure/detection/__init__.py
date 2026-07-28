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

from hazure.detection.autoregression import AutoregressionDetector
from hazure.detection.esd import EsdDetector
from hazure.detection.iqr import IqrDetector
from hazure.detection.level_shift import LevelShiftDetector
from hazure.detection.min_cluster import MinClusterDetector
from hazure.detection.multivariate_score import MultivariateScoreDetector
from hazure.detection.multivariate_signed_score import MultivariateSignedScoreDetector
from hazure.detection.outlier import OutlierDetector
from hazure.detection.pca import PcaDetector
from hazure.detection.quantile import QuantileDetector
from hazure.detection.regression import RegressionDetector
from hazure.detection.score import ScoreDetector
from hazure.detection.seasonal import SeasonalDetector
from hazure.detection.side import Side
from hazure.detection.signed_score import SignedScoreDetector
from hazure.detection.spike import SpikeDetector
from hazure.detection.threshold import ThresholdDetector
from hazure.detection.volatility_shift import VolatilityShiftDetector

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
