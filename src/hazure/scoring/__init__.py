"""Scoring: series in, "how unusual is each point" out.

A score is a continuous number per observation, on whatever scale the algorithm
naturally works in, where a larger magnitude means more unusual. Scores are
useful on their own — for ranking, for plotting, for feeding a model — and become
labels when passed through a :class:`hazure.BaseThreshold`.

Several scorers are *signed*, and the sign carries information: which way the
series moved. A detector can therefore use one scorer and still act on increases
only, or decreases only, without a second pass over the data.

Univariate scorers handle one series at a time and fan out over the columns of a
frame, each column learning its own normal. Multivariate ones need every column
at once, which is what lets them see an anomaly that lives in the relationship
between columns rather than in any one of them.
"""

from __future__ import annotations

from hazure.scoring.autoregression_residual import AutoregressionResidualScorer
from hazure.scoring.deviation import DeviationScorer
from hazure.scoring.double_rolling import DoubleRollingScorer
from hazure.scoring.min_cluster import MinClusterScorer
from hazure.scoring.outlier import OutlierScorer
from hazure.scoring.pca_reconstruction_error import PcaReconstructionErrorScorer
from hazure.scoring.regression_residual import RegressionResidualScorer
from hazure.scoring.rolling_aggregate import RollingAggregateScorer
from hazure.scoring.seasonal_residual import SeasonalResidualScorer

__all__ = [
    "AutoregressionResidualScorer",
    "DeviationScorer",
    "DoubleRollingScorer",
    "MinClusterScorer",
    "OutlierScorer",
    "PcaReconstructionErrorScorer",
    "RegressionResidualScorer",
    "RollingAggregateScorer",
    "SeasonalResidualScorer",
]
