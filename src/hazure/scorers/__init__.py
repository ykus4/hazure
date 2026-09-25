"""Scorers: series in, "how unusual is each point" out.

A score is a continuous number per observation, on whatever scale the algorithm
naturally works in, where a larger magnitude means more unusual. Scores are
useful on their own — for ranking, for plotting, for feeding a model — and become
labels when paired with a :mod:`threshold <hazure.thresholds>` in a
:class:`~hazure.Detector`.

Several scorers are *signed*, and the sign carries information: which way the
series moved. :class:`~hazure.thresholds.SignedThreshold` can therefore act on
increases only, or decreases only, without a second pass over the data.

Univariate scorers handle one series at a time and fan out over the columns of a
frame, each column learning its own normal. Multivariate ones need every column
at once, which is what lets them see an anomaly that lives in the relationship
between columns rather than in any one of them.

**Distance from normal**
    :class:`DeviationScorer` against a learned centre and spread;
    :class:`HampelScorer` and :class:`RollingQuantileScorer` against a local one
    that follows the series.
**What a model fails to explain**
    :class:`AutoregressionResidualScorer` for the series' own recent past,
    :class:`StlResidualScorer` and :class:`MstlResidualScorer` for one or several
    seasonal rhythms, :class:`SpectralResidualScorer` for a saliency map that
    assumes no period at all.
**Change points**
    :class:`PeltScorer`, implemented here, and :class:`RupturesScorer` for the
    other search strategies of ``ruptures``. They ask when the series *became* a
    different series rather than which points do not belong.
**Unusual shapes**
    :class:`MatrixProfileScorer` and :class:`DampScorer` score a subsequence
    rather than a point, so they find anomalies made entirely of ordinary values.
**Models of yours**
    :class:`OutlierScorer` and :class:`MinClusterScorer` adapt an outlier or a
    clustering model with a scikit-learn interface.

Any :mod:`transformer <hazure.transformers>` whose output is itself a score — a
rolling statistic, the difference between two adjacent windows, a seasonal
residual, a PCA reconstruction error, a regression residual — is used as one
through :class:`AsScorer`.

Everything here needs only numpy, except :class:`RupturesScorer`, the
matrix-profile scorers and the STL scorers, which import their backend lazily and
say how to install it if it is missing.
"""

from __future__ import annotations

from hazure.scorers.adapter import AsScorer
from hazure.scorers.autoregression import AutoregressionResidualScorer
from hazure.scorers.breakpoint import Cost
from hazure.scorers.damp import DampScorer
from hazure.scorers.deviation import DeviationScorer
from hazure.scorers.hampel import HampelScorer
from hazure.scorers.matrix_profile import MatrixProfileScorer
from hazure.scorers.min_cluster import MinClusterScorer
from hazure.scorers.mstl import MstlResidualScorer
from hazure.scorers.outlier import OutlierScorer
from hazure.scorers.pelt import PeltScorer
from hazure.scorers.rolling_quantile import RollingQuantileScorer
from hazure.scorers.ruptures import RupturesScorer
from hazure.scorers.spectral_residual import SpectralResidualScorer
from hazure.scorers.stl import StlResidualScorer

__all__ = [
    "AsScorer",
    "AutoregressionResidualScorer",
    "Cost",
    "DampScorer",
    "DeviationScorer",
    "HampelScorer",
    "MatrixProfileScorer",
    "MinClusterScorer",
    "MstlResidualScorer",
    "OutlierScorer",
    "PeltScorer",
    "RollingQuantileScorer",
    "RupturesScorer",
    "SpectralResidualScorer",
    "StlResidualScorer",
]
