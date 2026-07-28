"""Detection methods beyond the rolling-window rules.

Five families, each a scorer and — where the pairing is obvious — a ready-made
detector:

**Spectral residual**
    Saliency from a Fourier transform with its smooth amplitude envelope removed.
    Assumes no period, no trend and no distribution.
**Local order statistics**
    The Hampel filter and a rolling quantile band, for a normal range that drifts
    and a scale the outliers cannot inflate.
**Segmentation**
    Asks when the series *became* a different series rather than which points do
    not belong. PELT in numpy, plus an adapter for other search strategies.
**Matrix profile**
    The subsequences least like anything else in the series. Scores a shape
    rather than a point, so it finds anomalies made entirely of ordinary values.
**Seasonal-trend decomposition**
    Single- or multi-period, scoring what the decomposition cannot explain.

The spectral residual, the Hampel filter and PELT are implemented here and need
nothing but numpy. :class:`RupturesScorer`, the matrix-profile scorers and the
STL scorers import their backend lazily and say how to install it if it is
missing.
"""

from __future__ import annotations

from hazure.methods.breakpoint_scorer import Cost
from hazure.methods.damp_scorer import DampScorer
from hazure.methods.hampel_detector import HampelDetector
from hazure.methods.hampel_scorer import HampelScorer
from hazure.methods.matrix_profile_detector import MatrixProfileDetector
from hazure.methods.matrix_profile_scorer import MatrixProfileScorer
from hazure.methods.mstl_detector import MstlDetector
from hazure.methods.mstl_residual_scorer import MstlResidualScorer
from hazure.methods.pelt_detector import PeltDetector
from hazure.methods.pelt_scorer import PeltScorer
from hazure.methods.rolling_quantile_scorer import RollingQuantileScorer
from hazure.methods.ruptures_scorer import RupturesScorer
from hazure.methods.spectral_residual_detector import SpectralResidualDetector
from hazure.methods.spectral_residual_scorer import SpectralResidualScorer
from hazure.methods.stl_detector import StlDetector
from hazure.methods.stl_residual_scorer import StlResidualScorer

__all__ = [
    "Cost",
    "DampScorer",
    "HampelDetector",
    "HampelScorer",
    "MatrixProfileDetector",
    "MatrixProfileScorer",
    "MstlDetector",
    "MstlResidualScorer",
    "PeltDetector",
    "PeltScorer",
    "RollingQuantileScorer",
    "RupturesScorer",
    "SpectralResidualDetector",
    "SpectralResidualScorer",
    "StlDetector",
    "StlResidualScorer",
]
