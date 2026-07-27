"""Detection methods beyond the rolling-window rules.

Each module here is one technique, with its scorer, its ready-made detector, and
the reasoning behind both kept together:

``spectral``
    Saliency from the spectral residual: a Fourier transform, its smooth
    amplitude envelope removed, and back again. Assumes no period, no trend and
    no distribution.
``robust``
    The Hampel filter and a rolling quantile band — local order statistics, for a
    normal range that drifts and a scale the outliers cannot inflate.
``changepoint``
    Segmentation, which asks when the series *became* a different series rather
    than which points do not belong. PELT in numpy, plus an adapter for other
    search strategies.
``motif``
    Matrix profile discords: the subsequences least like anything else in the
    series. Scores a shape rather than a point, so it finds anomalies made
    entirely of ordinary values.
``stl``
    Seasonal-trend decomposition, single- or multi-period, scoring what the
    decomposition cannot explain.

The two implemented from scratch — ``spectral`` and ``robust``, along with PELT —
need nothing but numpy. The adapters (``changepoint.RupturesScorer``, ``motif``,
``stl``) import their backend lazily and say how to install it if it is missing.
"""

from __future__ import annotations

from hazure.methods.changepoint import (
    Cost,
    PeltDetector,
    PeltScorer,
    RupturesScorer,
)
from hazure.methods.motif import (
    DampScorer,
    MatrixProfileDetector,
    MatrixProfileScorer,
)
from hazure.methods.robust import (
    HampelDetector,
    HampelScorer,
    RollingQuantileScorer,
)
from hazure.methods.spectral import (
    SpectralResidualDetector,
    SpectralResidualScorer,
)
from hazure.methods.stl import (
    MstlDetector,
    MstlResidualScorer,
    StlDetector,
    StlResidualScorer,
)

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
