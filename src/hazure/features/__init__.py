"""Feature engineering: series in, series out.

A transformer turns a series into the series an anomaly detector should actually
look at — a rolling median, a lag matrix, a seasonal residual, a principal
component. Because the output is a ``TimeSeries`` like the input, transformers
chain with each other and sit upstream of scorers in a pipeline.

Univariate transformers handle one series at a time and fan out automatically
over the columns of a frame. Multivariate ones need every column at once, which
is what lets them find a point that is unremarkable in each column on its own
yet impossible taken together.
"""

from __future__ import annotations

from hazure.features.customized_transformer import CustomizedTransformer
from hazure.features.double_rolling_aggregate import DoubleRollingAggregate
from hazure.features.ordinary_least_squares import OrdinaryLeastSquares
from hazure.features.pca_projection import PcaProjection
from hazure.features.pca_reconstruction import PcaReconstruction
from hazure.features.pca_reconstruction_error import PcaReconstructionError
from hazure.features.regression_residual import RegressionResidual
from hazure.features.regressor import Regressor
from hazure.features.retrospect import Retrospect
from hazure.features.rolling_aggregate import RollingAggregate
from hazure.features.seasonal_decomposition import SeasonalDecomposition
from hazure.features.standard_scale import StandardScale
from hazure.features.sum_all import SumAll

__all__ = [
    "CustomizedTransformer",
    "DoubleRollingAggregate",
    "OrdinaryLeastSquares",
    "PcaProjection",
    "PcaReconstruction",
    "PcaReconstructionError",
    "RegressionResidual",
    "Regressor",
    "Retrospect",
    "RollingAggregate",
    "SeasonalDecomposition",
    "StandardScale",
    "SumAll",
]
