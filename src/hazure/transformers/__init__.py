"""Transformers: series in, series out.

A transformer turns a series into the series an anomaly detector should actually
look at — a rolling median, a lag matrix, a seasonal residual, a principal
component. Because the output is a ``TimeSeries`` like the input, transformers
chain with each other and sit upstream of scorers in a pipeline. One whose output
already measures how unusual each point is becomes a scorer through
:class:`~hazure.scorers.AsScorer`, so the arithmetic lives here once.

Univariate transformers handle one series at a time and fan out automatically
over the columns of a frame. Multivariate ones need every column at once, which
is what lets them find a point that is unremarkable in each column on its own
yet impossible taken together.

The window engine underneath — :func:`rolling`, :func:`double_rolling` and
:func:`parse_duration`, over the statistics in :data:`AGGREGATIONS` — works on
plain numpy arrays and is exported here for use on its own.
"""

from __future__ import annotations

from hazure._core.window import AGGREGATIONS, double_rolling, parse_duration, rolling
from hazure.transformers.customized_transformer import CustomizedTransformer
from hazure.transformers.double_rolling_aggregate import DoubleRollingAggregate
from hazure.transformers.ordinary_least_squares import OrdinaryLeastSquares
from hazure.transformers.pca_column_error import PcaColumnError
from hazure.transformers.pca_projection import PcaProjection
from hazure.transformers.pca_reconstruction import PcaReconstruction
from hazure.transformers.pca_reconstruction_error import PcaReconstructionError
from hazure.transformers.regression_residual import RegressionResidual
from hazure.transformers.regressor import Regressor
from hazure.transformers.retrospect import Retrospect
from hazure.transformers.rolling_aggregate import RollingAggregate
from hazure.transformers.seasonal_decomposition import SeasonalDecomposition
from hazure.transformers.standard_scale import StandardScale
from hazure.transformers.sum_all import SumAll

__all__ = [
    "AGGREGATIONS",
    "CustomizedTransformer",
    "DoubleRollingAggregate",
    "OrdinaryLeastSquares",
    "PcaColumnError",
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
    "double_rolling",
    "parse_duration",
    "rolling",
]
