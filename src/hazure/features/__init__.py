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

from hazure.features._multivariate import (
    CustomizedTransformer,
    OrdinaryLeastSquares,
    PcaProjection,
    PcaReconstruction,
    PcaReconstructionError,
    RegressionResidual,
    Regressor,
    SumAll,
)
from hazure.features._univariate import (
    DoubleRollingAggregate,
    Retrospect,
    RollingAggregate,
    SeasonalDecomposition,
    StandardScale,
)

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
