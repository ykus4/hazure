"""hazure — finding anomalies in time series without labelled examples.

The case this is built for: you have a metric, you suspect it occasionally
misbehaves, and you have no record of when it did. There is nothing to train a
classifier on, so the model has to describe what normal looks like and report
departures from it.

Everything is built from five composable pieces::

    Scorer      series -> continuous score     .score()      .fit_score()
    Threshold   score  -> binary labels        .apply()      .fit_apply()
    Detector    a Scorer and Threshold paired  .detect()     .fit_detect()
    Aggregator  several label series -> one    .aggregate()
    Transformer series -> series               .transform()  .fit_transform()

Asking "how unusual is this point" and asking "is that unusual enough to report"
are different questions, so they are separate types. One threshold policy is then
reusable across every scorer, a scorer can be swapped without revisiting the
policy, and a score is useful on its own for ranking rather than flagging.

Any pandas, polars or pyarrow object with a time axis is accepted, and results
come back in the flavour they went in as::

    >>> import numpy as np, pandas as pd
    >>> from hazure import SpikeDetector
    >>> index = pd.date_range("2024-01-01", periods=200, freq="h")
    >>> values = np.zeros(200)
    >>> values[120] = 9.0
    >>> flags = SpikeDetector(window=24).fit_detect(pd.Series(values, index=index))
    >>> bool(flags.idxmax() == index[120])
    True

Labels are ``1.0`` anomalous, ``0.0`` normal and ``NaN`` unknown — a point whose
score could not be computed is not quietly called normal.

Runtime dependencies are ``narwhals`` and ``numpy``. SciPy, scikit-learn,
statsmodels, matplotlib, stumpy and ruptures are extras, imported only by the
components that need them.

Every public name is importable straight from this package. They are also grouped
by subject, if you prefer to import from there:

``hazure.detection``
    Ready-made detectors, each a scorer paired with a threshold.
``hazure.scoring``
    Continuous scores, for ranking or for pairing with your own threshold.
``hazure.thresholds``
    Turning a score into labels.
``hazure.features``
    Feature engineering: rolling aggregates, lags, decomposition, projections.
``hazure.ensemble``
    Combining several verdicts into one.
``hazure.compose``
    :class:`Pipeline` for a chain, :class:`Graph` for anything branching.
``hazure.events``
    Moving between per-sample labels and anomalous intervals.
``hazure.evaluation``
    Metrics, and time-ordered folds to compute them over.
``hazure.methods``
    Further method families: spectral residual, Hampel filtering, change-point
    segmentation, matrix profile discords, STL residuals.
``hazure.plotting``
    One :func:`~hazure.plotting.plot` function, for looking at the result.
"""

from __future__ import annotations

from hazure._core import (
    AGGREGATIONS,
    BaseAggregator,
    BaseDetector,
    BaseScorer,
    BaseThreshold,
    BaseTransformer,
    Component,
    TimeSeries,
    double_rolling,
    parse_duration,
    rolling,
)
from hazure.compose import Graph, Node, Pipeline
from hazure.detection import (
    AutoregressionDetector,
    EsdDetector,
    IqrDetector,
    LevelShiftDetector,
    MinClusterDetector,
    MultivariateScoreDetector,
    MultivariateSignedScoreDetector,
    OutlierDetector,
    PcaDetector,
    QuantileDetector,
    RegressionDetector,
    ScoreDetector,
    SeasonalDetector,
    Side,
    SignedScoreDetector,
    SpikeDetector,
    ThresholdDetector,
    VolatilityShiftDetector,
)
from hazure.ensemble import (
    AndAggregator,
    CustomizedAggregator,
    OrAggregator,
    ScoreAggregator,
    VoteAggregator,
)
from hazure.evaluation import f1_score, iou, precision, recall, split_train_test
from hazure.events import Events, expand_events, to_events, to_labels, validate_series
from hazure.features import (
    CustomizedTransformer,
    DoubleRollingAggregate,
    OrdinaryLeastSquares,
    PcaProjection,
    PcaReconstruction,
    PcaReconstructionError,
    RegressionResidual,
    Regressor,
    Retrospect,
    RollingAggregate,
    SeasonalDecomposition,
    StandardScale,
    SumAll,
)
from hazure.methods import (
    DampScorer,
    HampelDetector,
    HampelScorer,
    MatrixProfileDetector,
    MatrixProfileScorer,
    MstlDetector,
    MstlResidualScorer,
    PeltDetector,
    PeltScorer,
    RollingQuantileScorer,
    RupturesScorer,
    SpectralResidualDetector,
    SpectralResidualScorer,
    StlDetector,
    StlResidualScorer,
)
from hazure.scoring import (
    AutoregressionResidualScorer,
    DeviationScorer,
    DoubleRollingScorer,
    MinClusterScorer,
    OutlierScorer,
    PcaReconstructionErrorScorer,
    RegressionResidualScorer,
    RollingAggregateScorer,
    SeasonalResidualScorer,
)
from hazure.thresholds import (
    MAD_SCALE,
    EsdThreshold,
    Factor,
    FactorSpec,
    FixedThreshold,
    IqrThreshold,
    MadThreshold,
    QuantileThreshold,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "AGGREGATIONS",
    "MAD_SCALE",
    "AndAggregator",
    "AutoregressionDetector",
    "AutoregressionResidualScorer",
    "BaseAggregator",
    "BaseDetector",
    "BaseScorer",
    "BaseThreshold",
    "BaseTransformer",
    "Component",
    "CustomizedAggregator",
    "CustomizedTransformer",
    "DampScorer",
    "DeviationScorer",
    "DoubleRollingAggregate",
    "DoubleRollingScorer",
    "EsdDetector",
    "EsdThreshold",
    "Events",
    "Factor",
    "FactorSpec",
    "FixedThreshold",
    "Graph",
    "HampelDetector",
    "HampelScorer",
    "IqrDetector",
    "IqrThreshold",
    "LevelShiftDetector",
    "MadThreshold",
    "MatrixProfileDetector",
    "MatrixProfileScorer",
    "MinClusterDetector",
    "MinClusterScorer",
    "MstlDetector",
    "MstlResidualScorer",
    "MultivariateScoreDetector",
    "MultivariateSignedScoreDetector",
    "Node",
    "OrAggregator",
    "OrdinaryLeastSquares",
    "OutlierDetector",
    "OutlierScorer",
    "PcaDetector",
    "PcaProjection",
    "PcaReconstruction",
    "PcaReconstructionError",
    "PcaReconstructionErrorScorer",
    "PeltDetector",
    "PeltScorer",
    "Pipeline",
    "QuantileDetector",
    "QuantileThreshold",
    "RegressionDetector",
    "RegressionResidual",
    "RegressionResidualScorer",
    "Regressor",
    "Retrospect",
    "RollingAggregate",
    "RollingAggregateScorer",
    "RollingQuantileScorer",
    "RupturesScorer",
    "ScoreAggregator",
    "ScoreDetector",
    "SeasonalDecomposition",
    "SeasonalDetector",
    "SeasonalResidualScorer",
    "Side",
    "SignedScoreDetector",
    "SpectralResidualDetector",
    "SpectralResidualScorer",
    "SpikeDetector",
    "StandardScale",
    "StlDetector",
    "StlResidualScorer",
    "SumAll",
    "ThresholdDetector",
    "TimeSeries",
    "VolatilityShiftDetector",
    "VoteAggregator",
    "__version__",
    "double_rolling",
    "expand_events",
    "f1_score",
    "iou",
    "parse_duration",
    "precision",
    "recall",
    "rolling",
    "split_train_test",
    "to_events",
    "to_labels",
    "validate_series",
]
