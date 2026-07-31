# Changelog

Notable changes to `hazure`, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html) — while the major
version is 0, a minor bump may still break an interface, and the changelog will
say so.

## 0.1.0 — 2026-07-31

The first release. Everything below is the initial development work, so there is
nothing to compare it against and no upgrade to plan; the list is what the library
contains.

### Added

- **The core.** `TimeSeries`, the dataframe boundary that reads any pandas,
  polars or pyarrow object with a time axis into a sorted `int64` array of UTC
  nanoseconds and returns results in the flavour they arrived in. `Component` and
  the five base classes, with automatic per-column fan-out for anything
  univariate. `rolling` and `double_rolling`, the window engine, in samples or in
  durations.
- **Detectors.** Distribution rules (`ThresholdDetector`, `QuantileDetector`,
  `IqrDetector`, `EsdDetector`), rolling comparisons (`SpikeDetector`,
  `LevelShiftDetector`, `VolatilityShiftDetector`), model residuals
  (`SeasonalDetector`, `AutoregressionDetector`, `RegressionDetector`,
  `PcaDetector`) and model adapters (`MinClusterDetector`, `OutlierDetector`).
- **Scorers and thresholds** for each of the above, usable separately:
  `ScoreDetector` pairs any scorer with any threshold.
- **Further method families** in `hazure.methods`: spectral residual, the Hampel
  filter, a rolling quantile band, PELT segmentation with a `ruptures` adapter,
  matrix-profile discords, and STL / MSTL residuals.
- **A detector for every method scorer**: `DampDetector`,
  `RollingQuantileDetector` and `RupturesDetector`, plus the `normalize` its
  scorer always took on `MatrixProfileDetector`.
- **Storing a fitted model.** `Component.to_dict` and
  `Configurable.from_dict` carry a fitted component — nested components, private
  learned state and all — through JSON, so a model fitted on a period you trust
  outlives the process that fitted it.
- **Per-column attribution for PCA anomalies.** `PcaColumnError` writes out the
  terms of the reconstruction error, one per column, so a flag can be taken apart
  without a second model that might disagree with the first.
- **How late, and how good the ranking is.** `detection_delay` and
  `detection_delays` say when an alert arrived rather than only whether it did,
  and `average_precision` and `roc_auc` score a scorer without a fence being
  chosen first.
- **Composition.** `Pipeline` for a chain, `Graph` for a directed acyclic graph
  of components, and the `And` / `Or` / `Vote` / customised aggregators.
  `ScoreAggregator` combines the scores themselves rather than the verdicts,
  normalising each input by rank or by its MAD first so that a scorer cannot
  dominate the ensemble by emitting larger numbers.
- **Events and evaluation.** `Events` as closed intervals in UTC nanoseconds,
  conversion to and from labels, point- and event-based precision, recall, F1 and
  IoU, and time-ordered `split_train_test` folds.
- **A fence at a false-alarm rate you choose.** `PotThreshold` fits a generalised
  Pareto distribution to the tail of the training scores by maximum likelihood, so
  the cut-off can be placed where exceedance has probability `1e-4` — beyond the
  largest score ever observed, which no quantile of a sample can reach. Its
  `update` drives the same fence online, absorbing each score into the tail and
  discarding the ones it flags, since an anomaly is not evidence about how normal
  behaves.
- **Detection on a series that is still arriving.** `Stream` keeps a buffer of the
  recent past and runs a fitted component over it once per observation, so the
  online answer is the batch answer rather than a second implementation that
  agrees with it on the tested cases. `Stream.prime` fills the buffer from history
  and *checks* it: too short for what the component looks back over, and it
  refuses to start rather than quietly computing from a window that was never
  full. A stream carries its buffer through `to_dict`, so a monitor can be stored
  and resumed.
- **Choosing the cut-off.** `tune_threshold` searches for the cut-off that scores
  best against labelled incidents, event-based by default.
  `budget_threshold` needs no labels at all: it lowers the fence as far as an
  alert budget — one page a week — allows, descending until the budget breaks so
  that the non-monotonicity of alert counts cannot flatter the result. Both return
  a `Calibration` carrying the whole curve, not only the winner.
- **Data to try things on**, in `hazure.datasets`. `make_series` plants anomalies
  of five shapes at a strength you set and hands back the ground truth with the
  series. `load_nab` fetches a labelled series from the Numenta Anomaly Benchmark
  on demand and caches it, needing nothing beyond the standard library.
  `compare` runs several detectors over either and lays the metrics out side by
  side, alert counts included.
- **Documentation** at <https://ykus4.github.io/hazure/>: quickstart, guide, six
  pages on the mathematics of every scorer, threshold and metric, and a full API
  reference. Every example in every docstring is executed by the test suite, so a
  documented output that has drifted from the code is a test failure.
