# Changelog

Notable changes to `hazure`, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[semantic versioning](https://semver.org/spec/v2.0.0.html) — while the major
version is 0, a minor bump may still break an interface, and the changelog will
say so.

## Unreleased

Everything below is the initial development work; nothing has been released yet.

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
- **Per-column attribution for PCA anomalies.** `PcaColumnError` writes out the
  terms of the reconstruction error, one per column, so a flag can be taken apart
  without a second model that might disagree with the first.
- **Composition.** `Pipeline` for a chain, `Graph` for a directed acyclic graph
  of components, and the `And` / `Or` / `Vote` / customised aggregators.
- **Events and evaluation.** `Events` as closed intervals in UTC nanoseconds,
  conversion to and from labels, point- and event-based precision, recall, F1 and
  IoU, and time-ordered `split_train_test` folds.
- **Documentation** at <https://ykus4.github.io/hazure/>: quickstart, guide, six
  pages on the mathematics of every scorer, threshold and metric, and a full API
  reference.
