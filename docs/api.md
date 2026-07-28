# API reference

Every public name, grouped by the module it lives in. The five base classes, the
`TimeSeries` boundary and the window engine are importable straight from
`hazure`; everything else from the module listed here.

## `hazure`

::: hazure
    options:
      members:
        - TimeSeries
        - Component
        - BaseScorer
        - BaseThreshold
        - BaseDetector
        - BaseTransformer
        - BaseAggregator
        - rolling
        - double_rolling
        - parse_duration

## `hazure.detection`

Ready-made detectors: a series in, binary labels out.

::: hazure.detection

## `hazure.scoring`

The scorers those detectors are built from. A score is worth having on its own,
for ranking.

::: hazure.scoring

## `hazure.thresholds`

Where to draw the line, independently of what produced the score.

::: hazure.thresholds

## `hazure.features`

Transformers: series in, series out. These sit upstream of a scorer.

::: hazure.features

## `hazure.ensemble`

Aggregators, for combining several detectors' verdicts.

::: hazure.ensemble

## `hazure.compose`

Chaining and wiring components into models.

::: hazure.compose

## `hazure.events`

Moving between one-label-per-sample and the anomalous-interval view of the same
thing.

::: hazure.events

## `hazure.evaluation`

Point- and event-based metrics, how late each alert was, threshold-free ranking
quality, and time-ordered folds to compute any of them over.

::: hazure.evaluation

## `hazure.methods`

Further method families: spectral residual, Hampel filtering and rolling
quantile bands, change-point segmentation, matrix-profile discords, and
STL / MSTL residuals.

::: hazure.methods

## `hazure.plotting`

Drawing a series, its verdicts and its scores. Needs the `viz` extra.

::: hazure.plotting
