# hazure

You have a metric — requests per second, queue depth, a sensor reading — you
suspect it occasionally misbehaves, and you have no record of when it did.
`hazure` gives you the tools to find those moments: rule-based and unsupervised
detectors that learn what "normal" looks like from the series itself, and that
say where it stopped holding.

It takes any pandas, polars or pyarrow object with a time axis and hands the
answer back in the same flavour. At runtime it needs `narwhals` and `numpy`, and
nothing else.

```bash
pip install hazure[pandas]
```

## The shortest useful example

```python
import numpy as np
import pandas as pd
from hazure.detection import SeasonalDetector

# Hourly traffic with a daily rhythm — and one afternoon that went wrong.
rng = np.random.default_rng(0)
index = pd.date_range("2024-03-01", periods=24 * 21, freq="h", name="time")
daily = 40 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
traffic = pd.Series(100 + daily + rng.normal(0, 3, len(index)), index=index, name="rps")
traffic.iloc[300:306] = 20.0

labels = SeasonalDetector(period=24).fit_detect(traffic)
```

`labels` is a pandas `Series` on the same index: `1.0` where the hour is
anomalous, `0.0` where it is not, `NaN` where the detector cannot say. Six hours
are flagged, and they are exactly the six that were planted. Read them as
intervals rather than as samples:

```python
from hazure.events import to_events

print(to_events(labels))
# Events([2024-03-13T12:00:00..2024-03-13T17:59:59.999999999])
```

## Five component types

Everything in `hazure` is one of five things, and each has `fit` plus one verb:

| Type | Takes | Gives | Verbs |
| --- | --- | --- | --- |
| `Scorer` | a series | a continuous score | `.score()`, `.fit_score()` |
| `Threshold` | a score | binary labels | `.apply()`, `.fit_apply()` |
| `Detector` | a series | binary labels | `.detect()`, `.fit_detect()` |
| `Aggregator` | several label series | one label series | `.aggregate()` |
| `Transformer` | a series | a series | `.transform()`, `.fit_transform()` |

A `Detector` is a `Scorer` and a `Threshold` paired, and the parts stay visible
as `.scorer` and `.threshold`. Asking *how unusual is this point* and asking *is
that unusual enough to report* are separate questions, and keeping them apart
buys three things:

- one threshold policy is reusable across every scorer;
- a scorer can be swapped without revisiting the policy;
- a score is useful on its own, for ranking the worst hours of a month even when
  none of them crosses a line.

Labels are `1.0` anomalous, `0.0` normal and `NaN` unknown. `NaN` is common and
meaningful: a window-based detector cannot judge the first few observations, and
saying so is more useful than calling them normal. A univariate component handed
a multi-column frame fits one independent copy per column, so each column learns
its own normal range.

## Composing them

A `Pipeline` is a straight chain, and behaves as a single detector:

```python
from hazure import Pipeline
from hazure.features import SeasonalDecomposition
from hazure.scoring import DeviationScorer
from hazure.thresholds import IqrThreshold

model = Pipeline(
    [
        ("deseasonalise", SeasonalDecomposition(period=24)),
        ("score", DeviationScorer()),
        ("cut", IqrThreshold(factor=3.0)),
    ]
)
labels = model.fit_detect(traffic)
```

A `Graph` is for anything that branches. Here a quantile rule alone raises six
false alarms over three weeks; requiring a seasonal violation to agree with it
leaves exactly the real one:

```python
from hazure import Graph, Node
from hazure.detection import QuantileDetector
from hazure.ensemble import AndAggregator

agree = Graph(
    [
        Node("unusual_value", QuantileDetector(low=0.01, high=0.99)),
        Node("off_pattern", SeasonalDetector(period=24)),
        Node("both", AndAggregator(), inputs=("unusual_value", "off_pattern")),
    ]
)
labels = agree.fit_detect(traffic)
print(agree.to_mermaid())
```

```mermaid
flowchart LR
  input(["input"])
  unusual_value["QuantileDetector<br/>unusual_value"]
  off_pattern["SeasonalDetector<br/>off_pattern"]
  both["AndAggregator<br/>both"]
  input --> unusual_value
  input --> off_pattern
  unusual_value --> both
  off_pattern --> both
  both --> output((["output"]))
```

`Graph.trace()` returns every node's output rather than only the last, which is
how you find out which branch disagreed.

## Any dataframe, and the same one back

pandas, polars and pyarrow are all first-class, and the result matches the input:
a pandas `Series` with a `DatetimeIndex` comes back as a pandas `Series` with the
same index, and a polars `DataFrame` with a time column comes back as a polars
`DataFrame` with that column in the same place — name, unit and time zone intact.
The numbers are identical across all three, because the time axis is read into a
sorted `int64` array of UTC nanoseconds once, on the way in, leaving no index
alignment to disagree about.

```python
import polars as pl

wide = pl.DataFrame({"time": index, "rps": traffic.to_numpy()})
same_answer = SeasonalDetector(period=24).fit_detect(wide)
```

## Dependencies

`narwhals` and `numpy` at runtime. Everything else is an extra, imported lazily
by the algorithm that needs it, so importing `hazure` never pulls in a plotting
stack or a solver you were not going to use.

| Extra | Brings | Needed for |
| --- | --- | --- |
| `pandas`, `polars`, `pyarrow` | that dataframe library | reading and returning that flavour |
| `stats` | scipy, statsmodels | `EsdThreshold`, STL and MSTL residuals |
| `sklearn` | scikit-learn | PCA, regression and clustering scorers |
| `mp` | stumpy | matrix-profile discords |
| `cpd` | ruptures | extra change-point search strategies |
| `viz` | matplotlib | `hazure.plotting.plot` |
| `all` | all of the above except `cpd` | |

An algorithm whose extra is missing raises on use, naming the extra to install —
never at import time.

## What is in the box

| Module | Holds |
| --- | --- |
| `hazure.detection` | ready-made detectors: outliers, spikes, level and volatility shifts, seasonal and autoregressive violations, PCA and regression residuals, clustering |
| `hazure.scoring` | the scorers those detectors are built from |
| `hazure.thresholds` | fixed, quantile, inter-quartile, MAD and generalised-ESD cut-offs |
| `hazure.features` | rolling and double-rolling aggregates, lagging, scaling, seasonal decomposition, PCA and regression residuals |
| `hazure.ensemble` | `And`, `Or`, `Vote` and customised aggregators |
| `hazure.compose` | `Pipeline`, `Graph`, `Node` |
| `hazure.events` | `Events`, and conversion between labels and intervals |
| `hazure.evaluation` | point- and event-based recall, precision, F1 and IoU, and time-ordered folds |
| `hazure.methods` | further method families: spectral residual, Hampel filtering, change-point segmentation, matrix profile, STL residuals |
| `hazure.plotting` | one `plot` function for a series, its verdicts and its scores |

## Details

- Python 3.11 and newer.
- Fully typed; `mypy --strict` clean.
- MIT licensed.
- Documentation: <https://ykus4.github.io/hazure/> — a
  [quickstart](https://ykus4.github.io/hazure/quickstart/), a
  [guide](https://ykus4.github.io/hazure/guide/), and the full
  [API reference](https://ykus4.github.io/hazure/api/).
