# Quickstart

This page finds one planted anomaly end to end: generate a series, detect,
convert the verdict to intervals, score it against the truth, and look at it.
Every block runs as written, top to bottom.

```bash
pip install hazure[pandas,viz]
```

## A series with something wrong in it

Three weeks of hourly traffic with a daily rhythm and a little noise, and one
afternoon where the service degraded and served almost nothing.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
index = pd.date_range("2024-03-01", periods=24 * 21, freq="h", name="time")
daily = 40 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
traffic = pd.Series(100 + daily + rng.normal(0, 3, len(index)), index=index, name="rps")

# Six hours of degradation, starting at 12:00 on 13 March.
traffic.iloc[300:306] = 20.0
```

Nothing here says *where* the anomaly is. That is the point: an unsupervised
detector learns normal behaviour from the same series it searches.

## Detect

The daily cycle is normal behaviour, so it belongs in the model rather than in
the anomalies. `SeasonalDetector` learns the average shape of one cycle and
flags the hours the shape fails to explain.

```python
from hazure.detection import SeasonalDetector

detector = SeasonalDetector(period=24)
labels = detector.fit_detect(traffic)

print(labels.value_counts(dropna=False).to_dict())
# {0.0: 498, 1.0: 6}
```

`labels` is a pandas `Series` on `traffic`'s own index — `1.0` anomalous, `0.0`
normal, `NaN` where the detector cannot say. `fit_detect` is the unsupervised
entry point: one series both defines normal and is searched for departures from
it. When you have a clean history to learn from, `fit` it once and `detect` on
later data instead.

## Read it as intervals

An anomaly is usually a stretch of time, not a set of points. On a regular
hourly axis a flagged sample stands for the hour it opens, so six consecutive
flags are one six-hour event rather than six one-hour ones.

```python
from hazure.events import to_events

found = to_events(labels)
print(found)
# Events([2024-03-13T12:00:00..2024-03-13T17:59:59.999999999])
print(found.n_events, found.durations // 3_600_000_000_000)
# 1 [6]
```

`Events` is a sorted, non-overlapping set of closed intervals in UTC
nanoseconds. `found.to_list()` gives them back as native timestamps, and
`found.to_frame()` as a two-column `start`/`end` frame in your own backend.

## Score it against the truth

Here we do know the answer, so we can measure. Build the ground truth the same
way — as labels, converted with the same period semantics — and the two are
directly comparable.

```python
from hazure.events import to_events
from hazure.evaluation import f1_score, iou, precision, recall

truth_labels = pd.Series(np.zeros(len(index)), index=index, name="rps")
truth_labels.iloc[300:306] = 1.0
truth = to_events(truth_labels)

print(recall(truth, found), precision(truth, found))
# 1.0 1.0
print(f1_score(truth, found), iou(truth, found))
# 1.0 1.0
```

Those numbers are *event-based*, because both arguments are `Events`: one true
interval, detected, and one predicted interval, real. Pass the label series
instead and the same functions count samples:

```python
print(recall(truth_labels, labels))
# 1.0
```

The two modes answer different questions. Event-based is usually what an
operator cares about — one page, one investigation — and an alert firing
part-way through a six-hour outage is a hit rather than a fraction of one. By
default an event counts as detected when half its duration is covered; pass
`thresh=` to change that.

## Look at the score behind the verdict

The detector is a scorer and a threshold paired. Ask the scorer directly and you
get the signed seasonal residual, which is worth ranking even where nothing
crosses a line.

```python
from hazure.scoring import SeasonalResidualScorer

scores = SeasonalResidualScorer(period=24).fit_score(traffic)
print(scores.iloc[300:303].round(1).to_list())
# [-76.1, -66.2, -56.7]
```

Swap the policy without touching the scorer. `ScoreDetector` pairs any scorer
with any threshold:

```python
from hazure.detection import ScoreDetector
from hazure.thresholds import QuantileThreshold

strict = ScoreDetector(
    SeasonalResidualScorer(period=24),
    QuantileThreshold(low=0.01, high=0.99),
)
print(int(strict.fit_detect(traffic).sum()))
# 12
```

## Plot it

`hazure.plotting.plot` draws the series, any number of named verdicts, and the
score on its own panel below — a score in the tens would otherwise flatten a
series in the hundreds.

```python
import matplotlib

matplotlib.use("Agg")  # any backend; this one just writes files

from hazure.plotting import plot

fig, axes = plot(
    traffic,
    anomaly={"detected": labels, "truth": truth},
    score=scores,
    title="a degraded afternoon",
)
fig.savefig("quickstart.png", dpi=120)
```

Two named verdicts, two colours, one time axis: whether the detector agreed with
the truth is now a glance. `plot` returns the `Figure` as well as the axes, so
you can save and close it, and it never touches `matplotlib.rcParams` — your
theme survives the call. Use `style="marker"` to mark the flagged observations on
the line instead of shading the interval.

## Any dataframe

The same code works on polars or pyarrow, and the result comes back in the
flavour it went in as.

```python
import polars as pl

wide = pl.DataFrame({"time": index, "rps": traffic.to_numpy()})
polars_labels = SeasonalDetector(period=24).fit_detect(wide)
print(type(polars_labels).__name__, int(polars_labels["rps"].sum()))
# DataFrame 6
```

## Where to go next

- [Guide](guide.md) — the five component types, which detector suits which kind
  of anomaly, and the two behaviours that surprise people most.
- [API reference](api.md) — every public name, module by module.
