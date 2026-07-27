# Guide

The [quickstart](quickstart.md) got one anomaly found. This page is about
choosing: which pieces exist, which detector suits which kind of anomaly, and
two behaviours that look like bugs the first time you meet them.

The code blocks build on one another, so running the page top to bottom
reproduces every number shown.

## The five component types

Everything is one of five things, and each has `fit` plus one verb:

| Type | Takes | Gives | Verbs |
| --- | --- | --- | --- |
| `Scorer` | a series | a continuous score | `.score()`, `.fit_score()` |
| `Threshold` | a score | binary labels | `.apply()`, `.fit_apply()` |
| `Detector` | a series | binary labels | `.detect()`, `.fit_detect()` |
| `Aggregator` | several label series | one label series | `.aggregate()` |
| `Transformer` | a series | a series | `.transform()`, `.fit_transform()` |

A score answers *how unusual is this point*, on whatever scale the algorithm
works in, higher being more unusual. A threshold answers *is that unusual enough
to report*. They are separate objects because they are separate questions: one
threshold policy is then reusable across every scorer, a scorer can be swapped
without revisiting the policy, and a score is useful on its own — for ranking the
worst hours of a month even when none of them crosses a line.

A `Detector` is the pairing of the two, since that is how detection is usually
wanted, and the parts stay reachable as `.scorer` and `.threshold`. Pull a
detector apart to inspect the raw score, or build your own pairing with
`ScoreDetector(scorer, threshold)`.

A `Transformer` sits upstream: a rolling aggregate, a lag, a seasonal
decomposition. An `Aggregator` sits downstream, reducing several detectors'
verdicts to one. `Pipeline` chains components; `Graph` wires them as a directed
acyclic graph when the model branches. Both are themselves components, so a
pipeline can be a node of a graph and either can be used anywhere a detector can.

Two conventions run through all of it:

- **Labels are `1.0` anomalous, `0.0` normal, `NaN` unknown.** Fitted attributes
  end in an underscore (`threshold_`, `period_`), and every constructor parameter
  is a plain keyword argument stored under its own name, so `get_params()`,
  `clone()` and `repr()` need no maintenance.
- **Results come back in the flavour they went in as.** pandas in, pandas out,
  with the same index; polars in, polars out, with the time column in the same
  place.

## Which detector for which anomaly

"Anomalous" is not one thing, and the choice of detector is mostly the choice of
which of these you mean.

| The anomaly is | Reach for | Because |
| --- | --- | --- |
| a value outside its usual range | `IqrDetector`, `QuantileDetector`, `EsdDetector`, `ThresholdDetector` | the value alone is enough to judge; no context needed |
| a **spike** — local and temporary | `SpikeDetector` | compares each point with the window just before it, so a slow drift is invisible to it |
| a **level shift** — permanent | `LevelShiftDetector` | compares two long windows either side of each point; a single odd value cannot move a wide median |
| a **volatility shift** | `VolatilityShiftDetector` | measures spread rather than position, relatively, so a doubling of noise counts equally on a quiet series and a loud one |
| a violation of a **seasonal pattern** | `SeasonalDetector` | the cycle is normal behaviour, so it belongs in the model rather than in the anomalies |
| a violation of the series' own **dynamics** | `AutoregressionDetector` | asks whether a value is unusual *given* where the series just was, which catches a break at a perfectly ordinary level |
| a broken relationship **between columns** | `RegressionDetector`, `PcaDetector` | both columns can be in their usual range and still be impossible together |
| a rare **combination** of readings | `MinClusterDetector`, `OutlierDetector` | nothing needs to be said about what anomalous looks like; the shape of the data decides |

The distinctions matter more than they look. Here are three series, each with one
thing wrong, and the detector that suits each:

```python
import numpy as np
import pandas as pd
from hazure.detection import (
    LevelShiftDetector,
    SpikeDetector,
    VolatilityShiftDetector,
)
from hazure.events import to_events

rng = np.random.default_rng(4)
index = pd.date_range("2024-01-01", periods=600, freq="h", name="time")

values = rng.normal(10.0, 1.0, 600)
values[150] = 22.0  # one hour, gone by the next
spiky = pd.Series(values, index=index, name="t")

values = rng.normal(10.0, 1.0, 600)
values[300:] += 6.0  # it never comes back
stepped = pd.Series(values, index=index, name="t")

values = rng.normal(10.0, 1.0, 600)
values[300:] = 10.0 + rng.normal(0.0, 4.0, 300)  # same level, more noise
noisy = pd.Series(values, index=index, name="t")

for name, detector, series in (
    ("spike", SpikeDetector(window=24, factor=6.0), spiky),
    ("level shift", LevelShiftDetector(window=24), stepped),
    ("volatility shift", VolatilityShiftDetector(window=24, factor=6.0), noisy),
):
    events = to_events(detector.fit_detect(series))
    first = np.datetime64(int(events.bounds[0, 0]), "ns")
    print(f"{name:17s} {events.n_events} event(s), first at {first}")
# spike             1 event(s), first at 2024-01-07T06:00:00.000000000
# level shift       1 event(s), first at 2024-01-13T00:00:00.000000000
# volatility shift  2 event(s), first at 2024-01-12T21:00:00.000000000
```

`values[150]` is 2024-01-07 06:00, and the spike detector lands on it exactly.
The shift detectors land on a window *around* 2024-01-13 12:00, which is
`values[300]` — the point where the change happened. That difference is the
subject of the first section below.

Every one of these takes a `factor`, which is how many spreads from normal is too
far, and a `side` of `"both"`, `"positive"` or `"negative"`. Detecting only the
increases is not a different algorithm, just a filter on the same one.

## Univariate and multivariate

A univariate component works on one series at a time. Hand it a frame and it does
not fail and does not pool the columns: it fits **one independent copy of itself
per column**, so each column learns its own normal range and each gets its own
verdict column back. That is what "separable across series" means — the algorithm
never needs to see two columns at once, so `hazure` can fan it out for you.

```python
from hazure.detection import IqrDetector

frame = pd.DataFrame(
    {"celsius": rng.normal(20.0, 1.0, 200), "pascals": rng.normal(1e5, 500.0, 200)},
    index=index[:200],
)
frame.iloc[50, 0] = 40.0
frame.iloc[150, 1] = 2e5

detector = IqrDetector().fit(frame)
labels = detector.detect(frame)
print(list(labels.columns), labels.sum().to_dict())
# ['celsius', 'pascals'] {'celsius': 1.0, 'pascals': 1.0}
```

Two columns on wildly different scales, and one fence each. Pooling them would
have made either fence meaningless.

A multivariate component is one whose algorithm genuinely needs every column at
once, and it returns a **single** verdict rather than one per column, because the
anomaly is a property of the combination:

```python
from hazure.detection import RegressionDetector

requests = 100 + 20 * np.sin(np.arange(400) / 12) + rng.normal(0, 2, 400)
cpu = 0.4 * requests + rng.normal(0, 1, 400)
cpu[250:256] += 30.0  # CPU burns without the traffic to explain it
pair = pd.DataFrame({"requests": requests, "cpu": cpu}, index=index[:400])

verdict = RegressionDetector(target="cpu").fit_detect(pair)
print(list(verdict.columns))
# ['anomaly']
```

Neither column leaves its usual range; the relationship between them is what
broke. A multivariate component also remembers the columns it was fitted on, and
reorders a later frame to match, because its coefficients are positional.

## Missing values

A missing observation is `NaN`, and it stays `NaN`. Nothing is imputed, and no
gap is filled with a guess, because a guess would be indistinguishable from an
observation in the output.

```python
gappy = spiky.copy()
gappy.iloc[100:105] = np.nan

verdict = IqrDetector().fit_detect(gappy)
print(verdict.iloc[99:106].to_list())
# [0.0, nan, nan, nan, nan, nan, 0.0]
```

`NaN` in the output means *unknown*, which is different from `0.0`. It also
appears where a window-based detector has too little history to judge — the first
`window` observations of a rolling comparison — and saying so is more useful than
calling them normal. Both `to_events` and the point-based metrics read `NaN` as
not-anomalous, so an unknown region never invents an event, but it never counts
as a correct negative either.

`validate_series` applies the same normalisation every detector applies
internally — read the time axis, sort it, keep the first observation at each
timestamp, cast to float — and hands the result back, so a surprising detection
can be traced to a reordered axis or a dropped duplicate rather than guessed at.

## Two things that surprise people

### A shift detector reports the change point, not the anomalous interval

`LevelShiftDetector` answers *when did the series become a different series*. It
compares the window before each point with the window after it, so it fires
around the moment of change and goes quiet again once both windows sit at the new
level — even though, in an operational sense, everything after the change is
still wrong.

That is by design, and it means **evaluating a shift detector against
interval-shaped ground truth will show near-zero recall even when it is working
perfectly**:

```python
from hazure.events import Events, expand_events
from hazure.evaluation import f1_score, precision, recall

rng = np.random.default_rng(1)
axis = pd.date_range("2024-01-01", periods=480, freq="h", name="time")
values = rng.normal(10.0, 1.0, 480)
values[240:] += 6.0
series = pd.Series(values, index=axis, name="temperature")

found = to_events(LevelShiftDetector(window=24).fit_detect(series))
print(found)
# Events([2024-01-10T12:00:00..2024-01-11T12:59:59.999999999])

# "Everything from the shift onwards was wrong" — true, but not what was asked.
as_interval = Events.from_any([(axis[240], axis[-1])])
print(recall(as_interval, found), precision(as_interval, found))
# 0.0 1.0
```

Recall is zero because the detector covered 25 hours of a 240-hour interval, and
an event counts as detected only when half its duration is covered. Nothing is
broken. The comparison is.

The correct comparison reduces the truth to the change point too, and allows a
tolerance — the detector cannot localise the change more finely than the window
it used, so give it that window:

```python
as_change_point = expand_events(Events.from_any([axis[240]]), before="12h", after="12h")
print(recall(as_change_point, found), precision(as_change_point, found))
# 1.0 1.0
print(f1_score(as_change_point, found))
# 1.0
```

The same applies to `VolatilityShiftDetector` and to the change-point methods in
`hazure.methods.changepoint`. If what you actually want is the whole degraded
stretch, detect the change points and fill between them, or use a detector that
judges each point on its own — a level shift leaves every subsequent value
outside the *training* range, which is exactly what `IqrDetector` fitted on the
clean history will tell you.

### An inter-quartile fence widens as the fraction of unusual scores grows

`IqrDetector` and `IqrThreshold` use the box-plot rule: everything outside
`[Q1 - factor * IQR, Q3 + factor * IQR]`. Quartiles ignore the tails, which is
the whole point — the outliers being looked for do not widen the range that is
supposed to exclude them.

But quartiles are not immune, only resistant. Push enough of the series into the
unusual state and a quartile lands *inside* it, the inter-quartile range grows to
span both regimes, and the fence stops separating them:

```python
from hazure.detection import IqrDetector

axis = pd.date_range("2024-01-01", periods=200, freq="h", name="time")
for fraction in (0.05, 0.2, 0.4, 0.6):
    values = rng.normal(0.0, 1.0, 200)
    values[: int(200 * fraction)] += 8.0
    flagged = IqrDetector(factor=1.5).fit_detect(pd.Series(values, index=axis))
    print(f"{fraction:>4.0%} of the series raised -> {int(flagged.sum()):3d} flagged")
#   5% of the series raised ->  10 flagged
#  20% of the series raised ->  40 flagged
#  40% of the series raised ->   0 flagged
#  60% of the series raised ->   0 flagged
```

At 40% the third quartile has moved into the raised group, so `IQR` is about 8
rather than about 1.3, and a fence 1.5 IQRs wide covers everything. **This is the
definition working as intended, not a failure.** A rule that says "unusual
relative to this series" cannot flag the majority of the series; if the anomalous
state is the majority, it is the normal state as far as the rule can tell.

What to do about it, in order of preference:

1. **Fit on a clean period and apply to the suspect one.** `fit` on the history
   you trust, `detect` on the rest — the fence then comes from data that has not
   been contaminated. `split_train_test` builds time-ordered folds for this.
2. **Say what normal is.** `ThresholdDetector(low=..., high=...)` learns nothing,
   so nothing can contaminate it.
3. **Ask a local question instead.** `SpikeDetector` and `LevelShiftDetector`
   compare each point with its own neighbourhood, so a regime lasting most of the
   series is still a departure from the hour before it began.
4. **Detect the transition.** If most of the series is in the bad state, the
   interesting event is when it entered it — see the previous section.

The same reasoning applies to `QuantileThreshold`, whose cut-off is a percentile
of the scores it was fitted on and so always flags roughly a fixed fraction, and
to `MadThreshold`, which is more resistant than the IQR fence but not immune
either.

## Where to go next

- [API reference](api.md) — every public name, module by module.
