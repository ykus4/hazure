# hazure

You have a metric — requests per second, queue depth, a sensor reading — you
suspect it occasionally misbehaves, and you have no record of when it did.
`hazure` finds those moments: rule-based and unsupervised detectors that learn
what "normal" looks like from the series itself, and that say where it stopped
holding.

Any pandas, polars or pyarrow object with a time axis goes in, and the answer
comes back in the same flavour. At runtime it needs `narwhals` and `numpy` and
nothing else; every heavier dependency is an extra, imported lazily by the
algorithm that needs it.

```bash
pip install hazure[pandas]
```

```python
import numpy as np
import pandas as pd
from hazure.detection import SeasonalDetector
from hazure.events import to_events

# Hourly traffic with a daily rhythm — and one afternoon that went wrong.
rng = np.random.default_rng(0)
index = pd.date_range("2024-03-01", periods=24 * 21, freq="h", name="time")
daily = 40 * np.sin(2 * np.pi * np.arange(len(index)) / 24)
traffic = pd.Series(100 + daily + rng.normal(0, 3, len(index)), index=index, name="rps")
traffic.iloc[300:306] = 20.0

labels = SeasonalDetector(period=24).fit_detect(traffic)
print(to_events(labels))
# Events([2024-03-13T12:00:00..2024-03-13T17:59:59.999999999])
```

`labels` sits on `traffic`'s own index — `1.0` anomalous, `0.0` normal, `NaN`
unknown — and the six flagged hours are exactly the six that were planted.

Everything is one of five composable component types (`Scorer`, `Threshold`,
`Detector`, `Aggregator`, `Transformer`), chained with `Pipeline` and wired with
`Graph` when the model branches.

## Documentation

<https://ykus4.github.io/hazure/>

- [Quickstart](https://ykus4.github.io/hazure/quickstart/) — one planted anomaly
  end to end: detect, convert to intervals, score, plot.
- [Guide](https://ykus4.github.io/hazure/guide/) — which detector suits which
  kind of anomaly, and the two behaviours that surprise people most.
- [How it works](https://ykus4.github.io/hazure/algorithms/) — the mathematics of
  every scorer, threshold and metric, and where each one's assumptions run out.
- [API reference](https://ykus4.github.io/hazure/api/) — every public name.

## Details

- Python 3.11 and newer.
- Fully typed; `mypy --strict` clean.
- MIT licensed.
