# Univariate scores

[Rolling comparisons](rolling.md) covered the scores built from windows. The
three here are built from a **model of the series** instead: a level and a
spread, a repeating shape, or the series' own dynamics. Each one subtracts what
it expects and hands the leftover to a threshold.

All three are separable per column. Handed a frame, each fits one independent
copy of itself per column, so every column learns its own normal.

## Deviation from a learned centre

`DeviationScorer` is the simplest thing that can be called a model: one centre
and one spread, learned from the training series as a whole.

$$
s_t = \frac{x_t - c}{\sigma}
$$

`center` picks $c$ — `"median"` or `"mean"`. `scale` picks $\sigma$:

| `scale` | $\sigma$ | Breakdown point |
| --- | --- | --- |
| `"iqr"` (default) | $Q(0.75) - Q(0.25)$ | 25 % |
| `"idr"` | $Q(0.90) - Q(0.10)$ | 10 % |
| `"mad"` | $1.4826 \cdot \operatorname{median}\lvert x - \operatorname{median}(x)\rvert$ | 50 % |
| `"std"` | $\sqrt{\tfrac{1}{m-1}\sum(x - \bar{x})^2}$ | $1/m$ |

The score is **signed**, so the direction of the excursion survives to the
detector and `side` can act on it.

Because $c$ and $\sigma$ are fitted once and reused, this is the scorer to reach
for when you have a clean period to learn from: `fit` on the good week, `score`
forever after, and the yardstick does not move when the data goes wrong. That is
the opposite of `StandardScale`, which looks identical but recomputes its
statistics from whatever series it is handed — a transformer for putting columns
on a comparable footing, not a model of normal.

A constant training series gives $\sigma = 0$, and rather than divide by it the
score becomes $0$ where $x_t = c$ and $\pm\infty$ either side. An infinite score
is the honest answer: any departure at all from a series that never varied is
unboundedly surprising.

## Seasonal residual

When a daily or weekly cycle is normal behaviour, it belongs in the model rather
than in the anomalies. `SeasonalResidualScorer` learns the average shape of one
cycle and scores what that shape fails to explain:

$$
s_t = x_t - T_t - S_{\phi(t)}
$$

with $S$ the seasonal profile, $\phi(t)$ the phase of $t$, and $T_t$ a trend that
is only estimated when `trend=True` (otherwise $T_t \equiv 0$ and the profile
carries the level).

### The profile

Phase is positional: $\phi = t \bmod P$ over the observations, so with $P = 24$
on hourly data, phase 7 is "the same hour of the day as the eighth observation of
training". The profile is the mean of each phase over the observed data,

$$
S_p = \operatorname{mean}\bigl\{\,\tilde{x}_t \;:\; \phi(t) = p\,\bigr\}
$$

where $\tilde{x}$ is the series after detrending. A phase that was never observed
during training raises rather than silently producing a `NaN` column.

With `trend=True` the profile is re-centred to mean zero, so the level lives in
the trend and the seasonal term is a pure deviation.

### The trend

$T$ is a centred moving average of exactly one period — the standard classical
decomposition filter, with the even case weighted so the window stays symmetric:

$$
T_t = \begin{cases}
\dfrac{1}{P}\displaystyle\sum_{j=-k}^{k} x_{t+j} & P = 2k+1\\[12pt]
\dfrac{1}{P}\left(\tfrac{1}{2}x_{t-k} + \displaystyle\sum_{j=-k+1}^{k-1} x_{t+j} + \tfrac{1}{2}x_{t+k}\right) & P = 2k
\end{cases}
$$

Both sets of weights sum to 1, and both span exactly one cycle, which is what
makes the filter annihilate the seasonal component instead of smearing it. The
half-weights at the ends of the even filter exist because $2k$ consecutive points
would cover one phase twice.

The cost is a margin: $\lfloor P/2 \rfloor$ rows at each end are `NaN`, and —
unlike `rolling`, which skips missing values — the convolution propagates them,
so one gap blanks a whole window's worth of trend. `trend=False` avoids all of
this, and is the default, because most metrics worth monitoring are stationary
enough over a few cycles that the profile alone is a good model.

Note that the trend is re-estimated on whatever series is being scored, while the
profile comes from training. A later series is judged against the shape that used
to hold, at the level it currently has.

### Detecting the period

With `period=None` the cycle length is found from the autocorrelation, computed
by FFT, which is the same quantity as the direct sum but $O(n \log n)$:

$$
\hat{\gamma}(\ell) = \mathcal{F}^{-1}\!\left[\bigl|\mathcal{F}[\tilde{x}]\bigr|^2\right](\ell),
\qquad
\hat{\rho}(\ell) = \frac{\hat{\gamma}(\ell)}{\hat{\gamma}(0)}
$$

Missing observations are set to the mean so they contribute nothing to the sum,
and the series is zero-padded to the next power of two beyond $2n$ so the
circular convolution the FFT actually computes cannot wrap the end of the series
onto its beginning. The period is then the **highest strict local maximum** of
$\hat\rho$ over lags $1 \dots \lfloor n/2 \rfloor$ that reaches at least $0.3$.
If nothing clears that floor, it raises rather than guessing.

Two things follow. It picks the strongest cycle, not the shortest: on hourly data
with both a daily and a weekly rhythm it may well return 168 rather than 24. And
it can only see what is inside the window, hence lags no longer than $n/2$ — two
full cycles are the minimum for a period to be visible at all, and that minimum
is enforced separately when the profile is fitted.

Passing `period=` explicitly is usually better, because you know the physics of
the metric and the autocorrelation only knows the sample.

### Scoring later data

Phase is recovered arithmetically from the training anchor, not from position:

$$
\phi(t) = \left\lfloor \frac{t - t_0}{\Delta} \right\rfloor \bmod P
$$

so a gap in the series being scored is harmless — every timestamp knows its own
phase. A different sampling interval, or an axis offset that is not a whole
number of steps from the training anchor, raises. Silently scoring Tuesday's
profile against Wednesday's data would be worse than an error.

## Autoregressive residual

The previous two scores ask whether a value is unusual on its own terms.
`AutoregressionResidualScorer` asks whether it is unusual **given where the
series just was**:

$$
s_t = x_t - \hat{f}\bigl(x_{t-s},\ x_{t-2s},\ \dots,\ x_{t - n s}\bigr)
$$

for `n_steps` lags spaced `step_size` apart. With the default
`OrdinaryLeastSquares` that is a linear AR model fitted by least squares:

$$
s_t = x_t - \left(\beta_0 + \sum_{k=1}^{n} \beta_k\, x_{t-ks}\right)
$$

Mechanically it is a lag matrix and a regression: `Retrospect` builds the columns
`t-0, t-s, …, t-ns`, and `RegressionResidual` takes `t-0` as the target and the
rest as features. Any regressor with `fit` / `predict` can be substituted — the
scorer deep-copies it, so a frame fanned out across columns fits one model per
column instead of overwriting one shared object.

What this catches that the others do not is a **break in dynamics at an ordinary
level**. A metric that ramps smoothly and then jumps to a value it would have
reached ten minutes later has, at every instant, a perfectly normal value; only
the step from the previous one is wrong. Conversely, it is blind by construction
to anything the lags predict — a level shift is unremarkable to an AR model one
step after it happens, because the model's input has shifted too.

The first $n \cdot s$ rows are `NaN` — no history — and a series too short or too
gappy to yield a single complete lag row scores all `NaN` rather than raising.
`step_size` is the way to reach a seasonal lag without paying for everything in
between: `n_steps=2, step_size=24` regresses on the same hour of the last two
days.

## Every detector, disassembled

A `Detector` is a scorer and a threshold, and both parts stay reachable as
`.scorer` and `.threshold`. Below is what each ready-made one is made of. Every
`IqrThreshold(None, f)` is one-sided — the fence applies to $|s_t|$, and `side`
filters the sign afterwards.

| Detector | Scorer | Threshold |
| --- | --- | --- |
| `ThresholdDetector` | *(none — the value is the score)* | `FixedThreshold(low, high)` |
| `QuantileDetector` | *(none)* | `QuantileThreshold(low, high)` |
| `IqrDetector` | *(none)* | `IqrThreshold(factor=3.0)` |
| `EsdDetector` | *(none)* | `EsdThreshold(alpha=0.05)` |
| `SpikeDetector` | `DoubleRollingScorer((w, 1), "median", "diff")` | `IqrThreshold((None, 3.0))` |
| `LevelShiftDetector` | `DoubleRollingScorer(w, "median", "diff")` | `IqrThreshold((None, 6.0))` |
| `VolatilityShiftDetector` | `DoubleRollingScorer(w, "std", "rel_diff")` | `IqrThreshold((None, 6.0))` |
| `SeasonalDetector` | `SeasonalResidualScorer(period, trend)` | `IqrThreshold((None, 3.0))` |
| `AutoregressionDetector` | `AutoregressionResidualScorer(n, s)` | `IqrThreshold((None, 3.0))` |
| `RegressionDetector` | `RegressionResidualScorer(target)` | `IqrThreshold((None, 3.0))` |
| `PcaDetector` | `PcaReconstructionErrorScorer(k)` | `IqrThreshold((None, 5.0))` |
| `MinClusterDetector` | `MinClusterScorer(model)` | `FixedThreshold(high=0.5)` |
| `OutlierDetector` | `OutlierScorer(model)` | `FixedThreshold(high=0.5)` |
| `SpectralResidualDetector` | `SpectralResidualScorer(...)` | `IqrThreshold((None, 3.0))` |
| `HampelDetector` | `HampelScorer(window, center)` | `FixedThreshold(high=factor)` |
| `PeltDetector` | `PeltScorer(...)` | `FixedThreshold(high=0.0)` |
| `MatrixProfileDetector` | `MatrixProfileScorer(window)` | `IqrThreshold((None, 3.0))` |
| `StlDetector`, `MstlDetector` | `StlResidualScorer` / `MstlResidualScorer` | `IqrThreshold((None, 3.0))` |

The two `FixedThreshold` rows are the interesting ones. Everything else learns a
cut-off because its score is in arbitrary units; a Hampel score is already in
robust standard deviations and a cluster score is already binary, so learning a
fence for either would add a failure mode without adding information.

Nothing about these pairings is privileged. `ScoreDetector(scorer, threshold)`
builds any other combination, and a scorer used on its own is often the more
useful object — a ranking of the worst hours of a month does not need a line
drawn through it.
