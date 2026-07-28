# How it works

The [guide](../guide.md) says which detector to reach for. This section says what
each one actually computes — the formula, the assumption behind it, and the
conditions under which the assumption stops holding.

None of it is needed to use the library. It is here because an unsupervised
detector cannot be checked against labels you do not have, so the next best thing
is to know exactly what question it answered.

## Notation

A series is a value $x_t$ observed at timestamp $t$, on an axis sorted ascending
with duplicate timestamps dropped. Where the axis is regular, $\Delta$ is its
sampling interval, and $x_1, \dots, x_n$ index the observations in time order. A
frame is $p$ columns $x_{t1}, \dots, x_{tp}$ sharing one axis.

Three symbols recur:

| Symbol | Is | Produced by |
| --- | --- | --- |
| $s_t$ | a **score** — how unusual $t$ is, higher being more unusual | a `Scorer` |
| $\ell$, $u$ | a **pair of fences** on the score | a `Threshold` |
| $y_t \in \{0, 1, \mathrm{NaN}\}$ | a **label** — normal, anomalous, unknown | a `Detector` |

A score may be **signed**, in which case its magnitude carries the strength and
its sign carries a direction — the value was above, or below, what was expected.
Detectors built on a signed score threshold $|s_t|$ and then use the sign to
honour `side`; see [sides](thresholds.md#sides).

## The shape every detector has

$$
x_t \;\xrightarrow[\text{transform}]{}\; \tilde{x}_t
    \;\xrightarrow[\text{score}]{}\; s_t
    \;\xrightarrow[\text{threshold}]{}\; y_t
    \;\xrightarrow[\text{aggregate}]{}\; y^{\ast}_t
$$

Only the middle two steps are compulsory, and even the first of *those* is
optional: `IqrDetector`, `QuantileDetector`, `EsdDetector` and
`ThresholdDetector` have no scorer at all, because the value is already the
score.

Two properties hold throughout, and both fall out of the arithmetic rather than
being policy laid over it:

- **A point that cannot be scored is `NaN`, never `0`.** Missing input, too
  little history to fill a window, a degenerate denominator — each produces an
  unknown, and unknown survives to the output as a third state.
- **Fitting and scoring are separate operations.** Every cut-off is a function of
  the data the threshold was *fitted* on. Fit on a clean period and the fence
  describes that period; call `fit_detect` and the fence describes a population
  that contains the anomalies being looked for. The second is the normal
  unsupervised mode, and it is sound up to the contamination limit spelled out
  under [thresholds](thresholds.md#what-contamination-does-to-a-fitted-fence).

## The pages

- **[Rolling comparisons](rolling.md)** — the window engine, and the
  single- and double-window statistics most detectors are built from.
- **[Univariate scores](univariate.md)** — deviation, seasonal residual,
  autoregressive residual.
- **[Thresholds](thresholds.md)** — fixed, quantile, IQR, MAD, and the
  generalized ESD test.
- **[Multivariate scores](multivariate.md)** — regression residual, PCA
  reconstruction error, and the clustering and outlier-model adapters.
- **[Method families](methods.md)** — spectral residual, Hampel, change-point
  segmentation, matrix profile, STL/MSTL.
- **[Events and metrics](evaluation.md)** — what an interval is in exact
  arithmetic, and what precision and recall mean over intervals rather than
  samples.
