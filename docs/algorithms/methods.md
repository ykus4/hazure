# Method families

`hazure.methods` collects the techniques that are families of their own rather
than variations on the rolling comparison. Three of them — spectral residual,
Hampel and PELT — need nothing but numpy. The rest are adapters over a backend,
imported lazily, and each says how to install itself if it is missing.

## Spectral residual

Borrowed from visual saliency detection[^hou] and adapted to time series[^ren].
The claim it rests on is an empirical regularity: the **log amplitude spectrum of
a natural signal is locally smooth**. Whatever is smooth is the redundant,
predictable part; what is left over after subtracting the smooth part is where
the surprise is.

For the series $x$ (gaps linearly interpolated first), with $F$ the discrete
Fourier transform:

$$
L(f) = \log\bigl(|F(f)| + \varepsilon\bigr), \qquad
A(f) = \bigl(h_q * L\bigr)(f), \qquad
R(f) = L(f) - A(f)
$$

$A$ is a moving average of width `window` over the frequency bins — the smooth
expectation — and $R$ is the **spectral residual**. Transforming back, keeping
the original phase and using only the residual amplitude, gives the saliency map:

$$
S(t) = \Bigl|\,\mathcal{F}^{-1}\bigl[\exp\bigl(R(f)\bigr)\,e^{\,i\,\varphi(f)}\bigr]\,\Bigr|
$$

Keeping the phase is what puts the saliency back **where it happened**: amplitude
says how much of each frequency is present, phase says where. Finally the map is
divided by its own trailing local mean over `score_window` points, so the score
is a dimensionless ratio:

$$
s_t = \frac{S(t)}{\frac{1}{|W_t|}\sum_{u \in W_t} S(u)}
$$

Two consequences worth internalising. **A clean series scores about 1, not
about 0** — the score is a ratio to the local average, so "unremarkable" is
unity. And the score has no `NaN` margin at either end: before the transform the
series is extended a few points to the right by linear extrapolation from the
last `series_window` observations, which absorbs the wrap-around artefact the DFT
would otherwise leave at the boundary. The extension is discarded afterwards.

Missing observations are interpolated for the transform, then restored to `NaN`
in the score — the interpolation is there so the spectrum is defined, not to
invent verdicts.

The window to think about is `window`, in **frequency bins**, not time. Making it
larger smooths the expectation more, which leaves more in the residual.

## Hampel filter

A median filter with a robust scale attached — the classic decision-theoretic
outlier rule, applied in a moving window.

$$
m_t = \operatorname{median}(W_t), \qquad
d_t = |x_t - m_t|, \qquad
\hat{\sigma}_t = 1.4826 \cdot \operatorname{median}\bigl(\{d_u : u \in W_t\}\bigr)
$$

$$
s_t = \frac{d_t}{\hat{\sigma}_t}
$$

The score is in **local robust standard deviations**, already interpretable,
which is why `HampelDetector` pairs it with a `FixedThreshold(high=factor)` and
is `trainable=False`: it needs no fit, and the conventional $3\sigma$ means what
it usually means. See [MadThreshold](thresholds.md#madthreshold) for where 1.4826
comes from.

Both passes are median-based, so the rule tolerates up to half the window being
corrupted — the highest breakdown point of anything in this library. The window
is centred by default, which makes it non-causal (each verdict uses the points
after it) and costs a `NaN` margin at both ends; `center=False` makes it usable
in a streaming setting at the price of only seeing the past.

One implementation detail worth knowing if you compare against a textbook: the
scale is the rolling median of the *pointwise* deviations $|x_u - m_u|$, each
from its own window's median, rather than $\operatorname{median}_{u \in W_t}|x_u
- m_t|$ from the single median at $t$. The two agree wherever the level is
locally constant and the first is far cheaper. Where the spread is exactly zero —
a flat window — the score is `NaN`, not infinity: a constant stretch is
undefined, not infinitely anomalous.

## Rolling quantile band

`RollingQuantileScorer` draws a band from two rolling quantiles and measures how
far outside it each point falls, in the **units of the series**:

$$
s_t = \max\bigl(x_t - Q_{high}(W_t),\ Q_{low}(W_t) - x_t,\ 0\bigr)
$$

Zero inside the band is a measurement, not a missing value: the point was in the
normal range, and by exactly nothing.

The window is trailing and **includes the point being scored**, which is a
conservative bias — an extreme point widens the very band it is being compared
with, so the score understates it. Widening the window dilutes that; excluding
the point would remove it, at the cost of a band that reacts to a level change
one window later.

That same inclusion is why `RollingQuantileDetector` fits an IQR fence on the
excursions rather than reporting every point that leaves the band. On a series
with a trend the newest observation is routinely the largest in its own window,
and so sits *at* the band's edge and a little outside it. Treating any non-zero
excursion as an anomaly flags a sizeable fraction of a plainly ordinary series —
17 of 40 points on the drifting series in the class's own docstring. The
excursions are a distribution like any other; what matters is one out of
proportion to the rest.

The band and the fence therefore do different jobs: the quantiles decide what
counts as an excursion at all, and `factor` decides which excursions are worth
reporting. Widen the band far enough and every excursion shrinks toward zero,
leaving the fence nothing to separate — a window spanning most of the series
flags nothing, which is correct.

## Change-point segmentation

Different question again: not *which points are unusual* but *where does the
series stop being one series*. Given a cost $C$ measuring how badly a segment
fits a single constant, find the partition minimising

$$
\min_{0 = \tau_0 < \tau_1 < \cdots < \tau_K = n}\
\sum_{k=1}^{K} \Bigl[\,C\bigl(x_{\tau_{k-1}:\tau_k}\bigr) + \lambda\,\Bigr]
$$

The penalty $\lambda$ is what stops the trivial answer — one segment per point,
cost zero. Every additional change point has to pay for itself.

**PELT**[^killick] solves this exactly with dynamic programming plus a pruning
rule. Writing $F(t)$ for the optimal cost of $x_{0:t}$:

$$
F(t) = \min_{s < t}\ \bigl[F(s) + C(x_{s:t}) + \lambda\bigr]
$$

and the pruning: if $F(s) + C(x_{s:t}) > F(t)$, then $s$ can never be the optimal
last change point for **any** end beyond $t$, so it is discarded permanently.
That is what turns $O(n^2)$ into roughly $O(n)$ on data with changes spread
through it, while still returning the exact optimum rather than a greedy
approximation.

The two costs:

$$
C_2(s, e) = \sum_{i=s}^{e-1} \bigl(x_i - \bar{x}\bigr)^2
\qquad\text{(\texttt{"l2"}, changes in mean)}
$$

$$
C_1(s, e) = \sum_{i=s}^{e-1} \bigl|x_i - \operatorname{median}(x)\bigr|
\qquad\text{(\texttt{"l1"}, resistant to outliers)}
$$

`"l2"` runs in $O(1)$ per segment from cumulative sums, which is where the speed
comes from; `"l1"` has no such summary and is correspondingly slower. Missing
observations are skipped, never interpolated.

With `penalty=None` the default is a BIC-flavoured rule scaled to the noise:

$$
\hat{\sigma} = \frac{1.4826 \cdot \operatorname{median}\bigl(|\Delta x|\bigr)}{\sqrt{2}},
\qquad
\lambda = \begin{cases} 2\,\hat{\sigma}^2 \ln n & \texttt{"l2"}\\ 2\,\hat{\sigma}\,\ln n & \texttt{"l1"}\end{cases}
$$

Estimating the noise from successive **differences** rather than from the series
is what makes it usable on data that already contains the shifts being looked
for: a level change contributes one large difference, not a wholesale inflation
of the spread. The $\sqrt{2}$ removes the variance doubling that differencing
introduces, the $\ln n$ is the BIC term for one extra parameter, and the factor 2
counts the two a segment costs — where it happens and what the new level is.

`penalty` is the knob: raise it for fewer, more confident changes; lower it for
more.

### From breakpoints to scores

A change point is not a point-in-time anomaly, so the score is deliberately
sparse — the magnitude of the level change **at** the breakpoint, and exactly
zero everywhere else:

$$
s_t = \begin{cases}
\bigl|\,\text{level}_{k} - \text{level}_{k-1}\,\bigr| & t = \tau_k\\
0 & \text{otherwise}
\end{cases}
$$

with the level being a median for `"l1"` and a mean for `"l2"`. `PeltDetector`
thresholds it at `FixedThreshold(high=0.0)`, so every breakpoint with a non-zero
measured change is flagged, and the penalty — not a fence — controls how many
there are.

Breakpoints are stored as **timestamps** and relocated by timestamp when scoring
a different series, not by position, so an axis that starts elsewhere still lines
up.

### RupturesScorer and RupturesDetector

An adapter over [ruptures](https://centre-borelli.github.io/ruptures-docs/) for
the search strategies not implemented here: `binseg` (greedy binary
segmentation), `window` (a sliding two-sample discrepancy), `dynp` (exact dynamic
programming for a **known** number of changes), `bottomup`. Its `cost` string is
passed straight through, so kernel costs like `"rbf"` and `"normal"` are
available — those detect changes in distribution rather than only in mean.

Two behavioural differences from `PeltScorer`: gaps are linearly interpolated
rather than skipped, and `n_bkps` asks for an exact number of change points
instead of a penalty. Needs `pip install hazure[cpd]`, which requires Python
below 3.14; on newer interpreters `PeltScorer` covers the mean-shift case with no
dependency at all.

`RupturesDetector` pairs it with `FixedThreshold(high=0.0)`, exactly as
`PeltDetector` does — with no factor to tune, because the search has already
decided which changes earn a segment. Second-guessing that with a rule on the
score would answer the same question twice, with less information. Report fewer
changes by raising `penalty`, or by naming `n_bkps`.

## Matrix profile

For each length-$m$ subsequence of the series, find its nearest neighbour
elsewhere in the same series and record that distance. A subsequence whose
nearest neighbour is far away is a **discord**: a shape that happens nowhere
else.

$$
s_i = \min_{j \,\notin\, \mathcal{T}(i)}\ d_z\bigl(T_{i,m},\ T_{j,m}\bigr),
\qquad
d_z(A, B) = \left\lVert \frac{A - \mu_A}{\sigma_A} - \frac{B - \mu_B}{\sigma_B} \right\rVert_2
$$

$\mathcal{T}(i)$ is the trivial-match exclusion zone around $i$ — without it
every subsequence's nearest neighbour would be the one starting one sample later.

The distance is **z-normalised** by default, so matching is on shape: the same
pattern at a different level or amplitude still matches. `normalize=False`
switches to plain Euclidean distance, where level and amplitude are part of the
identity of the shape.

This is the only method here that needs no notion of what anomalous looks like,
and the only one that finds *shapes*. Its blind spot is the mirror image of its
strength: an anomaly that repeats is not a discord. Two identical outages in a
month are each other's nearest neighbour, and both score low.

A distance is computed per subsequence and then broadcast back to points: each
point takes the **maximum** distance over the subsequences covering it, so the
flagged region is $m$ points wide. Maximum rather than mean, because a point
inside one strange window is worth flagging even if its other windows are
ordinary.

`DampScorer`, and `DampDetector` pairing it with the same IQR fence, restrict
that comparison to the past — each subsequence is judged against its nearest
neighbour that **started earlier**, so a novel shape scores high the first time it
occurs rather than being explained away by its own later recurrence. That is the
one to reach for when an anomaly might repeat, and when the question is "was this
novel at the time" rather than "is this unique in the record". The first $2m$
points are `NaN` while there is too little history for the comparison to mean
anything. Both need `pip install hazure[mp]`.

## STL and MSTL residuals

Seasonal-trend decomposition by loess[^stl]: the series is split into trend,
seasonal and remainder by iterated local regression, and the score is the
magnitude of the remainder.

$$
x_t = T_t + S_t + R_t, \qquad s_t = |R_t|
$$

Compared with the [seasonal profile](univariate.md#seasonal-residual) in
`hazure.scoring`, STL buys a **seasonal shape that is allowed to evolve** and a
loess trend rather than a moving average — worth it when the daily pattern in
December is not the one from June. `robust=True`, the default, adds
iteratively-reweighted fitting so that the anomalies you are looking for do not
distort the decomposition that is supposed to reveal them.

MSTL extends it to several periods at once, fitting each in turn — a daily and a
weekly cycle superimposed, which is the usual situation for anything driven by
human activity.

Two caveats. The score is **not standardised** — it is in the units of the
series, so how large a residual counts as large depends on how well the
decomposition fitted, which is exactly why the detectors pair it with a *learned*
IQR fence rather than a fixed one. And loess is fitted over the whole series, so
every residual depends on the entire input, future included. That is fine for a
retrospective search and wrong for anything claiming to be online. Needs
`pip install hazure[stats]`.

With `period=None`, STL infers one observation-count period from the sampling
interval: a day's worth of observations for sub-daily data, a week for daily
data. Anything else needs `period=` explicitly, and MSTL always does.

[^hou]: Hou, X. & Zhang, L. (2007). *Saliency Detection: A Spectral Residual
    Approach.* CVPR.
[^ren]: Ren, H. et al. (2019). *Time-Series Anomaly Detection Service at
    Microsoft.* KDD.
[^killick]: Killick, R., Fearnhead, P. & Eckley, I. A. (2012). *Optimal Detection
    of Changepoints with a Linear Computational Cost.* JASA 107(500), 1590–1598.
[^stl]: Cleveland, R. B., Cleveland, W. S., McRae, J. E. & Terpenning, I. (1990).
    *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.* Journal of
    Official Statistics 6(1), 3–73.
