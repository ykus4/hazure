# Thresholds

A threshold answers one question — *is this score unusual enough to report* —
and answers it the same way whatever produced the score. That is why it is a
separate object: replacing `SeasonalResidualScorer` with `HampelScorer` does not
change what "unusual enough" means, and should not require rewriting it.

Every rule below reduces to a pair of fences $\ell$ and $u$ learned from the
scores it was fitted on. The labelling step is shared:

$$
y_t =
\begin{cases}
\mathrm{NaN} & s_t = \mathrm{NaN}\\
1 & s_t > u \ \text{ or } \ s_t < \ell\\
0 & \text{otherwise}
\end{cases}
$$

Both comparisons are **strict**, so the normal region is the closed interval
$[\ell, u]$ and a score sitting exactly on a fence is normal. A missing fence is
$\pm\infty$ rather than a special case, so a one-sided rule needs no separate
code path. If every training score was missing, both fences are `NaN` and *every*
label is `NaN` — an unfitted fence never silently passes everything.

## FixedThreshold

The one rule that learns nothing: $\ell$ and $u$ are the numbers you gave it.
`trainable = False`, so it works without `fit`, and no amount of contaminated
data can move it. Passing neither bound raises — a threshold that can flag
nothing is a mistake, not a configuration.

Use it when the limits come from outside the data: an SLO, a sensor range, a
saturation point, or a score that is already in interpretable units (which is
exactly why `HampelDetector` uses it).

## QuantileThreshold

$$
\ell = Q(\alpha_\ell), \qquad u = Q(\alpha_u)
$$

with $Q$ the empirical quantile function of the fitted scores, evaluated by
linear interpolation between order statistics (numpy's default, type 7). The
parameters are **levels in $[0, 1]$**, not values.

Its defining property is also its catch: `QuantileThreshold(high=0.99)` puts 1 %
of the *training* scores above the fence by construction, whatever the data looks
like. On a series with no anomalies in it at all, it still cuts at the 99th
percentile. It is a rule for "show me the worst 1 %", not for "show me the ones
that are wrong" — right when a fixed alert budget matters more than fidelity.

## IqrThreshold

Tukey's box-plot rule. With $Q_1$, $Q_3$ the quartiles of the fitted scores and
$\mathrm{IQR} = Q_3 - Q_1$:

$$
\ell = Q_1 - f_\ell \cdot \mathrm{IQR}, \qquad u = Q_3 + f_u \cdot \mathrm{IQR}
$$

The two fences hang off *different* centres — the lower off $Q_1$, the upper off
$Q_3$ — so the normal region contains the middle half of the training scores no
matter how small $f$ is.

`factor` is one number for both tails or a `(low, high)` pair, and `None` on a
side leaves that side unbounded. This is how every compound detector in the
library thresholds a magnitude:

```python
IqrThreshold(factor=(None, 3.0))  # -inf below, Q3 + 3 IQR above
```

The default is `3.0` rather than Tukey's conventional `1.5` because what gets
thresholded here is usually already a residual: the easy structure has been
subtracted, and what remains has heavier tails than the raw data did.

Why quartiles rather than a mean and a standard deviation — the quantity wanted
is *the spread of the normal points*, and the sample standard deviation of a
series containing a large outlier is dominated by that outlier. The fence would
widen to admit exactly the point it was meant to exclude.

## MadThreshold

The same idea, one step more resistant. With $m = \operatorname{median}(s)$:

$$
\mathrm{MAD} = \operatorname{median}\bigl(|s - m|\bigr), \qquad
\hat{\sigma} = 1.4826 \cdot \mathrm{MAD}
$$

$$
\ell = m - f_\ell\,\hat{\sigma}, \qquad u = m + f_u\,\hat{\sigma}
$$

Both fences share one centre here, so the region is symmetric about the median.

The constant is not arbitrary. For $X \sim \mathcal{N}(\mu, \sigma^2)$,

$$
\operatorname{median}|X - \mu| = \sigma\,\Phi^{-1}(0.75) \approx 0.6745\,\sigma
\qquad\Longrightarrow\qquad
\frac{1}{\Phi^{-1}(0.75)} \approx 1.4826
$$

is precisely the factor that makes $\hat\sigma$ unbiased for $\sigma$ on normal
data. It is exported as `MAD_SCALE`. So `factor=3.0` means "three standard
deviations" when the data happens to be normal, and keeps meaning "three robust
spreads" when it does not.

## What contamination does to a fitted fence

A fitted rule estimates normality from a sample that contains the anomalies. The
**breakdown point** is the fraction of that sample that has to be corrupted
before the estimate can be dragged arbitrarily far:

| Estimator | Breakdown point |
| --- | --- |
| mean, standard deviation | $1/n$ — one bad point is enough |
| quartiles, $\mathrm{IQR}$ | $25\%$ |
| median, $\mathrm{MAD}$ | $50\%$ |

Below that fraction the estimate is *resistant*: the outliers sit beyond the
quantiles being read, so they cannot move them. Above it, a quantile lands inside
the anomalous population, the spread estimate inflates to span both regimes, and
the fence stops separating them. `IqrDetector` flagging nothing once 40 % of a
series has shifted is [this arithmetic working
correctly](../guide.md#an-inter-quartile-fence-widens-as-the-fraction-of-unusual-scores-grows),
not a failure: no rule of the form "unusual relative to this sample" can call the
majority of the sample unusual.

The ways out, in order of preference — fit on a clean period and `detect` on the
suspect one; use a `FixedThreshold`, which has no breakdown point because it
estimates nothing; or ask a local question (`SpikeDetector`,
`LevelShiftDetector`), where the comparison population is a neighbouring window
rather than the whole series.

## EsdThreshold

The rules above pick a fence and count what falls outside it. The generalized
extreme Studentized deviate test[^rosner] asks a statistical question instead —
*is the most extreme point too extreme to be the tail of a normal sample of this
size* — and answers it at a stated significance level $\alpha$.

Fitting removes points one at a time. On the surviving set $S$, with $\bar{x}_S$
its mean and $s_S$ its sample standard deviation ($\mathrm{ddof}=1$):

$$
R = \frac{\max_{x \in S}\,|x - \bar{x}_S|}{s_S}
$$

and the critical value at removal step $i$, writing $m = n - i$ for the number of
points still in play and $\nu = m - 1$:

$$
\lambda_i = \frac{m\, t_{p,\nu}}{\sqrt{\bigl(\nu + t_{p,\nu}^{2}\bigr)(m+1)}},
\qquad p = 1 - \frac{\alpha}{2(m+1)}
$$

$t_{p,\nu}$ is the $p$-quantile of Student's $t$ on $\nu$ degrees of freedom. The
$2(m+1)$ in the tail probability is a Bonferroni-style correction: each step
tests the *maximum* of $m$ deviates rather than one deviate fixed in advance, so
the per-comparison level has to shrink with the number of candidates for the
overall level to stay near $\alpha$. If $R > \lambda_i$ the extreme point is
removed and the test repeats on the rest; otherwise iteration stops, and what
remains is the estimated normal population.

Two implementation notes. The scores are sorted once, so the most extreme
survivor is always at one end of a contiguous slice and each step costs $O(1)$ —
the whole fit is $O(n \log n)$. And only sufficient statistics of the survivors
are kept (`count_`, `sum_`, `sum_squares_`, `critical_value_`), never the data.

Scoring a later point $x$ puts it back into that retained population, of size
$N = \texttt{count\_} + 1$, and runs one step of the same test:

$$
\bar{x} = \frac{\texttt{sum\_} + x}{N}, \qquad
s^2 = \frac{\texttt{sum\_squares\_} + x^2 - N\bar{x}^2}{N - 1}, \qquad
y = \mathbb{1}\!\left[\frac{|x - \bar{x}|}{s} > \texttt{critical\_value\_}\right]
$$

That is what makes the test usable out of sample: the fitted state is a
*population* a new observation can be tested against, rather than a frozen fence.

The assumption to check is normality. ESD is a test *about a normal sample*; on a
skewed or bimodal score distribution it will reject perfectly ordinary points
from the long side. It needs $n \ge 2$ to say anything, $\alpha$ strictly inside
$(0,1)$, and the `stats` extra for SciPy's $t$ distribution.

[^rosner]: Rosner, B. (1983). *Percentage Points for a Generalized ESD
    Many-Outlier Procedure.* Technometrics 25(2), 165–172.

## Sides

`side` belongs to the detector rather than the threshold, but it is best
understood here, because it does not change the fence. A signed-score detector
thresholds the **magnitude**, then filters on the **sign**:

$$
y_t =
\begin{cases}
\mathbb{1}\bigl[\,|s_t| > u\,\bigr] & \texttt{side="both"}\\
\mathbb{1}\bigl[\,|s_t| > u\,\bigr]\cdot\mathbb{1}[s_t > 0] & \texttt{side="positive"}\\
\mathbb{1}\bigl[\,|s_t| > u\,\bigr]\cdot\mathbb{1}[s_t < 0] & \texttt{side="negative"}
\end{cases}
$$

$u$ is fitted on $|s_t|$ over *all* training points whatever `side` says, so the
three settings share one cut-off and differ only in which crossings survive.
Detecting only the increases is a filter on one algorithm, not a different
algorithm — and specifically it is **not** the same as fitting a one-sided fence
on the positive scores alone, which would be a different, narrower cut-off.

Gating only ever clears a label to `0`; it never creates one. A `NaN` label stays
`NaN`, because withholding a direction is a statement about scores that exist,
not about scores that are missing.
