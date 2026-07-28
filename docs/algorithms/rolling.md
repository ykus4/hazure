# Rolling comparisons

Most of the detectors in this library are one idea in three costumes: **compare a
point, or a stretch of points, with its own neighbourhood**. A spike is a point
unlike the hour before it. A level shift is an hour unlike the day before it. A
volatility shift is the same comparison made on spread instead of position.

The engine underneath is two public functions, `rolling` and `double_rolling`,
and everything on this page is a way of parameterising them.

## What a window is

`rolling(values, window, agg)` returns $y_i = \operatorname{agg}(W_i)$, where
$W_i$ is the set of **observed** values inside the window at row $i$.

An integer `window` counts samples; a string, `timedelta` or `np.timedelta64`
counts time (`"24h"`, `"7d"`). The distinction matters as soon as the axis has
gaps: `window=24` on hourly data with a missing afternoon reaches back further
than 24 hours, while `window="24h"` always covers exactly a day and simply has
fewer points in it.

By default the window is **trailing and right-closed** — it ends at the current
point and includes it:

$$
W_i = \{x_j : i - w + 1 \le j \le i\}
$$

`closed` moves the endpoints (`"left"` excludes the current point, `"both"`
widens by one, `"neither"` narrows by one), and `center=True` puts the point in
the middle: for odd $w$ the window is symmetric, and for even $w$ the extra
observation goes on the **left**, matching pandas.

Two rules govern what comes out:

- **`NaN` is skipped, not propagated.** A missing observation shrinks $|W_i|$
  rather than poisoning the result.
- **`min_periods` decides reportability.** If $|W_i|$ falls below it the row is
  `NaN`. It defaults to the full window for sample counts — hence the familiar
  $w - 1$ leading `NaN` — and to `1` for durations, whose width legitimately
  varies with the data. A centred integer window splits the margin instead:
  $\lceil (w-1)/2 \rceil$ leading, $\lfloor (w-1)/2 \rfloor$ trailing.

### The aggregations

`AGGREGATIONS` holds fifteen names. Most are what you would expect; these are the
ones with a choice buried in them:

| Name | Computes | Note |
| --- | --- | --- |
| `median` | $Q(0.5)$ | linear interpolation between order statistics |
| `std`, `var` | $s^2 = \frac{1}{n-1}\sum (x - \bar{x})^2$ | **ddof = 1**; a window of one observation gives `NaN` |
| `quantile` | $Q(q)$ | needs `q`; linear interpolation |
| `iqr` | $Q(0.75) - Q(0.25)$ | |
| `idr` | $Q(0.90) - Q(0.10)$ | wider, and less sensitive to a single extreme than `std` |
| `count` | $n_i$ | `NaN` below `min_periods`, unlike pandas, which reports the count anyway |
| `nnz`, `nunique` | non-zero and distinct counts | `nunique` compares exact floats |

`skew` and `kurt` are the bias-corrected sample statistics — the same ones pandas
reports. With central moments $m_k = \frac{1}{n}\sum (x - \bar{x})^k$:

$$
G_1 = \frac{\sqrt{n(n-1)}}{n-2}\cdot\frac{m_3}{m_2^{3/2}},
\qquad
G_2 = \frac{(n^2-1)\,m_4/m_2^{2} - 3(n-1)^2}{(n-2)(n-3)}
$$

$G_1$ needs $n \ge 3$, $G_2$ needs $n \ge 4$, and both are `NaN` on a window with
no spread rather than dividing by zero.

## Two windows, one boundary

`double_rolling` is where the detectors actually live. It summarises the stretch
**before** each point and the stretch **from** each point, and compares the two:

$$
L_i = \operatorname{agg}\bigl(\{x_j : i - w_L \le j \le i-1\}\bigr), \qquad
R_i = \operatorname{agg}\bigl(\{x_j : i \le j \le i + w_R - 1\}\bigr)
$$

The left window is left-closed and the right window right-closed, so the two
**partition** the neighbourhood at the boundary just before row $i$, with no
observation counted twice and none skipped. The current point belongs to the
right — the future — side. That is the convention that makes the score mean
"the series changed *here*".

Both windows, both aggregations and both `min_periods` can be given as a
`(left, right)` pair.

The comparison is chosen by `diff`:

| `diff` | Score | Signed |
| --- | --- | --- |
| `"l1"` (default), `"l2"` | $\lvert R_i - L_i \rvert$ | no |
| `"diff"` | $R_i - L_i$ | yes — positive means the future side is larger |
| `"rel_diff"` | $\dfrac{R_i - L_i}{L_i}$ | yes |
| `"abs_rel_diff"` | $\left\lvert \dfrac{R_i - L_i}{L_i} \right\rvert$ | no |

The relative forms divide by the **past** aggregate alone, not by a pooled
scale. That is what makes a doubling of noise score the same on a quiet series
and a loud one — and it is also the caveat: on a series that crosses zero, $L_i$
near zero sends the ratio to $\pm\infty$, and $L_i$ exactly zero with $R_i$ zero
gives `NaN`. Relative comparisons are for positive quantities.

Structurally the first $w_L$ rows and the last $w_R - 1$ rows are `NaN`: neither
end has both windows.

## The three detectors this produces

Each is the same scorer with a different shape of window, thresholded by a
one-sided IQR fence on the magnitude ([why one-sided](thresholds.md#sides)):

| Detector | Windows | `agg` | `diff` | Default `factor` |
| --- | --- | --- | --- | --- |
| `SpikeDetector` | $(w,\ 1)$ | `"median"` | `"diff"` | 3.0 |
| `LevelShiftDetector` | $(w,\ w)$ | `"median"` | `"diff"` | 6.0 |
| `VolatilityShiftDetector` | $(w,\ w)$ | `"std"` | `"rel_diff"` | 6.0 |

**`SpikeDetector`** has a right window of exactly one observation, so $R_i = x_i$
and the score collapses to

$$
s_i = x_i - \operatorname{median}(x_{i-w}, \dots, x_{i-1})
$$

— this point against the recent past. A slow drift is invisible to it, because
the reference median drifts along with the series. That is a feature: whatever
the level was yesterday, it is the last hour that decides whether now is a spike.

**`LevelShiftDetector`** makes both windows long. A single odd value cannot move
a wide median, so the score only rises when the two windows genuinely straddle
two different levels. Note what it therefore reports: the moment of change, not
the stretch that follows it — [the first of the two surprises in the
guide](../guide.md#a-shift-detector-reports-the-change-point-not-the-anomalous-interval).

**`VolatilityShiftDetector`** measures spread instead of position, relatively.
`agg` may be any of `std`, `var`, `iqr`, `idr`; the last two are more resistant if
the noisy regime also contains outliers. Note the asymmetry that a ratio carries:
a relative *increase* is unbounded above, while a relative *decrease* is bounded
below by $-1$. A halving of the noise can therefore never score as strongly as a
doubling, and with `side="both"` and a fence fitted on magnitudes, quietening is
systematically harder to detect than getting louder.

Why `median` is the default for the first two and `std` for the third: the whole
point of the comparison is that one window contains the anomaly. A mean would
absorb it and shrink the very difference being measured, while a median resists
it until the anomaly makes up half the window. For volatility, absorbing it *is*
the measurement.

## Choosing the window

The window is a statement about the timescale of the thing you are looking for,
and there are only really three constraints:

1. **Long enough for the aggregate to be stable.** A median over five points
   moves around on its own; the score then has to clear a fence fitted on that
   noise.
2. **Short enough that the reference is still "normal".** Once a window spans a
   daily cycle, the comparison is between different times of day rather than
   between normal and abnormal. On seasonal data either use a window that is an
   exact multiple of the period, or take the seasonality out first with
   `SeasonalDecomposition`.
3. **No longer than the anomaly you will tolerate missing.** A level shift
   detector with a 7-day window cannot resolve two changes three days apart; it
   sees one blurred transition.

And a shift detector cannot localise a change more finely than the window it
used, which is why evaluating one needs a matching tolerance —
`expand_events(truth, before=w, after=w)`.
