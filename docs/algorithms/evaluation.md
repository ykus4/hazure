# Events and metrics

A detector emits one label per sample. An operator thinks in outages. Most of the
awkwardness in evaluating anomaly detection lives in the gap between those two
views, so it is worth being exact about how this library crosses it.

## What an event is

`Events` is a sorted, non-overlapping set of **closed** intervals $[a, b]$ in
UTC nanoseconds. Closed at both ends means an instant is representable as
$[t, t]$, and it fixes the duration as

$$
d = b - a + 1 \ \text{ns}
$$

The $+1$ looks like an off-by-one and is not. Integer nanosecond bounds are
inclusive, so $[a, b]$ occupies the continuous half-open span $[a, b+1)$: an
instant lasts one nanosecond, the finest representable moment, rather than zero
time. The payoff is that an event covering $k$ consecutive samples spaced
$\Delta$ apart lasts exactly $k\Delta$ — the duration ratio and the sample count
agree, so an event-based metric and a point-based one measure the same thing.

Two intervals whose gap is exactly 1 ns are **fused**, for the same reason: a
1 ns gap is an artefact of inclusive endpoints, not a moment of normality.
Overlapping and nested intervals collapse too, so the invariants always hold —
sorted, disjoint, non-touching. Union and intersection are boundary sweeps that
re-normalise through the same path.

## Labels to intervals and back

`to_events` reads a label series as intervals. A label counts as anomalous when
it clips to exactly 1 — so `NaN` is **not** anomalous, and neither is `0.5`.
Unknown never invents an event.

What each flagged sample becomes depends on whether the axis is regular:

$$
t \;\longmapsto\;
\begin{cases}
[\,t,\ t + \Delta - 1\,] & \text{period semantics (regular axis)}\\
[\,t,\ t\,] & \text{instant semantics}
\end{cases}
$$

Period semantics is what you want, and what you get by default when the sampling
interval $\Delta$ can be inferred: a flagged sample **stands for the interval it
opens**. Six consecutive hourly flags then become one six-hour event rather than
six one-hour ones, because consecutive intervals land exactly 1 ns apart and
fuse. Under instant semantics the same six flags stay six events of 1 ns each.

$\Delta$ is inferred only from an axis of at least three samples with a constant
step; otherwise it is `None` and instant semantics apply silently. `as_periods=`
overrides the inference in either direction, and `as_periods=True` on an
irregular axis raises rather than guessing.

`to_labels` goes back, marking every sample whose own period overlaps an event.
Bounds falling between samples are snapped **outwards** to the samples they
touch, so the round trip is lossless for events that `to_events` could have
produced on that axis, and lossy — necessarily — for arbitrary bounds.

`expand_events(events, before=, after=)` widens each interval and re-merges,
which is how a detection tolerance is expressed. A bare integer margin is
**nanoseconds**; `"1h"`, `timedelta` and `np.timedelta64` all work and are
clearer.

## Point-based metrics

Pass label series and the four metrics count samples. With $T$ and $P$ the sets
of true and predicted anomalous samples:

$$
\text{recall} = \frac{|T \cap P|}{|T|}, \qquad
\text{precision} = \frac{|T \cap P|}{|P|}, \qquad
\text{IoU} = \frac{|T \cap P|}{|T \cup P|}
$$

$$
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

An empty denominator gives `NaN`, not zero — no true anomalies means recall is
undefined, and reporting `0.0` would let it be averaged into a summary as if it
were a failure.

Note the consequence of `NaN` labels reading as not-anomalous: an unknown region
never invents a detection, but it never counts as a correct negative either. It
simply does not appear in $T$ or $P$.

## Event-based metrics

Pass `Events` and the same functions switch to intervals. An event counts as
detected when the *fraction of its duration* covered by the other set reaches
`thresh`:

$$
\text{recall} = \frac{\#\bigl\{\,i \;:\; \operatorname{dur}(E_i \cap P) \ \ge\ \theta \cdot \operatorname{dur}(E_i)\,\bigr\}}{\#\{E_i\}}
$$

$$
\text{precision} = \frac{\#\bigl\{\,j \;:\; \operatorname{dur}(F_j \cap T) \ \ge\ \theta \cdot \operatorname{dur}(F_j)\,\bigr\}}{\#\{F_j\}}
$$

with $E$ the true events and $F$ the predicted ones. The criterion is applied to
the event **being scored** — to true events for recall, to predicted events for
precision — never to both at once. $\theta$ defaults to `0.5` and must lie in
$(0, 1]$; since it is strictly positive and every event lasts at least 1 ns, an
uncovered event can never pass.

$$
\text{IoU} = \frac{\operatorname{dur}(T \cap P)}{\operatorname{dur}(T \cup P)}
$$

IoU is the one metric with the same definition in both modes, which makes it the
one to compare across them.

This is usually the mode that matters. An alert firing part-way through a
six-hour outage is one page and one investigation — a hit, not a fraction of one.
Conversely a detector that fires for one minute in the middle of that outage
scores full recall on the event and would score $1/360$ on the samples. Which of
those is the honest number depends on what happens next: if the alert wakes
someone up, event-based is right; if the labels feed a downstream aggregate,
point-based is.

`f1_score` takes **two** thresholds, `recall_thresh` and `precision_thresh`,
because the two questions are genuinely different. How much of an outage must be
covered before you call it caught, and how much of an alert must be real before
you call it justified, need not be the same number.

The failure mode to watch for is comparing shapes that are not comparable. A
change-point detector produces short intervals around transitions; scored
against interval-shaped ground truth it will show near-zero recall while working
perfectly. [The guide works through the
example](../guide.md#a-shift-detector-reports-the-change-point-not-the-anomalous-interval);
the fix is to reduce the truth to change points and allow a tolerance with
`expand_events`.

## Time-ordered splitting

`split_train_test` builds folds that never train on the future. Shuffled
$k$-fold would leak: a model fitted on Wednesday and validated on Tuesday has
seen the answer.

Four layouts, chosen by `mode`, with $L$ the fold length:

| `mode` | Train | Test | Shape |
| --- | --- | --- | --- |
| 1 | first $\theta$ of each block | rest of that block | disjoint consecutive blocks |
| 2 | $[0, \theta u_k)$ | $[\theta u_k, u_k)$ | nested, all starting at the beginning |
| 3 | $[0, c_k)$ | next $L$ samples | expanding window, fixed test block |
| 4 | $[0, c_k)$ | everything after | expanding window, test to the end |

with $u_k = (k+1)L$ and $c_k = (k+1)L$. Modes 1 and 2 divide by `n_splits`;
modes 3 and 4 divide by `n_splits + 1`, since the last fold needs data left over
to test on. `train_ratio` applies to modes 1 and 2 only — in 3 and 4 the split
point *is* the fold boundary. The final fold absorbs the remainder rather than
dropping it, and every fold is checked to be non-empty on both sides.

Mode 3 is the usual choice for "does this still work as more history
accumulates". Mode 1 is the one to reach for when the series is long enough that
distant history is not representative.

## Combining verdicts

An `Aggregator` reduces several label series to one, and the interesting part is
what it does with `NaN`. Unknown is a third state, and the combination rules are
Kleene's three-valued logic:

| | `OrAggregator` | `AndAggregator` |
| --- | --- | --- |
| any input is `1` | **1** | `NaN` if any unknown, else 1 |
| any input is `0` | `NaN` if any unknown, else 0 | **0** |
| all unknown | `NaN` | `NaN` |

The asymmetry is the point. For "or", one confident detection settles it —
whatever the others could not say cannot make it *less* true. For "and", one
confident *normal* settles it. Only where the missing verdict could change the
answer does the result become unknown. Anything non-zero and known counts as
anomalous, so these compose with scores as well as with labels.

`VoteAggregator` takes a fraction of the **known** inputs:

$$
y_t = \mathbb{1}\!\left[\ \frac{\#\{j : x_{tj} \text{ anomalous}\}}{\#\{j : x_{tj} \ne \mathrm{NaN}\}} \ \ge\ \theta\ \right]
$$

Unknowns abstain rather than voting against — a detector still inside its warm-up
window should not drag the vote down. The comparison is non-strict, so the
default $\theta = 0.5$ is a simple majority with ties counting as anomalous, and
a row where every input is unknown is `NaN` rather than a division by zero.

Inputs are outer-joined on the time axis first, so a timestamp missing from one
input arrives as unknown, and detectors on different but overlapping axes combine
without alignment work.
