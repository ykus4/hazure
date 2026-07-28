# Multivariate scores

A univariate component handed a frame fits one independent copy of itself per
column. A multivariate one cannot: its algorithm needs every column at once,
because the anomaly is a property of the **combination**. Both columns can sit
in their usual range and still be impossible together.

Consequences that follow from that, and hold for every component on this page:

- The verdict is **one column**, named `anomaly` (or `residual`, `error`,
  `min_cluster`, `outlier` for the raw scores), not one per input column.
- The fitted column names and their **order** are remembered. A later frame is
  reordered to match and extra columns are dropped, because coefficients are
  positional and silently transposing two columns would be a wrong answer rather
  than an error.
- A row is usable only if **every** column in it is present. Rows with any
  missing value are excluded from fitting and score `NaN` — there is no partial
  evidence, because the algorithms read a point in $\mathbb{R}^p$.

## Regression residual

Pick one column as the target $y$; the rest are features $x \in \mathbb{R}^{p-1}$
in training order. Fit any regressor with a scikit-learn-shaped
`fit` / `predict` interface, and score the **signed, unstandardised** residual:

$$
s_t = y_t - \hat{f}(x_t)
$$

Positive means the target came in above what its features predicted. That sign is
what `side` filters on, so `RegressionDetector(target="cpu", side="positive")`
means "CPU burning hotter than the traffic explains" and ignores the reverse.

The default regressor, `OrdinaryLeastSquares`, solves least squares with an
explicit intercept column:

$$
\hat{\beta} = \arg\min_{\beta}\ \bigl\lVert [\mathbf{1}\ \ X]\,\beta - y \bigr\rVert_2^2,
\qquad \hat{y} = \beta_0 + x^{\top}\beta_{1:}
$$

via `numpy.linalg.lstsq` with `rcond=None`, so a rank-deficient design — two
perfectly collinear features, say — returns the **minimum-norm** solution rather
than raising. The fit succeeds; the coefficients just are not individually
identified, which does not affect the residual.

`RegressionDetector` pairs the residual with a one-sided IQR fence on its
magnitude, $|s_t| > Q_3(|s|) + 3\,\mathrm{IQR}(|s|)$, then applies `side`.

Anything with `fit(X, y)` and `predict(X)` can be substituted — a gradient
boosting model, an isotonic fit, a physical model wrapped in two methods. The
scorer deep-copies it at fit time, so the object you passed stays unfitted and
reusable.

## PCA reconstruction error

The idea: if $p$ correlated metrics really move along a $k$-dimensional
manifold, then a point that leaves that manifold is anomalous **even when every
coordinate is ordinary**.

Fitting centres the complete training rows and takes the thin SVD:

$$
X_c = X - \mathbf{1}\mu^{\top} = U\Sigma V^{\top}, \qquad
W = \bigl[v_1 \cdots v_k\bigr]^{\top} \in \mathbb{R}^{k \times p}
$$

The rows of $W$ are the $k$ leading right singular vectors — the eigenvectors of
the covariance, in decreasing order of explained variance. Each is sign-flipped
so its largest-magnitude entry is positive, which does not change the subspace
but makes `components_` reproducible instead of leaving the sign to LAPACK.

Projection, reconstruction, and the score:

$$
z_t = W(x_t - \mu), \qquad
\hat{x}_t = \mu + W^{\top}z_t = \mu + W^{\top}W(x_t - \mu)
$$

$$
s_t = \lVert x_t - \hat{x}_t \rVert_2^2
    = \bigl\lVert (I - W^{\top}W)(x_t - \mu) \bigr\rVert_2^2
    = \sum_{j > k} \langle x_t - \mu,\ v_j \rangle^2
$$

$W^{\top}W$ is the orthogonal projector onto the retained subspace, so the score
is the squared length of the part of the centred point that the model has no
room for — the energy in the discarded directions. It is $\ge 0$ by
construction, which is why `PcaDetector` bounds it above only.

Three things to be deliberate about:

**The data is centred but not scaled.** The SVD is of the covariance, not the
correlation, so a column in the hundreds of thousands dominates one in the
single digits and the leading component will simply track it. Where the columns
carry different units, put `StandardScale` upstream and the components describe
correlation structure instead.

**The score is squared, with no square root.** Squaring is monotone, so it does
not change the ranking, but it does change the *distribution* — squaring stretches
the right tail — and the fence is fitted on that distribution. Hence
`PcaDetector`'s default `factor=5.0` where most detectors use `3.0`.

**$k$ is a modelling choice, not a tuning knob.** It says how many degrees of
freedom normal behaviour has. Too large and the residual subspace is empty, so
nothing can look wrong; too small and ordinary variation lands in the residual.
`PcaProjection` and `PcaReconstruction` are exported so the intermediate
quantities can be looked at directly.

### Which column carried the error

The verdict is one column because the anomaly is a property of the combination.
That is right, and it is also unhelpful the moment a flag has to be investigated.
But the squared norm is a sum over coordinates, so the score comes apart
term by term with nothing left over:

$$
s_t = \lVert x_t - \hat{x}_t \rVert_2^2 = \sum_{j=1}^{p} (x_{tj} - \hat{x}_{tj})^2
$$

`PcaColumnError` emits those $p$ terms under the input's own column names, so its
rows sum back to `PcaReconstructionError` to floating point. It is a
decomposition, not an approximation of one: there is no separate model to
disagree with the score being explained, and no residual share to argue about.

Note carefully what the terms are and are not. Each is a coordinate of
$(I - W^{\top}W)(x_t - \mu)$, the part of the point the retained subspace could not
express — so a large term says the model failed to place that column, which is
not the same as that column being wrong. Two columns that normally move together
and now disagree are jointly unplaceable, and the residual vector is orthogonal to
the fitted subspace, which fixes how the error divides between them by geometry
rather than by fault. The split can even put the larger share on the column that
never moved. Naming the culprit inside a pair needs a third measurement the pair
can be checked against, and PCA was given no such thing — it was given the pair.

The centring-not-scaling point above applies with force here, because shares
invite comparison in a way a single score does not: an unscaled column with a
large variance carries more absolute squared error than a small one whatever
either is doing. Across columns in different units, `StandardScale` upstream is
what makes the shares mean anything.

## Clustering and outlier-model adapters

Two thin adapters exist so that anything with a scikit-learn-shaped interface can
be a `hazure` component without `hazure` importing scikit-learn. Both emit a
$\{0, 1\}$ score, so the paired detectors use `FixedThreshold(high=0.5)` — the
"threshold" is a formality; the model already decided.

**`MinClusterScorer`** — cluster the complete training rows, then treat
membership of the smallest cluster as the anomaly:

$$
c^{\star} = \arg\min_{c}\ n_c, \qquad
s_t = \mathbb{1}\bigl[\hat{\ell}(x_t) = c^{\star}\bigr]
$$

with $n_c$ the number of training rows assigned to cluster $c$, ties broken to
the lowest label, and $s_t \equiv 0$ if the model produced only one cluster. Note
what this is **not**: it is not a distance to a centroid and not a density. The
counts are used only to pick which label means "rare"; the score itself is
membership. The model needs a real `predict` — a transductive clusterer, DBSCAN
among them, is rejected at fit time, because scoring later data requires
assigning points the model never saw.

**`OutlierScorer`** — delegate entirely, on the scikit-learn convention that an
outlier model returns $-1$:

$$
s_t = \mathbb{1}\bigl[m(x_t) = -1\bigr]
$$

`IsolationForest`, `OneClassSVM`, `EllipticEnvelope`, `LocalOutlierFactor` all
fit. If the model has `predict` it is fitted once and reused; if it only has
`fit_predict` (LOF in its default mode) it re-judges each batch against itself,
which is the semantics of that estimator rather than a workaround.

Because these read a **label** and not `decision_function`, the underlying
continuous score is discarded. That is the price of the adapter being universal.
Where the ranking matters — and for anomaly detection it usually does — wrap the
model's own score in a `CustomizedTransformer` and threshold it yourself.
