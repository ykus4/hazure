"""Ready-made detectors, one per kind of anomaly.

Each function here returns a :class:`~hazure.Detector` — a scorer and a threshold
chosen together for one phenomenon — so detection is one call::

    >>> import numpy as np, pandas as pd
    >>> from hazure import detectors
    >>> index = pd.date_range("2024-01-01", periods=200, freq="h")
    >>> values = np.zeros(200)
    >>> values[120] = 9.0
    >>> flags = detectors.spike(window=24).fit_detect(pd.Series(values, index=index))
    >>> bool(flags.idxmax() == index[120])
    True

and what comes back is the same class you would build by hand, so it prints what
it is made of, reconfigures through ``set_params`` and stores with ``to_dict``::

    >>> detectors.iqr(factor=1.5)
    Detector(scorer=None, threshold=IqrThreshold(factor=1.5))

Pick by what "anomalous" means for your series:

=================================== ==============================================
The anomaly is                      Reach for
=================================== ==============================================
a value outside its usual range     :func:`iqr`, :func:`quantile`, :func:`esd`,
                                    :func:`limits`
a **spike** — local and temporary   :func:`spike`, :func:`hampel`
a **level shift** — permanent       :func:`level_shift`
a **volatility shift**              :func:`volatility_shift`
a break in a seasonal pattern       :func:`seasonal`, :func:`stl`, :func:`mstl`
a break in the series' own dynamics :func:`autoregression`
a departure from a drifting range   :func:`rolling_quantile`
something nobody has characterised  :func:`spectral_residual`
a change of regime                  :func:`pelt`, :func:`ruptures`
an unusual *shape*                  :func:`matrix_profile`, :func:`damp`
columns that stop agreeing          :func:`regression`, :func:`pca`,
                                    :func:`outlier`, :func:`min_cluster`
=================================== ==============================================

Labels are ``1.0`` anomalous, ``0.0`` normal and ``NaN`` unknown. NaN is common
and meaningful: a window-based detector cannot judge the first few observations,
and saying so is more useful than calling them normal.
"""

from __future__ import annotations

from hazure.detectors.changes import level_shift, spike, volatility_shift
from hazure.detectors.local import hampel, rolling_quantile
from hazure.detectors.patterns import (
    autoregression,
    mstl,
    seasonal,
    spectral_residual,
    stl,
)
from hazure.detectors.relations import min_cluster, outlier, pca, regression
from hazure.detectors.segments import pelt, ruptures
from hazure.detectors.shapes import damp, matrix_profile
from hazure.detectors.values import esd, iqr, limits, quantile

__all__ = [
    "autoregression",
    "damp",
    "esd",
    "hampel",
    "iqr",
    "level_shift",
    "limits",
    "matrix_profile",
    "min_cluster",
    "mstl",
    "outlier",
    "pca",
    "pelt",
    "quantile",
    "regression",
    "rolling_quantile",
    "ruptures",
    "seasonal",
    "spectral_residual",
    "spike",
    "stl",
    "volatility_shift",
]
