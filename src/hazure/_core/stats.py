"""Constants of robust statistics, shared by scorers and thresholds alike."""

from __future__ import annotations

from typing import Final

__all__ = ["MAD_SCALE"]

#: Scale factor that turns a median absolute deviation into an estimate of the
#: standard deviation of a normal sample. For X ~ N(mu, sigma) the median of
#: |X - mu| is sigma * Phi^-1(0.75) = 0.6745 * sigma, so dividing the MAD by that
#: constant — equivalently multiplying by 1 / 0.6745 = 1.4826 — puts the MAD on
#: the same scale as a standard deviation. Without it, ``factor=3`` would mean
#: three MADs, which is only two standard deviations.
MAD_SCALE: Final = 1.482602218505602
