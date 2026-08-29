"""What to do about gaps in an array of observations.

Three answers, and which one applies is a property of the algorithm rather than
of the data. A model that reads a whole row at once — PCA, a regression on
several columns — cannot form an opinion about a row with a hole in it, so it
works on the rows that are complete and reports NaN for the rest: that is
:func:`complete_rows`. An algorithm that needs an unbroken sequence to run at
all — an FFT, a seasonal decomposition, a change-point search — has no such
option, so the gaps are bridged before it runs and its answers there are marked
unknown afterwards: that is :func:`fill_gaps`.

The third answer is for the one array that is not a set of observations at all.
A label is a claim that something happened, so a gap in one is not an unknown to
be worked around but a claim nobody made: a missing label is not an alarm. That
is :func:`as_flags`, and every part of the library that reads labels — the
converters, the metrics — reads them through it, so that the rule is stated
once.

Keeping all three here, rather than beside the dataframe boundary in
:mod:`hazure._core.series`, keeps them findable together and makes the choice
between them an explicit one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["as_flags", "complete_rows", "fill_gaps"]


def complete_rows(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Mark rows with no missing value, the only rows a row-wise model can use.

    Parameters
    ----------
    values
        2-D array of observations.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``len(values)``.
    """
    usable: NDArray[np.bool_] = np.asarray(
        ~np.isnan(values).any(axis=1), dtype=np.bool_
    )
    return usable


def fill_gaps(
    column: NDArray[np.float64], missing: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Replace missing observations by linear interpolation between neighbours.

    Parameters
    ----------
    column
        1-D array, not entirely missing.
    missing
        Where ``column`` is NaN.

    Returns
    -------
    numpy.ndarray
        A copy with no missing values. Gaps at either end are held flat at the
        nearest observation, which is what :func:`numpy.interp` does outside its
        range.
    """
    if not bool(missing.any()):
        return column
    positions = np.arange(column.shape[0], dtype=np.float64)
    filled = column.copy()
    filled[missing] = np.interp(
        positions[missing], positions[~missing], column[~missing]
    )
    return filled


def as_flags(values: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Read a column of labels as booleans, treating NaN as not anomalous.

    Parameters
    ----------
    values
        1-D array of labels. Values outside ``[0, 1]`` are clipped into it, so a
        label of 2 is as anomalous as a 1 and a -1 is as ordinary as a 0.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``len(values)``, true exactly where the clipped
        label is 1. Anything strictly between the two — a score of 0.7, say — is
        not a label at all, and counts as not anomalous rather than being
        rounded into a claim nobody made.
    """
    flags: NDArray[np.bool_] = np.asarray(
        np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0) == 1.0
    )
    return flags
