"""Time-ordered train/test folds, so a model never trains on its future."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hazure import TimeSeries

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "split_train_test",
]


_MODES = (1, 2, 3, 4)


# Positions of one fold: ((train_start, train_stop), (test_start, test_stop)).
_Fold = tuple[tuple[int, int], tuple[int, int]]


def split_train_test(
    data: Any,
    mode: int = 1,
    n_splits: int = 1,
    train_ratio: float = 0.7,
) -> list[tuple[Any, Any]]:
    """Split a time series into ``n_splits`` (train, test) folds.

    Parameters
    ----------
    data
        Anything :meth:`hazure.TimeSeries.from_any` accepts. The slices come
        back in the same flavour.
    mode
        Which of the four splitting schemes to use, 1 to 4. See Notes.
    n_splits
        Number of folds to produce, at least 1.
    train_ratio
        Fraction of each fold given to training, strictly between 0 and 1. Modes
        3 and 4 derive their cut points from ``n_splits`` alone and ignore it.

    Returns
    -------
    list of tuple
        ``n_splits`` ``(train, test)`` pairs of native objects, in time order.

    Raises
    ------
    TypeError
        ``data`` is not a recognised dataframe, or an argument has the wrong
        type.
    ValueError
        ``mode`` is not 1-4, ``n_splits`` is below 1, ``train_ratio`` is not
        strictly between 0 and 1, or the requested split would leave a fold with
        an empty training or testing side.

    Notes
    -----
    In the diagrams below, ``1`` marks a training position, ``2`` a testing
    position and ``0`` a position that fold does not use. Each row is one fold.
    All four show a 40-sample series with ``n_splits=4, train_ratio=0.7``.

    **Mode 1** — ``n_splits`` equal disjoint folds, each cut at
    ``train_ratio``. Every observation is used exactly once, so the folds are
    independent, but a later fold trains on no more history than an earlier
    one::

        1111111222000000000000000000000000000000
        0000000000111111122200000000000000000000
        0000000000000000000011111112220000000000
        0000000000000000000000000000001111111222

    **Mode 2** — nested folds, all starting at the first observation, each cut
    at ``train_ratio``. Fold *k* covers the first ``k / n_splits`` of the
    series, so training and testing both grow::

        1111111222000000000000000000000000000000
        1111111111111122222200000000000000000000
        1111111111111111111112222222220000000000
        1111111111111111111111111111222222222222

    **Mode 3** — an expanding training window with a fixed-size test block
    appended. Every fold tests on the same number of observations, which makes
    scores directly comparable across folds::

        1111111122222222000000000000000000000000
        1111111111111111222222220000000000000000
        1111111111111111111111112222222200000000
        1111111111111111111111111111111122222222

    **Mode 4** — an expanding training window tested against the whole
    remainder. This is the "what would I have known at time *t*" question, at
    the cost of test sets of unequal size::

        1111111122222222222222222222222222222222
        1111111111111111222222222222222222222222
        1111111111111111111111112222222222222222
        1111111111111111111111111111111122222222

    Examples
    --------
    >>> import pandas as pd
    >>> index = pd.date_range("2024-01-01", periods=10, freq="D")
    >>> data = pd.DataFrame({"x": range(10)}, index=index)
    >>> [(len(train), len(test)) for train, test in split_train_test(data)]
    [(7, 3)]
    >>> [
    ...     (len(train), len(test))
    ...     for train, test in split_train_test(data, mode=4, n_splits=2)
    ... ]
    [(3, 7), (6, 4)]
    """
    _check_types(mode, n_splits, train_ratio)
    if mode not in _MODES:
        msg = f"mode must be one of {list(_MODES)}, got {mode}."
        raise ValueError(msg)
    if n_splits < 1:
        msg = f"n_splits must be at least 1, got {n_splits}."
        raise ValueError(msg)
    if not 0.0 < train_ratio < 1.0:
        msg = (
            f"train_ratio must be strictly between 0 and 1, got {train_ratio}; "
            f"both ends are excluded because an empty train or test set cannot "
            f"be scored."
        )
        raise ValueError(msg)

    ts = TimeSeries.from_any(data)
    n_rows = ts.n_rows
    # Modes 3 and 4 hold back one extra block, which becomes the first fold's
    # test set. Floor division rather than rounding, so the cumulative cuts can
    # never run past the end of the series.
    fold_len = n_rows // (n_splits if mode in (1, 2) else n_splits + 1)

    folds = [
        _clamp(fold, n_rows)
        for fold in _cuts(mode, n_rows, n_splits, train_ratio, fold_len)
    ]
    for position, ((train_start, train_stop), (test_start, test_stop)) in enumerate(
        folds
    ):
        if train_stop - train_start < 1 or test_stop - test_start < 1:
            msg = (
                f"Fold {position} would be empty on one side "
                f"({train_stop - train_start} training rows, "
                f"{test_stop - test_start} testing rows) for {n_rows} "
                f"observations with mode={mode}, n_splits={n_splits}, "
                f"train_ratio={train_ratio}. Lower n_splits, or move "
                f"train_ratio closer to 0.5."
            )
            raise ValueError(msg)

    return [(_slice(ts, *train), _slice(ts, *test)) for train, test in folds]


def _check_types(mode: Any, n_splits: Any, train_ratio: Any) -> None:
    """Reject wrong argument types before any arithmetic happens."""
    for name, value in (("mode", mode), ("n_splits", n_splits)):
        if not isinstance(value, int) or isinstance(value, bool):
            msg = f"{name} must be an int, got {type(value).__name__}."
            raise TypeError(msg)
    if not isinstance(train_ratio, float | int) or isinstance(train_ratio, bool):
        msg = f"train_ratio must be a number, got {type(train_ratio).__name__}."
        raise TypeError(msg)


def _cuts(
    mode: int,
    n_rows: int,
    n_splits: int,
    train_ratio: float,
    fold_len: int,
) -> Iterator[_Fold]:
    """Yield the positions of each fold.

    The four modes differ only in this arithmetic, so keeping the slicing out of
    here leaves each one a single readable expression.
    """
    if mode == 1:
        start = 0
        for _ in range(n_splits - 1):
            cut = start + round(fold_len * train_ratio)
            yield (start, cut), (cut, start + fold_len)
            start += fold_len
        # The last fold absorbs the division remainder, so nothing is dropped.
        cut = start + round((n_rows - start) * train_ratio)
        yield (start, cut), (cut, n_rows)
    elif mode == 2:
        for k in range(n_splits - 1):
            stop = fold_len * (k + 1)
            cut = round(stop * train_ratio)
            yield (0, cut), (cut, stop)
        cut = round(n_rows * train_ratio)
        yield (0, cut), (cut, n_rows)
    elif mode == 3:
        for k in range(n_splits - 1):
            cut = (k + 1) * fold_len
            yield (0, cut), (cut, cut + fold_len)
        # The final test block also absorbs the remainder.
        cut = n_splits * fold_len
        yield (0, cut), (cut, n_rows)
    else:
        for k in range(n_splits):
            cut = (k + 1) * fold_len
            yield (0, cut), (cut, n_rows)


def _clamp(fold: _Fold, n_rows: int) -> _Fold:
    """Hold both slices inside ``[0, n_rows]`` so their lengths are honest."""
    (train_start, train_stop), (test_start, test_stop) = fold
    return (
        (min(train_start, n_rows), min(train_stop, n_rows)),
        (min(test_start, n_rows), min(test_stop, n_rows)),
    )


def _slice(ts: TimeSeries, start: int, stop: int) -> Any:
    """Return rows ``[start, stop)`` as a native object.

    ``freq`` is carried across rather than re-inferred: a positional slice of a
    regular series is still regular, even when it is now too short for the
    inference rule to say so on its own.
    """
    return TimeSeries(
        time=ts.time[start:stop],
        values=ts.values[start:stop],
        columns=ts.columns,
        freq=ts.freq,
        origin=ts.origin,
    ).to_native()
