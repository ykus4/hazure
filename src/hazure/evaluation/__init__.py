"""Judging a model: metrics, and the folds to compute them over.

Each metric accepts either label series (scored per sample) or
:class:`~hazure.events.Events` (scored per interval), and a dict of either to score
several anomaly types at once.

:func:`split_train_test` builds time-ordered folds. Shuffling is not an option for
a time series, so the four schemes it offers differ in how the training window
grows and where the test block sits.
"""

from __future__ import annotations

from hazure.evaluation._metrics import f1_score, iou, precision, recall
from hazure.evaluation._split import split_train_test

__all__ = ["f1_score", "iou", "precision", "recall", "split_train_test"]
