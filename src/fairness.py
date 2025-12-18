from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroupMetrics:
    group_value: Any
    n: int
    positive_rate: float | None
    tpr: float | None
    fpr: float | None


def demographic_parity_by_group(y_pred: np.ndarray, group: np.ndarray, positive_label: int = 1) -> dict[Any, float]:
    """Positive prediction rate per group (demographic parity slice).

    This is a slice metric, not a causal statement.
    """
    rates: dict[Any, float] = {}
    for g in np.unique(group):
        mask = group == g
        if mask.sum() == 0:
            continue
        rates[g] = float(np.mean(y_pred[mask] == positive_label))
    return rates


def tpr_fpr_by_group(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray, positive_label: int = 1) -> dict[Any, tuple[float | None, float | None]]:
    """Compute TPR/FPR per group for binary classification."""
    out: dict[Any, tuple[float | None, float | None]] = {}
    y_true_bin = (y_true == positive_label).astype(int)
    y_pred_bin = (y_pred == positive_label).astype(int)

    for g in np.unique(group):
        mask = group == g
        yt = y_true_bin[mask]
        yp = y_pred_bin[mask]
        if yt.size == 0:
            continue
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        tpr = (tp / (tp + fn)) if (tp + fn) > 0 else None
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else None
        out[g] = (tpr, fpr)

    return out


