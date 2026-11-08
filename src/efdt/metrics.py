"""Multi-label metrics and validation-only threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score, average_precision_score


def _macro_valid_metric(y_true: np.ndarray, y_score: np.ndarray, metric) -> float:
    values = []
    for k in range(y_true.shape[1]):
        if np.unique(y_true[:, k]).size < 2:
            continue
        try:
            values.append(float(metric(y_true[:, k], y_score[:, k])))
        except ValueError:
            continue
    return float(np.mean(values)) if values else float("nan")


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Choose one F1-maximizing threshold per label using validation data only."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_prob must have identical (N,K) shapes")
    grid = np.linspace(0.05, 0.95, 91)
    thresholds = np.full(y_true.shape[1], 0.5, dtype=float)
    for k in range(y_true.shape[1]):
        if not np.any(y_true[:, k] == 1) or not np.any(y_true[:, k] == 0):
            continue
        scores = [f1_score(y_true[:, k], y_prob[:, k] >= t, zero_division=0) for t in grid]
        thresholds[k] = float(grid[int(np.argmax(scores))])
    return thresholds


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None) -> dict[str, float | list[float]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_prob must have identical (N,K) shapes")
    thresholds = np.asarray(thresholds if thresholds is not None else np.full(y_true.shape[1], 0.5))
    pred = (y_prob >= thresholds[None, :]).astype(int)
    per_f1 = f1_score(y_true, pred, average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_f1)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_auroc": _macro_valid_metric(y_true, y_prob, roc_auc_score),
        "macro_auprc": _macro_valid_metric(y_true, y_prob, average_precision_score),
        "hamming_loss": float(hamming_loss(y_true, pred)),
        "per_class_f1": [float(v) for v in per_f1],
        "thresholds": [float(v) for v in thresholds],
    }
