from __future__ import annotations

import numpy as np


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((y_pred[valid] - y_true[valid]) ** 2)))


def masked_r2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(bool) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan")
    truth = y_true[valid]
    pred = y_pred[valid]
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    xv = x[valid]
    yv = y[valid]
    xv = xv - np.mean(xv)
    yv = yv - np.mean(yv)
    denom = float(np.sqrt(np.sum(xv * xv) * np.sum(yv * yv)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(xv * yv) / denom)
