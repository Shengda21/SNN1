"""Metrics and cluster bootstrap."""
import numpy as np


def nmse(pred, Y, scale):
    """Normalized MSE identical to w^T C w for a single predictor:
    mean over rows of (1/d) sum_l ((pred - y)_l / s_l)^2."""
    r = (pred - Y) / scale
    return float(np.mean(np.sum(r ** 2, 1) / Y.shape[1]))


def nmse_rows(pred, Y, scale):
    r = (pred - Y) / scale
    return np.sum(r ** 2, 1) / Y.shape[1]


def r2_pooled(pred, Y):
    y = Y.ravel(); p = pred.ravel()
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def per_joint_rmse(pred, Y):
    return np.sqrt(np.mean((pred - Y) ** 2, 0))


def per_joint_r2(pred, Y):
    return 1.0 - np.sum((pred - Y) ** 2, 0) / np.sum((Y - Y.mean(0)) ** 2, 0)


def cluster_bootstrap(values, clusters, n_boot=1000, seed=0, stat=np.mean):
    """values: [N] per-unit values; clusters: dict(level -> [N] ids). Resamples
    every clustering level independently (multi-way cluster bootstrap), returning
    (point, lo, hi) with 95% percentile interval. Shared seed -> shared indices
    across methods, so paired differences are obtained by passing differences."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    levels = list(clusters.values())
    uniq = [np.unique(l) for l in levels]
    point = float(stat(values))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        wts = np.ones(len(values))
        for l, u in zip(levels, uniq):
            draw = rng.choice(len(u), len(u), replace=True)
            counts = np.bincount(draw, minlength=len(u))
            wts *= counts[np.searchsorted(u, l)]
        if wts.sum() == 0:
            boots[b] = point; continue
        boots[b] = float(np.sum(wts * values) / np.sum(wts)) if stat is np.mean else stat(np.repeat(values, wts.astype(int)))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
