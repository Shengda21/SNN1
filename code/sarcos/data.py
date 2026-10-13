"""SARCOS protocol: official 44,484 training rows -> 35,588 train /
4,448 dev / 4,448 cal by a fixed shuffle; official 4,449 test rows untouched.
Normalization fitted on D_train only. Conditions are applied to standardized
inputs with per-role fixed noise seeds so that every method sees identical
perturbed inputs."""
import json
import os
import numpy as np
from scipy.io import loadmat

SPLIT_SEED = 2025
N_TRAIN, N_DEV, N_CAL = 35588, 4448, 4448
Q_QD_CHANNELS = list(range(14))   # 7 q + 7 qdot (acceleration channels are 14..20)


def _load(path):
    mat = loadmat(path)
    key = [k for k in mat if not k.startswith("_")][0]
    return np.asarray(mat[key], dtype=np.float64)


def load_sarcos(data_dir, manifest_path=None):
    tr = _load(os.path.join(data_dir, "sarcos_inv.mat"))
    te = _load(os.path.join(data_dir, "sarcos_inv_test.mat"))
    assert tr.shape == (44484, 28) and te.shape == (4449, 28)
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(tr))
    idx = {"train": perm[:N_TRAIN], "dev": perm[N_TRAIN:N_TRAIN + N_DEV],
           "cal": perm[N_TRAIN + N_DEV:N_TRAIN + N_DEV + N_CAL]}
    X = {r: tr[i, :21] for r, i in idx.items()}; Y = {r: tr[i, 21:] for r, i in idx.items()}
    X["test"], Y["test"] = te[:, :21], te[:, 21:]
    xm, xs = X["train"].mean(0), X["train"].std(0) + 1e-8
    ym, ys = Y["train"].mean(0), Y["train"].std(0) + 1e-8
    out = {"X": {r: ((X[r] - xm) / xs).astype(np.float32) for r in X},
           "Y": {r: ((Y[r] - ym) / ys).astype(np.float32) for r in Y},
           "Y_phys": {r: Y[r] for r in Y},
           "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
           "n_in": 21, "n_out": 7, "rows": {r: idx[r].tolist() for r in idx}}
    if manifest_path:
        with open(manifest_path, "w") as f:
            json.dump({"split_seed": SPLIT_SEED, "sizes": {r: len(v) for r, v in idx.items()},
                       "rows": out["rows"], "test": "official sarcos_inv_test.mat, 4449 rows"}, f)
    return out


CONDITION_SEEDS = {"dev": 11, "cal": 22, "test": 33}


def apply_condition(X, role, cond):
    """cond: 'N' | 'O1:<rho>' (all 21 channels) | 'O1qqd:<rho>' (q, qdot only).
    rho is the noise std in units of the training std of each channel."""
    if cond == "N":
        return X
    kind, rho = cond.split(":")
    rho = float(rho)
    seed = CONDITION_SEEDS[role] * 1000 + int(round(rho * 10000)) + (7 if kind == "O1qqd" else 0)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(X.shape).astype(np.float32) * rho
    if kind == "O1qqd":
        mask = np.zeros(X.shape[1], np.float32); mask[Q_QD_CHANNELS] = 1
        noise *= mask
    return X + noise
