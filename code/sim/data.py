"""Simulation data loader with the same interface as sarcos.data.

Roles: train, dev, cal, test (nominal plant) and cal_P1, test_P1 (payload plant).
X = standardized [q, qd, qdd] (21), Y = standardized tau (7); standardization
fitted on train. groups[role] gives the trajectory id of every row (independent
unit for calibration and statistics). Conditions on standardized inputs:
  'N' ; 'O1:<rho>' i.i.d. noise on q and qd channels (rho x training std);
  'O2B:<rho>' per-trajectory constant bias on q and qd channels.
P1 roles accept only 'N' (the plant, not the observation, changed)."""
import os
import numpy as np

_GROUPS = {}
CONDITION_SEEDS = {"dev": 11, "cal": 22, "test": 33, "cal_P1": 44, "test_P1": 55}
Q_QD = list(range(14))


def _flatten(npz):
    X = np.concatenate([npz["q"], npz["qd"], npz["qdd"]], 2)   # [n, T, 21]
    Y = npz["tau"]
    n, T, _ = X.shape
    groups = np.repeat(np.arange(n), T)
    return X.reshape(n * T, 21), Y.reshape(n * T, 7), groups


def load_sim(data_dir):
    roles = {"train": "train_N", "dev": "dev_N", "cal": "cal_N", "test": "test_N",
             "cal_P1": "cal_P1", "test_P1": "test_P1"}
    raw = {r: _flatten(np.load(os.path.join(data_dir, f + ".npz"))) for r, f in roles.items()}
    xm, xs = raw["train"][0].mean(0), raw["train"][0].std(0) + 1e-8
    ym, ys = raw["train"][1].mean(0), raw["train"][1].std(0) + 1e-8
    out = {"X": {}, "Y": {}, "Y_phys": {}, "groups": {}, "x_mean": xm, "x_std": xs, "y_mean": ym,
           "y_std": ys, "n_in": 21, "n_out": 7}
    for r, (X, Y, g) in raw.items():
        out["X"][r] = ((X - xm) / xs).astype(np.float32)
        out["Y"][r] = ((Y - ym) / ys).astype(np.float32)
        out["Y_phys"][r] = Y
        out["groups"][r] = g
        _GROUPS[r] = g
    return out


def apply_condition(X, role, cond):
    if cond == "N":
        return X
    if role.endswith("_P1") or role == "train":
        return None
    kind, rho = cond.split(":")
    rho = float(rho)
    seed = CONDITION_SEEDS[role] * 1000 + int(round(rho * 10000)) + (7 if kind == "O2B" else 0)
    rng = np.random.default_rng(seed)
    mask = np.zeros(X.shape[1], np.float32); mask[Q_QD] = 1
    if kind == "O1":
        return X + rng.standard_normal(X.shape).astype(np.float32) * rho * mask
    if kind == "O2B":
        g = _GROUPS[role]
        nb = rng.standard_normal((g.max() + 1, X.shape[1])).astype(np.float32) * rho * mask
        return X + nb[g]
    raise ValueError(cond)
