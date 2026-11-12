"""Shared helpers for E27-E29 (task layer). New file: the modules of the original code are
imported, never modified.

Conventions taken line by line from run_closedloop_aug.py (E13) and analysis_lowfreq_matched.py
(E13c):
  * calibration: m = 10 trajectories per group (groups 0-2 of D_cal; cal_P1 under P1), condition
    applied with sim.data.apply_condition, residual matrix in standardized units, W-Q(k) by
    budgeted_qp, W-S by rule_WS;
  * every executed member output is clipped to the effort range (standardized units) before the
    weighted sum; the null predictor is zero in standardized units;
  * input estimator: W = Sig (Sig + diag(v_est))^-1, v_est = max(var(cal inputs) - var(train), 0),
    x_est = mu + (x - mu) W^T, with the variances of the calibration group.
"""
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.signal import lfilter, lfilter_zi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.models import predict  # noqa: E402
from common import calibration as cal  # noqa: E402
from dump_preds import load_member  # noqa: E402
from sim.data import load_sim, apply_condition  # noqa: E402
from sim.panda_env import TORQUE_LIMIT  # noqa: E402

KP = np.array([100.0, 100, 100, 100, 40, 40, 40])
KD = np.array([8.0, 8, 8, 8, 3.2, 3.2, 3.2])
JOINT_DAMPING = 1.0
GROUPS = [0, 1, 2]
NOISE_SEED = {"N": 900, "O1M": 901, "P1": 902}
TS = 0.01
FS = 100.0
FCS = [2, 3, 5, 8, 12, 16, 20, 25, 30, 40]
L_EQ_JOINT = 2.0 * (KD + JOINT_DAMPING) / (KP * TS)      # 18 (joints 1-4), 21 (joints 5-7)


# ----------------------------------------------------------------------------- filters
def lp_alpha(fc):
    """Coefficient of the causal first-order low-pass y_k = y_{k-1} + a (x_k - y_{k-1}) with cutoff fc
    (pole matched to exp(-2 pi fc Ts))."""
    return 1.0 - math.exp(-2.0 * math.pi * fc * TS)


def leq_exact(fc):
    """White-noise power reduction of the discrete filter: 1 / (a / (2 - a))."""
    a = lp_alpha(fc)
    return (2.0 - a) / a


def leq_cont(fc):
    """Continuous approximation of the plan (A5): fs / (pi fc), at least 1."""
    return max(1.0, FS / (math.pi * fc))


def fc_from_leq_exact(L):
    if not np.isfinite(L) or L <= 1.0:
        return np.nan
    a = 2.0 / (L + 1.0)
    return -math.log(1.0 - a) / (2.0 * math.pi * TS)


def fc_from_leq_cont(L):
    if not np.isfinite(L) or L <= 0:
        return np.nan
    return FS / (math.pi * L)


def lowpass(E, fc, axis=1):
    """Causal first-order low-pass along `axis` (time), initialized at the first sample."""
    if fc is None or not np.isfinite(fc):
        return E
    a = lp_alpha(fc)
    b, aa = [a], [1.0, -(1.0 - a)]
    x0 = np.take(E, [0], axis=axis)
    zi = lfilter_zi(b, aa)[0] * x0
    y, _ = lfilter(b, aa, E, axis=axis, zi=zi)
    return y


def moving_average_3d(E, L):
    """E [n_traj, T, d] -> L-step moving average along each trajectory [n_traj, T-L+1, d]
    (identical to analysis_lowfreq.moving_average on the flattened array)."""
    c = np.cumsum(np.concatenate([np.zeros_like(E[:, :1]), E], 1), 1)
    return (c[:, L:] - c[:, :-L]) / L


def slow_white(m10, m25):
    """Fit slow(L) = S + W / L through the moving-average risks at L = 10 and 25."""
    W = (m10 - m25) / (1.0 / 10 - 1.0 / 25)
    S = m25 - W / 25.0
    return S, W


# ----------------------------------------------------------------------------- setup
def setup(data_dir, frozen_path, device=None):
    frozen = json.load(open(frozen_path))
    D = load_sim(data_dir)
    ctx = {"D": D, "frozen": frozen, "strong": frozen["strong_surrogate"], "rho": frozen["O1_M"],
           "xm": D["x_mean"], "xs": D["x_std"], "ym": D["y_mean"], "ys": D["y_std"],
           "device": device or torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    Xtr = D["X"]["train"].astype(np.float64)
    ctx["mu"], ctx["Sig"], ctx["v_train"] = Xtr.mean(0), np.cov(Xtr.T), Xtr.var(0)
    ctx["clip_lo"] = (-TORQUE_LIMIT - ctx["ym"]) / ctx["ys"]
    ctx["clip_hi"] = (TORQUE_LIMIT - ctx["ym"]) / ctx["ys"]
    return ctx


def load_config(path):
    return json.load(open(path))


def resolve(cfg, name, seed, strong):
    """Candidate -> list of (member id or None for the null predictor, behind_estimator)."""
    spec = cfg["candidates"][name]
    items = []
    for blk in spec["blocks"]:
        mem = blk["members"]
        tmpl = cfg["member_sets"][mem] if isinstance(mem, str) else mem
        items += [(t.format(seed=seed, strong=strong), bool(blk.get("est", False))) for t in tmpl]
    if spec.get("null"):
        items.append((None, False))
    return items


def cand_groups(cfg, name):
    return cfg["candidates"][name].get("groups", GROUPS)


class Members:
    """Lazy cache of the frozen members of one seed and of their (clipped, standardized) predictions."""

    def __init__(self, ctx, runs, seed):
        self.ctx, self.runs, self.seed = ctx, runs, seed
        self.models = {}
        self.cache = {}

    def model(self, mid):
        if mid not in self.models:
            self.models[mid] = load_member(self.runs, mid, 21, 7, self.ctx["device"])
        return self.models[mid]

    def pred(self, mid, X, tag=None):
        """Clipped standardized prediction (float64) of member mid on standardized inputs X."""
        key = (mid, tag)
        if tag is not None and key in self.cache:
            return self.cache[key]
        if mid is None:
            P = np.zeros((len(X), 7))
        else:
            P = predict(self.model(mid), np.asarray(X, np.float32), self.ctx["device"]).astype(np.float64)
            P = np.clip(P, self.ctx["clip_lo"], self.ctx["clip_hi"])
        if tag is not None:
            self.cache[key] = P
        return P

    def drop(self, prefix):
        for k in [k for k in self.cache if k[1] is not None and k[1].startswith(prefix)]:
            del self.cache[k]


def estimator_matrix(ctx, Xcal_group):
    v_est = np.maximum(np.asarray(Xcal_group, np.float64).var(0) - ctx["v_train"], 0.0)
    W = ctx["Sig"] @ np.linalg.inv(ctx["Sig"] + np.diag(v_est))
    return W, v_est


def apply_estimator(ctx, X, W):
    return ctx["mu"] + (np.asarray(X, np.float64) - ctx["mu"]) @ W.T


def cal_split(ctx, cond, group, xform=None):
    """Calibration inputs of one group under a closed-loop condition (N, O1M, P1), as in E13.
    xform(X_std [n_traj*T, 21], role) may replace the acceleration channels (setting b of E27)."""
    D = ctx["D"]
    if cond == "N":
        role, tag = "cal", "N"
    elif cond == "O1M":
        role, tag = "cal", f"O1:{ctx['rho']}"
    elif cond == "P1":
        role, tag = "cal_P1", "N"
    else:
        raise ValueError(cond)
    X = apply_condition(D["X"][role], role, tag).astype(np.float64)
    if xform is not None:
        X = xform(X, role)
    gcal = D["groups"][role]
    idx = np.where(np.isin(gcal, np.arange(10 * group, 10 * group + 10)))[0]
    W, v_est = estimator_matrix(ctx, X[idx])
    return {"X": X[idx], "Y": D["Y"][role][idx].astype(np.float64), "g": gcal[idx], "W": W, "v_est": v_est,
            "role": role, "cond": cond}


def test_inputs(ctx, cond):
    """Open-loop test inputs (standardized) and targets of a condition."""
    D = ctx["D"]
    if cond == "N":
        role, tag = "test", "N"
    elif cond == "O1M":
        role, tag = "test", f"O1:{ctx['rho']}"
    elif cond == "P1":
        role, tag = "test_P1", "N"
    else:
        raise ValueError(cond)
    X = apply_condition(D["X"][role], role, tag).astype(np.float64)
    return {"X": X, "Y": D["Y"][role].astype(np.float64), "Y_phys": D["Y_phys"][role], "g": D["groups"][role],
            "role": role}


# ----------------------------------------------------------------------------- weights
def fit_weights(rule, Pc, Yc, gc, extra=None):
    """Pc: list of clipped standardized calibration predictions (zeros for the null)."""
    M = len(Pc)
    one = np.ones(Yc.shape[1])
    if rule == "fixed":
        assert M == 1
        return np.ones(1)
    if rule == "WS":
        return cal.rule_WS(cal.residual_matrix(Pc, Yc, one, gc)[0])
    if rule.startswith("WQ") and rule[2:].isdigit():
        k = int(rule[2:])
        return cal.budgeted_qp(cal.residual_matrix(Pc, Yc, one, gc)[0], min(k, M))["w"]
    if rule == "W1":
        return cal.budgeted_qp(cal.residual_matrix(Pc, Yc, one, gc)[0], M)["w"]
    if rule == "W3":
        Wj = np.zeros((M, Yc.shape[1]))
        for l in range(Yc.shape[1]):
            C = cal.residual_matrix([p[:, [l]] for p in Pc], Yc[:, [l]], np.ones(1), gc)[0]
            Wj[:, l] = cal.budgeted_qp(C, M)["w"]
        return Wj
    if rule.startswith("C:"):
        # weights from a supplied second-moment matrix (task-weighted calibration, E29)
        k = int(rule.split(":")[1])
        return cal.budgeted_qp(extra, min(k, M))["w"]
    raise ValueError(rule)


def combine(P_list, w):
    w = np.asarray(w, np.float64)
    out = np.zeros_like(P_list[0])
    for i, p in enumerate(P_list):
        wi = w[i]
        if np.all(np.abs(wi) <= 1e-12):
            continue
        out = out + wi * p
    return out


def weight_summary(items, w):
    w = np.asarray(w, np.float64)
    wsum = w if w.ndim == 1 else w.mean(1)
    null_w = float(sum(wsum[i] for i in range(len(items)) if items[i][0] is None))
    est_w = float(sum(wsum[i] for i in range(len(items)) if items[i][0] is not None and items[i][1]))
    n_exec = int(sum(1 for i in range(len(items)) if items[i][0] is not None and np.any(np.abs(w[i]) > 1e-12)))
    return {"null_weight": null_w, "est_weight": est_w, "n_exec": n_exec}


def candidate_weights(ctx, cfg, members, name, cs, cal_tag):
    """Fit the weights of a candidate on one calibration split (cs from cal_split)."""
    items = resolve(cfg, name, members.seed, ctx["strong"])
    spec = cfg["candidates"][name]
    Xe = None
    Pc = []
    for mid, est in items:
        if est:
            if Xe is None:
                Xe = apply_estimator(ctx, cs["X"], cs["W"])
            Pc.append(members.pred(mid, Xe, cal_tag + ":est"))
        else:
            Pc.append(members.pred(mid, cs["X"], cal_tag + ":raw"))
    if spec["weights"].startswith("e21:"):
        w = e21_weights(spec, members.seed, cs["cond"], len(items))
    else:
        w = fit_weights(spec["weights"], Pc, cs["Y"], cs["g"])
    return items, w


def candidate_predict(ctx, members, items, w, X, W_est, tag_raw, tag_est):
    """Standardized prediction of a weighted candidate on standardized inputs X."""
    Xe = None
    P = []
    w = np.asarray(w, np.float64)
    for i, (mid, est) in enumerate(items):
        if np.all(np.abs(w[i]) <= 1e-12) or mid is None:
            P.append(np.zeros((len(X), 7)))
            continue
        if est:
            if Xe is None:
                Xe = apply_estimator(ctx, X, W_est)
            P.append(members.pred(mid, Xe, tag_est))
        else:
            P.append(members.pred(mid, X, tag_raw))
    return combine(P, w)


def make_ff(ctx, members, items, w, W_est):
    """Callable for the closed loop: physical X [B, 21] -> physical torque [B, 7] (as make_ff of E13,
    with optional per-joint weights)."""
    xm, xs, ym, ys = ctx["xm"], ctx["xs"], ctx["ym"], ctx["ys"]
    dev = ctx["device"]
    lo = torch.as_tensor(ctx["clip_lo"], dtype=torch.float64, device=dev)
    hi = torch.as_tensor(ctx["clip_hi"], dtype=torch.float64, device=dev)
    w = np.asarray(w, np.float64)
    supp = [i for i in range(len(items)) if np.any(np.abs(w[i]) > 1e-12) and items[i][0] is not None]
    mus = ctx["mu"]
    Wt = None if W_est is None else W_est.T
    wt = [torch.as_tensor(np.broadcast_to(w[i], (7,)).copy(), dtype=torch.float64, device=dev) for i in range(len(items))]

    def ff(X):
        xstd = (X - xm) / xs
        xb = torch.as_tensor(xstd, dtype=torch.float32, device=dev)
        xe = None
        out = torch.zeros((len(X), 7), dtype=torch.float64, device=dev)
        with torch.no_grad():
            for i in supp:
                mid, est = items[i]
                mdl = members.model(mid)
                if est:
                    if xe is None:
                        xe = torch.as_tensor(mus + (xstd - mus) @ Wt, dtype=torch.float32, device=dev)
                    p = mdl(xe)[-1].double()
                else:
                    p = mdl(xb)[-1].double()
                out = out + wt[i] * torch.clamp(p, lo, hi)
        return out.cpu().numpy() * ys + ym
    n_exec = len(supp)
    return ff, n_exec


E21_PATH = os.path.join(HERE, "..", "results", "sim", "e21_ladder_seeds.csv")
E21_COND = {"N": "N", "O1M": "O1-M", "P1": "P1"}
_E21 = {}


def e21_weights(spec, seed, cond, n_items):
    """Weights of E21's optimum (e21_ladder_seeds.csv: platform sim, source orig, full D_cal of the
    condition) for method spec['e21_method'] in space W1 (5 + null) or W3 (per joint, [7 x 5] stored
    joint-major, null = 1 - row sum). Returned as [n_items] (W1) or [n_items, 7] (W3), null last."""
    import pandas as pd
    if "df" not in _E21:
        _E21["df"] = pd.read_csv(E21_PATH)
    df = _E21["df"]
    space = spec["weights"].split(":")[1]
    r = df[(df.platform == "sim") & (df.source == "orig") & (df.cond == E21_COND[cond]) & (df.seed == seed)
           & (df.method == spec["e21_method"]) & (df.space == space)]
    assert len(r) == 1, (spec, seed, cond, len(r))
    w = np.asarray(json.loads(r.iloc[0]["weights"]), np.float64)
    if space == "W1":
        out = np.r_[w, max(0.0, 1.0 - w.sum())]
    elif space == "W3":
        Wl = w.reshape(7, -1)                         # [joint, member]
        out = np.vstack([Wl.T, np.clip(1.0 - Wl.sum(1), 0.0, None)[None, :]])
    else:
        raise ValueError(space)
    assert len(out) == n_items, (len(out), n_items)
    return out


# ----------------------------------------------------------------------------- joint-loop task filter (E29)
def effective_inertia(xml_dir, Q, payload_kg=0.0, n_samples=400, seed=0):
    """Mean diagonal of the joint-space mass matrix (armature included) over sampled configurations."""
    import mujoco
    from sim.panda_env import PandaSim
    sim = PandaSim(xml_dir, payload_kg)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Q), min(n_samples, len(Q)), replace=False)
    Mfull = np.zeros((sim.model.nv, sim.model.nv))
    diag = []
    for i in idx:
        sim.data.qpos[:] = Q[i]
        mujoco.mj_forward(sim.model, sim.data)
        try:
            mujoco.mj_fullM(sim.model, sim.data, Mfull)          # mujoco >= 3.4 signature
        except TypeError:
            mujoco.mj_fullM(sim.model, Mfull, sim.data.qM)
        diag.append(np.diag(Mfull).copy())
    return np.mean(diag, 0)


def joint_loop_filters(J, kp=KP, kd=KD, damping=JOINT_DAMPING, ts=TS):
    """Per joint, the discrete closed-loop response from a feedforward torque error r[k] (zero-order
    hold) to the tracking error e[k] = q[k] - q_ref[k] of the sampled PD loop
        J e'' + D e' = r[k] - kp e[k] - kd e'[k]   (decoupled, linear).
    Returns a list of (b, a) for scipy.signal.lfilter."""
    from scipy.signal import cont2discrete, ss2tf
    out = []
    for l in range(len(J)):
        A = np.array([[0.0, 1.0], [0.0, -damping / J[l]]]); B = np.array([[0.0], [1.0 / J[l]]])
        Cm = np.array([[1.0, 0.0]]); Dm = np.zeros((1, 1))
        Ad, Bd, _, _, _ = cont2discrete((A, B, Cm, Dm), ts, method="zoh")
        K = np.array([[kp[l], kd[l]]])
        Acl = Ad - Bd @ K
        num, den = ss2tf(Acl, Bd, Cm, Dm)
        out.append((np.real(num[0]), np.real(den)))
    return out


def task_filter(E, filters):
    """E [n_traj, T, 7] torque error (N m) -> tracking error (rad) through the joint-loop filters."""
    out = np.zeros_like(E)
    for l, (b, a) in enumerate(filters):
        out[:, :, l] = lfilter(b, a, E[:, :, l], axis=1)
    return out
