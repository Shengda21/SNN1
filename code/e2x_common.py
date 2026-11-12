"""Shared utilities for the zero-training 'reach' analyses E21-E26.

Everything works on per-unit second moments of frozen predictions:
  FF[u, l, i, j] = sum_{rows r in u} F_i(r, l) F_j(r, l)     (same-joint products)
  FY[u, l, i]    = sum_{rows r in u} F_i(r, l) Y(r, l)
  YY[u, l, l']   = sum_{rows r in u} Y(r, l) Y(r, l')
  n[u]           = number of rows of sub-unit u
plus, for selected blocks of members, the full cross-joint products used by
linear stacking (W4):  XFF[u] = sum_r f_r f_r^T, XFY[u] = sum_r f_r Y_r^T with
f_r = vec(F_block(r, :)) ordered member-major (index = j * d + k).
All predictions and targets are in standardized units (D_tau^{-1}); the risk of a
predictor F is R(F) = mean over rows of (1/d) sum_l (F_l - Y_l)^2, identical to the
original w^T C w (sim: rows are averaged per trajectory and trajectories have equal
length, so the pooled mean equals the mean of trajectory means).
All arithmetic float64.
"""
import itertools
import json
import math
import os
import sys

import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from common import calibration as cal  # noqa: E402

SURR = ["atan", "sigmoid", "ste", "triangular", "sre"]
STRONG = {"sim": "sigmoid", "sarcos": "triangular"}
SEEDS = [0, 1, 2, 3, 4]
# fixed seeds of the paired standard-normal noise xi (one array per role, independent of rho)
XI_SEED = {"dev": 2901, "cal": 2902, "test": 2903, "cal_P1": 2904, "test_P1": 2905}
# seeds of the simulated calibration noise (E23) and of misspecified true noise
SIM_SEED = {"cal": 2912, "cal_P1": 2914}
MIS_SEED = {"cal": 2922, "test": 2923}
STATIC_SEED = 2931
FROZEN_LEVELS = {"sim": {"L": 0.35, "M": 0.7, "H": 1.0}, "sarcos": {"L": 0.25, "M": 0.5, "H": 0.7}}


def cname(c):
    return c.replace(":", "_")


def noise_mask(platform, n_in=21):
    """Channels that receive observation noise: sim q, qdot (0-13); sarcos all 21."""
    m = np.zeros(n_in)
    if platform == "sim":
        m[:14] = 1.0
    else:
        m[:] = 1.0
    return m


# ----------------------------------------------------------------------------
# member sets
# ----------------------------------------------------------------------------
def member_blocks(platform, seed, which="full"):
    s = STRONG[platform]
    B = {
        "dm": [f"dm_{x}_s{seed}" for x in SURR],
        "ds": [f"ds_{s}_s{seed}_m{i}" for i in range(5)],
        "dt": [f"dt_{s}_s{seed}_c{i}" for i in range(5)],
        "bl": [f"bl_learnable_s{seed}"],
        "ba": [f"ba_{s}_s{seed}"],
        "bn": [f"bn_mlp_s{seed}"],
    }
    if which == "core":
        return B
    B.update({
        "dma": [f"dma_{x}_s{seed}" for x in SURR],
        "dsa": [f"dsa_{s}_s{seed}_m{i}" for i in range(5)],
        "dmas": [f"dmas_{s}_s{seed}"],
    })
    if platform == "sim":
        B.update({
            "dmq": [f"dmq_{x}_s{seed}" for x in SURR],
            "dsq": [f"dsq_{s}_s{seed}_m{i}" for i in range(5)],
            "bnqs": [f"bnqs_mlp_s{seed}"],
            "dmqs": [f"dmqs_{s}_s{seed}"],
        })
    return B


def flat_members(blocks):
    out = []
    for k in blocks:
        out += blocks[k]
    return out


# ----------------------------------------------------------------------------
# unit layout
# ----------------------------------------------------------------------------
def unit_layout(platform, D, role, seg=1, block_rows=100):
    """Contiguous sub-units. sim: trajectory (1000 steps) split into `seg` equal
    segments; sarcos: blocks of `block_rows` consecutive rows of the role (last block
    partial). Returns sub [N] sub-unit id per row, parent [U] unit id per sub-unit."""
    N = len(D["Y"][role])
    if platform == "sim":
        traj = np.asarray(D["groups"][role])
        assert np.all(np.diff(traj) >= 0)
        T = int(np.bincount(traj)[0])
        assert np.all(np.bincount(traj) == T) and T % seg == 0
        L = T // seg
        sub = traj * seg + (np.arange(N) - np.searchsorted(traj, traj)) // L
        parent = np.repeat(np.arange(traj.max() + 1), seg)
        return sub.astype(np.int64), parent.astype(np.int64)
    sub = np.arange(N) // block_rows
    parent = np.arange(sub.max() + 1)
    return sub.astype(np.int64), parent.astype(np.int64)


def _padded(A, sub):
    """A: [..., N, d] rows contiguous by sub-unit -> [..., U, L, d] zero padded."""
    U = int(sub.max()) + 1
    counts = np.bincount(sub, minlength=U)
    L = int(counts.max())
    if np.all(counts == L):
        return A.reshape(A.shape[:-2] + (U, L, A.shape[-1])), counts
    out = np.zeros(A.shape[:-2] + (U, L, A.shape[-1]), dtype=A.dtype)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for u in range(U):
        out[..., u, :counts[u], :] = A[..., starts[u]:starts[u] + counts[u], :]
    return out, counts


def compute_moments(P, Y, sub, xblocks=None):
    """P: [M, N, d] float64 predictions; Y: [N, d]; sub: [N] contiguous sub-unit ids.
    xblocks: dict name -> list of member indices for cross-joint moments.
    Returns dict of float64 arrays (sums over the rows of every sub-unit)."""
    P = np.asarray(P, np.float64); Y = np.asarray(Y, np.float64)
    M, N, d = P.shape
    Pp, counts = _padded(P, sub)                  # [M, U, L, d]
    Yp, _ = _padded(Y, sub)                       # [U, L, d]
    Pt = np.ascontiguousarray(Pp.transpose(1, 3, 0, 2))      # [U, d, M, L]
    Yt = np.ascontiguousarray(Yp.transpose(0, 2, 1))         # [U, d, L]
    out = {"FF": Pt @ Pt.transpose(0, 1, 3, 2),             # [U, d, M, M]
           "FY": (Pt @ Yt[..., None])[..., 0],              # [U, d, M]
           "YY": Yt @ Yt.transpose(0, 2, 1),                # [U, d, d]
           "n": counts.astype(np.int64)}
    del Pt
    if xblocks:
        for name, idx in xblocks.items():
            F = Pp[idx]                                     # [Mb, U, L, d]
            Mb = len(idx)
            Fx = np.ascontiguousarray(F.transpose(1, 2, 0, 3)).reshape(F.shape[1], F.shape[2], Mb * d)  # [U, L, Mb*d]
            out[f"XFF_{name}"] = Fx.transpose(0, 2, 1) @ Fx            # [U, p, p]
            out[f"XFY_{name}"] = Fx.transpose(0, 2, 1) @ Yp           # [U, p, d]
    return out


# ----------------------------------------------------------------------------
# moment files
# ----------------------------------------------------------------------------
class MomFile:
    """Lazy reader of a moment npz written by e2x_dump_paired.py / e23_dump.py."""

    def __init__(self, path):
        self.path = path
        self.z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(self.z["meta"]))
        self.n = self.z["n"].astype(np.float64)
        self.parent = self.z["parent"]
        self.YY = self.z["YY"]
        self.seeds = [int(s) for s in self.meta["seeds"]]
        self.names = {s: [str(x) for x in self.z[f"names_s{s}"]] for s in self.seeds}
        self._c = {}

    def arr(self, key):
        if key not in self._c:
            self._c[key] = self.z[key]
        return self._c[key]

    def has_members(self, seed, names):
        return seed in self.names and all(n in self.names[seed] for n in names)

    def sel(self, seed, names):
        """Sub-unit sums for the listed members: FF [U,d,M,M], FY [U,d,M], YYd [U,d], n [U]."""
        idx = [self.names[seed].index(n) for n in names]
        FF = self.arr(f"FF_s{seed}")[:, :, idx][:, :, :, idx]
        FY = self.arr(f"FY_s{seed}")[:, :, idx]
        YYd = np.einsum("ull->ul", self.YY)
        return Mom(FF, FY, YYd, self.n, self.parent)

    def xsel(self, seed, block):
        k = f"XFF_s{seed}_{block}"
        if k not in self.z.files:
            return None
        return XMom(self.arr(k), self.arr(f"XFY_s{seed}_{block}"), self.YY, self.n, self.parent)


class Mom:
    """Same-joint moments of a member list over sub-units (sums)."""

    def __init__(self, FF, FY, YYd, n, parent):
        self.FF, self.FY, self.YYd, self.n, self.parent = FF, FY, YYd, n, parent

    def subset(self, mask):
        return Mom(self.FF[mask], self.FY[mask], self.YYd[mask], self.n[mask], self.parent[mask])

    def units(self, unit_ids):
        return self.subset(np.isin(self.parent, unit_ids))

    def pooled(self):
        N = self.n.sum()
        return self.FF.sum(0) / N, self.FY.sum(0) / N, self.YYd.sum(0) / N

    def per_unit(self):
        """Means per parent unit: Phi [G,d,M,M], g [G,d,M], Py [G,d], n [G], ids."""
        ids = np.unique(self.parent)
        pos = np.searchsorted(ids, self.parent)
        G = len(ids)
        FF = np.zeros((G,) + self.FF.shape[1:]); FY = np.zeros((G,) + self.FY.shape[1:])
        YY = np.zeros((G,) + self.YYd.shape[1:]); n = np.zeros(G)
        np.add.at(FF, pos, self.FF); np.add.at(FY, pos, self.FY); np.add.at(YY, pos, self.YYd); np.add.at(n, pos, self.n)
        return FF / n[:, None, None, None], FY / n[:, None, None], YY / n[:, None], n, ids

    def per_sub(self):
        n = self.n
        return self.FF / n[:, None, None, None], self.FY / n[:, None, None], self.YYd / n[:, None], n


class XMom:
    """Cross-joint moments of one block (sums over sub-units)."""

    def __init__(self, XFF, XFY, YY, n, parent):
        self.XFF, self.XFY, self.YY, self.n, self.parent = XFF, XFY, YY, n, parent

    def subset(self, mask):
        return XMom(self.XFF[mask], self.XFY[mask], self.YY[mask], self.n[mask], self.parent[mask])

    def units(self, unit_ids):
        return self.subset(np.isin(self.parent, unit_ids))

    def pooled(self):
        N = self.n.sum()
        return self.XFF.sum(0) / N, self.XFY.sum(0) / N, self.YY.sum(0) / N

    def per_unit(self):
        ids = np.unique(self.parent)
        pos = np.searchsorted(ids, self.parent)
        G = len(ids)
        A = np.zeros((G,) + self.XFF.shape[1:]); B = np.zeros((G,) + self.XFY.shape[1:])
        C = np.zeros((G,) + self.YY.shape[1:]); n = np.zeros(G)
        np.add.at(A, pos, self.XFF); np.add.at(B, pos, self.XFY); np.add.at(C, pos, self.YY); np.add.at(n, pos, self.n)
        return A / n[:, None, None], B / n[:, None, None], C / n[:, None, None], n, ids


def save_moments(path, meta, n, parent, YY, groups):
    """groups: dict seed -> dict(names=[...], FF=..., FY=..., X={block: (XFF, XFY)})."""
    arrs = {"meta": np.array(json.dumps(meta)), "n": n, "parent": parent, "YY": YY}
    for s, gr in groups.items():
        arrs[f"names_s{s}"] = np.array(gr["names"])
        arrs[f"FF_s{s}"] = gr["FF"]
        arrs[f"FY_s{s}"] = gr["FY"]
        for b, (A, B) in gr.get("X", {}).items():
            arrs[f"XFF_s{s}_{b}"] = A
            arrs[f"XFY_s{s}_{b}"] = B
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# risk algebra
# ----------------------------------------------------------------------------
def C_matrix(Phi, g, Py, null=False):
    """Residual second-moment matrix C_ij = <F_i - Y, F_j - Y>; with null=True the
    null member F_0 = 0 (e_0 = -Y) is appended as the last index."""
    M = len(g)
    C = Phi - g[:, None] - g[None, :] + Py
    if not null:
        return C
    Ce = np.empty((M + 1, M + 1))
    Ce[:M, :M] = C
    Ce[M, :M] = Ce[:M, M] = Py - g
    Ce[M, M] = Py
    return Ce


def face_qp(C, faces, tol=1e-10):
    """min w^T C w over the union of the simplices of the given faces (index tuples).
    Exact by face enumeration (closed-form affine minimizer, SLSQP fallback on singular
    faces). Ties within tol: fewer members, then lexicographic (as cal.budgeted_qp)."""
    M = C.shape[0]
    cands = []
    for T in faces:
        CT = C[np.ix_(T, T)]
        x, ok = cal._affine_minimizer(CT)
        if not ok or x is None:
            x = cal._slsqp_face(CT)
            if x is None:
                continue
        if np.all(x >= -1e-12):
            x = np.clip(x, 0, None); x = x / x.sum()
            w = np.zeros(M); w[list(T)] = x
            cands.append((float(w @ C @ w), len(T), tuple(T), w))
    cands.sort(key=lambda t: (t[0], t[1], t[2]))
    v0 = cands[0][0]
    ties = [c for c in cands if c[0] <= v0 + tol * (1 + abs(v0))]
    ties.sort(key=lambda t: (t[1], t[2]))
    return ties[0][3], ties[0][0]


def kkt_simplex(C, w, full=True):
    g = C @ w
    v = float(w @ g)
    supp = np.where(w > 0)[0]
    r = float(np.max(np.abs(g[supp] - v))) if len(supp) else 0.0
    if full:
        r = max(r, float(np.max(np.clip(v - g, 0, None))))
    return r


def simplex_solve(Phi, g, Py, null=False, k=None):
    """W0 (null=False) or W1 (null=True) with at most k executed (non-null) members;
    the null member is free. Returns (w over real members, null weight, value, kkt)."""
    M = len(g)
    C = C_matrix(Phi, g, Py, null)
    kk = M if k is None else min(k, M)
    faces = []
    for size in range(1, kk + 1):
        for T in itertools.combinations(range(M), size):
            faces.append(T)
            if null:
                faces.append(T + (M,))
    if null:
        faces.append((M,))
    w, v = face_qp(C, faces)
    kkt = kkt_simplex(C, w, full=(k is None or kk >= M))
    if null:
        return w[:M], float(w[M]), v, kkt
    return w, 0.0, v, kkt


def cone_nnls(Phi, g):
    """min w^T Phi w - 2 g^T w s.t. w >= 0 by NNLS on the Cholesky factor."""
    from scipy.optimize import nnls
    M = len(g)
    jit = 1e-12 * max(1.0, float(np.trace(Phi)) / M)
    Lc = np.linalg.cholesky(Phi + jit * np.eye(M))
    b = np.linalg.solve(Lc, g)
    w, _ = nnls(Lc.T, b, maxiter=1000)
    return w


def w1_via_nnls(Phi, g, Py):
    """W1 optimum via the cone: if the cone optimum has sum <= 1 it is feasible and
    optimal; otherwise the sum constraint is active and W1 = W0."""
    w = cone_nnls(Phi, g)
    if w.sum() <= 1.0:
        val = float(w @ Phi @ w - 2 * g @ w + Py)
        return w, val
    w0, _, v0, _ = simplex_solve(Phi, g, Py, null=False)
    return w0, v0


def kkt_w1(Phi, g, w):
    """KKT residual of min w'Phi w - 2g'w s.t. w >= 0, 1'w <= 1 (gradient / 2)."""
    grad = Phi @ w - g
    s = w.sum()
    supp = w > 1e-14
    if s < 1 - 1e-9:
        mu = 0.0
    else:
        mu = float(-np.mean(grad[supp])) if supp.any() else 0.0
        mu = max(mu, 0.0)
    r = grad + mu
    res = np.abs(r[supp]).max() if supp.any() else 0.0
    res = max(res, float(np.clip(-r[~supp], 0, None).max()) if (~supp).any() else 0.0)
    return float(res)


def pj_risk(W, Phil, gl, Pyl):
    """Risk of per-joint weights W [d, M] (null weight implicit)."""
    FFt = np.einsum("lm,lmn,ln->", W, Phil, W) / len(Pyl)
    FYt = np.einsum("lm,lm->", W, gl) / len(Pyl)
    Py = float(np.mean(Pyl))
    return float(FFt - 2 * FYt + Py), float(FFt), float(FYt), Py


def pj_unit_risk(W, Phiu, gu, Pyu):
    """Per-unit risks for per-joint weights: Phiu [G,d,M,M], gu [G,d,M], Pyu [G,d]."""
    d = Pyu.shape[1]
    FF = np.einsum("lm,glmn,ln->g", W, Phiu, W) / d
    FY = np.einsum("lm,glm->g", W, gu) / d
    return FF - 2 * FY + Pyu.mean(1)


def geometry(FFt, FYt, Py):
    """beta, ||nu||^2, slope s, angle risk, scale cost of a predictor from <F,F>, <F,Y>, Py."""
    R = FFt - 2 * FYt + Py
    beta = FYt / Py
    nu2 = FFt - beta ** 2 * Py
    s = FYt / FFt if FFt > 0 else np.nan
    ang = Py - FYt ** 2 / FFt if FFt > 0 else Py
    scale = FFt * (1 - s) ** 2 if FFt > 0 else 0.0
    return {"risk": R, "beta": beta, "nu2": nu2, "slope": s, "angle_risk": ang, "scale_cost": scale,
            "out_power": FFt, "Py": Py}


# ----------------------------------------------------------------------------
# weight spaces on (Phil, gl, Pyl)  -- Phil [d,M,M], gl [d,M], Pyl [d]
# ----------------------------------------------------------------------------
def solve_space(space, Phil, gl, Pyl, k=None):
    d, M = gl.shape
    Phi, g, Py = Phil.mean(0), gl.mean(0), float(Pyl.mean())
    info = {}
    if space in ("W0", "W1"):
        w, w0, v, kkt = simplex_solve(Phi, g, Py, null=(space == "W1"), k=k)
        W = np.tile(w, (d, 1))
        info.update(kkt=kkt, value_cal=v)
        if space == "W1" and k is None:
            wn, vn = w1_via_nnls(Phi, g, Py)
            info["nnls_gap"] = abs(vn - v)
            info["kkt_w1"] = kkt_w1(Phi, g, w)
        return W, info
    if space in ("W2", "W3"):
        W = np.zeros((d, M)); kk, gaps = 0.0, 0.0
        for l in range(d):
            w, w0, v, kkt = simplex_solve(Phil[l], gl[l], float(Pyl[l]), null=(space == "W3"), k=k)
            W[l] = w; kk = max(kk, kkt)
            if space == "W3" and k is None:
                wn, vn = w1_via_nnls(Phil[l], gl[l], float(Pyl[l]))
                gaps = max(gaps, abs(vn - v))
        info.update(kkt=kk)
        if space == "W3" and k is None:
            info["nnls_gap"] = gaps
        return W, info
    raise ValueError(space)


def ridge_pj_fit(Phil, gl, lam):
    d, M = gl.shape
    W = np.zeros((d, M))
    for l in range(d):
        W[l] = np.linalg.solve(Phil[l] + lam * np.eye(M), gl[l])
    return W


def lam_grid(trace_scale, n=13):
    return trace_scale * np.logspace(-7, 1, n)


def cv_select(fit, risk, folds, lams):
    """Generic K-fold selection: fit(train_mask, lam) -> params; risk(params, val_mask)."""
    tot = np.zeros(len(lams))
    for f in folds:
        tr = ~f
        for i, lam in enumerate(lams):
            tot[i] += risk(fit(tr, lam), f)
    i = int(np.argmin(tot))
    return float(lams[i]), tot / len(folds)


def solve_W4d(mom, folds):
    """Per-joint unconstrained linear stacking with ridge; lambda by K-fold over folds
    (list of boolean masks over sub-units of mom)."""
    Phil, gl, Pyl = mom.pooled()
    sc = float(np.mean([np.trace(Phil[l]) / Phil.shape[1] for l in range(Phil.shape[0])]))
    lams = lam_grid(sc)

    def fit(mask, lam):
        P, g, _ = mom.subset(mask).pooled()
        return ridge_pj_fit(P, g, lam)

    def rk(W, mask):
        P, g, y = mom.subset(mask).pooled()
        return pj_risk(W, P, g, y)[0]

    if folds is None or len(folds) < 2:
        lam = float(lams[len(lams) // 2]); cvr = None
    else:
        lam, cvr = cv_select(fit, rk, folds, lams)
    return ridge_pj_fit(Phil, gl, lam), {"lam": lam, "lam_rel": lam / sc}


def x_risk(A, G, B, YY):
    """Risk of a linear stack A [p, d]: (1/d) tr(A'GA - 2A'B + YY)."""
    d = YY.shape[0]
    FF = float(np.einsum("pl,pq,ql->", A, G, A)) / d
    FY = float(np.einsum("pl,pl->", A, B)) / d
    Py = float(np.trace(YY)) / d
    return FF - 2 * FY + Py, FF, FY, Py


def x_unit_risk(A, Gu, Bu, YYu):
    d = YYu.shape[1]
    FF = np.einsum("pl,gpq,ql->g", A, Gu, A) / d
    FY = np.einsum("pl,gpl->g", A, Bu) / d
    return FF - 2 * FY + np.einsum("gll->g", YYu) / d


def solve_W4(xmom, folds):
    """Full linear stacking (all joints of all members -> each joint) with ridge."""
    G, B, YY = xmom.pooled()
    p = G.shape[0]
    sc = float(np.trace(G)) / p
    lams = lam_grid(sc)

    def fit(mask, lam):
        Gm, Bm, _ = xmom.subset(mask).pooled()
        return np.linalg.solve(Gm + lam * np.eye(p), Bm)

    def rk(A, mask):
        Gm, Bm, Ym = xmom.subset(mask).pooled()
        return x_risk(A, Gm, Bm, Ym)[0]

    if folds is None or len(folds) < 2:
        lam = float(lams[len(lams) // 2])
    else:
        lam, _ = cv_select(fit, rk, folds, lams)
    return np.linalg.solve(G + lam * np.eye(p), B), {"lam": lam, "lam_rel": lam / sc}


def make_folds(mom, mode="units", k=None):
    """Boolean masks over sub-units: one fold per parent unit (or per sub-unit if a
    single unit), or k contiguous groups of parent units."""
    ids = np.unique(mom.parent)
    if len(ids) < 2:
        U = len(mom.parent)
        if U < 2:
            return None
        return [np.arange(U) == u for u in range(U)]
    if k is None or k >= len(ids):
        return [mom.parent == u for u in ids]
    parts = np.array_split(ids, k)
    return [np.isin(mom.parent, p) for p in parts]


# ----------------------------------------------------------------------------
# bootstrap (count-weighted version of common.metrics.cluster_bootstrap; with equal
# weights it reproduces the original exactly, same random draws)
# ----------------------------------------------------------------------------
def cboot(values, weights, clusters, n_boot=1000, seed=0):
    values = np.asarray(values, float)
    w0 = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    rng = np.random.default_rng(seed)
    levels = list(clusters.values())
    uniq = [np.unique(l) for l in levels]
    pos = [np.searchsorted(u, l) for l, u in zip(levels, uniq)]
    point = float(np.sum(w0 * values) / np.sum(w0))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        wts = w0.copy()
        for u, p in zip(uniq, pos):
            draw = rng.choice(len(u), len(u), replace=True)
            counts = np.bincount(draw, minlength=len(u))
            wts = wts * counts[p]
        s = wts.sum()
        boots[b] = point if s == 0 else float(np.sum(wts * values) / s)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def jdump(x, nd=5):
    return json.dumps([round(float(v), nd) for v in np.ravel(x)])
