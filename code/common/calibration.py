"""Residual second moments, weight rules and budgeted least-squares averaging.

Implements the residual geometry and the calibration rules of the paper exactly:
  e_j = D_tau^{-1}(f_j - y) / sqrt(d)                              (eq. 1)
  C_ij = E_Z[ (1/T_Z) sum_t <e_i, e_j> ]                           (eq. 2)
  R(w) = w^T C w ; ambiguity decomposition                          (eq. 3)
  two-member witness alpha*, improvement                            (eq. 5-7)
  W_k = {w in simplex : |supp w| <= k}, v_k, G_k                    (eq. 8)
  g_cal,k = Rhat(w_alg) - vhat_k                                    (eq. 17)
  eps_m = B_e^2 sqrt(2/m log(2 M^2 / delta))                        (eq. 18)
Weight rules: W-S, W-E, W-B (BIC-inspired), W-D (direct choice among
7 candidates), W-Q(k) (exact budgeted least-squares convex average by face
enumeration with KKT certificate).
All arithmetic is float64.
"""
import itertools
import math
import numpy as np


# ----------------------------------------------------------------------------
# residual geometry
# ----------------------------------------------------------------------------
def normalized_residuals(preds, Y, scale):
    """preds: list of [N,d] arrays (physical or normalized units, same as Y);
    scale: [d] output scale D_tau. Returns E: [M, N, d] float64."""
    d = Y.shape[1]
    Y64 = np.asarray(Y, dtype=np.float64); sc = np.asarray(scale, dtype=np.float64)
    E = np.stack([(np.asarray(p, dtype=np.float64) - Y64) / sc for p in preds], 0) / math.sqrt(d)
    return E


def per_unit_moments(E, groups=None):
    """E: [M,N,d]. groups: [N] int ids of independent units (trajectories) or None
    (every row is its own unit). Returns A: [G, M, M] per-unit second-moment matrices
    and the unit ids."""
    M, N, d = E.shape
    if groups is None:
        A = np.einsum("ind,jnd->nij", E, E)
        return A, np.arange(N)
    groups = np.asarray(groups)
    ids = np.unique(groups)
    A = np.zeros((len(ids), M, M))
    order = np.argsort(groups, kind="stable")
    bounds = np.searchsorted(groups[order], ids, side="left").tolist() + [N]
    for gi in range(len(ids)):
        idx = order[bounds[gi]:bounds[gi + 1]]
        Eg = E[:, idx, :].reshape(M, -1)
        A[gi] = Eg @ Eg.T / len(idx)
    return A, ids


def chat_from_units(A):
    """C-hat = mean over independent units (eq. 16)."""
    return A.mean(0)


def residual_matrix(preds, Y, scale, groups=None):
    E = normalized_residuals(preds, Y, scale)
    A, ids = per_unit_moments(E, groups)
    return chat_from_units(A), A, ids


def risk(w, C):
    w = np.asarray(w, float)
    return float(w @ C @ w)


def ambiguity_decomposition(w, C):
    """eq. (3): R_a - R(w) = D(w) - sum_j w_j (R_j - R_a), a = argmin_j C_jj."""
    w = np.asarray(w, float)
    R = np.diag(C)
    a = int(np.argmin(R))
    Dw = 0.5 * sum(w[i] * w[j] * (R[i] + R[j] - 2 * C[i, j])
                   for i in range(len(w)) for j in range(len(w)))
    weak = float(np.sum(w * (R - R[a])))
    net = R[a] - risk(w, C)
    return {"best": a, "net_gain": float(net), "disagreement": float(Dw),
            "weak_member_cost": weak, "identity_residual": float(net - (Dw - weak))}


def two_member_witness(C):
    """eq. (5)-(7). Returns None if no strictly improving average exists."""
    R = np.diag(C)
    a = int(np.argmin(R))
    tol = 1e-10 * (1.0 + abs(R[a]))
    cands = [b for b in range(len(R)) if b != a and C[a, b] < R[a] - tol]
    if not cands:
        return {"best": a, "exists": False}
    out = []
    for b in cands:
        D = R[a] + R[b] - 2 * C[a, b]
        alpha = (R[a] - C[a, b]) / D
        gain = (R[a] - C[a, b]) ** 2 / D
        out.append((b, float(alpha), float(gain)))
    b, alpha, gain = max(out, key=lambda t: t[2])
    return {"best": a, "exists": True, "partner": int(b), "alpha": alpha, "gain": gain,
            "all_partners": out}


# ----------------------------------------------------------------------------
# exact simplex / budgeted QP by face enumeration
# ----------------------------------------------------------------------------
def _affine_minimizer(C_T):
    """Minimizer of w^T C w on the affine hull {sum w = 1} for the face T."""
    n = C_T.shape[0]
    if n == 1:
        return np.array([1.0]), True
    try:
        if np.linalg.cond(C_T) > 1e12:
            return None, False
        x = np.linalg.solve(C_T, np.ones(n))
        s = x.sum()
        if not np.isfinite(s) or abs(s) < 1e-14:
            return None, False
        return x / s, True
    except np.linalg.LinAlgError:
        return None, False


def _slsqp_face(C_T):
    from scipy.optimize import minimize
    n = C_T.shape[0]
    best = None
    for start in [np.ones(n) / n] + [np.eye(n)[i] for i in range(n)]:
        res = minimize(lambda w: w @ C_T @ w, start, jac=lambda w: 2 * C_T @ w,
                       bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                                          "jac": lambda w: np.ones(n)}],
                       method="SLSQP", options={"ftol": 1e-15, "maxiter": 500})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    w = np.clip(best.x, 0, None); w /= w.sum()
    return w


def budgeted_qp(C, k, tol=1e-10):
    """min_{w in W_k} w^T C w by enumerating every face T with |T| <= k.
    On each face the affine minimizer is exact (closed form) when C_T is
    nonsingular; singular faces fall back to SLSQP. The face optimum is feasible
    iff all its weights are >= 0 (then it is the optimum of the simplex of T only
    if it is interior, but the global optimum over W_k always lies in the relative
    interior of some face of size <= k, so the minimum over feasible face
    minimizers is the exact budgeted optimum).
    Ties within tol (relative) are broken by fewer members, then lexicographic.
    Returns dict(w, value, support, kkt_residual, n_faces, candidates)."""
    C = np.asarray(C, float)
    M = C.shape[0]
    cands = []
    n_faces = 0
    for size in range(1, k + 1):
        for T in itertools.combinations(range(M), size):
            n_faces += 1
            C_T = C[np.ix_(T, T)]
            x, ok = _affine_minimizer(C_T)
            if not ok or x is None:
                x = _slsqp_face(C_T)
                if x is None:
                    continue
            if np.all(x >= -1e-12):
                x = np.clip(x, 0, None); x = x / x.sum()
                w = np.zeros(M); w[list(T)] = x
                cands.append((float(w @ C @ w), size, T, w))
    cands.sort(key=lambda t: (t[0], t[1], t[2]))
    v0 = cands[0][0]
    ties = [c for c in cands if c[0] <= v0 + tol * (1 + abs(v0))]
    ties.sort(key=lambda t: (t[1], t[2]))
    value, size, T, w = ties[0]
    # KKT certificate on the chosen face: (C w)_j equal on support; for k = M also
    # (C w)_j >= value for all j (eq. 4).
    g = C @ w
    supp = np.where(w > 0)[0]
    kkt = float(np.max(np.abs(g[supp] - value))) if len(supp) else 0.0
    if k >= M:
        kkt = max(kkt, float(np.max(np.clip(value - g, 0, None))))
    return {"w": w, "value": float(value), "support": [int(j) for j in supp],
            "kkt_residual": kkt, "n_faces": n_faces,
            "all_face_values": [(c[0], list(c[2])) for c in cands]}


def value_curve(C, kmax=None):
    """v_k and G_k = v_1 - v_k for k = 1..M (eq. 8-9)."""
    M = C.shape[0]
    kmax = kmax or M
    v = [budgeted_qp(C, k)["value"] for k in range(1, kmax + 1)]
    return {"v": v, "G": [v[0] - vk for vk in v]}


# ----------------------------------------------------------------------------
# weight rules
# ----------------------------------------------------------------------------
def rule_WS(C):
    M = C.shape[0]
    w = np.zeros(M); w[int(np.argmin(np.diag(C)))] = 1.0
    return w


def rule_WE(C):
    M = C.shape[0]
    return np.ones(M) / M


def rule_WB(mse_list, n_scalar):
    """BIC-inspired weights: w_j propto exp(-(n/2) log MSE_j), n = number of
    scalar calibration observations (pooled rows x d).
    Computed with the log-sum-exp trick."""
    mse = np.asarray(mse_list, float)
    logw = -0.5 * n_scalar * np.log(np.maximum(mse, 1e-300))
    logw -= logw.max()
    w = np.exp(logw)
    return w / w.sum()


def rule_WD(C, mse_list, n_scalar):
    """Direct choice among 7 candidates (5 singles, equal, BIC) by calibration risk."""
    M = C.shape[0]
    cands = [np.eye(M)[j] for j in range(M)] + [rule_WE(C), rule_WB(mse_list, n_scalar)]
    names = [f"single{j}" for j in range(M)] + ["equal", "bic"]
    vals = [risk(w, C) for w in cands]
    j = int(np.argmin(vals))
    return cands[j], names[j]


def rule_WQ(C, k):
    return budgeted_qp(C, k)


def g_cal(w_alg, Chat, k):
    """Empirical suboptimality within the same budget family (eq. 17)."""
    return float(risk(w_alg, Chat) - budgeted_qp(Chat, k)["value"])


def certificate(w_alg, Chat, eps):
    """Computable lower bound on the population gain v_1 - R(w_alg) on the
    Hoeffding event: vhat_1 - Rhat(w_alg) - 2 eps."""
    return float(np.min(np.diag(Chat)) - risk(w_alg, Chat) - 2.0 * eps)


def eps_m(B_e2, m, M, delta=0.05):
    """eq. (18): B_e^2 sqrt(2/m log(2 M^2/delta))."""
    return float(B_e2 * math.sqrt(2.0 / m * math.log(2.0 * M * M / delta)))


def B_e2_from_limits(U, s, d):
    """eq. (15): (4/d) sum_l (U_l / s_l)^2 for clipped predictors and bounded labels."""
    return float(4.0 / d * np.sum((np.asarray(U) / np.asarray(s)) ** 2))


# ----------------------------------------------------------------------------
# self tests: vertex, two-member, dense, high-common-error cases
# ----------------------------------------------------------------------------
def self_test():
    rng = np.random.default_rng(0)
    rep = {}
    # dense example eq. (11)-(12): C = beta 11^T + s2 I  -> v_k = beta + s2/k
    M, beta, s2 = 5, 0.3, 0.2
    C = beta * np.ones((M, M)) + s2 * np.eye(M)
    vc = value_curve(C)
    rep["dense_vk_err"] = max(abs(vc["v"][k - 1] - (beta + s2 / k)) for k in range(1, M + 1))
    # vertex optimal: C_ab >= R_a for all b
    C2 = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 1.5], [1.0, 1.5, 3.0]])
    q = budgeted_qp(C2, 3)
    rep["vertex_support"] = q["support"]; rep["vertex_value_err"] = abs(q["value"] - 1.0)
    rep["vertex_no_witness"] = not two_member_witness(C2)["exists"]
    # two-member witness formula vs QP on the edge
    C3 = np.array([[1.0, 0.2], [0.2, 1.5]])
    tw = two_member_witness(C3)
    q3 = budgeted_qp(C3, 2)
    rep["two_member_gain_err"] = abs((C3[0, 0] - q3["value"]) - tw["gain"])
    rep["two_member_alpha_err"] = abs(q3["w"][1] - tw["alpha"])
    # ambiguity identity on random PSD matrices and random weights
    errs = []
    for _ in range(50):
        X = rng.standard_normal((5, 40)); C4 = X @ X.T / 40
        w = rng.dirichlet(np.ones(5))
        errs.append(abs(ambiguity_decomposition(w, C4)["identity_residual"]))
    rep["ambiguity_identity_max_err"] = max(errs)
    # monotonicity v_1 >= v_2 >= ... and G_M > 0 iff G_2 > 0 (eq. 9-10) on random PSD
    mono, iff = True, True
    for _ in range(50):
        X = rng.standard_normal((5, 30)); C5 = X @ X.T / 30 + 0.01 * np.eye(5)
        v = value_curve(C5)["v"]
        mono &= all(v[i] >= v[i + 1] - 1e-12 for i in range(4))
        G = [v[0] - x for x in v]
        iff &= ((G[4] > 1e-12) == (G[1] > 1e-12))
        # cross-check with SLSQP on full simplex
        ws = _slsqp_face(C5)
        if ws is not None and risk(ws, C5) < v[4] - 1e-8:
            mono = False
    rep["monotone"] = mono; rep["G_M_iff_G_2"] = iff
    # BIC rule sanity: strongly concentrated for large n
    rep["bic_large_n_max"] = float(rule_WB([1.0, 1.1, 1.2], 10000).max())
    return rep


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=1, default=str))
