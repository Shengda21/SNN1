"""Numerical self-check of the theory in Supplementary Material S6 (new version).

Checks
  (a) identities (I1)-(I3), R_circ = min_c R(cF) at c = s, scaling of (beta, |nu|^2),
      C = P_y (1-beta)(1-beta)^T + N, conditional mean on the curve |nu|^2 = beta(1-beta)P_y
  (b) Lemma 2: correlation (1+1/SNR)^(-1/2), P_flip = arctan(SNR^(-1/2))/pi against Monte Carlo
      (SNR 0.1-100), the integral formula for an offset threshold, monotonicity in SNR at a
      fixed standardized offset, and the counterexample at a fixed absolute offset
  (c) Proposition 3: closed-form slope s against the regression slope, closed-form risks,
      non-negativity of the gap and its equality condition Sigma_delta A^T = k Sigma A^T
  (d) Proposition 4: closed-form critical level against numerical root finding and against the
      support change of the exact simplex solution on a fine grid
  (e) Lemma 1: hand-derived surrogate back-propagation against torch autograd on the SNN class
      of common/models.py; state independence of the straight-through operator
  (f) Proposition 5 and the split-sample certificate: ratio -> 1, excess-risk rate, coverage
  (g) Proposition 6: label-free moment matrix against the target-labelled one; payload counterexample

Run:  python theory_selfcheck.py            (about one to three minutes on a laptop CPU)
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, optimize, stats

RNG = np.random.default_rng(20261111)
ROWS = []          # (check, quantity, value, tolerance or expectation, verdict)


def report(check, quantity, value, target, ok):
    ROWS.append((check, quantity, value, target, "PASS" if ok else "FAIL"))
    print(f"[{check}] {quantity}: {value}  ({target})  -> {'PASS' if ok else 'FAIL'}")


# ----------------------------------------------------------------------------------------------
# helpers: trajectory-averaged inner product and the simplex solver of S6.5
# ----------------------------------------------------------------------------------------------
def make_ip(traj_id):
    """<U,V> = E_Z[(1/T_Z) sum_t <U_t, V_t>] with the empirical law over trajectories."""
    ids, inv, counts = np.unique(traj_id, return_inverse=True, return_counts=True)
    wts = 1.0 / (len(ids) * counts[inv])          # each row weighted by 1/(n_traj * T_Z)

    def ip(U, V):
        return float(np.sum(wts * np.sum(U * V, axis=1)))
    return ip


def simplex_opt(C, kmax=None, tol=1e-12):
    """Exact minimiser of w^T C w over {w >= 0, 1^T w = 1, |supp w| <= kmax} by face enumeration."""
    M = C.shape[0]
    kmax = M if kmax is None else kmax
    best = (np.inf, None, None)
    for r in range(1, kmax + 1):
        for S in itertools.combinations(range(M), r):
            CS = C[np.ix_(S, S)]
            try:
                z = np.linalg.solve(CS, np.ones(r))
            except np.linalg.LinAlgError:
                continue
            if z.sum() <= 0:
                continue
            wS = z / z.sum()
            if np.any(wS < -1e-12):
                continue
            v = 1.0 / z.sum()
            if best[1] is None or v < best[0] - tol * max(1.0, abs(best[0])):
                w = np.zeros(M)
                w[list(S)] = np.clip(wS, 0, None)
                w /= w.sum()
                best = (v, w, S)
    return best


# ----------------------------------------------------------------------------------------------
# (a) identities
# ----------------------------------------------------------------------------------------------
def check_identities():
    n_traj, d, M = 40, 7, 5
    lens = RNG.integers(50, 200, n_traj)
    traj = np.repeat(np.arange(n_traj), lens)
    n = traj.size
    ip = make_ip(traj)
    Y = RNG.standard_normal((n, d)) @ RNG.standard_normal((d, d)) / math.sqrt(d)
    F = [RNG.uniform(0.3, 1.2) * Y + RNG.uniform(0.2, 0.8) * RNG.standard_normal((n, d)) / math.sqrt(d)
         + 0.1 * RNG.standard_normal((1, d)) for _ in range(M)]
    Py = ip(Y, Y)
    beta = np.array([ip(f, Y) / Py for f in F])
    nu = [f - b * Y for f, b in zip(F, beta)]
    err = {"I1": 0.0, "orth": 0.0, "I2": 0.0, "Rcirc": 0.0, "argmin": 0.0, "scale": 0.0, "I3": 0.0,
           "C": 0.0, "Phi": 0.0}
    for f, b, v in zip(F, beta, nu):
        R = ip(f - Y, f - Y)
        err["I1"] = max(err["I1"], abs(R - ((1 - b) ** 2 * Py + ip(v, v))))
        err["orth"] = max(err["orth"], abs(ip(v, Y)))
        s = ip(f, Y) / ip(f, f)
        cos2 = ip(f, Y) ** 2 / (ip(f, f) * Py)
        Rc, scost = Py * (1 - cos2), ip(f, f) * (1 - s) ** 2
        err["I2"] = max(err["I2"], abs(R - (Rc + scost)))
        res = optimize.minimize_scalar(lambda c: ip(c * f - Y, c * f - Y), bounds=(-5, 5), method="bounded",
                                       options={"xatol": 1e-12})
        err["Rcirc"] = max(err["Rcirc"], abs(res.fun - Rc))
        err["argmin"] = max(err["argmin"], abs(res.x - s))
        c = RNG.uniform(0.2, 2.0)
        bc = ip(c * f, Y) / Py
        nc = ip(c * f - bc * Y, c * f - bc * Y)
        err["scale"] = max(err["scale"], abs(bc - c * b), abs(nc - c ** 2 * ip(v, v)))
    N = np.array([[ip(a, b2) for b2 in nu] for a in nu])
    E = [f - Y for f in F]
    C = np.array([[ip(a, b2) for b2 in E] for a in E])
    Phi = np.array([[ip(a, b2) for b2 in F] for a in F])
    g = np.array([ip(f, Y) for f in F])
    err["C"] = np.max(np.abs(C - (Py * np.outer(1 - beta, 1 - beta) + N)))
    for _ in range(20):
        w = RNG.normal(size=M)                          # arbitrary weights, no sum constraint
        Fw = sum(wj * f for wj, f in zip(w, F))
        bw = ip(Fw, Y) / Py
        nw = Fw - bw * Y
        err["I3"] = max(err["I3"], abs(bw - w @ beta), abs(ip(nw, nw) - w @ N @ w))
        err["Phi"] = max(err["Phi"], abs(ip(Fw - Y, Fw - Y) - (w @ Phi @ w - 2 * g @ w + Py)))
    for k, v in err.items():
        report("a", f"max abs error {k}", f"{v:.2e}", "< 1e-9", v < 1e-9)

    # conditional mean: linear-Gaussian sample, (i) exact in-sample projection, (ii) population formula
    p, dd, nn = 6, 3, 400_000
    A = RNG.standard_normal((dd, p))
    L = RNG.standard_normal((p, p)); Sig = L @ L.T / p + 0.2 * np.eye(p)
    Sd = np.diag(RNG.uniform(0.1, 1.0, p))
    x = RNG.multivariate_normal(np.zeros(p), Sig, nn)
    xt = x + RNG.multivariate_normal(np.zeros(p), Sd, nn)
    y = x @ A.T + 0.3 * RNG.standard_normal((nn, dd))
    ipn = make_ip(np.zeros(nn, int))
    B_ls, *_ = np.linalg.lstsq(xt, y, rcond=None)      # in-sample projection onto linear maps of x~
    G = xt @ B_ls
    K = Sig @ np.linalg.inv(Sig + Sd)
    Gpop = xt @ (A @ K).T
    for name, Gm, tol in [("in-sample projection", G, 1e-9), ("population E[y|x~]", Gpop, 1e-2)]:
        Py2 = ipn(y, y); b = ipn(Gm, y) / Py2; s = ipn(Gm, y) / ipn(Gm, Gm)
        nu2 = ipn(Gm - b * y, Gm - b * y)
        dev = abs(nu2 - b * (1 - b) * Py2) / Py2
        report("a", f"conditional mean ({name}): |s-1|, rel. distance to curve",
               f"{abs(s - 1):.1e}, {dev:.1e}", f"< {tol:g}", abs(s - 1) < tol and dev < tol)


# ----------------------------------------------------------------------------------------------
# (b) Lemma 2
# ----------------------------------------------------------------------------------------------
def pflip_offset(snr, kappa):
    """E[ bar Phi( sqrt(SNR) |Z - kappa| ) ], Z ~ N(0,1), computed with u = sqrt(SNR)(Z - kappa):
    P = SNR^{-1/2} int_0^inf [phi(kappa + u/sqrt(SNR)) + phi(kappa - u/sqrt(SNR))] barPhi(u) du."""
    q = math.sqrt(snr)
    f = lambda u: (stats.norm.pdf(kappa + u / q) + stats.norm.pdf(kappa - u / q)) * stats.norm.sf(u)
    val, _ = integrate.quad(f, 0, np.inf, limit=400, epsabs=1e-14, epsrel=1e-11)
    return val / q


def check_lemma2():
    n = 2_000_000
    snrs = np.logspace(-1, 2, 13)
    max_dc, max_z, max_rel = 0.0, 0.0, 0.0
    for snr in snrs:
        X = RNG.standard_normal(n)
        Nn = RNG.standard_normal(n) / math.sqrt(snr)
        Xt = X + Nn
        rc = np.corrcoef(X, Xt)[0, 1]
        max_dc = max(max_dc, abs(rc - (1 + 1 / snr) ** -0.5))
        pf = np.mean((X > 0) != (Xt > 0))
        p0 = math.atan(snr ** -0.5) / math.pi
        se = math.sqrt(p0 * (1 - p0) / n)
        max_z = max(max_z, abs(pf - p0) / se)
        max_rel = max(max_rel, abs(pf - p0) / p0)
        # quadrant formula
        assert abs(0.5 - math.asin((1 + 1 / snr) ** -0.5) / math.pi - p0) < 1e-12
    report("b", "corr(w'x, w'x~) vs (1+1/SNR)^(-1/2), max abs diff (13 SNR in [0.1,100])",
           f"{max_dc:.1e}", "< 2e-3", max_dc < 2e-3)
    report("b", "P_flip MC vs arctan(SNR^-1/2)/pi: max |z|, max rel. error",
           f"{max_z:.2f}, {max_rel:.1e}", "|z| < 4", max_z < 4)
    # offset threshold: integral formula vs MC, monotonicity in SNR at fixed kappa, max at kappa=0
    mz, mono, peak = 0.0, True, True
    for kappa in [0.25, 0.5, 1.0, 2.0]:
        prev = None
        for snr in snrs:
            pi_ = pflip_offset(snr, kappa)
            if prev is not None and not pi_ < prev:
                mono = False
            prev = pi_
            if pi_ > math.atan(snr ** -0.5) / math.pi + 1e-12:
                peak = False
        for snr in [0.3, 3.0, 30.0]:
            X = RNG.standard_normal(n)
            Xt = X + RNG.standard_normal(n) / math.sqrt(snr)
            pf = np.mean((X > kappa) != (Xt > kappa))
            p0 = pflip_offset(snr, kappa)
            mz = max(mz, abs(pf - p0) / math.sqrt(p0 * (1 - p0) / n))
    report("b", "offset threshold: MC vs E[barPhi(sqrt(SNR)|Z-kappa|)], max |z|", f"{mz:.2f}", "|z| < 4", mz < 4)
    report("b", "P_flip strictly decreasing in SNR at fixed kappa (0.25,0.5,1,2)", str(mono), "True", mono)
    report("b", "P_flip(kappa) <= P_flip(0) for every SNR", str(peak), "True", peak)
    # fixed absolute offset h: not monotone in the signal variance
    h, sn = 2.0, 1.0
    vals = []
    for ss in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        vals.append(pflip_offset((ss / sn) ** 2, h / ss))
    nonmono = not all(np.diff(vals) < 0)
    report("b", "fixed absolute offset h=2, sigma_n=1, sigma_s=0.25..8: P_flip",
           ", ".join(f"{v:.4f}" for v in vals), "non-monotone in SNR (statement needs fixed kappa)", nonmono)
    # small-noise order: P_flip ~ rho/pi
    rho = np.array([1e-3, 1e-2, 0.05])
    sl = np.polyfit(np.log(rho), np.log(np.arctan(rho) / math.pi), 1)[0]
    report("b", "log-log slope of P_flip in rho = SNR^-1/2 at small rho", f"{sl:.4f}", "= 1 (order rho, not rho^2)",
           abs(sl - 1) < 1e-3)
    worst = 0.0
    for kappa in [0.0, 0.5, 1.0, 2.0]:
        r = 1e-3
        worst = max(worst, abs(pflip_offset(1 / r ** 2, kappa) / (r * math.exp(-kappa ** 2 / 2) / math.pi) - 1))
    report("b", "small noise: P_flip / (exp(-kappa^2/2) SNR^-1/2 / pi) at SNR^-1/2 = 1e-3, kappa in {0,.5,1,2}",
           f"max |ratio-1| = {worst:.1e}", "< 1e-2", worst < 1e-2)


# ----------------------------------------------------------------------------------------------
# (c) Proposition 3
# ----------------------------------------------------------------------------------------------
def prop3_closed(Ab, Sig, Sd, Peta):
    Ssig = np.trace(Ab @ Sig @ Ab.T)
    Snoi = np.trace(Ab @ Sd @ Ab.T)
    s = Ssig / (Ssig + Snoi)
    RF = Snoi + Peta
    scost = Snoi ** 2 / (Ssig + Snoi)
    Rcirc = Peta + Ssig * Snoi / (Ssig + Snoi)
    par = Sig @ np.linalg.solve(Sig + Sd, Sd)
    Rstar = Peta + np.trace(Ab @ par @ Ab.T)
    excess_s8 = np.trace(Ab @ Sd @ np.linalg.solve(Sig + Sd, Sd) @ Ab.T)
    return dict(Ssig=Ssig, Snoi=Snoi, s=s, RF=RF, scost=scost, Rcirc=Rcirc, Rstar=Rstar, excess_s8=excess_s8)


def check_prop3():
    p, d, n = 6, 3, 1_000_000
    Ab = RNG.standard_normal((d, p)) / math.sqrt(d)
    L = RNG.standard_normal((p, p)); Sig = L @ L.T / p + 0.1 * np.eye(p)
    Ld = RNG.standard_normal((p, p)); Sd = 0.5 * Ld @ Ld.T / p + np.diag(RNG.uniform(0.05, 0.5, p))
    Seta = np.diag(RNG.uniform(0.01, 0.1, d)); Peta = np.trace(Seta)
    cf = prop3_closed(Ab, Sig, Sd, Peta)
    x = RNG.multivariate_normal(np.zeros(p), Sig, n)
    xt = x + RNG.multivariate_normal(np.zeros(p), Sd, n)
    Y = x @ Ab.T + RNG.multivariate_normal(np.zeros(d), Seta, n)
    F = xt @ Ab.T
    ip = lambda U, V: float(np.mean(np.sum(U * V, 1)))
    slope = ip(F, Y) / ip(F, F)                     # least-squares slope of Y on F
    K = Sig @ np.linalg.inv(Sig + Sd)
    G = xt @ (Ab @ K).T
    emp = dict(s=slope, RF=ip(F - Y, F - Y), Rcirc=ip(slope * F - Y, slope * F - Y), Rstar=ip(G - Y, G - Y))
    for k in ["s", "RF", "Rcirc", "Rstar"]:
        rel = abs(emp[k] - cf[k]) / abs(cf[k])
        report("c", f"closed form vs sample (n=1e6): {k}", f"{cf[k]:.4f} vs {emp[k]:.4f} (rel {rel:.1e})", "rel < 1e-2",
               rel < 1e-2)
    report("c", "s < 1 when S_noise > 0", f"{cf['s']:.4f}", "< 1", cf["s"] < 1)
    dec = abs((cf["RF"] - cf["Rstar"]) - (cf["scost"] + (cf["Rcirc"] - cf["Rstar"])))
    dec2 = abs((cf["RF"] - cf["Rstar"]) - cf["excess_s8"])
    report("c", "R(F)-R* = scale cost + gap, and = S8.1 excess", f"{dec:.1e}, {dec2:.1e}", "< 1e-12",
           dec < 1e-12 and dec2 < 1e-12)
    # non-negativity over random instances
    gaps = []
    for _ in range(20000):
        p2, d2 = RNG.integers(1, 8), RNG.integers(1, 5)
        A2 = RNG.standard_normal((d2, p2))
        L2 = RNG.standard_normal((p2, p2)); S2 = L2 @ L2.T + 1e-3 * np.eye(p2)
        if RNG.random() < 0.5:
            D2 = np.diag(RNG.uniform(0, 2, p2))
        else:
            Q = RNG.standard_normal((p2, p2)); D2 = Q @ Q.T * RNG.uniform(0, 2)
        c2 = prop3_closed(A2, S2, D2, 0.0)
        gaps.append((c2["Rcirc"] - c2["Rstar"]) / max(c2["Ssig"], 1e-12))
    gaps = np.array(gaps)
    report("c", "gap R_circ - R* >= 0 on 20,000 random (A, Sigma, Sigma_delta): min relative gap",
           f"{gaps.min():.2e}", ">= -1e-12", gaps.min() >= -1e-12)
    # equality cases
    kap = 0.7
    eq1 = prop3_closed(Ab, Sig, kap * Sig, Peta)
    # rank-one A whose row is a generalised eigenvector of (Sd, Sig); Sd not proportional to Sig
    evals, evecs = np.linalg.eig(np.linalg.solve(Sig, Sd))
    a = np.real(evecs[:, 0])
    A1 = np.vstack([a, 2 * a])
    eq2 = prop3_closed(A1, Sig, Sd, Peta)
    gap1, gap2 = eq1["Rcirc"] - eq1["Rstar"], eq2["Rcirc"] - eq2["Rstar"]
    report("c", "equality: Sigma_delta = k Sigma -> gap; rows of A in one eigenspace of Sigma^-1 Sigma_delta -> gap",
           f"{gap1:.1e}, {gap2:.1e}", "both 0 (|.| < 1e-12)", abs(gap1) < 1e-12 and abs(gap2) < 1e-12)
    A3 = np.vstack([np.real(evecs[:, 0]), np.real(evecs[:, 1])])   # two different eigenvalues
    eq3 = prop3_closed(A3, Sig, Sd, Peta)
    # per-joint scale: R_circ >= R_joint >= R*, equality with R* iff each row is a generalised eigenvector
    def r_joint(Ab_, Sig_, Sd_, Peta_):
        tot = Peta_
        for a_ in Ab_:
            ss, sn = a_ @ Sig_ @ a_, a_ @ Sd_ @ a_
            tot += ss * sn / (ss + sn)
        return tot
    Rj = r_joint(Ab, Sig, Sd, Peta)
    A4 = np.vstack([np.real(evecs[:, 0]), np.real(evecs[:, 1])])
    c4 = prop3_closed(A4, Sig, Sd, Peta)
    Rj4 = r_joint(A4, Sig, Sd, Peta)
    okj = cf["Rcirc"] >= Rj - 1e-12 >= cf["Rstar"] - 2e-12 and abs(Rj4 - c4["Rstar"]) < 1e-12
    report("c", "per-joint scale: R_circ >= R_joint >= R*; rows in different eigenspaces -> R_joint = R*",
           f"{cf['Rcirc']:.4f} >= {Rj:.4f} >= {cf['Rstar']:.4f}; {Rj4 - c4['Rstar']:.1e}", "ordering, 0", okj)
    report("c", "rows in two different eigenspaces -> gap > 0", f"{eq3['Rcirc'] - eq3['Rstar']:.3e}", "> 0",
           eq3["Rcirc"] - eq3["Rstar"] > 1e-8)


# ----------------------------------------------------------------------------------------------
# (d) Proposition 4
# ----------------------------------------------------------------------------------------------
def first_change(Cfun, a, grid, tol=1e-9):
    """Smallest rho on the grid at which the exact simplex optimum leaves the vertex a, refined by
    bisection on the same predicate to tolerance tol."""
    moved = lambda r: simplex_opt(Cfun(r), tol=0.0)[2] != (a,)
    prev = grid[0]
    for r in grid[1:]:
        if moved(r):
            lo, hi = prev, r
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if moved(mid) else (mid, hi)
            return hi
        prev = r
    return np.nan


def check_prop4():
    max_cf, max_grid, n_ok, n_tot = 0.0, 0.0, 0, 0
    max_cf2 = 0.0
    step = 1e-2
    for trial in range(60):
        M = 4
        # residuals e_j(rho) = b_j + sqrt(rho) m_j + rho n_j with mutually orthogonal blocks, so that
        # C(rho) = B + rho G1 + rho^2 G2 exactly; member 0 is the optimal vertex at rho = 0
        b0 = RNG.standard_normal(6) * 0.3
        Bm = np.vstack([b0] + [(1 + RNG.uniform(0.1, 1.0)) * b0 + RNG.standard_normal(6) * 0.3 *
                               np.r_[0, 0, 0, 1, 1, 1] for _ in range(M - 1)])
        Bm[:, 3:] -= 0                                    # b_0 lives in the first three coordinates
        Bm[0, 3:] = 0
        B = Bm @ Bm.T
        Gm = RNG.standard_normal((M, 4))
        Gm[0] *= RNG.uniform(1.5, 2.5)                    # member 0 degrades fastest
        G2 = Gm @ Gm.T
        Hm = RNG.standard_normal((M, 2)) * 0.3
        Hm[0] *= 1.5
        G1 = Hm @ Hm.T
        a = 0
        if not np.all(B[a, 1:] > B[a, a]):
            continue
        # quadratic model C = B + rho^2 G2
        roots_cf, roots_num = [], []
        for b in range(1, M):
            D0, D2 = B[a, b] - B[a, a], G2[a, a] - G2[a, b]
            if D2 > 0:
                roots_cf.append(math.sqrt(D0 / D2))
                f = lambda r: (B[a, b] + r ** 2 * G2[a, b]) - (B[a, a] + r ** 2 * G2[a, a])
                roots_num.append(optimize.brentq(f, 0, 50))
        if not roots_cf:
            continue
        n_tot += 1
        rs_cf, rs_num = min(roots_cf), min(roots_num)
        max_cf = max(max_cf, abs(rs_cf - rs_num))
        grid = np.arange(0, rs_num * 1.5 + 0.01, step)
        rg = first_change(lambda r: B + r ** 2 * G2, a, grid)
        if np.isfinite(rg):
            max_grid = max(max_grid, abs(rg - rs_num))
            n_ok += abs(rg - rs_num) <= 1e-6
        # linear-quadratic model C = B + rho G1 + rho^2 G2
        roots_cf2, roots_num2 = [], []
        for b in range(1, M):
            D0, D1, D2 = B[a, b] - B[a, a], G1[a, a] - G1[a, b], G2[a, a] - G2[a, b]
            disc = D1 ** 2 + 4 * D0 * D2
            if disc >= 0 and D1 + math.sqrt(disc) > 0:
                roots_cf2.append(2 * D0 / (D1 + math.sqrt(disc)))
                f = lambda r: D0 - D1 * r - D2 * r ** 2
                hi = roots_cf2[-1] * 1.0000001 + 1e-9
                roots_num2.append(optimize.brentq(f, 0, hi + 1e-6) if f(hi + 1e-6) < 0 else np.nan)
        if roots_cf2:
            max_cf2 = max(max_cf2, np.nanmax(np.abs(np.array(roots_cf2) - np.array(roots_num2))))
    report("d", f"quadratic model: closed-form rho* vs brentq ({n_tot} random dictionaries)", f"{max_cf:.1e}",
           "< 1e-9", max_cf < 1e-9)
    report("d", f"support change of the exact simplex optimum (grid {step:g} + bisection) vs rho*",
           f"max |diff| = {max_grid:.1e}; {n_ok}/{n_tot} agree to 1e-6", "all agree", n_ok == n_tot and n_tot > 20)
    report("d", "linear-quadratic model: 2D0/(D1+sqrt(D1^2+4D0D2)) vs brentq", f"{max_cf2:.1e}", "< 1e-6",
           max_cf2 < 1e-6)


# ----------------------------------------------------------------------------------------------
# (e) Lemma 1 against autograd
# ----------------------------------------------------------------------------------------------
def check_lemma1():
    try:
        import torch
    except ImportError:
        report("e", "Lemma 1", "skipped (no torch)", "-", True)
        return
    code = Path(__file__).resolve().parent
    if not code.exists():
        report("e", "Lemma 1", f"skipped ({code} not found)", "-", True)
        return
    sys.path.insert(0, str(code))
    from common.models import SNN
    from common.surrogates import surrogate_curve
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    def forward_record(net, x):
        B = x.shape[0]; H = net.hidden
        lifs, fcs = [net.lif1, net.lif2, net.lif3], [net.fc1, net.fc2, net.fc3]
        m = [torch.zeros(B, H) for _ in range(3)]
        m4 = torch.zeros(B, H); m5 = torch.zeros(B, net.n_out)
        rec = []
        for t in range(net.T):
            a, st = x, {}
            for l in range(3):
                lam, th = lifs[l].beta.detach().clamp(0, 1), lifs[l].thr.detach()
                v = lam * m[l] + fcs[l](a).detach()
                z = (v - th > 0).double()
                st[f"u{l+1}"], st[f"a{l}"] = v - th, a
                m[l] = v - th * z
                a = z
            h = torch.tanh(m[2]); st["h"] = h
            m4 = net.li4.beta.detach().clamp(0, 1) * m4 + net.fc4(h).detach()
            m5 = net.li5.beta.detach().clamp(0, 1) * m5 + net.fc5(m4).detach()
            st["yhat"] = m5
            rec.append(st)
        return rec

    def lemma_grads(net, x, y, gfun, temporal):
        """Recursions of Lemma 1; temporal(lam, th, g) is the factor multiplying eps_{t+1}."""
        rec = forward_record(net, x)
        T, B, d, H = net.T, x.shape[0], net.n_out, net.hidden
        lam = [l.beta.detach().clamp(0, 1) for l in [net.lif1, net.lif2, net.lif3]]
        th = [l.thr.detach() for l in [net.lif1, net.lif2, net.lif3]]
        lam4, lam5 = net.li4.beta.detach().clamp(0, 1), net.li5.beta.detach().clamp(0, 1)
        W = [None, net.fc1.weight.detach(), net.fc2.weight.detach(), net.fc3.weight.detach(),
             net.fc4.weight.detach(), net.fc5.weight.detach()]
        e5n, e4n = torch.zeros(B, d), torch.zeros(B, H)
        en = [torch.zeros(B, H) for _ in range(3)]
        gW = [torch.zeros_like(W[l + 1]) for l in range(3)]
        for t in reversed(range(T)):
            st = rec[t]
            e5 = 2 * (st["yhat"] - y) / (T * B * d) + lam5 * e5n          # MSELoss over [T, B, d]
            e4 = e5 @ W[5] + lam4 * e4n
            g3 = gfun(st["u3"])
            e3 = (1 - th[2] * g3) * ((1 - st["h"] ** 2) * (e4 @ W[4]) + lam[2] * en[2])
            g2 = gfun(st["u2"])
            e2 = g2 * (e3 @ W[3]) + temporal(lam[1], th[1], g2) * en[1]
            g1 = gfun(st["u1"])
            e1 = g1 * (e2 @ W[2]) + temporal(lam[0], th[0], g1) * en[0]
            for l, e in enumerate([e1, e2, e3]):
                gW[l] += e.T @ st[f"a{l}"]
            e5n, e4n, en = e5, e4, [e1, e2, e3]
        return gW

    worst_ok, worst_alt = 0.0, np.inf
    for name, width in [("atan", 4.0), ("sigmoid", 10.0), ("triangular", 0.5), ("sre", 10.0), ("ste", None)]:
        net = SNN(5, 3, hidden=9, T=5, surrogate=name, width=width).double()
        with torch.no_grad():
            for l in [net.lif1, net.lif2, net.lif3]:
                l.thr.uniform_(0.5, 1.5); l.beta.uniform_(0.4, 1.0)
            for fc in [net.fc1, net.fc2, net.fc3]:
                fc.weight.mul_(3.0); fc.bias.uniform_(0.0, 1.0)
        x, y = torch.randn(128, 5), torch.randn(128, 3)
        out = net(x)
        loss = torch.nn.functional.mse_loss(out, y.unsqueeze(0).expand_as(out))
        net.zero_grad(); loss.backward()
        auto = [net.fc1.weight.grad, net.fc2.weight.grad, net.fc3.weight.grad]
        gfun = lambda u, name=name, width=width: torch.as_tensor(surrogate_curve(name, width, u.numpy()))
        ok = lemma_grads(net, x, y, gfun, lambda lam, th, g: lam * (1 - th * g))
        alt = lemma_grads(net, x, y, gfun, lambda lam, th, g: lam - th * g)
        rel = lambda P, Q: max(float((p - q).abs().max() / p.abs().max()) for p, q in zip(P, Q))
        worst_ok = max(worst_ok, rel(auto, ok))
        worst_alt = min(worst_alt, rel(auto, alt))
    report("e", "autograd vs Lemma 1 recursion, temporal factor lam*(1-theta*g), 5 surrogates: max rel. error",
           f"{worst_ok:.1e}", "< 1e-12", worst_ok < 1e-12)
    report("e", "autograd vs the alternative factor lam - theta*g: min over surrogates of max rel. error",
           f"{worst_alt:.1e}", "> 1e-3 (alternative is not the implemented recursion)", worst_alt > 1e-3)
    # straight-through at initialisation (thr = 1 exactly)
    net = SNN(5, 3, hidden=9, T=5, surrogate="ste").double()
    x, y = torch.randn(128, 5), torch.randn(128, 3)
    out = net(x)
    torch.nn.functional.mse_loss(out, y.unsqueeze(0).expand_as(out)).backward()
    gz = max(float(f.weight.grad.abs().max()) for f in [net.fc1, net.fc2, net.fc3])
    gt = float(net.lif3.thr.grad.abs().max())
    report("e", "STE at initialisation (theta = 1): max |dL/dW_1..3|, max |dL/dtheta_3|", f"{gz:.1e}, {gt:.2e}",
           "0 and > 0", gz == 0.0 and gt > 0)
    # state independence: inject a fixed error at the post-reset membrane of layer 2 and read dL/db_1
    def bias_grad(net, x, r):
        B = x.shape[0]
        m1 = torch.zeros(B, net.hidden); m2 = torch.zeros_like(m1)
        Lr = 0.0
        for t in range(net.T):
            s1, m1 = net.lif1(net.fc1(x), m1)
            s2, m2 = net.lif2(net.fc2(s1), m2)
            Lr = Lr + (m2 * r[t]).sum() / B
        net.zero_grad(); Lr.backward()
        return net.fc1.bias.grad.clone()
    diffs = {}
    for name, width in [("ste", None), ("atan", 4.0)]:
        net = SNN(5, 3, hidden=9, T=5, surrogate=name, width=width).double()
        with torch.no_grad():
            for l in [net.lif1, net.lif2]:
                l.thr.uniform_(0.5, 1.5); l.beta.uniform_(0.4, 1.0)
            net.fc1.weight.mul_(3.0)
        r = torch.randn(net.T, net.hidden)
        g_a = bias_grad(net, torch.randn(64, 5), r)
        g_b = bias_grad(net, 3 * torch.randn(64, 5) + 1, r)
        diffs[name] = float((g_a - g_b).abs().max() / g_a.abs().max())
    report("e", "same injected layer-2 error, two input batches: rel. change of the credit at layer 1 (STE, ATan)",
           f"{diffs['ste']:.1e}, {diffs['atan']:.1e}", "STE 0 (state independent), ATan > 0",
           diffs["ste"] < 1e-12 and diffs["atan"] > 1e-3)


# ----------------------------------------------------------------------------------------------
# (f) Proposition 5 and the certificate
# ----------------------------------------------------------------------------------------------
def check_prop5():
    M, T = 4, 50
    b = np.array([0.30, 0.32, 0.35, 0.40])                 # common bias part of the residuals
    Lm = RNG.standard_normal((M, M)) * 0.15
    Smu = Lm @ Lm.T                                         # trajectory-level random effect
    Le = RNG.standard_normal((M, M)) * 0.5
    Seps = Le @ Le.T + 0.05 * np.eye(M)                     # step-level noise
    C = np.outer(b, b) + Smu + Seps

    def draw_A(m):
        mu = RNG.multivariate_normal(np.zeros(M), Smu, m)
        eps = RNG.multivariate_normal(np.zeros(M), Seps, (m, T))
        e = b[None, None, :] + mu[:, None, :] + eps
        return np.einsum("rti,rtj->rij", e, e) / T

    v, wstar, Sstar = simplex_opt(C)
    ms = [2, 5, 10, 20, 50, 100, 200, 400]
    exc, ratio = [], []
    for m in ms:
        ex = []
        for _ in range(300):
            Ch = draw_A(m).mean(0)
            _, wh, _ = simplex_opt(Ch)
            ex.append(wh @ C @ wh - v)
        exc.append(np.mean(ex)); ratio.append(1 + np.mean(ex) / v)
    slope = np.polyfit(np.log(ms[3:]), np.log(exc[3:]), 1)[0]
    report("f", f"simplex: optimum support {Sstar}; R(w_hat)/inf R at m = 2..400",
           ", ".join(f"{r:.4f}" for r in ratio), "decreasing to 1", ratio[-1] < 1.002 and ratio[-1] < ratio[0])
    report("f", "log-log slope of the mean excess risk in m (m >= 20)", f"{slope:.2f}", "about -1 (O_p(1/m))",
           -1.3 < slope < -0.7)
    # budget family W_2 (finite union of faces)
    v2, w2, S2 = simplex_opt(C, kmax=2)
    ex2 = []
    for _ in range(300):
        Ch = draw_A(100).mean(0)
        _, wh, _ = simplex_opt(Ch, kmax=2)
        ex2.append(wh @ C @ wh - v2)
    report("f", "budget family W_2 at m = 100: mean ratio", f"{1 + np.mean(ex2) / v2:.4f}", "close to 1",
           1 + np.mean(ex2) / v2 < 1.01)
    # split-sample certificate: coverage of the one-sided 95% t lower bound
    for m in [10, 20, 40]:
        cover, cert, reps = 0, 0, 2000
        for _ in range(reps):
            A = draw_A(m)
            A1, A2 = A[: m // 2], A[m // 2:]
            C1 = A1.mean(0)
            _, wh, _ = simplex_opt(C1)
            ah = int(np.argmin(np.diag(C1)))
            G = A2[:, ah, ah] - np.einsum("i,rij,j->r", wh, A2, wh)
            lb = G.mean() - stats.t.ppf(0.95, len(G) - 1) * G.std(ddof=1) / math.sqrt(len(G))
            Gam = C[ah, ah] - wh @ C @ wh
            cover += lb <= Gam
            cert += lb > 0
        report("f", f"certificate m = {m}: coverage of the 95% lower bound, certification rate",
               f"{cover / reps:.3f}, {cert / reps:.3f}", "coverage >= 0.93", cover / reps >= 0.93)


# ----------------------------------------------------------------------------------------------
# (g) Proposition 6
# ----------------------------------------------------------------------------------------------
def check_prop6():
    p, M, H = 4, 3, 12
    Wm = [RNG.standard_normal((H, p)) for _ in range(M)]
    tm = [RNG.standard_normal(H) * 0.5 for _ in range(M)]
    am = [RNG.standard_normal(H) / H for _ in range(M)]
    wtrue = RNG.standard_normal(p)
    m0 = lambda x: np.sin(x @ wtrue) + 0.5 * x[:, 0]

    def members(x):                                   # hard-threshold members (piecewise constant)
        return np.stack([(x @ W.T > t).astype(float) @ a * 3 + 0.8 * m0(x) * (j + 1) / M
                         for j, (W, t, a) in enumerate(zip(Wm, tm, am))], 1)

    def Cmat(x_obs, y):
        E = members(x_obs) - y[:, None]
        return E.T @ E / len(y)

    v_true = np.array([0.5, 0.5, 0.0, 0.0]) ** 2
    # reference: target-labelled moment matrix from a very large sample
    xr = RNG.standard_normal((4_000_000, p))
    yr = m0(xr) + 0.1 * RNG.standard_normal(len(xr))
    C_tgt = Cmat(xr + RNG.standard_normal(xr.shape) * np.sqrt(v_true), yr)
    w_tgt = simplex_opt(C_tgt)[1]
    out = []
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        xn = RNG.standard_normal((n, p)); yn = m0(xn) + 0.1 * RNG.standard_normal(n)      # nominal, labelled
        xu = RNG.standard_normal((n, p)) + RNG.standard_normal((n, p)) * np.sqrt(v_true)  # target, unlabelled
        vhat = np.maximum(0, xu.var(0) - xn.var(0))
        C_lf = Cmat(xn + RNG.standard_normal(xn.shape) * np.sqrt(vhat), yn)
        w_lf = simplex_opt(C_lf)[1]
        out.append((n, np.max(np.abs(C_lf - C_tgt)) / np.max(np.abs(C_tgt)), w_lf @ C_tgt @ w_lf - w_tgt @ C_tgt @ w_tgt))
    report("g", "label-free C (nominal labels + simulated noise with v_hat) vs target C: max rel |dC|, excess risk",
           "; ".join(f"n={n}: {d:.1e}, {e:.1e}" for n, d, e in out), "decreasing in n",
           out[-1][1] < out[0][1] and out[-1][2] <= out[0][2] + 1e-12)
    # payload counterexample: same inputs, labels scaled by 1 + kappa
    yp = 0.6 * m0(xr) + 0.1 * RNG.standard_normal(len(xr))
    C_pay = Cmat(xr + RNG.standard_normal(xr.shape) * np.sqrt(v_true), yp)
    w_pay = simplex_opt(C_pay)[1]
    report("g", "plant change (labels x0.6, inputs unchanged): label-free weights vs target-optimal weights",
           f"{np.round(w_tgt, 3)} vs {np.round(w_pay, 3)}", "differ (target labels needed)",
           np.max(np.abs(w_tgt - w_pay)) > 0.05)


if __name__ == "__main__":
    print("== (a) identities ==");      check_identities()
    print("== (b) Lemma 2 ==");         check_lemma2()
    print("== (c) Proposition 3 ==");   check_prop3()
    print("== (d) Proposition 4 ==");   check_prop4()
    print("== (e) Lemma 1 ==");         check_lemma1()
    print("== (f) Proposition 5 ==");   check_prop5()
    print("== (g) Proposition 6 ==");   check_prop6()
    print("\n| check | quantity | value | target | verdict |\n|---|---|---|---|---|")
    for r in ROWS:
        print("| " + " | ".join(str(c) for c in r) + " |")
    nfail = sum(r[-1] == "FAIL" for r in ROWS)
    print(f"\n{len(ROWS) - nfail}/{len(ROWS)} checks passed")
    sys.exit(1 if nfail else 0)
