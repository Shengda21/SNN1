"""E24: a usable gain certificate by sample splitting (replaces the Hoeffding bound).

Within each calibration group (sim: group g = trajectories 10g..10g+9, first m of them;
merged groups give m = 20, 40, 100; sarcos: consecutive 100-row blocks), the units are
split in two halves. The first half fixes the weights w_hat (W1 = simplex + null, also
W0) and the best single member a_hat; on the second half the per-unit gains
G_r = L_r(a_hat) - L_r(w_hat) are i.i.d. given the first half, and the one-sided 95%
lower bound of their mean is computed by the t interval and by the percentile bootstrap.
The halves are then swapped (cross-fitting). Certified: lower bound > 0. Coverage: the
lower bound does not exceed the true gain R_test(a_hat) - R_test(w_hat) on the test role
(cov_*), or the gain on the test role pooled with the calibration units that the
replicate did not use (cov_*_held; a less noisy reference for large m).
The Hoeffding certificate of the manuscript (eq. 18, B_e^2 from the torque limits) is
reported for the arm as a reference.

python e24_certificate.py --platform sim --res ../results/sim
"""
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2x_common import SEEDS, MomFile, member_blocks, solve_space, pj_risk, pj_unit_risk
from common import calibration as cal

B_E2_SIM = 5186.022241826433   # results/sim/theory_quantities.json of the manuscript


def conds(platform):
    if platform == "sim":
        return [("N", "cal", "test", "N"), ("O1-L", "cal", "test", "O1_0.35"), ("O1-M", "cal", "test", "O1_0.7"),
                ("O1-H", "cal", "test", "O1_1.0"), ("O2B", "cal", "test", "O2B_0.7"), ("P1", "cal_P1", "test_P1", "N")]
    return [("N", "cal", "test", "N"), ("O1-L", "cal", "test", "O1_0.25"), ("O1-M", "cal", "test", "O1_0.5"),
            ("O1-H", "cal", "test", "O1_0.7"), ("O1qqd", "cal", "test", "O1qqd_0.5")]


def replicates(platform, n_units):
    """list of (m, replicate id, unit ids)"""
    out = []
    if platform == "sim":
        for m in (4, 6, 8, 10):
            for g in range(10):
                out.append((m, g, np.arange(10 * g, 10 * g + m)))
        for m, k in ((20, 2), (40, 4)):
            for r in range(10 // k):
                out.append((m, r, np.arange(10 * k * r, 10 * k * (r + 1))))
        out.append((100, 0, np.arange(100)))
    else:
        for m in (4, 6, 10, 20, 40):
            for r in range(n_units // m):
                out.append((m, r, np.arange(m * r, m * (r + 1))))
    return out


def lower_bounds(G, rng, n_boot=2000):
    n = len(G)
    mu = float(np.mean(G)); sd = float(np.std(G, ddof=1)) if n > 1 else np.inf
    lb_t = mu - stats.t.ppf(0.95, n - 1) * sd / math.sqrt(n) if n > 1 else -np.inf
    bm = rng.choice(G, size=(n_boot, n), replace=True).mean(1)
    lb_b = float(np.percentile(bm, 5))
    return mu, sd, lb_t, lb_b


def job(a):
    platform, res, seed, cd = a
    lab, cr, tr, fl = cd
    Fc = MomFile(os.path.join(res, "orig", f"moments_{cr}_{fl}.npz"))
    Ft = MomFile(os.path.join(res, "orig", f"moments_{tr}_{fl}.npz"))
    B = member_blocks(platform, seed, "full")
    rows = []
    rng = np.random.default_rng(1000 + seed)
    for dn, mids in (("D-M", B["dm"]), ("D-S", B["ds"])):
        mc = Fc.sel(seed, mids)
        Pu, gu, yu, nu, ids = mc.per_unit()
        mtest = Ft.sel(seed, mids)
        Pt, gt, yt = mtest.pooled()
        n_test = float(mtest.n.sum())
        M = len(mids)
        for m, r, uids in replicates(platform, len(ids)):
            half = m // 2
            A_ids, B_ids = uids[:half], uids[half:2 * half]
            for direction, (fit_ids, ev_ids) in (("AB", (A_ids, B_ids)), ("BA", (B_ids, A_ids))):
                fi = np.isin(ids, fit_ids); ei = np.isin(ids, ev_ids)
                wf = nu[fi] / nu[fi].sum()
                Pf = np.einsum("g,gdij->dij", wf, Pu[fi]); gf = np.einsum("g,gdi->di", wf, gu[fi]); yf = wf @ yu[fi]
                Cf = Pf.mean(0) - gf.mean(0)[:, None] - gf.mean(0)[None, :] + yf.mean()
                a_hat = int(np.argmin(np.diag(Cf)))
                Ea = np.zeros((7, M)); Ea[:, a_hat] = 1.0
                for sp in ("W1", "W0"):
                    W, _ = solve_space(sp, Pf, gf, yf)
                    G = pj_unit_risk(Ea, Pu[ei], gu[ei], yu[ei]) - pj_unit_risk(W, Pu[ei], gu[ei], yu[ei])
                    mu, sd, lb_t, lb_b = lower_bounds(G, rng)
                    true_gain = pj_risk(Ea, Pt, gt, yt)[0] - pj_risk(W, Pt, gt, yt)[0]
                    # second reference: test units pooled with the calibration units not used by this replicate
                    un = ~np.isin(ids, uids)
                    if un.any():
                        gu_un = pj_unit_risk(Ea, Pu[un], gu[un], yu[un]) - pj_unit_risk(W, Pu[un], gu[un], yu[un])
                        held_gain = float((np.sum(nu[un] * gu_un) + n_test * true_gain) / (nu[un].sum() + n_test))
                    else:
                        held_gain = true_gain
                    rec = {"cond": lab, "seed": seed, "dict": dn, "space": sp, "m": m, "rep": r, "dir": direction,
                           "n_eval": int(ei.sum()), "a_hat": mids[a_hat], "mean_gain": mu, "sd_gain": sd,
                           "lb_t": lb_t, "lb_boot": lb_b, "true_gain": true_gain,
                           "cert_t": bool(lb_t > 0), "cert_boot": bool(lb_b > 0),
                           "cov_t": bool(lb_t <= true_gain), "cov_boot": bool(lb_b <= true_gain),
                           "heldout_gain": held_gain, "cov_t_held": bool(lb_t <= held_gain), "cov_boot_held": bool(lb_b <= held_gain),
                           "null_weight": float(1 - W[0].sum())}
                    if platform == "sim" and sp == "W0" and direction == "AB":
                        # manuscript certificate on all m units: vhat_1 - Rhat(w) - 2 eps_m
                        wall = nu[np.isin(ids, uids)]; sel = np.isin(ids, uids)
                        Pm = np.einsum("g,gdij->dij", wall / wall.sum(), Pu[sel]); gm = np.einsum("g,gdi->di", wall / wall.sum(), gu[sel])
                        ym = (wall / wall.sum()) @ yu[sel]
                        Cm = Pm.mean(0) - gm.mean(0)[:, None] - gm.mean(0)[None, :] + ym.mean()
                        q = cal.budgeted_qp(Cm, M)
                        eps = cal.eps_m(B_E2_SIM, m, M, 0.05)
                        rec["hoeffding_lb"] = float(np.min(np.diag(Cm)) - q["value"] - 2 * eps)
                    rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(args.platform, args.res, s, c) for c in conds(args.platform) for s in SEEDS]
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.res, "e24_rows.csv"), index=False)
    agg = df.groupby(["cond", "dict", "space", "m"]).agg(
        n_cases=("cert_t", "size"), cert_rate_t=("cert_t", "mean"), cert_rate_boot=("cert_boot", "mean"),
        coverage_t=("cov_t", "mean"), coverage_boot=("cov_boot", "mean"),
        coverage_t_held=("cov_t_held", "mean"), coverage_boot_held=("cov_boot_held", "mean"), mean_lb_t=("lb_t", "mean"),
        mean_lb_boot=("lb_boot", "mean"), mean_gain_eval=("mean_gain", "mean"), mean_true_gain=("true_gain", "mean"), mean_heldout_gain=("heldout_gain", "mean"),
        frac_true_gain_pos=("true_gain", lambda x: float(np.mean(x > 0)))).reset_index()
    if "hoeffding_lb" in df:
        h = df.dropna(subset=["hoeffding_lb"]).groupby(["cond", "dict", "space", "m"])["hoeffding_lb"].agg(["mean", "max"]).reset_index()
        h.columns = ["cond", "dict", "space", "m", "hoeffding_lb_mean", "hoeffding_lb_max"]
        agg = agg.merge(h, on=["cond", "dict", "space", "m"], how="left")
    # minimal m with certification rate >= 0.9
    mins = []
    for (c, dn, sp), sub in agg.groupby(["cond", "dict", "space"]):
        for meth in ("t", "boot"):
            ok = sub[sub[f"cert_rate_{meth}"] >= 0.9]["m"]
            mins.append({"cond": c, "dict": dn, "space": sp, "method": meth, "min_m_90": int(ok.min()) if len(ok) else None})
    agg.to_csv(os.path.join(args.res, "e24_certificate.csv"), index=False)
    pd.DataFrame(mins).to_csv(os.path.join(args.res, "e24_min_m.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s, {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()
