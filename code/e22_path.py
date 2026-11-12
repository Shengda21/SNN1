"""E22: noise path of the calibrated weights on the paired-noise grid delta = rho * xi.

For every rho of the grid (sim 0..1.2, sarcos 0..0.9, step 0.05) and every seed:
  * the extended residual matrix C(rho) (members + null, e_0 = -Y) of D-M and D-S on the
    calibration role (and on the test role),
  * W0 (simplex) and W1 (simplex + null) weights fitted on cal, their cal value and test risk,
  * the test risk, beta, ||nu||^2 and slope of every single member.
Interpolation of C(rho): entrywise polynomial with C(0) anchored at the measured nominal
matrix, family {a rho^2} or {a rho + b rho^2} chosen by leave-one-level-out over the
three frozen levels, refitted on all three. Predicted W0/W1 solutions on a fine grid give
the predicted support-change points (critical levels, Proposition MA-3 / Prop. 1(b) of
the manuscript), compared with the change points of the measured C(rho) interpolated
piecewise linearly between grid levels (criterion: |difference| <= one grid step).
Also: growth exponent kappa = dlog(R(rho)-R(0))/dlog rho, the D-M/D-S risk crossing and
the level where the straight-through member beats the best window member.

python e22_path.py --platform sim --res ../results/sim
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2x_common import (SEEDS, SURR, STRONG, FROZEN_LEVELS, MomFile, C_matrix, face_qp, simplex_solve, pj_risk,
                        geometry, member_blocks, jdump)

STEP = 0.05


def grid_files(res):
    out = {}
    for fn in glob.glob(os.path.join(res, "paired", "moments_cal_rho*.npz")):
        m = re.search(r"rho([0-9.]+?)(__\w+)?\.npz$", os.path.basename(fn))
        if m.group(2):
            continue
        rho = float(m.group(1))
        ft = fn.replace("moments_cal_", "moments_test_")
        if os.path.exists(ft):
            out[rho] = (fn, ft)
    return dict(sorted(out.items()))


def qp_ext(Ce, null):
    """Simplex QP on the extended matrix; null=False excludes the null index."""
    M = Ce.shape[0] - 1
    idx = list(range(M + 1)) if null else list(range(M))
    C = Ce[np.ix_(idx, idx)]
    import itertools
    faces = [T for s in range(1, len(idx) + 1) for T in itertools.combinations(range(len(idx)), s)]
    w, v = face_qp(C, faces)
    out = np.zeros(M + 1)
    out[idx] = w
    return out, v


def support(w, tol=1e-9):
    return frozenset(np.where(w > tol)[0].tolist())


def events_from_path(rhos, ws):
    ev = []
    prev = support(ws[0])
    for i in range(1, len(rhos)):
        cur = support(ws[i])
        for j in cur - prev:
            ev.append((0.5 * (rhos[i - 1] + rhos[i]), j, "enter"))
        for j in prev - cur:
            ev.append((0.5 * (rhos[i - 1] + rhos[i]), j, "leave"))
        prev = cur
    return ev


def fit_poly(rhos, Cs, C0, family):
    """Entrywise least squares of C(rho) - C0 on the basis of the family."""
    basis = {"q": lambda r: np.array([r ** 2]), "lq": lambda r: np.array([r, r ** 2]),
             "lqc": lambda r: np.array([r, r ** 2, r ** 3])}[family]
    X = np.stack([basis(r) for r in rhos])                # [L, p]
    Yv = np.stack([(C - C0).ravel() for C in Cs])          # [L, K]
    coef, *_ = np.linalg.lstsq(X, Yv, rcond=None)
    return lambda r: C0 + (basis(r) @ coef).reshape(C0.shape), coef


def crossing(x, y):
    """First rho where y changes sign (linear interpolation); None if never."""
    for i in range(1, len(x)):
        if np.sign(y[i - 1]) != np.sign(y[i]) and y[i - 1] != 0:
            return float(x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / (y[i - 1] - y[i]))
    return None


def seed_job(args):
    platform, res, seed, files = args
    B = member_blocks(platform, seed, "full")
    dicts = {"D-M": B["dm"], "D-S": B["ds"]}
    rhos = list(files.keys())
    rows = []
    Cstore = {}
    singles = None
    for rho, (fc, ft) in files.items():
        Fc, Ft = MomFile(fc), MomFile(ft)
        if singles is None:
            singles = Fc.names[seed]
        for dn, mids in dicts.items():
            mc, mt = Fc.sel(seed, mids), Ft.sel(seed, mids)
            Pc, gc, yc = mc.pooled(); Pt, gt, yt = mt.pooled()
            Cc = C_matrix(Pc.mean(0), gc.mean(0), float(yc.mean()), null=True)
            Ct = C_matrix(Pt.mean(0), gt.mean(0), float(yt.mean()), null=True)
            Cstore.setdefault(dn, {"cal": [], "test": []})
            Cstore[dn]["cal"].append(Cc); Cstore[dn]["test"].append(Ct)
            for W, null in (("W0", False), ("W1", True)):
                w, v = qp_ext(Cc, null)
                rt = float(w @ Ct @ w)
                M = len(mids)
                FF = float(w[:M] @ Pt.mean(0) @ w[:M]); FY = float(w[:M] @ gt.mean(0)); Py = float(yt.mean())
                geo = geometry(FF, FY, Py)
                rows.append({"kind": "dict", "seed": seed, "name": dn, "space": W, "rho": rho, "risk_cal": v,
                             "risk_test": rt, "beta": geo["beta"], "nu2": geo["nu2"], "slope": geo["slope"],
                             "angle_risk": geo["angle_risk"], "scale_cost": geo["scale_cost"],
                             "null_weight": float(w[M]), "weights": jdump(w, 4),
                             "support": json.dumps(sorted(support(w)))})
        ms, mt_ = Fc.sel(seed, singles), Ft.sel(seed, singles)
        Pc, gc, yc = ms.pooled(); Pt, gt, yt = mt_.pooled()
        for j, nm in enumerate(singles):
            FFc = float(Pc[:, j, j].mean()); FYc = float(gc[:, j].mean())
            FF = float(Pt[:, j, j].mean()); FY = float(gt[:, j].mean()); Py = float(yt.mean())
            geo = geometry(FF, FY, Py)
            rows.append({"kind": "member", "seed": seed, "name": nm, "space": "-", "rho": rho,
                         "risk_cal": FFc - 2 * FYc + float(yc.mean()), "risk_test": geo["risk"], "beta": geo["beta"],
                         "nu2": geo["nu2"], "slope": geo["slope"], "angle_risk": geo["angle_risk"],
                         "scale_cost": geo["scale_cost"], "null_weight": np.nan, "weights": "", "support": ""})
    return seed, rows, {dn: {k: np.stack(v) for k, v in d.items()} for dn, d in Cstore.items()}, rhos


def critical_job(args):
    """Predicted vs measured support-change points for one seed and dictionary."""
    platform, seed, dn, rhos, Ccal, Ctest, family, fit_levels = args
    rhos = np.asarray(rhos)
    i0 = int(np.argmin(np.abs(rhos)))
    C0 = Ccal[i0]
    li = [int(np.argmin(np.abs(rhos - r))) for r in fit_levels]
    pred, coef = fit_poly(rhos[li], [Ccal[i] for i in li], C0, family)
    fine = np.round(np.arange(0, rhos.max() + 1e-9, 0.005), 4)

    def interp(r):
        k = np.searchsorted(rhos, r)
        if k == 0:
            return Ccal[0]
        if k >= len(rhos):
            return Ccal[-1]
        a = (r - rhos[k - 1]) / (rhos[k] - rhos[k - 1])
        return (1 - a) * Ccal[k - 1] + a * Ccal[k]

    out_ev, out_pred = [], []
    M = Ccal.shape[1] - 1
    for W, null in (("W0", False), ("W1", True)):
        wm = [qp_ext(interp(r), null)[0] for r in fine]
        wp = [qp_ext(pred(r), null)[0] for r in fine]
        em = events_from_path(fine, wm)
        ep = events_from_path(fine, wp)
        used = set()
        for (r, j, kind) in em:
            cand = [(abs(rp - r), q) for q, (rp, jp, kp) in enumerate(ep) if jp == j and kp == kind and q not in used]
            if cand:
                dmin, q = min(cand)
                used.add(q)
                out_ev.append({"seed": seed, "dict": dn, "space": W, "member_idx": j, "member": "null" if j == M else j,
                               "direction": kind, "rho_measured": r, "rho_predicted": ep[q][0], "abs_diff": dmin,
                               "within_one_step": bool(dmin <= STEP + 1e-9)})
            else:
                out_ev.append({"seed": seed, "dict": dn, "space": W, "member_idx": j, "member": "null" if j == M else j,
                               "direction": kind, "rho_measured": r, "rho_predicted": np.nan, "abs_diff": np.nan,
                               "within_one_step": False})
        for q, (rp, jp, kp) in enumerate(ep):
            if q not in used:
                out_ev.append({"seed": seed, "dict": dn, "space": W, "member_idx": jp, "member": "null" if jp == M else jp,
                               "direction": kp, "rho_measured": np.nan, "rho_predicted": rp, "abs_diff": np.nan,
                               "within_one_step": False})
        # predicted risk at the grid levels vs measured QP value on cal, and test risk of the predicted weights
        for i, r in enumerate(rhos):
            wpr, vpr = qp_ext(pred(r), null)
            wme, vme = qp_ext(Ccal[i], null)
            out_pred.append({"seed": seed, "dict": dn, "space": W, "rho": float(r), "fit_level": bool(i in li or i == i0),
                             "value_pred": vpr, "value_meas": vme, "risk_cal_of_pred_w": float(wpr @ Ccal[i] @ wpr),
                             "risk_test_pred_w": float(wpr @ Ctest[i] @ wpr), "risk_test_meas_w": float(wme @ Ctest[i] @ wme),
                             "C_rel_err": float(np.linalg.norm(pred(r) - Ccal[i]) / np.linalg.norm(Ccal[i]))})
    # pairwise vertex roots (Prop. 1(b)): a = best single at rho, b any other incl. null
    pw = []
    diag_m = np.array([np.diag(C)[:M] for C in Ccal])
    for b in range(M + 1):
        for src, fC in (("measured", interp), ("predicted", pred)):
            vals = []
            for r in fine:
                C = fC(r)
                a = int(np.argmin(np.diag(C)[:M]))
                vals.append(np.nan if a == b else C[a, b] - C[a, a])
            vals = np.asarray(vals)
            root = None
            for k in range(1, len(fine)):
                if np.isfinite(vals[k - 1]) and np.isfinite(vals[k]) and vals[k - 1] > 0 >= vals[k]:
                    root = float(fine[k - 1] + 0.005 * vals[k - 1] / (vals[k - 1] - vals[k]))
                    break
            pw.append({"seed": seed, "dict": dn, "b": "null" if b == M else b, "source": src, "rho_root": root})
    return out_ev, out_pred, pw, coef


def lolo_error(rhos, Ccal_list, family, fit_levels):
    """Leave-one-frozen-level-out prediction error of C (relative Frobenius)."""
    rhos = np.asarray(rhos)
    i0 = int(np.argmin(np.abs(rhos)))
    errs = []
    for h in fit_levels:
        tr = [r for r in fit_levels if r != h]
        for Ccal in Ccal_list:
            li = [int(np.argmin(np.abs(rhos - r))) for r in tr]
            pred, _ = fit_poly(rhos[li], [Ccal[i] for i in li], Ccal[i0], family)
            ih = int(np.argmin(np.abs(rhos - h)))
            errs.append(np.linalg.norm(pred(h) - Ccal[ih]) / np.linalg.norm(Ccal[ih]))
    return float(np.mean(errs))


def kappa_fit(r, R, R0):
    m = (r > 0) & (R - R0 > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(r[m]), np.log(R[m] - R0), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--family", default="", help="override the LOLO choice (sensitivity run; outputs get a suffix)")
    args = ap.parse_args()
    t0 = time.time()
    suf = f"_{args.family}" if args.family else ""
    seeds = [int(s) for s in args.seeds.split(",")]
    files = grid_files(args.res)
    rhos = list(files.keys())
    print("grid:", rhos, flush=True)
    rows, Cs = [], {}
    with Pool(args.workers) as pool:
        for seed, r, C, _ in pool.imap_unordered(seed_job, [(args.platform, args.res, s, files) for s in seeds]):
            rows += r; Cs[seed] = C
            print("seed", seed, "path done", f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if not suf:
        df.to_csv(os.path.join(args.res, "e22_path.csv"), index=False)
    np.savez_compressed(os.path.join(args.res, "e22_C.npz"), rhos=np.array(rhos),
                        **{f"s{s}_{dn}_{role}": Cs[s][dn][role] for s in Cs for dn in Cs[s] for role in ("cal", "test")})
    # weight table w*(rho) for E23
    wt = df[df.kind == "dict"][["seed", "name", "space", "rho", "weights"]]
    wt.to_csv(os.path.join(args.res, "e22_wtable.csv"), index=False)
    # ---------------- polynomial family by leave-one-level-out
    fl = FROZEN_LEVELS[args.platform]
    fit_levels = [fl["L"], fl["M"], fl["H"]]
    allC = [Cs[s][dn]["cal"] for s in Cs for dn in Cs[s]]
    lolo = {fam: lolo_error(rhos, allC, fam, fit_levels) for fam in ("q", "lq")}
    family = args.family or min(lolo, key=lolo.get)
    print("LOLO errors", lolo, "->", family, flush=True)
    jobs = [(args.platform, s, dn, rhos, Cs[s][dn]["cal"], Cs[s][dn]["test"], family, fit_levels) for s in Cs for dn in Cs[s]]
    ev, pr, pw = [], [], []
    with Pool(args.workers) as pool:
        for e, p, w, _ in pool.imap_unordered(critical_job, jobs):
            ev += e; pr += p; pw += w
    evf = pd.DataFrame(ev)
    # name members
    def mname(row):
        if row["member"] == "null":
            return "null"
        mids = member_blocks(args.platform, int(row["seed"]), "full")["dm" if row["dict"] == "D-M" else "ds"]
        return mids[int(row["member_idx"])]
    if len(evf):
        evf["member_name"] = evf.apply(mname, axis=1)
    evf["family"] = family
    evf.to_csv(os.path.join(args.res, f"e22_critical{suf}.csv"), index=False)
    prf = pd.DataFrame(pr); prf["family"] = family
    prf.to_csv(os.path.join(args.res, f"e22_predicted_risk{suf}.csv"), index=False)
    pd.DataFrame(pw).to_csv(os.path.join(args.res, f"e22_pairwise_roots{suf}.csv"), index=False)
    if suf:
        print(f"sensitivity run ({family}) done in {time.time() - t0:.0f}s", flush=True)
        return
    # ---------------- kappa and crossings
    kap = []
    mem = df[df.kind == "member"]
    flev = [fl["M"], fl["H"]] if args.platform == "sarcos" else fit_levels
    for (s, nm), sub in mem.groupby(["seed", "name"]):
        sub = sub.sort_values("rho")
        r, R = sub["rho"].values, sub["risk_test"].values
        R0 = R[np.argmin(np.abs(r))]
        rec = {"seed": s, "name": nm, "R0": R0, "kappa_full": kappa_fit(r, R, R0),
               "kappa_frozen": kappa_fit(r[np.isin(np.round(r, 2), np.round(flev, 2))], R[np.isin(np.round(r, 2), np.round(flev, 2))], R0)}
        for lv in fit_levels:
            k = int(np.argmin(np.abs(r - lv)))
            if 0 < k < len(r) - 1:
                rec[f"kappa_local_{lv}"] = float((np.log(R[k + 1] - R0) - np.log(R[k - 1] - R0)) / (np.log(r[k + 1]) - np.log(r[k - 1])))
            rec[f"growth_per_rho2_{lv}"] = float((R[k] - R0) / r[k] ** 2)
        kap.append(rec)
    dsol = df[df.kind == "dict"]
    for (s, nm, sp), sub in dsol.groupby(["seed", "name", "space"]):
        sub = sub.sort_values("rho")
        r, R = sub["rho"].values, sub["risk_test"].values
        R0 = R[np.argmin(np.abs(r))]
        kap.append({"seed": s, "name": f"{nm}|{sp}", "R0": R0, "kappa_full": kappa_fit(r, R, R0),
                    "kappa_frozen": kappa_fit(r[np.isin(np.round(r, 2), np.round(flev, 2))], R[np.isin(np.round(r, 2), np.round(flev, 2))], R0)})
    kf = pd.DataFrame(kap)

    def mtype(nm):
        base = re.sub(r"_s\d+.*", "", nm)
        return base
    kf["type"] = kf["name"].apply(mtype)
    kf.to_csv(os.path.join(args.res, "e22_kappa.csv"), index=False)
    cr = []
    strong = STRONG[args.platform]
    win = [x for x in SURR if x != "ste"]
    piv = mem.pivot_table(index=["seed", "rho"], columns="name", values="risk_test").reset_index()
    curves = {}
    for s in seeds:
        p = piv[piv.seed == s].sort_values("rho")
        r = p["rho"].values
        ste = p[f"dm_ste_s{s}"].values
        best_w = np.min(np.stack([p[f"dm_{x}_s{s}"].values for x in win]), 0)
        cr.append({"type": "STE_vs_best_window", "seed": s, "rho_cross": crossing(r, ste - best_w)})
        curves.setdefault("ste", []).append(ste); curves.setdefault("bestw", []).append(best_w)
        for sp in ("W0", "W1"):
            a = dsol[(dsol.seed == s) & (dsol.name == "D-M") & (dsol.space == sp)].sort_values("rho")["risk_test"].values
            b = dsol[(dsol.seed == s) & (dsol.name == "D-S") & (dsol.space == sp)].sort_values("rho")["risk_test"].values
            cr.append({"type": f"DM_vs_DS_{sp}", "seed": s, "rho_cross": crossing(r, a - b)})
            curves.setdefault(f"dm{sp}", []).append(a); curves.setdefault(f"ds{sp}", []).append(b)
        mlp = p[f"bn_mlp_s{s}"].values
        cr.append({"type": "STE_vs_MLP", "seed": s, "rho_cross": crossing(r, ste - mlp)})
    r = np.array(rhos)
    # seed-mean curves: STE vs the best window member of the mean curves
    pm = mem.copy(); pm["base"] = pm["name"].str.replace(r"_s\d+", "", regex=True)
    mc = pm.groupby(["base", "rho"])["risk_test"].mean().unstack(0).sort_index()
    bestw_mean = mc[[f"dm_{x}" for x in win]].min(axis=1).values
    cr.append({"type": "STE_vs_best_window", "seed": "mean_curve", "rho_cross": crossing(r, mc["dm_ste"].values - bestw_mean)})
    for sp in ("W0", "W1"):
        cr.append({"type": f"DM_vs_DS_{sp}", "seed": "mean_curve",
                   "rho_cross": crossing(r, np.mean(curves[f"dm{sp}"], 0) - np.mean(curves[f"ds{sp}"], 0))})
    pd.DataFrame(cr).to_csv(os.path.join(args.res, "e22_crossings.csv"), index=False)
    json.dump({"family": family, "lolo_rel_err": lolo, "fit_levels": fit_levels, "grid": rhos},
              open(os.path.join(args.res, "e22_fit.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
