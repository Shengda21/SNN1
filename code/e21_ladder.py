"""E21: weight-space ladder for frozen members (zero training), from moment files.

For D-S, D-M, D-T and the single-model baselines B-F, B-L, B-A, B-N (plus the
augmented single models available on each platform) solve, on the calibration role of
the target condition, in the weight spaces
  W0  simplex (= W-Q(5) of the manuscript)
  W1  simplex + null  (= {w >= 0, sum w <= 1}); a single model: scalar c in [0, 1]
  W2  per-joint simplex (7 independent QPs)
  W3  per-joint simplex + null
  W4  full linear stacking (every joint of every member -> each joint), ridge, lambda by
      K-fold cross-fitting over calibration units
  W4d per-joint linear stacking (unconstrained), ridge, lambda by cross-fitting
  W5  label-free input estimator in front of the members (wfe), then W1
      (W5s: estimator then W0; W5o: oracle-variance estimator then W1, reference)
  W6  augmented members (D-MA for D-M, D-SA for D-S), simplex  (W6n: + null)
and evaluate on the test role: nMSE, beta, ||nu||^2, slope s, angle risk, scale cost,
executed members, null weight. Paired D-M - D-S differences with the two-level cluster
bootstrap (training seed x test unit, shared indices), and the attribution of the
D-S - D-M margin into scale and angle parts.

python e21_ladder.py --platform sim --res ../results/sim [--source orig|paired] [--mslice]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2x_common import (SEEDS, SURR, STRONG, MomFile, XMom, solve_space, solve_W4, solve_W4d, make_folds,
                        pj_risk, pj_unit_risk, x_risk, x_unit_risk, geometry, cboot, jdump, member_blocks, cname)

SPACES = ["W0", "W1", "W2", "W3", "W4", "W4d", "W5", "W5s", "W5o", "W6", "W6n"]


def conditions(platform, source):
    """(label, cal_role, test_role, file label)"""
    if platform == "sim":
        if source == "orig":
            return [("N", "cal", "test", "N"), ("O1-L", "cal", "test", "O1_0.35"), ("O1-M", "cal", "test", "O1_0.7"),
                    ("O1-H", "cal", "test", "O1_1.0"), ("O2B", "cal", "test", "O2B_0.7"), ("P1", "cal_P1", "test_P1", "N")]
        return [("N", "cal", "test", "rho0.00"), ("O1-L", "cal", "test", "rho0.35"), ("O1-M", "cal", "test", "rho0.70"),
                ("O1-H", "cal", "test", "rho1.00")]
    if source == "orig":
        return [("N", "cal", "test", "N"), ("O1-M", "cal", "test", "O1_0.5"), ("O1-H", "cal", "test", "O1_0.7"),
                ("O1-L", "cal", "test", "O1_0.25"), ("O1qqd", "cal", "test", "O1qqd_0.5")]
    return [("N", "cal", "test", "rho0.00"), ("O1-L", "cal", "test", "rho0.25"), ("O1-M", "cal", "test", "rho0.50"),
            ("O1-H", "cal", "test", "rho0.70")]


def methods(platform, seed):
    B = member_blocks(platform, seed, "full")
    s = STRONG[platform]
    out = {"D-M": ("dm", B["dm"]), "D-S": ("ds", B["ds"]), "D-T": ("dt", B["dt"]),
           "B-F": ("dm", [f"dm_{s}_s{seed}"]), "B-L": ("bl", B["bl"]), "B-A": ("ba", B["ba"]), "B-N": ("bn", B["bn"]),
           "B-A+": (None, B["dmas"])}
    if platform == "sim":
        out.update({"B-AQ+": (None, B["dmqs"]), "B-NQ+": (None, B["bnqs"])})
    return out


AUG = {"D-M": "dma", "D-S": "dsa"}


def xsub(xm, feat):
    if xm is None:
        return None
    return XMom(xm.XFF[:, feat][:, :, feat], xm.XFY[:, feat], xm.YY, xm.n, xm.parent)


def fit_eval(space, cal, test, xc=None, xt=None, folds=None, units=False):
    Phil, gl, Pyl = cal.pooled()
    d, M = gl.shape
    if space in ("W0", "W1", "W2", "W3"):
        W, info = solve_space(space, Phil, gl, Pyl); kind = "pj"
    elif space == "W4d":
        W, info = solve_W4d(cal, folds); kind = "pj"
    elif space == "W4":
        A, info = solve_W4(xc, folds); kind = "x"
    else:
        raise ValueError(space)
    row = {}
    ur = nu = None
    if kind == "pj":
        row["risk_cal"] = pj_risk(W, Phil, gl, Pyl)[0]
        Pt, gt, yt = test.pooled()
        R, FF, FY, Py = pj_risk(W, Pt, gt, yt)
        if units:
            Pu, gu, yu, nu, _ = test.per_unit()
            ur = pj_unit_risk(W, Pu, gu, yu)
        row["n_exec"] = int(np.sum(np.any(np.abs(W) > 1e-10, axis=0)))
        ws = W.sum(1)
        row["weight_sum"] = float(ws.mean())
        row["null_weight"] = float(1 - ws.mean()) if space in ("W1", "W3") else 0.0
        row["weights"] = jdump(W[0]) if space in ("W0", "W1") else jdump(W)
    else:
        G, B, YY = xc.pooled()
        row["risk_cal"] = x_risk(A, G, B, YY)[0]
        Gt, Bt, YYt = xt.pooled()
        R, FF, FY, Py = x_risk(A, Gt, Bt, YYt)
        if units:
            Gu, Bu, YYu, nu, _ = xt.per_unit()
            ur = x_unit_risk(A, Gu, Bu, YYu)
        row["n_exec"] = M
        row["weight_sum"] = np.nan; row["null_weight"] = np.nan
        row["weights"] = json.dumps({"fro_norm": float(np.linalg.norm(A))})
    geo = geometry(FF, FY, Py)
    row.update({"nmse": R, **{k: v for k, v in geo.items() if k != "risk"}})
    for k in ("kkt", "kkt_w1", "nnls_gap", "lam", "lam_rel"):
        if k in info:
            row[k] = info[k]
    return row, ur, nu


def job(args):
    platform, res, source, seed, cond, do_mslice = args
    lab, cal_role, test_role, flab = cond
    sub = "orig" if source == "orig" else "paired"
    f = lambda role, var="": os.path.join(res, sub, f"moments_{role}_{flab}{var}.npz")
    if not (os.path.exists(f(cal_role)) and os.path.exists(f(test_role))):
        return [], [], {}
    Fc, Ft = MomFile(f(cal_role)), MomFile(f(test_role))
    alt = {}
    if source == "orig":
        for var in ("wfe", "wfo"):
            if os.path.exists(f(cal_role, f"__{var}")) and os.path.exists(f(test_role, f"__{var}")):
                alt[var] = (MomFile(f(cal_role, f"__{var}")), MomFile(f(test_role, f"__{var}")))
    meths = methods(platform, seed)
    B = member_blocks(platform, seed, "full")
    d = 7
    rows, mrows, units = [], [], {}
    # calibration selections: full, and (sim) groups x m
    sels = [("all", -1, None)]
    if do_mslice and platform == "sim":
        for g in range(10):
            for m in (1, 2, 5, 10):
                sels.append((m, g, np.arange(10 * g, 10 * g + m)))
    for name, (xblock, mids) in meths.items():
        if not Fc.has_members(seed, mids):
            continue
        cal_all, test = Fc.sel(seed, mids), Ft.sel(seed, mids)
        xc_all = xt = None
        if xblock is not None:
            xc_all, xt = Fc.xsel(seed, xblock), Ft.xsel(seed, xblock)
            if name == "B-F" and xc_all is not None:
                j = B["dm"].index(mids[0])
                feat = list(range(j * d, (j + 1) * d))
                xc_all, xt = xsub(xc_all, feat), xsub(xt, feat)
        extra = {}
        for var, sp in (("wfe", "W5"), ("wfe", "W5s"), ("wfo", "W5o")):
            if var in alt and alt[var][0].has_members(seed, mids):
                extra[sp] = (alt[var][0].sel(seed, mids), alt[var][1].sel(seed, mids), "W1" if sp != "W5s" else "W0")
        if name in AUG and Fc.has_members(seed, B[AUG[name]]):
            ma = B[AUG[name]]
            extra["W6"] = (Fc.sel(seed, ma), Ft.sel(seed, ma), "W0")
            extra["W6n"] = (Fc.sel(seed, ma), Ft.sel(seed, ma), "W1")
        for m, g, uids in sels:
            if uids is None:
                cal, xc = cal_all, xc_all
            else:
                cal = cal_all.units(uids)
                xc = xc_all.units(uids) if xc_all is not None else None
            if platform == "sim":
                folds = make_folds(cal, k=10)
            else:
                folds = make_folds(cal, k=5)
            for sp in SPACES:
                if sp in ("W4",) and xc is None:
                    continue
                if sp in extra:
                    ccal, ctest, base = extra[sp]
                    if uids is not None:
                        ccal = ccal.units(uids)
                    try:
                        r, ur, nu = fit_eval(base, ccal, ctest, units=(uids is None))
                    except Exception as e:  # noqa
                        print("fail", name, sp, e, flush=True); continue
                elif sp in ("W5", "W5s", "W5o", "W6", "W6n"):
                    continue
                else:
                    r, ur, nu = fit_eval(sp, cal, test, xc, xt, folds, units=(uids is None))
                r.update({"platform": platform, "source": source, "cond": lab, "seed": seed, "method": name,
                          "space": sp, "m": m, "group": g, "n_members": len(mids)})
                if uids is None:
                    rows.append(r)
                    units[(lab, name, sp, seed)] = (ur, nu)
                else:
                    r.pop("weights", None)
                    mrows.append(r)
    return rows, mrows, units


PAIRS_METHOD = [("D-M", "D-S"), ("D-M", "D-T"), ("D-S", "D-T"), ("D-M", "B-N"), ("D-S", "B-N"), ("D-M", "B-A"),
                ("D-S", "B-A"), ("D-M", "B-F"), ("D-S", "B-F")]
PAIRS_CROSS = [(("D-S", "W1"), ("D-M", "W0")), (("B-N", "W1"), ("D-S", "W1")), (("B-N", "W1"), ("D-M", "W0")),
               (("B-N", "W1"), ("D-S", "W0")), (("B-F", "W1"), ("D-S", "W0")), (("B-A", "W1"), ("D-S", "W1")),
               (("D-M", "W1"), ("D-M", "W0")), (("D-S", "W1"), ("D-S", "W0")), (("D-M", "W3"), ("D-M", "W1")),
               (("D-S", "W3"), ("D-S", "W1")), (("D-M", "W4"), ("D-M", "W1")), (("D-S", "W4"), ("D-S", "W1"))]
PLAN_ATTR = {("sarcos", "O1-M"): (0.038, 0.012, 0.026, 0.32, 0.002), ("sarcos", "O1-H"): (0.091, 0.038, 0.053, 0.42, 0.005),
             ("sim", "O1-M"): (0.021, 0.021, 0.000, 1.00, -0.002), ("sim", "O1-H"): (0.093, 0.088, 0.005, 0.95, -0.002)}


def stack_units(units, keys_list, seeds):
    """Combine per-seed unit-risk vectors: value = sum_k coef_k * risk_k (same seed).
    keys_list: list of (coef, (cond, method, space)). Returns values, weights, seed ids, unit ids."""
    vals, wts, cs, cu = [], [], [], []
    for s in seeds:
        ok = all((k + (s,)) in units and units[k + (s,)][0] is not None for _, k in keys_list)
        if not ok:
            continue
        v = sum(c * units[k + (s,)][0] for c, k in keys_list)
        n = units[keys_list[0][1] + (s,)][1]
        vals.append(v); wts.append(n); cs.append(np.full(len(v), s)); cu.append(np.arange(len(v)))
    if not vals:
        return None
    return np.concatenate(vals), np.concatenate(wts), np.concatenate(cs), np.concatenate(cu)


def boot_samples(cols, w0, clusters, n_boot, seed=1):
    """Shared-draw bootstrap means of several value columns."""
    rng = np.random.default_rng(seed)
    levels = list(clusters.values())
    uniq = [np.unique(l) for l in levels]
    pos = [np.searchsorted(u, l) for l, u in zip(levels, uniq)]
    out = np.empty((n_boot, cols.shape[1]))
    for b in range(n_boot):
        wts = w0.copy()
        for u, p in zip(uniq, pos):
            draw = rng.choice(len(u), len(u), replace=True)
            wts = wts * np.bincount(draw, minlength=len(u))[p]
        out[b] = (wts @ cols) / wts.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--source", default="orig")
    ap.add_argument("--mslice", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    conds = conditions(args.platform, args.source)
    jobs = [(args.platform, args.res, args.source, s, c, args.mslice) for c in conds for s in seeds]
    rows, mrows, units = [], [], {}
    with Pool(args.workers) as pool:
        for r, mr, u in pool.imap_unordered(job, jobs):
            rows += r; mrows += mr; units.update(u)
            print(f"job done ({len(rows)} rows, {time.time() - t0:.0f}s)", flush=True)
    suf = "" if args.source == "orig" else "_pairednoise"
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.res, f"e21_ladder_seeds{suf}.csv"), index=False)
    if mrows:
        mf = pd.DataFrame(mrows)
        mf.to_csv(os.path.join(args.res, f"e21_mslice_rows{suf}.csv"), index=False)
        agg = mf.groupby(["cond", "method", "space", "m"]).agg(
            nmse=("nmse", "mean"), beta=("beta", "mean"), slope=("slope", "mean"), scale_cost=("scale_cost", "mean"),
            angle_risk=("angle_risk", "mean"), null_weight=("null_weight", "mean"), n_exec=("n_exec", "mean"),
            n=("nmse", "size")).reset_index()
        sd = mf.groupby(["cond", "method", "space", "m", "seed"])["nmse"].mean().groupby(["cond", "method", "space", "m"]).std().rename("seed_sd")
        agg = agg.merge(sd.reset_index(), on=["cond", "method", "space", "m"])
        full = df.groupby(["cond", "method", "space"]).agg(nmse=("nmse", "mean")).reset_index()
        full["m"] = 100 if args.platform == "sim" else -1
        full_sd = df.groupby(["cond", "method", "space"])["nmse"].std().rename("seed_sd").reset_index()
        full = full.merge(full_sd, on=["cond", "method", "space"])
        agg = pd.concat([agg, full], ignore_index=True)
        agg.to_csv(os.path.join(args.res, f"e21_mslice{suf}.csv"), index=False)
    # ---------------- summary with cluster bootstrap over seeds and test units
    summ = []
    for (lab, meth, sp), sub in df.groupby(["cond", "method", "space"]):
        st = stack_units(units, [(1.0, (lab, meth, sp))], seeds)
        pt, lo, hi = cboot(st[0], st[1], {"seed": st[2], "unit": st[3]}, n_boot=args.boot) if st else (np.nan,) * 3
        rec = {"cond": lab, "method": meth, "space": sp, "nmse_mean": sub["nmse"].mean(), "ci_lo": lo, "ci_hi": hi,
               "seed_sd": sub["nmse"].std(ddof=1), "n_seeds": len(sub)}
        for k in ("risk_cal", "beta", "nu2", "slope", "angle_risk", "scale_cost", "out_power", "Py", "n_exec",
                  "weight_sum", "null_weight", "kkt", "nnls_gap", "lam_rel"):
            if k in sub:
                rec[k] = sub[k].mean()
        summ.append(rec)
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(args.res, f"e21_ladder{suf}.csv"), index=False)
    # ---------------- paired differences
    prs = []
    labs = list(dict.fromkeys(df["cond"]))
    for lab in labs:
        for sp in SPACES:
            for a, b in PAIRS_METHOD:
                st = stack_units(units, [(1.0, (lab, a, sp)), (-1.0, (lab, b, sp))], seeds)
                if st is None:
                    continue
                pt, lo, hi = cboot(st[0], st[1], {"seed": st[2], "unit": st[3]}, n_boot=args.boot, seed=1)
                per_seed = [float(np.average(st[0][st[2] == s], weights=st[1][st[2] == s])) for s in np.unique(st[2])]
                prs.append({"cond": lab, "a": f"{a}|{sp}", "b": f"{b}|{sp}", "diff": pt, "ci_lo": lo, "ci_hi": hi,
                            "seeds_a_better": int(sum(x < 0 for x in per_seed)), "n_seeds": len(per_seed)})
        for (a, sa), (b, sb) in PAIRS_CROSS:
            st = stack_units(units, [(1.0, (lab, a, sa)), (-1.0, (lab, b, sb))], seeds)
            if st is None:
                continue
            pt, lo, hi = cboot(st[0], st[1], {"seed": st[2], "unit": st[3]}, n_boot=args.boot, seed=1)
            per_seed = [float(np.average(st[0][st[2] == s], weights=st[1][st[2] == s])) for s in np.unique(st[2])]
            prs.append({"cond": lab, "a": f"{a}|{sa}", "b": f"{b}|{sb}", "diff": pt, "ci_lo": lo, "ci_hi": hi,
                        "seeds_a_better": int(sum(x < 0 for x in per_seed)), "n_seeds": len(per_seed)})
    pd.DataFrame(prs).to_csv(os.path.join(args.res, f"e21_paired{suf}.csv"), index=False)
    # ---------------- attribution of the D-S - D-M margin
    att = []
    parts = {
        "margin_W0": [(1, "D-S", "W0"), (-1, "D-M", "W0")],
        "scale_part": [(1, "D-S", "W0"), (-1, "D-S", "W1"), (-1, "D-M", "W0"), (1, "D-M", "W1")],
        "angle_part": [(1, "D-S", "W1"), (-1, "D-M", "W1")],
        "rem_est_W1": [(1, "D-S", "W5"), (-1, "D-M", "W5")],
        "rem_est_W0": [(1, "D-S", "W5s"), (-1, "D-M", "W5s")],
        "rem_est_oracle_W1": [(1, "D-S", "W5o"), (-1, "D-M", "W5o")],
        "rem_perjoint_W3": [(1, "D-S", "W3"), (-1, "D-M", "W3")],
        "rem_perjoint_W2": [(1, "D-S", "W2"), (-1, "D-M", "W2")],
        "rem_stack_W4": [(1, "D-S", "W4"), (-1, "D-M", "W4")],
        "rem_aug_W6": [(1, "D-S", "W6"), (-1, "D-M", "W6")],
        "scale_cost_DS": [(1, "D-S", "W0"), (-1, "D-S", "W1")],
        "scale_cost_DM": [(1, "D-M", "W0"), (-1, "D-M", "W1")],
    }
    for lab in labs:
        cols, names, base = [], [], None
        for pn, terms in parts.items():
            st = stack_units(units, [(c, (lab, mth, sp)) for c, mth, sp in terms], seeds)
            if st is None:
                continue
            if base is None:
                base = st
            if len(st[0]) != len(base[0]):
                continue
            cols.append(st[0]); names.append(pn)
        if base is None:
            continue
        C = np.stack(cols, 1)
        w0 = base[1]
        pts = (w0 @ C) / w0.sum()
        bs = boot_samples(C, w0, {"seed": base[2], "unit": base[3]}, args.boot)
        rec = {"cond": lab}
        for i, pn in enumerate(names):
            rec[pn] = pts[i]; rec[pn + "_lo"] = np.percentile(bs[:, i], 2.5); rec[pn + "_hi"] = np.percentile(bs[:, i], 97.5)
        if "scale_part" in names and "margin_W0" in names:
            i, j = names.index("scale_part"), names.index("margin_W0")
            rec["scale_share"] = pts[i] / pts[j]
            r = bs[:, i] / bs[:, j]
            rec["scale_share_lo"], rec["scale_share_hi"] = np.percentile(r, 2.5), np.percentile(r, 97.5)
            for pn in ("rem_est_W1", "rem_est_W0", "rem_perjoint_W3", "rem_perjoint_W2", "rem_stack_W4"):
                if pn in names and "angle_part" in names:
                    rec[f"{pn}_frac_of_angle"] = pts[names.index(pn)] / pts[names.index("angle_part")]
        pl = PLAN_ATTR.get((args.platform, lab))
        if pl:
            rec.update({"plan_margin": pl[0], "plan_scale": pl[1], "plan_angle": pl[2], "plan_share": pl[3], "plan_rem_est": pl[4]})
        att.append(rec)
    pd.DataFrame(att).to_csv(os.path.join(args.res, f"e21_attribution{suf}.csv"), index=False)
    np.savez_compressed(os.path.join(args.res, f"e21_units{suf}.npz"),
                        **{"|".join(map(str, k)): np.stack([v[0], v[1]]) for k, v in units.items() if v[0] is not None})
    print(f"done in {time.time() - t0:.0f}s: {len(df)} rows, {len(mrows)} m-slice rows", flush=True)


if __name__ == "__main__":
    main()
