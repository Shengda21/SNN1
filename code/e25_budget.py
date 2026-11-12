"""E25: execution budget in the relaxed weight space and per-joint supports.

Budget family "at most k executed members + a free null" (the null needs no forward
pass) versus the manuscript's W-Q(k) (no null), k = 1..5, on D-M and D-S (and D-T);
calibration on the full calibration role and, on the arm, on each group of m = 10
trajectories (as E04). Retention of the improvement by two members (+ null) relative to
five members (+ null), with the best single member (W-S) and the scaled best single
member (k = 1 + null) as references. Per-joint weights (W2, W3): size of the union of the
per-joint supports (= executed members) and the weight of the straight-through member
on every joint.

python e25_budget.py --platform sim --res ../results/sim
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2x_common import SEEDS, MomFile, member_blocks, simplex_solve, solve_space, pj_risk, jdump
from e24_certificate import conds


def job(a):
    platform, res, seed, cd = a
    lab, cr, tr, fl = cd
    Fc = MomFile(os.path.join(res, "orig", f"moments_{cr}_{fl}.npz"))
    Ft = MomFile(os.path.join(res, "orig", f"moments_{tr}_{fl}.npz"))
    B = member_blocks(platform, seed, "full")
    rows, pj = [], []
    sels = [(-1, -1, None)]   # m = -1: the full calibration role
    if platform == "sim":
        sels += [(10, g, np.arange(10 * g, 10 * g + 10)) for g in range(10)]
    for dn, mids in (("D-M", B["dm"]), ("D-S", B["ds"]), ("D-T", B["dt"])):
        mc_all = Fc.sel(seed, mids)
        Pt, gt, yt = Ft.sel(seed, mids).pooled()
        M = len(mids)
        for m, g, uids in sels:
            mc = mc_all if uids is None else mc_all.units(uids)
            Pc, gc, yc = mc.pooled()
            Phi, gg, Py = Pc.mean(0), gc.mean(0), float(yc.mean())
            for null in (False, True):
                for k in range(1, M + 1):
                    w, w0, v, kkt = simplex_solve(Phi, gg, Py, null=null, k=k)
                    W = np.tile(w, (7, 1))
                    rows.append({"cond": lab, "seed": seed, "dict": dn, "m": m, "group": g,
                                 "family": "Wk+null" if null else "Wk", "k": k, "nmse": pj_risk(W, Pt, gt, yt)[0],
                                 "value_cal": v, "n_exec": int(np.sum(w > 1e-10)), "null_weight": w0, "kkt": kkt,
                                 "weights": jdump(w, 4)})
            if uids is None:
                for sp in ("W2", "W3"):
                    W, _ = solve_space(sp, Pc, gc, yc)
                    supp = np.any(W > 1e-10, axis=0)
                    rec = {"cond": lab, "seed": seed, "dict": dn, "space": sp, "nmse": pj_risk(W, Pt, gt, yt)[0],
                           "union_support": int(supp.sum()), "mean_joint_support": float(np.mean(np.sum(W > 1e-10, 1))),
                           "null_weight_by_joint": jdump(1 - W.sum(1), 4)}
                    for j, nm in enumerate(mids):
                        rec[f"w{j}_{nm}"] = jdump(W[:, j], 4)
                    if dn == "D-M":
                        ste = mids.index(f"dm_ste_s{seed}")
                        rec["ste_weight_by_joint"] = jdump(W[:, ste], 4)
                        rec["ste_joints"] = ",".join(str(l + 1) for l in range(7) if W[l, ste] > 1e-3)
                    pj.append(rec)
    return rows, pj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(args.platform, args.res, s, c) for c in conds(args.platform) for s in SEEDS]
    rows, pj = [], []
    with Pool(args.workers) as pool:
        for r, p in pool.imap_unordered(job, jobs):
            rows += r; pj += p
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.res, "e25_budget_rows.csv"), index=False)
    agg = df.groupby(["cond", "dict", "m", "family", "k"]).agg(
        nmse=("nmse", "mean"), nmse_sd=("nmse", "std"), n_exec=("n_exec", "mean"), null_weight=("null_weight", "mean")).reset_index()
    # retention (per seed and group, then averaged)
    ret = []
    for (c, dn, m, s, g), sub in df.groupby(["cond", "dict", "m", "seed", "group"]):
        R = {(f, k): v for f, k, v in zip(sub["family"], sub["k"], sub["nmse"])}
        ws, ws0 = R[("Wk", 1)], R[("Wk+null", 1)]
        for ref_name, ref in (("W-S", ws), ("k1+null", ws0)):
            den = ref - R[("Wk+null", 5)]
            for k in (2, 3):
                ret.append({"cond": c, "dict": dn, "m": m, "seed": s, "group": g, "ref": ref_name, "k": k,
                            "retention_null": (ref - R[("Wk+null", k)]) / den if abs(den) > 1e-12 else np.nan,
                            "retention_plain_vs_plain5": (R[("Wk", 1)] - R[("Wk", k)]) / (R[("Wk", 1)] - R[("Wk", 5)])
                            if abs(R[("Wk", 1)] - R[("Wk", 5)]) > 1e-12 else np.nan})
    rf = pd.DataFrame(ret)
    rag = rf.groupby(["cond", "dict", "m", "ref", "k"]).agg(retention_null=("retention_null", "mean"),
                                                             retention_plain=("retention_plain_vs_plain5", "mean")).reset_index()
    agg.to_csv(os.path.join(args.res, "e25_budget.csv"), index=False)
    rag.to_csv(os.path.join(args.res, "e25_retention.csv"), index=False)
    pd.DataFrame(pj).to_csv(os.path.join(args.res, "e25_perjoint.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
