"""E26 (optional): choosing the weight space from the calibration data.

Cross-fitting on the calibration subset (folds: trajectories, or the five 200-step
segments when m = 1; sarcos: 100-row blocks) selects the space with the smallest held-out
risk among W0 (simplex), W1 (+ null), W2 (per-joint simplex), W3 (per-joint + null),
W4 (full linear stacking with ridge; lambda by an inner cross-fit on the training folds).
The selected space is refitted on the whole subset and evaluated on the test role,
next to every fixed space and the post-hoc best space.

python e26_select.py --platform sim --res ../results/sim
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
from e2x_common import (SEEDS, MomFile, member_blocks, solve_space, solve_W4, make_folds, pj_risk, x_risk)
from e24_certificate import conds

SPACES = ["W0", "W1", "W2", "W3", "W4"]


def fit(sp, mom, xm, inner_k):
    if sp == "W4":
        A, _ = solve_W4(xm, make_folds(xm, k=inner_k))
        return ("x", A)
    P, g, y = mom.pooled()
    W, _ = solve_space(sp, P, g, y)
    return ("pj", W)


def risk(sol, mom, xm):
    kind, X = sol
    if kind == "pj":
        return pj_risk(X, *mom.pooled())[0]
    return x_risk(X, *xm.pooled())[0]


def job(a):
    platform, res, seed, cd = a
    lab, cr, tr, fl = cd
    Fc = MomFile(os.path.join(res, "orig", f"moments_{cr}_{fl}.npz"))
    Ft = MomFile(os.path.join(res, "orig", f"moments_{tr}_{fl}.npz"))
    B = member_blocks(platform, seed, "full")
    rows = []
    for dn, blk in (("D-M", "dm"), ("D-S", "ds")):
        mids = B[blk]
        mc_all, mt = Fc.sel(seed, mids), Ft.sel(seed, mids)
        xc_all, xt = Fc.xsel(seed, blk), Ft.xsel(seed, blk)
        if platform == "sim":
            sels = [(m, g, np.arange(10 * g, 10 * g + m)) for m in (1, 10) for g in range(10)] + [(100, -1, np.arange(100))]
            kfold = 10
        else:
            nb = len(np.unique(mc_all.parent))
            sels = [(10, r, np.arange(10 * r, 10 * r + 10)) for r in range(nb // 10)] + [(nb, -1, np.arange(nb))]
            kfold = 5
        for m, g, uids in sels:
            mc, xc = mc_all.units(uids), xc_all.units(uids)
            folds = make_folds(mc, k=kfold)
            cv = {}
            for sp in SPACES:
                tot = 0.0
                for f in folds:
                    sol = fit(sp, mc.subset(~f), xc.subset(~f), kfold)
                    tot += risk(sol, mc.subset(f), xc.subset(f))
                cv[sp] = tot / len(folds)
            test = {sp: risk(fit(sp, mc, xc, kfold), mt, xt) for sp in SPACES}
            sel = min(cv, key=cv.get)
            best = min(test, key=test.get)
            rec = {"cond": lab, "seed": seed, "dict": dn, "m": m, "group": g, "selected": sel,
                   "test_selected": test[sel], "posthoc_best": best, "test_posthoc_best": test[best],
                   "regret": test[sel] - test[best]}
            for sp in SPACES:
                rec[f"cv_{sp}"] = cv[sp]; rec[f"test_{sp}"] = test[sp]
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
            print(f"{len(rows)} rows {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.res, "e26_rows.csv"), index=False)
    out = []
    for (c, dn, m), sub in df.groupby(["cond", "dict", "m"]):
        rec = {"cond": c, "dict": dn, "m": m, "n": len(sub), "test_selected": sub["test_selected"].mean(),
               "regret": sub["regret"].mean()}
        for sp in SPACES:
            rec[f"freq_{sp}"] = float(np.mean(sub["selected"] == sp))
            rec[f"test_{sp}"] = sub[f"test_{sp}"].mean()
        out.append(rec)
    pd.DataFrame(out).to_csv(os.path.join(args.res, "e26_select.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
