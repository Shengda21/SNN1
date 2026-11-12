"""E26 control: choose the weight space by split-conformal interval length ([ZVWZ25], Sec. 4.4)
and compare with the cross-fitting selection of e26_select.py.

For one calibration set of m units (sim: m trajectories, m = 1 -> the five 200-step segments
of one trajectory; sarcos: m consecutive 100-row blocks) the units are split at random into
halves A (fit) and B (calibrate). Every space W0-W4 is fitted on A (same solvers and ridge
cross-fit as e26_select.py), the per-row residual norm ||F_w(x_r) - y_r||_2 (standardized
units, all d joints) is computed on the rows of B, and the split-conformal half-width is its
ceil((n+1)(1-alpha))/n empirical quantile, alpha = 0.1. The space with the shortest half-width
is selected. It is then refitted on all m units and evaluated on the test role (risk =
mean row-wise squared error / d, identical to e26_select.py), next to the cross-fitting
selection of the same calibration set (recomputed here with the e26_select functions so that
every m has one; for the m values that e26_rows.csv contains it is checked against that file).
Ten random splits per calibration set.

Residual rows need the stored predictions (runs25/<platform>/<member>/pred_cal_<cond>.npy,
float32, clipped to the torque limits on the arm as in e2x_dump_paired.py); fits and
test risks use the moment files, so the sub-unit layout must be the one of the moment files.

python e26_conformal.py --platform sim --data_dir ../data_sim --runs ../runs25/sim --res ../results/sim
"""
import argparse
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
from e2x_common import SEEDS, MomFile, member_blocks, make_folds, unit_layout, cname
from e26_select import SPACES, fit, risk

ALPHA = 0.1
N_SPLIT = 10
CONDS = {
    "sim": [("N", "N"), ("O1-L", "O1:0.35"), ("O1-M", "O1:0.7"), ("O1-H", "O1:1.0"), ("O2B", "O2B:0.7")],
    "sarcos": [("N", "N"), ("O1-M", "O1:0.5"), ("O1-H", "O1:0.7"), ("O1qqd", "O1qqd:0.5")],
}
MS = {"sim": (1, 2, 5, 10), "sarcos": (4, 10, 40)}
KFOLD = {"sim": 10, "sarcos": 5}
SEG = 5  # sim: segments per calibration trajectory in the moment files

_G = {}  # per-process cache: data


def get_data(platform, data_dir):
    if platform not in _G:
        from run_train import load_platform
        D, _ = load_platform(platform, data_dir)
        Y = D["Y"]["cal"].astype(np.float64)
        sub, parent = unit_layout(platform, D, "cal", seg=SEG if platform == "sim" else 1)
        clip = None
        if platform == "sim":
            from sim.panda_env import TORQUE_LIMIT
            clip = ((-TORQUE_LIMIT - D["y_mean"]) / D["y_std"], (TORQUE_LIMIT - D["y_mean"]) / D["y_std"])
        _G[platform] = (Y, sub, parent, clip)
    return _G[platform]


def conformal_q(r, alpha=ALPHA):
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(r)[min(k, n) - 1])


def predict_rows(sol, P, Px):
    """Combined prediction [N, d] of a fitted space. P: [M, N, d]; Px: [N, M*d] member-major."""
    kind, X = sol
    if kind == "pj":
        return np.einsum("mnl,lm->nl", P, X)
    return Px @ X


def split_units(platform, m, g, parent, U):
    """Calibration set g of size m as (mask over sub-units, list of split units (sub-unit id arrays))."""
    if platform == "sim":
        traj = np.arange(10 * g, 10 * g + m) if m > 1 else np.array([10 * g])
        gm = np.isin(parent, traj)
        if m == 1:
            units = [np.array([u]) for u in np.where(gm)[0]]
        else:
            units = [np.where(parent == t)[0] for t in traj]
    else:
        ids = np.arange(m * g, m * (g + 1))
        gm = np.isin(parent, ids)
        units = [np.where(parent == t)[0] for t in ids]
    return gm, units


def crossfit(mc, xc, mt, xt, kfold):
    folds = make_folds(mc, k=kfold)
    cv = {}
    for sp in SPACES:
        tot = 0.0
        for f in folds:
            sol = fit(sp, mc.subset(~f), xc.subset(~f), kfold)
            tot += risk(sol, mc.subset(f), xc.subset(f))
        cv[sp] = tot / len(folds)
    return cv


def job(a):
    platform, data_dir, runs, res, seed, ci = a
    lab, cond = CONDS[platform][ci]
    fl = cname(cond)
    Y, sub, parent_rows, clip = get_data(platform, data_dir)
    Fc = MomFile(os.path.join(res, "orig", f"moments_cal_{fl}.npz"))
    Ft = MomFile(os.path.join(res, "orig", f"moments_test_{fl}.npz"))
    U = len(Fc.parent)
    assert U == int(sub.max()) + 1 and np.array_equal(np.bincount(sub), Fc.n.astype(int)), "sub-unit layout mismatch"
    B = member_blocks(platform, seed, "full")
    rows, checks = [], []
    kfold = KFOLD[platform]
    for di, (dn, blk) in enumerate((("D-M", "dm"), ("D-S", "ds"))):
        mids = B[blk]
        P = np.stack([np.load(os.path.join(runs, mid, f"pred_cal_{fl}.npy")).astype(np.float64) for mid in mids])
        if clip is not None:
            P = np.clip(P, clip[0], clip[1])
        Px = np.ascontiguousarray(P.transpose(1, 0, 2)).reshape(P.shape[1], -1)   # f_r, index j*d + k
        mc_all, mt = Fc.sel(seed, mids), Ft.sel(seed, mids)
        xc_all, xt = Fc.xsel(seed, blk), Ft.xsel(seed, blk)
        for m in MS[platform]:
            ng = 10 if platform == "sim" else int(len(np.unique(Fc.parent)) // m)
            for g in range(ng):
                gm, units = split_units(platform, m, g, Fc.parent, U)
                mc, xc = mc_all.subset(gm), xc_all.subset(gm)
                cv = crossfit(mc, xc, mt, xt, kfold)
                cf_sel = min(cv, key=cv.get)
                sols_all = {sp: fit(sp, mc, xc, kfold) for sp in SPACES}
                test = {sp: risk(sols_all[sp], mt, xt) for sp in SPACES}
                best = min(test, key=test.get)
                rng = np.random.default_rng([2926, seed, ci, di, m, g])
                hw = np.zeros((N_SPLIT, len(SPACES)))
                selc = []
                test_conf, test_conf_a = [], []
                for s in range(N_SPLIT):
                    perm = rng.permutation(len(units))
                    nA = len(units) - len(units) // 2          # fit half takes the extra unit
                    A_u = np.concatenate([units[i] for i in perm[:nA]])
                    B_u = np.concatenate([units[i] for i in perm[nA:]])
                    mA = np.zeros(U, bool); mA[A_u] = True
                    mA_c = mA[gm]                              # mask within mc/xc (sub-units in gm order)
                    solsA = {}
                    rowsB = np.isin(sub, B_u)
                    for k, sp in enumerate(SPACES):
                        sol = fit(sp, mc.subset(mA_c), xc.subset(mA_c), kfold)
                        res_norm = np.linalg.norm(predict_rows(sol, P[:, rowsB], Px[rowsB]) - Y[rowsB], axis=1)
                        hw[s, k] = conformal_q(res_norm)
                        solsA[sp] = sol
                    sp_sel = SPACES[int(np.argmin(hw[s]))]
                    selc.append(sp_sel)
                    test_conf.append(test[sp_sel])
                    test_conf_a.append(risk(solsA[sp_sel], mt, xt))
                rec = {"cond": lab, "seed": seed, "dict": dn, "m": m, "group": g, "n_split": N_SPLIT,
                       "cf_selected": cf_sel, "test_cf": test[cf_sel], "posthoc_best": best, "test_posthoc_best": test[best],
                       "test_conf": float(np.mean(test_conf)), "test_conf_sd": float(np.std(test_conf)),
                       "test_conf_Afit": float(np.mean(test_conf_a)),
                       "agree": float(np.mean([s == cf_sel for s in selc]))}
                rec["diff"] = rec["test_conf"] - rec["test_cf"]
                rec["regret_conf"] = rec["test_conf"] - test[best]
                rec["regret_cf"] = rec["test_cf"] - test[best]
                for k, sp in enumerate(SPACES):
                    rec[f"freq_{sp}"] = float(np.mean([s == sp for s in selc]))
                    rec[f"hw_{sp}"] = float(hw[:, k].mean())
                    rec[f"test_{sp}"] = test[sp]
                    rec[f"cv_{sp}"] = cv[sp]
                rows.append(rec)
        # row-wise vs moment check of the full-calibration W0 / W1 risks (once per member set)
        mcal_all = mc_all
        for sp in ("W0", "W1"):
            sol = fit(sp, mcal_all, xc_all, kfold)
            rr = np.mean((predict_rows(sol, P, Px) - Y) ** 2)
            checks.append({"cond": lab, "seed": seed, "dict": dn, "space": sp, "rowwise": float(rr),
                           "moment": float(risk(sol, mcal_all, xc_all)),
                           "absdiff": abs(float(rr) - float(risk(sol, mcal_all, xc_all)))})
    return rows, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(args.platform, args.data_dir, args.runs, args.res, s, ci)
            for ci in range(len(CONDS[args.platform])) for s in SEEDS]
    rows, checks = [], []
    with Pool(args.workers) as pool:
        for r, c in pool.imap_unordered(job, jobs):
            rows += r; checks += c
            print(f"{len(rows)} rows {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["cond", "dict", "m", "seed", "group"]).reset_index(drop=True)
    df.to_csv(os.path.join(args.res, "e26_conformal.csv"), index=False)
    ck = pd.DataFrame(checks)
    print("row-wise vs moment risk max |diff|:", ck["absdiff"].max(), flush=True)

    # consistency of the recomputed cross-fitting selection with e26_rows.csv (shared m values)
    old = pd.read_csv(os.path.join(args.res, "e26_rows.csv"))
    mm = df.merge(old, on=["cond", "seed", "dict", "m", "group"], suffixes=("", "_old"))
    if len(mm):
        print("cross-fit vs e26_rows.csv:", len(mm), "common rows; same selection:",
              float((mm["cf_selected"] == mm["selected"]).mean()),
              "max |test diff|:", float((mm["test_cf"] - mm["test_selected"]).abs().max()), flush=True)

    out = []
    for (c, dn, m), sub in df.groupby(["cond", "dict", "m"], sort=False):
        seed_diff = sub.groupby("seed")["diff"].mean()
        rec = {"cond": c, "dict": dn, "m": m, "n": len(sub),
               "agree": sub["agree"].mean(),
               "test_conf": sub["test_conf"].mean(), "test_cf": sub["test_cf"].mean(),
               "diff": sub["diff"].mean(), "diff_seed_sd": float(seed_diff.std(ddof=1)),
               "regret_conf": sub["regret_conf"].mean(), "regret_cf": sub["regret_cf"].mean(),
               "regret_conf_q90": float(sub["regret_conf"].quantile(0.9)),
               "regret_cf_q90": float(sub["regret_cf"].quantile(0.9)),
               "regret_conf_max": sub["regret_conf"].max(), "regret_cf_max": sub["regret_cf"].max(),
               "test_conf_Afit": sub["test_conf_Afit"].mean()}
        for sp in SPACES:
            rec[f"freq_conf_{sp}"] = sub[f"freq_{sp}"].mean()
            rec[f"freq_cf_{sp}"] = float(np.mean(sub["cf_selected"] == sp))
            rec[f"hw_{sp}"] = sub[f"hw_{sp}"].mean()
            rec[f"test_{sp}"] = sub[f"test_{sp}"].mean()
        out.append(rec)
    pd.DataFrame(out).to_csv(os.path.join(args.res, "e26_conformal_summary.csv"), index=False)
    ck.to_csv(os.path.join(args.res, "e26_conformal_check.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
