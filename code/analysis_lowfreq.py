"""E13b: slow and fast parts of the open-loop residual on the simulated arm.

The observation noise is white in time (independent at every 10-ms control step) while the
reference and the torque vary at 0.1-0.4 Hz. A moving average of the residual over L control
steps along each test trajectory therefore keeps the slow, systematic part of the error (the
part a plant with inertia cannot average out) and suppresses the white part by a factor L.
For every method the script reports the total test nMSE and the nMSE of the L-step moving
average of its residual (the slow part), under N and O1-M, per seed.

python analysis_lowfreq.py --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json \
       --out ../results/sim/e13b_lowfreq.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.surrogates import SURROGATE_NAMES
from common import calibration as cal
from sim.data import load_sim
from sim.panda_env import TORQUE_LIMIT


def moving_average(E, g, L):
    out = []
    for t in np.unique(g):
        e = E[g == t]
        c = np.cumsum(np.vstack([np.zeros((1, e.shape[1])), e]), 0)
        out.append((c[L:] - c[:-L]) / L)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--frozen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--windows", default="10,25,50")
    args = ap.parse_args()
    frozen = json.load(open(args.frozen)); strong = frozen["strong_surrogate"]
    D = load_sim(args.data_dir)
    lo = (-TORQUE_LIMIT - D["y_mean"]) / D["y_std"]; hi = (TORQUE_LIMIT - D["y_mean"]) / D["y_std"]
    d = D["n_out"]; one = np.ones(d)
    Ycal, Ytest = D["Y"]["cal"].astype(np.float64), D["Y"]["test"].astype(np.float64)
    gcal, gtest = D["groups"]["cal"], D["groups"]["test"]
    Ls = [int(x) for x in args.windows.split(",")]

    def pred(mid, role, cond, var=""):
        f = os.path.join(args.runs, mid, f"pred_{role}_{cond.replace(':', '_')}{('__' + var) if var else ''}.npy")
        return np.clip(np.load(f).astype(np.float64), lo, hi)

    rows = []
    for seed in range(5):
        dm = [f"dm_{x}_s{seed}" for x in SURROGATE_NAMES]
        ds = [f"ds_{strong}_s{seed}_m{i}" for i in range(5)]
        dsq = [f"dsq_{strong}_s{seed}_m{i}" for i in range(5)]
        for cond in ["N", f"O1:{frozen['O1_M']}"]:
            zc, zt = [np.zeros_like(Ycal)], [np.zeros_like(Ytest)]

            def PP(mids, var=""):
                return [pred(m, "cal", cond, var) for m in mids], [pred(m, "test", cond, var) for m in mids]

            cands = {"D-S:W-Q5": PP(ds), "D-M:W-Q5": PP(dm), "D-SQ:W-Q5": PP(dsq),
                     "D-S|est:W-Q5": PP(ds, "wfe"), "D-M|est:W-Q5": PP(dm, "wfe")}
            cands["D-S+null:W-Q5"] = (cands["D-S:W-Q5"][0] + zc, cands["D-S:W-Q5"][1] + zt)
            cands["D-M+null:W-Q5"] = (cands["D-M:W-Q5"][0] + zc, cands["D-M:W-Q5"][1] + zt)
            cands["D-S,est-optional:W-Q5"] = (cands["D-S:W-Q5"][0] + cands["D-S|est:W-Q5"][0], cands["D-S:W-Q5"][1] + cands["D-S|est:W-Q5"][1])
            outs = {}
            for name, (Pc, Pt) in cands.items():
                C = cal.residual_matrix(Pc, Ycal, one, gcal)[0]
                w = cal.budgeted_qp(C, min(5, len(Pc)))["w"]
                outs[name] = sum(wj * p for wj, p in zip(w, Pt))
            Cm = cal.residual_matrix(cands["D-M:W-Q5"][0], Ycal, one, gcal)[0]
            outs["D-M:W-S"] = cands["D-M:W-Q5"][1][int(np.argmin(np.diag(Cm)))]
            for name, mid, var in [("B-N", f"bn_mlp_s{seed}", ""), ("B-N|est", f"bn_mlp_s{seed}", "wfe"), ("B-A", f"ba_{strong}_s{seed}", ""),
                                   ("B-A+(q,qd)", f"dmqs_{strong}_s{seed}", "")]:
                outs[name] = pred(mid, "test", cond, var)
            for name, P in outs.items():
                E = (P - Ytest)
                r = {"seed": seed, "cond": cond, "method": name, "nmse_total": float(np.mean(np.sum(E ** 2, 1)) / d)}
                for L in Ls:
                    M = moving_average(E, gtest, L)
                    r[f"nmse_slow_L{L}"] = float(np.mean(np.sum(M ** 2, 1)) / d)
                rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["cond", "method"], sort=False).mean(numeric_only=True).drop(columns=["seed"]).round(4).to_string())


if __name__ == "__main__":
    main()
