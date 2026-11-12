"""Cross-checks of the moment-based E21 results against the manuscript's result files.

  orig source:  W0 = W-Q(5) (e01_test_table), singles (e01), W1 = W-Q5 on D-x+0 (e09_rows,
                budget 5 of 6 candidates), W5s = W-Q5 on D-x|wfe, W5 = D-x|wfe+0, W6 = D-xA.
  paired grid:  rho = 0 equals the nominal predictions (checked at dump time); D-M W0 at the
                frozen levels within +-0.01 of the manuscript.
python e2x_verify.py --platform sim --res ../results/sim --orig ../results/sim
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

LAB = {"sim": {"N": "N", "O1:0.35": "O1-L", "O1:0.7": "O1-M", "O1:1.0": "O1-H", "O2B:0.7": "O2B", "P1": "P1"},
       "sarcos": {"N": "N", "O1:0.5": "O1-M", "O1:0.7": "O1-H", "O1qqd:0.5": "O1qqd"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--orig", required=True)
    args = ap.parse_args()
    lab = LAB[args.platform]
    out = {}
    mine = pd.read_csv(os.path.join(args.res, "e21_ladder_seeds.csv"))
    e01 = pd.read_csv(os.path.join(args.orig, "e01_test_table.csv"))
    e01 = e01[e01.cond.isin(lab)].copy(); e01["cond"] = e01["cond"].map(lab)
    e09 = pd.read_csv(os.path.join(args.orig, "e09_rows.csv"))
    e09 = e09[e09.cond.isin(lab)].copy(); e09["cond"] = e09["cond"].map(lab)
    checks = []
    for (meth, sp, src, key) in [("D-M", "W0", e01, ("W-Q5", "D-M")), ("D-S", "W0", e01, ("W-Q5", "D-S")),
                                 ("D-T", "W0", e01, ("W-Q5", "D-T")), ("B-F", "W0", e01, ("B-F", "D-M")),
                                 ("B-L", "W0", e01, ("B-L", "-")), ("B-A", "W0", e01, ("B-A", "-")), ("B-N", "W0", e01, ("B-N", "-")),
                                 ("D-M", "W1", e09, ("W-Q5", "D-M+0")), ("D-S", "W1", e09, ("W-Q5", "D-S+0")),
                                 ("D-M", "W5s", e09, ("W-Q5", "D-M|wfe")), ("D-S", "W5s", e09, ("W-Q5", "D-S|wfe")),
                                 ("D-M", "W5", e09, ("W-Q5", "D-M|wfe+0")), ("D-S", "W5", e09, ("W-Q5", "D-S|wfe+0")),
                                 ("D-M", "W5o", e09, ("W-Q5", "D-M|wfo+0")), ("D-M", "W6", e09, ("W-Q5", "D-MA")),
                                 ("D-S", "W6", e09, ("W-Q5", "D-SA")), ("B-N", "W5", e09, ("B-N|wfe", "-")),
                                 ("B-A+", "W0", e09, ("B-A+", "-")), ("B-AQ+", "W0", e09, ("B-AQ+", "-"))]:
        a = mine[(mine.method == meth) & (mine.space == sp)][["cond", "seed", "nmse"]]
        b = src[(src.method == key[0]) & (src.dictionary == key[1])][["cond", "seed", "nmse"]]
        m = a.merge(b, on=["cond", "seed"], suffixes=("_mine", "_orig"))
        if not len(m):
            continue
        m["d"] = m["nmse_mine"] - m["nmse_orig"]
        for c, sub in m.groupby("cond"):
            checks.append({"method": meth, "space": sp, "orig": "|".join(key), "cond": c, "n": len(sub),
                           "mine_mean": sub["nmse_mine"].mean(), "orig_mean": sub["nmse_orig"].mean(),
                           "max_abs_diff": float(sub["d"].abs().max())})
    cf = pd.DataFrame(checks)
    cf.to_csv(os.path.join(args.res, "e2x_verify_orig.csv"), index=False)
    print(cf.to_string())
    # paired-noise source at frozen levels vs manuscript (D-M and D-S W0)
    p = os.path.join(args.res, "e21_ladder_pairednoise.csv")
    if os.path.exists(p):
        pp = pd.read_csv(p)
        om = pd.read_csv(os.path.join(args.res, "e21_ladder.csv"))
        mm = pp.merge(om, on=["cond", "method", "space"], suffixes=("_paired", "_orig"))
        mm = mm[mm.space.isin(["W0", "W1", "W2", "W3"])][["cond", "method", "space", "nmse_mean_paired", "nmse_mean_orig"]]
        mm["diff"] = mm["nmse_mean_paired"] - mm["nmse_mean_orig"]
        mm["within_0.01"] = mm["diff"].abs() <= 0.01
        mm.to_csv(os.path.join(args.res, "e2x_verify_paired.csv"), index=False)
        print(mm[mm.method.isin(["D-M", "D-S"])].to_string())


if __name__ == "__main__":
    main()
