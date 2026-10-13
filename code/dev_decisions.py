"""Freeze the development-stage decisions from D_dev only.

Reads the width-search members (seed 0) and writes frozen.json with:
  widths[surrogate]      : dev-selected width per family (min dev nMSE, nominal)
  strong_surrogate       : family with the lowest dev nMSE at its chosen width
  O1_M, O1_L, O1_H       : observation-noise levels; O1_M is the largest level in the
                           candidate grid at which the strong single model keeps dev
                           R^2 >= 0.5 (usable accuracy), O1_L = O1_M/2, O1_H = 2*O1_M
  aug_sigma              : training augmentation level for B-A (= O1_M)
  conds                  : the condition list used for every formal prediction dump
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.surrogates import SURROGATE_NAMES, WIDTH_GRID
from common.metrics import r2_pooled, nmse

CAND = [0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.platform == "sarcos":
        from sarcos.data import load_sarcos
        D = load_sarcos(args.data_dir)
    else:
        from sim.data import load_sim
        D = load_sim(args.data_dir)
    Ydev = D["Y"]["dev"]
    table = {}
    for s in SURROGATE_NAMES:
        for w in WIDTH_GRID[s]:
            d = os.path.join(args.runs, f"width_{s}_{w}_s0")
            if not os.path.exists(os.path.join(d, "config.json")):
                print("missing", d); continue
            cfg = json.load(open(os.path.join(d, "config.json")))
            P = np.load(os.path.join(d, "pred_dev_N.npy"))
            table[(s, w)] = {"dev_nmse": nmse(P, Ydev, np.ones(7)), "dev_r2": r2_pooled(P, Ydev),
                             "best_epoch": cfg["best_epoch"], "train_time_s": cfg["train_time_s"]}
    widths = {}
    for s in SURROGATE_NAMES:
        cands = [(table[(s, w)]["dev_nmse"], w) for w in WIDTH_GRID[s] if (s, w) in table]
        widths[s] = min(cands)[1]
    strong = min(SURROGATE_NAMES, key=lambda s: table[(s, widths[s])]["dev_nmse"])
    # noise rule on the strong single model (dev predictions under candidate levels)
    d = os.path.join(args.runs, f"width_{strong}_{widths[strong]}_s0")
    import torch
    from common.models import SNN, predict
    if args.platform == "sarcos":
        from sarcos.data import apply_condition
    else:
        from sim.data import apply_condition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNN(D["n_in"], D["n_out"], surrogate=strong, width=widths[strong])
    model.load_state_dict(torch.load(os.path.join(d, "model.pt"), map_location=device)); model = model.to(device).eval()
    r2_levels = {}
    for rho in CAND:
        P = predict(model, apply_condition(D["X"]["dev"], "dev", f"O1:{rho}"), device)
        r2_levels[rho] = r2_pooled(P, Ydev)
    ok = [rho for rho in CAND if r2_levels[rho] >= 0.5]
    O1_M = max(ok) if ok else CAND[0]
    frozen = {"platform": args.platform, "widths": widths, "strong_surrogate": strong,
              "O1_M": O1_M, "O1_L": O1_M / 2, "O1_H": (CAND[CAND.index(O1_M) + 1] if CAND.index(O1_M) + 1 < len(CAND) else O1_M * 2), "aug_sigma": O1_M,
              "dev_r2_strong_by_level": r2_levels,
              "width_table": {f"{k[0]}:{k[1]}": v for k, v in table.items()}}
    O1_H = frozen["O1_H"]
    if args.platform == "sarcos":
        frozen["conds"] = ["N", f"O1:{O1_M}", f"O1:{O1_H}", f"O1qqd:{O1_M}"]
    else:
        frozen["conds"] = ["N", f"O1:{O1_M / 2}", f"O1:{O1_M}", f"O1:{O1_H}", f"O2B:{O1_M}"]
    with open(args.out, "w") as f:
        json.dump(frozen, f, indent=1)
    print(json.dumps(frozen, indent=1))


if __name__ == "__main__":
    main()
