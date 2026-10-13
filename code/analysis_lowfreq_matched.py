"""E13c: open-loop total and slow-part nMSE of every E13 feedforward model under the weights the
closed-loop run used (m = 10 trajectories, calibration groups 0-2; estimator variances from the
inputs of the same group), per seed, condition, and group. The weight fitting mirrors
run_closedloop_aug.py line by line; the metrics mirror analysis_lowfreq.py (full-Dcal weights).
Post-processing (figures/build_e13_tables.py) pairs each row with the tracking RMSE of the same
controller, seed, and group in e13_episodes_*.csv.

python analysis_lowfreq_matched.py --data_dir ../data_sim --runs ../runs/sim --frozen ../runs/sim/frozen.json \
       --out ../results/sim/e13c_lowfreq_matched.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models import predict
from common.surrogates import SURROGATE_NAMES
from common import calibration as cal
from dump_preds import load_member
from sim.data import load_sim, apply_condition
from sim.panda_env import TORQUE_LIMIT
from analysis_lowfreq import moving_average

GROUPS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--frozen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--conds", default="N,O1M")
    ap.add_argument("--windows", default="10,25,50")
    args = ap.parse_args()
    frozen = json.load(open(args.frozen)); strong = frozen["strong_surrogate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load_sim(args.data_dir)
    ym, ys = D["y_mean"], D["y_std"]
    Xtr = D["X"]["train"].astype(np.float64)
    mu, Sig, v_train = Xtr.mean(0), np.cov(Xtr.T), Xtr.var(0)
    clip_lo, clip_hi = (-TORQUE_LIMIT - ym) / ys, (TORQUE_LIMIT - ym) / ys
    clip = lambda P: np.clip(P, clip_lo, clip_hi)
    d = D["n_out"]; one = np.ones(d)
    Ytest = D["Y"]["test"].astype(np.float64); gtest = D["groups"]["test"]
    Ls = [int(x) for x in args.windows.split(",")]
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mids = {"D-M": [f"dm_{x}_s{seed}" for x in SURROGATE_NAMES],
                "D-S": [f"ds_{strong}_s{seed}_m{i}" for i in range(5)],
                "D-SQ": [f"dsq_{strong}_s{seed}_m{i}" for i in range(5)],
                "B-N": [f"bn_mlp_s{seed}"], "B-A": [f"ba_{strong}_s{seed}"], "B-A+(q,qd)": [f"dmqs_{strong}_s{seed}"]}
        models = {n: [load_member(args.runs, m, 21, 7, device) for m in lst] for n, lst in mids.items()}
        P = lambda name, X: [clip(predict(m, X.astype(np.float32), device).astype(np.float64)) for m in models[name]]
        for cond in args.conds.split(","):
            cond_tag = "N" if cond == "N" else f"O1:{frozen['O1_M']}"
            Xcal_c = apply_condition(D["X"]["cal"], "cal", cond_tag).astype(np.float64)
            Ycal = D["Y"]["cal"].astype(np.float64); gcal = D["groups"]["cal"]
            Xte_c = apply_condition(D["X"]["test"], "test", cond_tag).astype(np.float64)
            raw_te = {n: P(n, Xte_c) for n in ["D-M", "D-S", "D-SQ", "B-N", "B-A", "B-A+(q,qd)"]}
            zero_te = [np.zeros_like(Ytest)]
            for g in GROUPS:
                idx = np.where(np.isin(gcal, np.arange(10 * g, 10 * g + 10)))[0]
                Yc, gc = Ycal[idx], gcal[idx]
                v_est = np.maximum(Xcal_c[idx].var(0) - v_train, 0.0)
                W = Sig @ np.linalg.inv(Sig + np.diag(v_est))
                Xc_raw = Xcal_c[idx]
                Xc_est = mu + (Xcal_c[idx] - mu) @ W.T
                Xte_est = mu + (Xte_c - mu) @ W.T
                raw = {n: P(n, Xc_raw) for n in ["D-M", "D-S", "D-SQ"]}
                est = {n: P(n, Xc_est) for n in ["D-M", "D-S"]}
                est_te = {n: P(n, Xte_est) for n in ["D-M", "D-S", "B-N"]}
                zero = [np.zeros_like(Yc)]

                def Chat(Pl):
                    return cal.residual_matrix(Pl, Yc, one, gc)[0]

                wq = lambda Pl: cal.budgeted_qp(Chat(Pl), min(5, len(Pl)))["w"]
                # name -> (test predictions of the candidates, weights, null flags, estimator flags)
                ctl = {}
                if g == GROUPS[0]:
                    ctl["B-N"] = (raw_te["B-N"], [1.0], [0], [0])
                    ctl["B-A"] = (raw_te["B-A"], [1.0], [0], [0])
                    ctl["B-A+(q,qd)"] = (raw_te["B-A+(q,qd)"], [1.0], [0], [0])
                ctl["B-N|est"] = (est_te["B-N"], [1.0], [0], [1])
                ctl["D-M:W-S"] = (raw_te["D-M"], cal.rule_WS(Chat(raw["D-M"])), [0] * 5, [0] * 5)
                ctl["D-S:W-Q5"] = (raw_te["D-S"], wq(raw["D-S"]), [0] * 5, [0] * 5)
                ctl["D-M:W-Q5"] = (raw_te["D-M"], wq(raw["D-M"]), [0] * 5, [0] * 5)
                ctl["D-S+null:W-Q5"] = (raw_te["D-S"] + zero_te, wq(raw["D-S"] + zero), [0] * 5 + [1], [0] * 6)
                ctl["D-M+null:W-Q5"] = (raw_te["D-M"] + zero_te, wq(raw["D-M"] + zero), [0] * 5 + [1], [0] * 6)
                ctl["D-S|est:W-Q5"] = (est_te["D-S"], wq(est["D-S"]), [0] * 5, [1] * 5)
                ctl["D-M|est:W-Q5"] = (est_te["D-M"], wq(est["D-M"]), [0] * 5, [1] * 5)
                ctl["D-S,est-optional:W-Q5"] = (raw_te["D-S"] + est_te["D-S"], wq(raw["D-S"] + est["D-S"]), [0] * 10, [0] * 5 + [1] * 5)
                ctl["D-SQ:W-Q5"] = (raw_te["D-SQ"], wq(raw["D-SQ"]), [0] * 5, [0] * 5)
                for name, (Pt, w, is_null, is_est) in ctl.items():
                    w = np.asarray(w, dtype=np.float64)
                    Pw = sum(float(wj) * p for wj, p in zip(w, Pt))
                    E = Pw - Ytest
                    r = {"seed": seed, "cond": cond, "group": g, "method": name,
                         "nmse_total": float(np.mean(np.sum(E ** 2, 1)) / d)}
                    for L in Ls:
                        M = moving_average(E, gtest, L)
                        r[f"nmse_slow_L{L}"] = float(np.mean(np.sum(M ** 2, 1)) / d)
                    r["n_exec"] = int(sum(1 for i in range(len(w)) if w[i] > 1e-12 and not is_null[i]))
                    r["null_weight"] = float(sum(w[i] for i in range(len(w)) if is_null[i]))
                    r["est_weight"] = float(sum(w[i] for i in range(len(w)) if is_est[i] and not is_null[i]))
                    r["v_est_mean_qqd"] = float(v_est[:14].mean())
                    r["w"] = json.dumps([round(float(x), 4) for x in w])
                    rows.append(r)
                print(f"seed {seed} {cond} g{g}: " + ", ".join(f"{n} {ctl[n][0] and rows[-len(ctl) + i]['nmse_total']:.4f}" for i, n in enumerate(ctl)), flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(df.groupby(["cond", "method"], sort=False)[["nmse_total"] + [f"nmse_slow_L{L}" for L in Ls]].mean().round(4).to_string())
    print("E13C_DONE")


if __name__ == "__main__":
    main()
