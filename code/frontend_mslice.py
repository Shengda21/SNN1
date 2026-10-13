"""E11/E12 calibration-budget slice: how many calibration units do the null candidate
and the label-free input estimator need?

For every seed, calibration group g and budget m (simulated arm: m trajectories of group
g; SARCOS: n rows drawn with the group seed, exactly as in E03), the noise variances of
the input estimator are estimated from the inputs of that calibration subset only, the
frozen members of D-M and D-S are evaluated behind the resulting estimator on the subset
and on the test set, and W-Q(5) is fitted on the subset for five candidate sets:
plain, + null, estimator, estimator + null, and plain united with estimator versions.
Nothing is saved but the result rows.

python frontend_mslice.py --platform sim --data_dir ../data_sim --runs ../runs/sim \
       --frozen ../runs/sim/frozen.json --out ../results/sim/e11_m_slice.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.surrogates import SURROGATE_NAMES
from common import calibration as cal
from common.metrics import nmse
from common.models import predict
from dump_preds import load_member
from run_train import load_platform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--frozen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--groups", type=int, default=10)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, apply_condition = load_platform(args.platform, args.data_dir)
    frozen = json.load(open(args.frozen))
    sim = args.platform == "sim"
    strong = frozen["strong_surrogate"]
    if sim:
        from sim.panda_env import TORQUE_LIMIT
        lo = (-TORQUE_LIMIT - D["y_mean"]) / D["y_std"]; hi = (TORQUE_LIMIT - D["y_mean"]) / D["y_std"]
        clip = lambda P: np.clip(P, lo, hi)
        budgets = [1, 2, 5, 10]
    else:
        clip = lambda P: P
        budgets = [50, 200, 1000, 4448]
    d = D["n_out"]; scale = np.ones(d)
    Xtr = D["X"]["train"].astype(np.float64)
    mu, Sig, v_train = Xtr.mean(0), np.cov(Xtr.T), Xtr.var(0)
    Ycal, Ytest = D["Y"]["cal"].astype(np.float64), D["Y"]["test"].astype(np.float64)
    gcal_all = D["groups"]["cal"] if sim else None

    def subset(m, g):
        if sim:
            return np.where(np.isin(gcal_all, np.arange(10 * g, 10 * g + m)))[0]
        return np.random.default_rng(500 + g).permutation(len(Ycal))[:m]

    def wq5(Pc, Yc, gc):
        C, _, _ = cal.residual_matrix(Pc, Yc, scale, gc)
        return cal.budgeted_qp(C, min(5, len(Pc)))["w"]

    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mids = {"D-M": [f"dm_{x}_s{seed}" for x in SURROGATE_NAMES], "D-S": [f"ds_{strong}_s{seed}_m{i}" for i in range(5)]}
        models = {dn: [load_member(args.runs, m, D["n_in"], d, device) for m in ms] for dn, ms in mids.items()}
        for cond in ["N", f"O1:{frozen['O1_M']}"]:
            Xc_cal = apply_condition(D["X"]["cal"], "cal", cond).astype(np.float64)
            Xc_test = apply_condition(D["X"]["test"], "test", cond).astype(np.float64)
            raw = {dn: {"cal": [clip(predict(mo, Xc_cal.astype(np.float32), device).astype(np.float64)) for mo in models[dn]],
                        "test": [clip(predict(mo, Xc_test.astype(np.float32), device).astype(np.float64)) for mo in models[dn]]} for dn in mids}
            for g in range(args.groups):
                for m in budgets:
                    idx = subset(m, g)
                    gc = gcal_all[idx] if sim else None
                    v_est = np.maximum(Xc_cal[idx].var(0) - v_train, 0.0)
                    Wm = Sig @ np.linalg.inv(Sig + np.diag(v_est))
                    Xh_cal = (mu + (Xc_cal[idx] - mu) @ Wm.T).astype(np.float32)
                    Xh_test = (mu + (Xc_test - mu) @ Wm.T).astype(np.float32)
                    Yc = Ycal[idx]
                    z_c, z_t = [np.zeros_like(Yc)], [np.zeros_like(Ytest)]
                    for dn in mids:
                        rc = [p[idx] for p in raw[dn]["cal"]]; rt = raw[dn]["test"]
                        fc = [clip(predict(mo, Xh_cal, device).astype(np.float64)) for mo in models[dn]]
                        ft = [clip(predict(mo, Xh_test, device).astype(np.float64)) for mo in models[dn]]
                        cands = {"plain": (rc, rt), "+null": (rc + z_c, rt + z_t), "estimator": (fc, ft),
                                 "estimator+null": (fc + z_c, ft + z_t), "plain+estimator": (rc + fc, rt + ft)}
                        for name, (Pc, Pt) in cands.items():
                            w = wq5(Pc, Yc, gc)
                            P = sum(wj * p for wj, p in zip(w, Pt))
                            rows.append({"seed": seed, "cond": cond, "group": g, "m": m, "dictionary": dn, "candidates": name,
                                         "nmse": nmse(P, Ytest, scale), "support_size": int(np.sum(w > 1e-12)),
                                         "null_weight": float(w[-1]) if name.endswith("null") else np.nan,
                                         "estimator_weight": float(w[5:].sum()) if name == "plain+estimator" else np.nan,
                                         "v_est_mean": float(v_est.mean())})
            print("seed", seed, cond, "done", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.groupby(["cond", "m", "dictionary", "candidates"]).nmse.mean().unstack("candidates").round(4).to_string())


if __name__ == "__main__":
    main()
