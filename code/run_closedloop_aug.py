"""E13: closed-loop follow-up with the attenuation candidates of E09-E12.

Same plant, references, gains, failure rule, calibration budget (m = 10, groups 0-2) and
shared feedback law as E06. The feedforward candidates are

  baselines (rerun so that every comparison is paired inside this run)
    PD only | B-N | D-M:W-S | D-S:W-Q5 | D-M:W-Q5
  attenuation candidates
    D-S+null:W-Q5, D-M+null:W-Q5      constant null predictor as an extra candidate
    D-S|est:W-Q5,  D-M|est:W-Q5       members behind the label-free linear input estimator,
                                      noise variances from the inputs of the calibration group
    D-S,est-optional:W-Q5             plain and estimator versions offered together (k <= 5)
    B-N|est                           the MLP behind the same estimator
  augmented members
    B-A | B-A+(q,qd) | D-SQ:W-Q5      frozen-protocol augmented network, the channel-matched
                                      network with augmented validation, augmented seed dictionary
  diagnostic (O1-M, group 0; not a deployable controller)
    <name>@true-fb                    feedback computed from the true state, feedforward still
                                      fed the perturbed observation

The observation-noise seed is fixed per condition, so the noise draws are identical for
every controller, seed and process. (E06 derived its seed from Python's string hash, which
differs between processes; its comparisons were paired inside its single run.)

python run_closedloop_aug.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda \
       --runs ../runs/sim --frozen ../runs/sim/frozen.json --out ../results/sim/e13_episodes_a.csv --seeds 0,1,2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models import predict
from common.surrogates import SURROGATE_NAMES
from common import calibration as cal
from dump_preds import load_member
from sim.data import load_sim, apply_condition
from sim.panda_env import BatchedClosedLoop, TORQUE_LIMIT

KP = np.array([100.0, 100, 100, 100, 40, 40, 40])
KD = np.array([8.0, 8, 8, 8, 3.2, 3.2, 3.2])
GROUPS = [0, 1, 2]
NOISE_SEED = {"N": 900, "O1M": 901, "P1": 902}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--frozen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--conds", default="N,O1M,P1")
    args = ap.parse_args()
    frozen = json.load(open(args.frozen))
    strong = frozen["strong_surrogate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load_sim(args.data_dir)
    xm, xs, ym, ys = D["x_mean"], D["x_std"], D["y_mean"], D["y_std"]
    q_scale, qd_scale = xs[:7], xs[7:14]
    Xtr = D["X"]["train"].astype(np.float64)
    mu, Sig, v_train = Xtr.mean(0), np.cov(Xtr.T), Xtr.var(0)
    refs_npz = np.load(os.path.join(args.data_dir, "closed_refs.npz"))
    refs = [{"q": refs_npz["q"][i], "qd": refs_npz["qd"][i], "qdd": refs_npz["qdd"][i]} for i in range(refs_npz["q"].shape[0])]
    clip_lo, clip_hi = (-TORQUE_LIMIT - ym) / ys, (TORQUE_LIMIT - ym) / ys
    clip = lambda P: np.clip(P, clip_lo, clip_hi)
    one = np.ones(7)
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mids = {"D-M": [f"dm_{x}_s{seed}" for x in SURROGATE_NAMES],
                "D-S": [f"ds_{strong}_s{seed}_m{i}" for i in range(5)],
                "D-SQ": [f"dsq_{strong}_s{seed}_m{i}" for i in range(5)],
                "B-N": [f"bn_mlp_s{seed}"], "B-A": [f"ba_{strong}_s{seed}"], "B-A+(q,qd)": [f"dmqs_{strong}_s{seed}"]}
        models = {n: [load_member(args.runs, m, 21, 7, device) for m in lst] for n, lst in mids.items()}

        def make_ff(cands, w, W):
            """cands: list of (model or None for the null predictor, behind_estimator flag)."""
            supp = [i for i in range(len(cands)) if w[i] > 1e-12]

            def ff(X):
                xstd = (X - xm) / xs
                xb = torch.as_tensor(xstd, dtype=torch.float32, device=device)
                xe = None
                out = np.zeros((len(X), 7))
                with torch.no_grad():
                    for i in supp:
                        mdl, behind = cands[i]
                        if mdl is None:
                            continue            # null predictor: zero in standardized units
                        if behind:
                            if xe is None:
                                xe = torch.as_tensor(mu + (xstd - mu) @ W.T, dtype=torch.float32, device=device)
                            p = mdl(xe)[-1].double().cpu().numpy()
                        else:
                            p = mdl(xb)[-1].double().cpu().numpy()
                        out = out + w[i] * clip(p)
                return out * ys + ym
            n_exec = sum(1 for i in supp if cands[i][0] is not None)
            return ff, n_exec

        for cond in args.conds.split(","):
            if cond == "N":
                cal_role, payload, noise, cond_tag = "cal", 0.0, None, "N"
            elif cond == "O1M":
                cal_role, payload, noise, cond_tag = "cal", 0.0, frozen["O1_M"], f"O1:{frozen['O1_M']}"
            else:
                cal_role, payload, noise, cond_tag = "cal_P1", 1.0, None, "N"
            Xcal_c = apply_condition(D["X"][cal_role], cal_role, cond_tag).astype(np.float64)
            Ycal = D["Y"][cal_role].astype(np.float64); gcal = D["groups"][cal_role]
            for g in GROUPS:
                idx = np.where(np.isin(gcal, np.arange(10 * g, 10 * g + 10)))[0]
                Yc, gc = Ycal[idx], gcal[idx]
                v_est = np.maximum(Xcal_c[idx].var(0) - v_train, 0.0)
                W = Sig @ np.linalg.inv(Sig + np.diag(v_est))
                Xc_raw = Xcal_c[idx].astype(np.float32)
                Xc_est = (mu + (Xcal_c[idx] - mu) @ W.T).astype(np.float32)
                P = lambda name, X: [clip(predict(m, X, device).astype(np.float64)) for m in models[name]]
                raw = {n: P(n, Xc_raw) for n in ["D-M", "D-S", "D-SQ"]}
                est = {n: P(n, Xc_est) for n in ["D-M", "D-S"]}
                zero = [np.zeros_like(Yc)]

                def Chat(Pl):
                    return cal.residual_matrix(Pl, Yc, one, gc)[0]

                wq = lambda Pl: cal.budgeted_qp(Chat(Pl), min(5, len(Pl)))["w"]
                C = lambda name, behind=False: [(m, behind) for m in models[name]]
                NULL = [(None, False)]
                ctl = {}
                if g == GROUPS[0]:
                    ctl["PD"] = (None, None)
                    ctl["B-N"] = (C("B-N"), [1.0])
                    ctl["B-A"] = (C("B-A"), [1.0])
                    ctl["B-A+(q,qd)"] = (C("B-A+(q,qd)"), [1.0])
                ctl["B-N|est"] = (C("B-N", True), [1.0])
                ctl["D-M:W-S"] = (C("D-M"), cal.rule_WS(Chat(raw["D-M"])))
                ctl["D-S:W-Q5"] = (C("D-S"), wq(raw["D-S"]))
                ctl["D-M:W-Q5"] = (C("D-M"), wq(raw["D-M"]))
                ctl["D-S+null:W-Q5"] = (C("D-S") + NULL, wq(raw["D-S"] + zero))
                ctl["D-M+null:W-Q5"] = (C("D-M") + NULL, wq(raw["D-M"] + zero))
                ctl["D-S|est:W-Q5"] = (C("D-S", True), wq(est["D-S"]))
                ctl["D-M|est:W-Q5"] = (C("D-M", True), wq(est["D-M"]))
                ctl["D-S,est-optional:W-Q5"] = (C("D-S") + C("D-S", True), wq(raw["D-S"] + est["D-S"]))
                ctl["D-SQ:W-Q5"] = (C("D-SQ"), wq(raw["D-SQ"]))
                runs_ = [(n, c, w, False) for n, (c, w) in ctl.items()]
                if cond == "O1M" and g == GROUPS[0]:
                    runs_ += [(n + "@true-fb", ctl[n][0], ctl[n][1], True)
                              for n in ["PD", "B-N", "B-N|est", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-S|est:W-Q5"]]
                for name, cands, w, fb_true in runs_:
                    if cands is None:
                        ff, n_exec, null_w, est_w = None, 0, np.nan, np.nan
                    else:
                        ff, n_exec = make_ff(cands, w, W)
                        null_w = float(sum(w[i] for i in range(len(cands)) if cands[i][0] is None))
                        est_w = float(sum(w[i] for i in range(len(cands)) if cands[i][0] is not None and cands[i][1]))
                    t0 = time.time()
                    loop = BatchedClosedLoop(args.xml_dir, refs, KP, KD, payload_kg=payload, obs_noise=noise,
                                             noise_seed=NOISE_SEED[cond], q_scale=q_scale, qd_scale=qd_scale)
                    res = loop.run(ff, fb_true=fb_true)
                    wall = time.time() - t0
                    for e in range(len(refs)):
                        rows.append({"seed": seed, "cond": cond, "group": g, "controller": name, "episode": e,
                                     "rmse": res["rmse"][e], "maxerr": res["maxerr"][e], "tau_rms": res["tau_rms"][e],
                                     "tau_fb_rms": res["tau_fb_rms"][e], "tau_ff_rms": res["tau_ff_rms"][e],
                                     "saturation": res["saturation"][e], "failed": bool(res["failed"][e]),
                                     "n_exec": n_exec, "null_weight": null_w, "est_weight": est_w,
                                     "v_est_mean_qqd": float(v_est[:14].mean()), "wall_s": wall})
                    print(f"seed {seed} {cond} g{g} {name}: rmse {res['rmse'].mean():.4f} fail {res['failed'].mean():.2f} "
                          f"sat {res['saturation'].mean():.3f} wall {wall:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("E13_DONE")


if __name__ == "__main__":
    main()
