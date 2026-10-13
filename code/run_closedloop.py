"""E06: closed-loop tracking with a shared feedback law and frozen feedforward
predictors. Controllers:
  CL0 PD only                       CL1 PD + B-N (MLP)
  CL2 PD + W-S on D-M (1 executed)  CL3 PD + strong single (B-L or B-A, chosen on D_dev)
  CL4 PD + ordinary dict W-Q(2)     CL5 PD + ordinary dict W-Q(5)   (D-S or D-T, chosen on D_dev)
  CL6 PD + D-M W-Q(2)               CL7 PD + D-M W-Q(5)
Calibration: m = 10 trajectories, three pre-specified groups (0, 1, 2) of D_cal of the
same condition; weights fitted here with the same rules as the open-loop analysis.
Conditions: N, O1-M, P1. 30 reference episodes shared by every controller; the
observation-noise draws of a condition are identical across controllers.
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
from common.models import SNN, MLP
from common.surrogates import SURROGATE_NAMES
from common import calibration as cal
from common.metrics import nmse
from sim.data import load_sim
from sim.panda_env import BatchedClosedLoop, TORQUE_LIMIT

KP = np.array([100.0, 100, 100, 100, 40, 40, 40])
KD = np.array([8.0, 8, 8, 8, 3.2, 3.2, 3.2])
GROUPS = [0, 1, 2]


def load_member(runs, mid, device):
    cfg = json.load(open(os.path.join(runs, mid, "config.json")))
    if cfg["block"] == "bn":
        model = MLP(21, 7)
    else:
        model = SNN(21, 7, surrogate=cfg["surrogate"], width=cfg["width"], learnable_width=cfg["learnable_width"])
    model.load_state_dict(torch.load(os.path.join(runs, mid, "model.pt"), map_location=device))
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--conds", default="N,O1M,P1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frozen = json.load(open(args.frozen))
    strong = frozen["strong_surrogate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = load_sim(args.data_dir)
    xm, xs, ym, ys = D["x_mean"], D["x_std"], D["y_mean"], D["y_std"]
    q_scale, qd_scale = xs[:7], xs[7:14]
    refs_npz = np.load(os.path.join(args.data_dir, "closed_refs.npz"))
    refs = [{"q": refs_npz["q"][i], "qd": refs_npz["qd"][i], "qdd": refs_npz["qdd"][i]} for i in range(refs_npz["q"].shape[0])]
    clip_lo, clip_hi = (-TORQUE_LIMIT - ym) / ys, (TORQUE_LIMIT - ym) / ys
    rows, traces = [], {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        mids = {"D-M": [f"dm_{x}_s{seed}" for x in SURROGATE_NAMES],
                "D-S": [f"ds_{strong}_s{seed}_m{i}" for i in range(5)],
                "D-T": [f"dt_{strong}_s{seed}_c{i}" for i in range(5)],
                "B-L": [f"bl_learnable_s{seed}"], "B-A": [f"ba_{strong}_s{seed}"], "B-N": [f"bn_mlp_s{seed}"]}
        models = {}
        for name, lst in mids.items():
            models[name] = [load_member(args.runs, m, device) for m in lst]
        # dev-only choices: strong single (B-L vs B-A) and ordinary dictionary (D-S vs D-T)
        dev_nmse = lambda mid: nmse(np.load(os.path.join(args.runs, mid, "pred_dev_N.npy")), D["Y"]["dev"], np.ones(7))
        strong_single = "B-L" if dev_nmse(mids["B-L"][0]) <= dev_nmse(mids["B-A"][0]) else "B-A"
        ord_dict = "D-S" if np.mean([dev_nmse(m) for m in mids["D-S"]]) <= np.mean([dev_nmse(m) for m in mids["D-T"]]) else "D-T"
        print("seed", seed, "strong single", strong_single, "ordinary dict", ord_dict, flush=True)

        def make_ff(members, w):
            supp = [i for i in range(len(members)) if w[i] > 0]
            mods = [members[i][0] for i in supp]; ws = [w[i] for i in supp]

            def ff(X):
                xb = torch.as_tensor((X - xm) / xs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    out = 0
                    for mdl, wj in zip(mods, ws):
                        p = mdl(xb)[-1].double().cpu().numpy()
                        out = out + wj * np.clip(p, clip_lo, clip_hi)
                return out * ys + ym
            return ff, len(supp)

        for cond in args.conds.split(","):
            if cond == "N":
                cal_role, payload, noise, cond_tag = "cal", 0.0, None, "N"
            elif cond == "O1M":
                cal_role, payload, noise, cond_tag = "cal", 0.0, frozen["O1_M"], f"O1:{frozen['O1_M']}"
            elif cond == "P1":
                cal_role, payload, noise, cond_tag = "cal_P1", 1.0, None, "N"
            Ycal = D["Y"][cal_role]; gcal = D["groups"][cal_role]
            for g in GROUPS:
                idx = np.where(np.isin(gcal, np.arange(10 * g, 10 * g + 10)))[0]
                weights = {}
                for name in ["D-M", ord_dict]:
                    preds = [np.clip(np.load(os.path.join(args.runs, m, f"pred_{cal_role}_{cond_tag.replace(':', '_')}.npy"))[idx].astype(np.float64), clip_lo, clip_hi)
                             for m in mids[name]]
                    Chat, _, _ = cal.residual_matrix(preds, Ycal[idx], np.ones(7), gcal[idx])
                    weights[name] = {"W-S": cal.rule_WS(Chat), "W-Q2": cal.budgeted_qp(Chat, 2)["w"],
                                     "W-Q5": cal.budgeted_qp(Chat, 5)["w"]}
                controllers = {
                    "CL0:PD": (None, 0),
                    "CL1:PD+B-N": make_ff(models["B-N"], [1.0]),
                    "CL2:PD+D-M:W-S": make_ff(models["D-M"], weights["D-M"]["W-S"]),
                    f"CL3:PD+{strong_single}": make_ff(models[strong_single], [1.0]),
                    f"CL4:PD+{ord_dict}:W-Q2": make_ff(models[ord_dict], weights[ord_dict]["W-Q2"]),
                    f"CL5:PD+{ord_dict}:W-Q5": make_ff(models[ord_dict], weights[ord_dict]["W-Q5"]),
                    "CL6:PD+D-M:W-Q2": make_ff(models["D-M"], weights["D-M"]["W-Q2"]),
                    "CL7:PD+D-M:W-Q5": make_ff(models["D-M"], weights["D-M"]["W-Q5"]),
                }
                for cname, (ff, nexec) in controllers.items():
                    if cname.startswith(("CL0", "CL1", "CL3")) and g != GROUPS[0]:
                        continue   # calibration-free controllers: one run per seed/condition
                    t0 = time.time()
                    loop = BatchedClosedLoop(args.xml_dir, refs, KP, KD, payload_kg=payload, obs_noise=noise,
                                             noise_seed=900 + hash(cond) % 1000, q_scale=q_scale, qd_scale=qd_scale)
                    res = loop.run(ff)
                    wall = time.time() - t0
                    for e in range(len(refs)):
                        rows.append({"seed": seed, "cond": cond, "group": g, "controller": cname, "episode": e,
                                     "rmse": res["rmse"][e], "maxerr": res["maxerr"][e], "tau_rms": res["tau_rms"][e],
                                     "saturation": res["saturation"][e], "failed": bool(res["failed"][e]),
                                     "fail_step": int(res["fail_step"][e]), "n_exec": nexec, "wall_s": wall})
                    if seed == 0 and g == GROUPS[0]:
                        traces[f"{cond}|{cname}"] = res["err"][:3].astype(np.float32)
                    print(f"seed {seed} {cond} g{g} {cname}: rmse {res['rmse'].mean():.4f} fail {res['failed'].mean():.2f} wall {wall:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "e06_episodes.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "e06_episodes.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, "e06_traces.npz"), **traces)
    with open(os.path.join(args.out, "e06_config.json"), "w") as f:
        json.dump({"kp": KP.tolist(), "kd": KD.tolist(), "groups": GROUPS, "fail_err_rad": 0.5,
                   "episodes": len(refs), "conds": args.conds}, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
