"""E29: tracking-task supplement on the E13 setting (same plant, 30 references, gains, failure rule,
noise seeds N 900 / O1-M 901 / P1 902, calibration m = 10 in groups 0-2).

Controllers (feedback law of E13; '@true-fb' = feedback from the true state, feedforward still fed
the perturbed observation, as the E13 diagnostic):
  * E13 reruns (reproduction check): D-S:W-Q5, B-N, D-M:W-Q5 with the shared feedback law
  * reference feedforward '@ref': the frozen predictor evaluated on (q_ref, qd_ref, qdd_ref) instead of
    the observed state; weights calibrated on clean calibration inputs (the inputs it receives)
  * task-weighted calibration 'W-QG5': budgeted simplex weights that minimize the second moment of
    the calibration residual after the decoupled joint-loop disturbance response (C_G)
  * O1-M, true-state feedback: every E13 feedforward model plus the two additions above (measured
    tracking error for the task-weighted prediction of e29_taskweighted.py)

python e29_tracking.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs25/sim \
    --frozen ../runs25/sim/frozen.json --cfg e27_candidates.json --out ../results/sim/e29_tracking_a.csv --seeds 0,1,2
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import e27_common as C
from common import calibration as cal
from sim.panda_env import BatchedClosedLoop

E13_NAMES = ["B-N", "B-N|est", "B-A", "B-A+(q,qd)", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-M+null:W-Q5",
             "D-S|est:W-Q5", "D-M|est:W-Q5", "D-S,est-optional:W-Q5", "D-SQ:W-Q5"]
QG = {"D-S:W-QG5": "D-S:W-Q5", "D-M:W-QG5": "D-M:W-Q5", "D-S+null:W-QG5": "D-S+null:W-Q5"}
REF = ["D-S:W-Q5", "B-N"]


def ref_wrapper(ff, refs):
    """Feed (q_ref, qd_ref, qdd_ref) instead of (q_obs, qd_obs, qdd_ref): BatchedClosedLoop.run calls the
    feedforward exactly once per control step, in order."""
    Qr = np.stack([r["q"] for r in refs], 0); QDr = np.stack([r["qd"] for r in refs], 0)
    st = {"k": 0}

    def f(X):
        k = st["k"]; st["k"] += 1
        return ff(np.concatenate([Qr[:, k], QDr[:, k], X[:, 14:]], 1))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--frozen", required=True)
    ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--conds", default="O1M,N,P1")
    ap.add_argument("--groups", default="0,1,2")
    args = ap.parse_args()
    t00 = time.time()
    ctx = C.setup(args.data_dir, args.frozen)
    cfg = C.load_config(args.cfg)
    D = ctx["D"]; xs, ys = ctx["xs"], ctx["ys"]
    q_scale, qd_scale = xs[:7], xs[7:14]
    rn = np.load(os.path.join(args.data_dir, "closed_refs.npz"))
    refs = [{"q": rn["q"][i], "qd": rn["qd"][i], "qdd": rn["qdd"][i]} for i in range(rn["q"].shape[0])]
    Qtr = (D["X"]["train"][:, :7] * xs[:7] + ctx["xm"][:7]).astype(np.float64)
    filt = {0.0: C.joint_loop_filters(C.effective_inertia(args.xml_dir, Qtr, 0.0)),
            1.0: C.joint_loop_filters(C.effective_inertia(args.xml_dir, Qtr, 1.0))}
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mem = C.Members(ctx, args.runs, seed)
        for cond in args.conds.split(","):
            payload = 1.0 if cond == "P1" else 0.0
            noise = ctx["rho"] if cond == "O1M" else None
            clean = "P1" if cond == "P1" else "N"
            for g in [int(x) for x in args.groups.split(",")]:
                cs = C.cal_split(ctx, cond, g)
                cs_clean = cs if clean == cond else C.cal_split(ctx, clean, g)
                ctag, ctag_c = f"cal:{cond}:g{g}", f"cal:{clean}:g{g}"
                runs_ = []   # (name, items, w, W_est, fb_true, ff_input)

                def add(name, fb_true=False, ff_input="obs"):
                    if g not in C.cand_groups(cfg, name):
                        return
                    c_s = cs if ff_input == "obs" else cs_clean
                    items, w = C.candidate_weights(ctx, cfg, mem, name, c_s, ctag if ff_input == "obs" else ctag_c)
                    runs_.append((name, items, w, c_s["W"], fb_true, ff_input))

                def add_qg(name, fb_true=False):
                    base = QG[name]
                    items = C.resolve(cfg, base, seed, ctx["strong"])
                    Xe = None
                    R = []
                    for mid, est in items:
                        if est:
                            Xe = C.apply_estimator(ctx, cs["X"], cs["W"]) if Xe is None else Xe
                            P = mem.pred(mid, Xe, ctag + ":est")
                        else:
                            P = mem.pred(mid, cs["X"], ctag + ":raw")
                        R.append(C.task_filter(((P - cs["Y"]) * ys).reshape(10, -1, 7), filt[payload]).reshape(-1, 7))
                    Rm = np.stack(R, 0)
                    CG = np.einsum("ind,jnd->ij", Rm, Rm) / (Rm.shape[1] * 7)
                    w = cal.budgeted_qp(CG, min(5, len(items)))["w"]
                    runs_.append((name, items, w, cs["W"], fb_true, "obs"))

                for n in ["D-S:W-Q5", "B-N", "D-M:W-Q5"]:
                    add(n)
                for n in QG:
                    add_qg(n)
                for n in REF:
                    add(n, ff_input="ref")
                if cond == "O1M":
                    for n in E13_NAMES:
                        add(n, fb_true=True)
                    for n in QG:
                        add_qg(n, fb_true=True)
                    for n in REF:
                        add(n, fb_true=True, ff_input="ref")
                for name, items, w, W_est, fb_true, ff_input in runs_:
                    ff, n_exec = C.make_ff(ctx, mem, items, w, W_est)
                    if ff_input == "ref":
                        ff = ref_wrapper(ff, refs)
                    ws = C.weight_summary(items, w)
                    t0 = time.time()
                    loop = BatchedClosedLoop(args.xml_dir, refs, C.KP, C.KD, payload_kg=payload, obs_noise=noise,
                                             noise_seed=C.NOISE_SEED[cond], q_scale=q_scale, qd_scale=qd_scale)
                    res = loop.run(ff, fb_true=fb_true)
                    wall = time.time() - t0
                    label = name + ("@ref" if ff_input == "ref" else "") + ("@true-fb" if fb_true else "")
                    rj = np.sqrt((res["err"] ** 2).mean(1))
                    for e in range(len(refs)):
                        r = {"seed": seed, "cond": cond, "group": g, "controller": label, "base": name, "ff_input": ff_input,
                             "fb": "true" if fb_true else "shared", "episode": e, "rmse": res["rmse"][e], "maxerr": res["maxerr"][e],
                             "tau_rms": res["tau_rms"][e], "tau_fb_rms": res["tau_fb_rms"][e], "tau_ff_rms": res["tau_ff_rms"][e],
                             "saturation": res["saturation"][e], "failed": bool(res["failed"][e]), **ws, "wall_s": wall}
                        for l in range(7):
                            r[f"rmse_j{l + 1}"] = rj[e, l]
                        rows.append(r)
                    print(f"seed {seed} {cond} g{g} {label}: rmse {res['rmse'].mean():.4f} fail {res['failed'].mean():.2f} "
                          f"wall {wall:.1f}s total {time.time() - t00:.0f}s", flush=True)
                mem.drop(ctag); mem.drop(ctag_c)
                pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("E29T_DONE", time.time() - t00)


if __name__ == "__main__":
    main()
