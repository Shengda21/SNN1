"""E29 analysis: tracking summary (reference feedforward, task-weighted calibration, E13 reproduction)
and the task-weighted prediction of the tracking error from the slow and white parts of the residual.

Inputs: e29_tracking_[abc].csv (e29_tracking.py), e27_slowwhite.csv (e27_filtered.py), the original
E13 episodes (../results/sim/e13_episodes_*.csv, for the N column of the models not rerun).
Prediction (plan, A5): e^2 = (1/7) sum_l (s_l / Kp_l)^2 (S_l + W_l / L_l), s_l = training torque std,
L_l = 2 (Kd_l + D) / (Kp_l Ts) = 18 (joints 1-4), 21 (joints 5-7);
  * 'agg': S, W = joint averages (the plan's version, uniform across joints);
  * 'joint': per-joint S_l, W_l;
  * 'cg': the open-loop test residual filtered through the decoupled discrete joint loop (C_G).
Measured: O1-M with true-state feedback (all 13 E13 models, 3 groups) and N (shared feedback = true state).
Outputs: e29_tracking.csv (episodes), e29_tracking_summary.csv, e29_tracking_paired.csv, e29_taskweighted.csv.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

import e27_common as C

E13_NAMES = ["B-N", "B-N|est", "B-A", "B-A+(q,qd)", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-M+null:W-Q5",
             "D-S|est:W-Q5", "D-M|est:W-Q5", "D-S,est-optional:W-Q5", "D-SQ:W-Q5"]


def boot(mat, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    ns, ne = mat.shape
    b = np.empty(n_boot)
    for i in range(n_boot):
        b[i] = mat[rng.integers(0, ns, ns)][:, rng.integers(0, ne, ne)].mean()
    return float(mat.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="../results/sim"); ap.add_argument("--data_dir", default="../data_sim")
    ap.add_argument("--xml_dir", default="../mujoco_menagerie/franka_emika_panda"); ap.add_argument("--runs", default="../runs25/sim")
    ap.add_argument("--frozen", default="../runs25/sim/frozen.json"); ap.add_argument("--cfg", default="e27_candidates.json")
    ap.add_argument("--e13", default="../results/sim")
    args = ap.parse_args()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
    E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.res, "e29_tracking_[abc].csv")))], ignore_index=True)
    E.to_csv(os.path.join(args.res, "e29_tracking.csv"), index=False)
    O = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.e13, "e13_episodes_*.csv")))], ignore_index=True)
    mg = E.merge(O, on=["seed", "cond", "group", "controller", "episode"], suffixes=("", "_e13"))
    repro = {"n_rows_matched": int(len(mg)), "max_abs_diff_rmse": float((mg.rmse - mg.rmse_e13).abs().max())}

    # ------------------------------------------------------------------ tracking summary
    cell = E.groupby(["cond", "controller", "seed", "episode"]).agg(rmse=("rmse", "mean"), failed=("failed", "mean"),
                                                                     saturation=("saturation", "mean"), tau_rms=("tau_rms", "mean"),
                                                                     tau_ff_rms=("tau_ff_rms", "mean"), tau_fb_rms=("tau_fb_rms", "mean"),
                                                                     null_weight=("null_weight", "mean")).reset_index()

    def mat(cond, ctl):
        s = cell[(cell.cond == cond) & (cell.controller == ctl)]
        return None if len(s) == 0 else s.pivot(index="seed", columns="episode", values="rmse").values
    rows, prs = [], []
    for cond in ["N", "O1M", "P1"]:
        for ctl in sorted(cell[cell.cond == cond].controller.unique()):
            m = mat(cond, ctl)
            mean, lo, hi = boot(m)
            s = cell[(cell.cond == cond) & (cell.controller == ctl)]
            rows.append({"cond": cond, "controller": ctl, "rmse": mean, "ci_lo": lo, "ci_hi": hi, "fail": s.failed.mean(),
                         "saturation": s.saturation.mean(), "tau_rms": s.tau_rms.mean(), "tau_ff_rms": s.tau_ff_rms.mean(),
                         "tau_fb_rms": s.tau_fb_rms.mean(), "null_weight": s.null_weight.mean(), "n_seeds": m.shape[0]})
            base = "D-S:W-Q5@true-fb" if "@true-fb" in ctl else "D-S:W-Q5"
            if ctl != base:
                mb = mat(cond, base)
                if mb is not None:
                    dm = m - mb
                    mean, lo, hi = boot(dm, seed=1)
                    prs.append({"cond": cond, "a": ctl, "b": base, "diff": mean, "ci_lo": lo, "ci_hi": hi, "rel": mean / mb.mean(),
                                "seeds_a_better": int((dm.mean(1) < 0).sum()), "n_seeds": dm.shape[0]})
            if ctl.startswith("D-S:W-Q5@ref") or ctl.startswith("B-N@ref"):
                b2 = ctl.replace("@ref", "")
                mb = mat(cond, b2) if b2 != base else None
                if mb is not None:
                    dm = m - mb
                    mean, lo, hi = boot(dm, seed=2)
                    prs.append({"cond": cond, "a": ctl, "b": b2, "diff": mean, "ci_lo": lo, "ci_hi": hi, "rel": mean / mb.mean(),
                                "seeds_a_better": int((dm.mean(1) < 0).sum()), "n_seeds": dm.shape[0]})
    S = pd.DataFrame(rows); P = pd.DataFrame(prs)
    S.to_csv(os.path.join(args.res, "e29_tracking_summary.csv"), index=False)
    P.to_csv(os.path.join(args.res, "e29_tracking_paired.csv"), index=False)
    print((S.set_index(["cond", "controller"])[["rmse", "ci_lo", "ci_hi"]] * 1000).round(2).to_string())
    print(P.round(4).to_string())

    # ------------------------------------------------------------------ task-weighted prediction
    ctx = C.setup(args.data_dir, args.frozen)
    cfg = C.load_config(args.cfg)
    ys = ctx["ys"]
    c_l = (ys / C.KP) ** 2 / 7.0                         # rad^2 per unit normalized MSE, per joint
    L_plan = np.round(C.L_EQ_JOINT)                      # 18, 21
    SW = pd.read_csv(os.path.join(args.res, "e27_slowwhite.csv"))
    SW = SW[SW.joint != "all"].copy(); SW["joint"] = SW.joint.astype(int)
    Qtr = (ctx["D"]["X"]["train"][:, :7] * ctx["xs"][:7] + ctx["xm"][:7]).astype(np.float64)
    filt = C.joint_loop_filters(C.effective_inertia(args.xml_dir, Qtr, 0.0))
    pred_rows = []
    for seed in range(5):
        mem = C.Members(ctx, args.runs, seed)
        for cond in ["N", "O1M"]:
            te = C.test_inputs(ctx, cond)
            for g in C.GROUPS:
                cs = C.cal_split(ctx, cond, g)
                ctag = f"cal:{cond}:g{g}"
                for name in E13_NAMES:
                    if g not in C.cand_groups(cfg, name):
                        continue
                    items, w = C.candidate_weights(ctx, cfg, mem, name, cs, ctag)
                    Pt = C.candidate_predict(ctx, mem, items, w, te["X"], cs["W"], f"test:{cond}", f"test:{cond}:g{g}:est")
                    R = ((Pt - te["Y"]) * ys).reshape(40, -1, 7)
                    eg = C.task_filter(R, filt)
                    cg_l = (eg ** 2).mean((0, 1)) / 7.0            # contribution of joint l to e^2 (rad^2)
                    sw = SW[(SW.seed == seed) & (SW.cond == cond) & (SW.group == g) & (SW.candidate == name)].sort_values("joint")
                    Sl, Wl = sw.S.values, sw.W.values
                    agg = c_l * (Sl.mean() + Wl.mean() / L_plan)
                    jnt = c_l * (Sl + Wl / L_plan)
                    r = {"seed": seed, "cond": cond, "group": g, "candidate": name,
                         "pred_agg": float(np.sqrt(agg.sum())), "pred_joint": float(np.sqrt(jnt.sum())), "pred_cg": float(np.sqrt(cg_l.sum())),
                         "S_mean": float(Sl.mean()), "W_mean": float(Wl.mean())}
                    for l in range(7):
                        r[f"S_j{l + 1}"] = float(Sl[l]); r[f"W_j{l + 1}"] = float(Wl[l])
                        r[f"pred_joint_j{l + 1}"] = float(np.sqrt(7 * jnt[l])); r[f"pred_cg_j{l + 1}"] = float(np.sqrt(7 * cg_l[l]))
                        r[f"share_joint_j{l + 1}"] = float(jnt[l] / jnt.sum()); r[f"share_slow_joint_j{l + 1}"] = float(c_l[l] * Sl[l] / (c_l * Sl).sum())
                    pred_rows.append(r)
                mem.drop(ctag); mem.drop(f"test:{cond}:g{g}")
            mem.drop(f"test:{cond}")
        print("pred seed", seed, flush=True)
    PR = pd.DataFrame(pred_rows)
    # measured: O1-M true-state feedback (this rerun), N (this rerun where available, else E13)
    meas = []
    for cond, suffix in [("O1M", "@true-fb"), ("N", "")]:
        for name in E13_NAMES:
            ctl = name + suffix
            src = E[(E.cond == cond) & (E.controller == ctl)]
            tag = "e29"
            if len(src) == 0:
                src = O[(O.cond == cond) & (O.controller == ctl)]; tag = "e13"
            if len(src) == 0:
                continue
            for (seed, g), s in src.groupby(["seed", "group"]):
                r = {"seed": seed, "cond": cond, "group": g, "candidate": name, "meas_source": tag,
                     "meas_rmse_mean": float(s.rmse.mean()), "meas_rmse_rms": float(np.sqrt((s.rmse ** 2).mean()))}
                for l in range(7):
                    col = f"rmse_j{l + 1}"
                    r[f"meas_j{l + 1}"] = float(np.sqrt((s[col] ** 2).mean())) if col in s and s[col].notna().all() else np.nan
                meas.append(r)
    M = pd.DataFrame(meas)
    TW = PR.merge(M, on=["seed", "cond", "group", "candidate"], how="left")
    # per seed: average over groups; then mean over seeds
    num = [c for c in TW.columns if c not in ("seed", "cond", "group", "candidate", "meas_source")]
    per_seed = TW.groupby(["cond", "candidate", "seed"], as_index=False)[num].mean()
    mean = per_seed.groupby(["cond", "candidate"], as_index=False)[num].mean(); mean["seed"] = "mean"
    out = pd.concat([per_seed, mean], ignore_index=True)
    for k in ("agg", "joint", "cg"):
        out[f"err_{k}_rel"] = out[f"pred_{k}"] / out["meas_rmse_mean"] - 1.0
    out.to_csv(os.path.join(args.res, "e29_taskweighted.csv"), index=False)
    view = mean.set_index(["cond", "candidate"])
    cols = ["meas_rmse_mean", "meas_rmse_rms", "pred_agg", "pred_joint", "pred_cg"]
    print((view[cols] * 1000).round(2).to_string())
    print(view[[f"share_joint_j{l}" for l in range(1, 8)] + [f"share_slow_joint_j{l}" for l in range(1, 8)]].round(3).to_string())
    from scipy.stats import spearmanr
    rho = {}
    for cond in ["O1M", "N"]:
        v = mean[mean.cond == cond]
        rho[cond] = {k: float(spearmanr(v[f"pred_{k}"], v.meas_rmse_mean)[0]) for k in ("agg", "joint", "cg")}
    json.dump({"e13_reproduction": repro, "spearman_pred_vs_meas": rho, "L_plan": L_plan.tolist(),
               "c_l_mrad2": (c_l * 1e6).tolist(), "joint2_weight": float(c_l[1] / c_l.sum())},
              open(os.path.join(args.res, "e29_taskweighted_meta.json"), "w"), indent=1)
    print(json.dumps({"repro": repro, "spearman": rho}))
    print("E29TW_DONE")


if __name__ == "__main__":
    main()
