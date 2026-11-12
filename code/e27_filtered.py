"""E27 (zero-cost part): spurious interaction torque of every E13 feedforward model.

For a test trajectory without interaction, the interaction-torque estimate tau_ext_hat = f_hat(x~) -
tau_motor equals the open-loop residual f_hat(x~) - ID(x) (plan, Sec. 3.5). For every candidate of
e27_candidates.json, with the weights of the closed-loop calibration groups (m = 10, groups 0-2, as
run_closedloop_aug.py / analysis_lowfreq_matched.py), on the 40 test trajectories under N, O1-M, P1:
  * residual in N m per joint, causal first-order low-pass at fc in FCS, RMSE per joint and the
    root mean square over the 7 joints, plus nMSE (per-joint normalized) -> e27_filtered.csv
  * slow / white split: L-step moving averages (L = 10, 25, 50) and slow(L) = S + W / L fitted on
    L = 10, 25, per joint and aggregate -> e27_slowwhite.csv; the S/W prediction of the filtered
    risk S + W / L_eq(fc) is written next to the measured value in e27_filtered.csv
  * crossing cut-off frequency of attenuating candidates against D-S (and MLP): measured on a fine
    fc grid and predicted from S, W -> e27_crossing.csv

python e27_filtered.py --data_dir ../data_sim --runs ../runs25/sim --frozen ../runs25/sim/frozen.json \
    --cfg e27_candidates.json --out_dir ../results/sim
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import e27_common as C

FC_FINE = np.geomspace(1.0, 50.0, 79)


def first_crossing(fc, diff):
    """First fc (log-linear interpolation) where diff changes sign from > 0 to <= 0, scanning upward.
    Returns (fc_cross, n_sign_changes, sign_low, sign_high)."""
    s = np.sign(diff)
    changes = int(np.sum(s[1:] != s[:-1]))
    fcx = np.nan
    for i in range(len(fc) - 1):
        if diff[i] > 0 >= diff[i + 1]:
            t = diff[i] / (diff[i] - diff[i + 1])
            fcx = float(np.exp(np.log(fc[i]) + t * (np.log(fc[i + 1]) - np.log(fc[i]))))
            break
    return fcx, changes, int(s[0]), int(s[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--runs", required=True)
    ap.add_argument("--frozen", required=True); ap.add_argument("--cfg", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--conds", default="N,O1M,P1")
    ap.add_argument("--e13c", default="../results/sim/e13c_lowfreq_matched.csv")
    args = ap.parse_args()
    t0 = time.time()
    ctx = C.setup(args.data_dir, args.frozen)
    cfg = C.load_config(args.cfg)
    names = list(cfg["candidates"])
    ys = ctx["ys"]
    rows_f, rows_sw, fine = [], [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        mem = C.Members(ctx, args.runs, seed)
        for cond in args.conds.split(","):
            te = C.test_inputs(ctx, cond)
            nT = len(np.unique(te["g"]))
            for g in C.GROUPS:
                cs = C.cal_split(ctx, cond, g)
                ctag = f"cal:{cond}:g{g}"
                for name in names:
                    if g not in C.cand_groups(cfg, name):
                        continue
                    items, w = C.candidate_weights(ctx, cfg, mem, name, cs, ctag)
                    P = C.candidate_predict(ctx, mem, items, w, te["X"], cs["W"], f"test:{cond}", f"test:{cond}:g{g}:est")
                    E = ((P - te["Y"]) * ys).reshape(nT, -1, 7)            # N m, [traj, T, joint]
                    En = E / ys                                             # per-joint normalized
                    ws = C.weight_summary(items, w)
                    base = {"seed": seed, "cond": cond, "group": g, "candidate": name}
                    # slow / white split (normalized), per joint
                    m = {1: (En ** 2).mean((0, 1))}
                    for L in (10, 25, 50):
                        m[L] = (C.moving_average_3d(En, L) ** 2).mean((0, 1))
                    S, Wt = C.slow_white(m[10], m[25])
                    for j in range(8):
                        sel = slice(None) if j == 7 else j
                        agg = (lambda v: float(np.mean(v))) if j == 7 else (lambda v: float(v[sel]))
                        Sj, Wj = agg(S), agg(Wt)
                        rows_sw.append({**base, "joint": "all" if j == 7 else j + 1, "total": agg(m[1]), "slow_L10": agg(m[10]),
                                        "slow_L25": agg(m[25]), "slow_L50": agg(m[50]), "S": Sj, "W": Wj, "S_plus_W": Sj + Wj,
                                        "pred_L50": Sj + Wj / 50.0,
                                        "S_Nm2": float(np.mean(S * ys ** 2)) if j == 7 else float(S[j] * ys[j] ** 2),
                                        "W_Nm2": float(np.mean(Wt * ys ** 2)) if j == 7 else float(Wt[j] * ys[j] ** 2),
                                        **ws, "w": json.dumps(np.round(np.asarray(w), 4).tolist())})
                    # filtered spurious torque
                    for fc in [np.inf] + C.FCS:
                        Ef = E if not np.isfinite(fc) else C.lowpass(E, fc)
                        mse = (Ef ** 2).mean((0, 1))
                        Le = 1.0 if not np.isfinite(fc) else C.leq_exact(fc)
                        Lc = 1.0 if not np.isfinite(fc) else C.leq_cont(fc)
                        pred_n = S + Wt / Le
                        for j in range(8):
                            if j == 7:
                                rows_f.append({**base, "fc": fc, "joint": "all", "rmse_Nm": float(np.sqrt(mse.mean())),
                                               "nmse": float(np.mean(mse / ys ** 2)), "leq_exact": Le, "leq_cont": Lc,
                                               "nmse_pred_sw": float(pred_n.mean()),
                                               "rmse_pred_sw_Nm": float(np.sqrt(np.mean(pred_n * ys ** 2)))})
                            else:
                                rows_f.append({**base, "fc": fc, "joint": j + 1, "rmse_Nm": float(np.sqrt(mse[j])),
                                               "nmse": float(mse[j] / ys[j] ** 2), "leq_exact": Le, "leq_cont": Lc,
                                               "nmse_pred_sw": float(pred_n[j]),
                                               "rmse_pred_sw_Nm": float(np.sqrt(pred_n[j] * ys[j] ** 2))})
                    # fine grid for the crossing analysis
                    for fc in FC_FINE:
                        mse = (C.lowpass(E, fc) ** 2).mean((0, 1))
                        fine.append({**base, "fc": float(fc), "mse_all": float(mse.mean()), "nmse": float(np.mean(mse / ys ** 2)),
                                     "mse_j2": float(mse[1])})
                mem.drop(ctag)
                mem.drop(f"test:{cond}:g{g}")
                print(f"seed {seed} {cond} g{g} done {time.time() - t0:.0f}s", flush=True)
            mem.drop(f"test:{cond}")
            pd.DataFrame(rows_f).to_csv(os.path.join(args.out_dir, "e27_filtered.csv"), index=False)
            pd.DataFrame(rows_sw).to_csv(os.path.join(args.out_dir, "e27_slowwhite.csv"), index=False)
    F = pd.DataFrame(rows_f); SW = pd.DataFrame(rows_sw); FI = pd.DataFrame(fine)
    F.to_csv(os.path.join(args.out_dir, "e27_filtered.csv"), index=False)
    SW.to_csv(os.path.join(args.out_dir, "e27_slowwhite.csv"), index=False)
    FI.to_csv(os.path.join(args.out_dir, "e27_filtered_fine.csv"), index=False)

    # ------------------------------------------------------------------ crossing
    # group-averaged per seed (E13 convention: group-dependent candidates averaged over the groups)
    fin = FI.groupby(["cond", "candidate", "seed", "fc"], as_index=False)[["mse_all", "nmse", "mse_j2"]].mean()
    swa = SW[SW.joint == "all"].groupby(["cond", "candidate", "seed"], as_index=False)[["S", "W", "S_Nm2", "W_Nm2"]].mean()
    swj2 = SW[SW.joint == 2].groupby(["cond", "candidate", "seed"], as_index=False)[["S_Nm2", "W_Nm2"]].mean()
    rows_c = []
    for cond in args.conds.split(","):
        for a, b in cfg["crossing_pairs"]:
            for metric, col, scol, wcol, swtab in [("nmse", "nmse", "S", "W", swa), ("rms_Nm", "mse_all", "S_Nm2", "W_Nm2", swa),
                                                    ("joint2_Nm", "mse_j2", "S_Nm2", "W_Nm2", swj2)]:
                curves = {}
                for who in (a, b):
                    s = fin[(fin.cond == cond) & (fin.candidate == who)]
                    curves[who] = s.pivot(index="seed", columns="fc", values=col)
                if curves[a].empty or curves[b].empty:
                    continue
                seeds = sorted(set(curves[a].index) & set(curves[b].index))
                fcs = curves[a].columns.values
                per = []
                for sd in seeds + ["mean"]:
                    if sd == "mean":
                        da = curves[a].loc[seeds].mean(axis=0).values; db = curves[b].loc[seeds].mean(axis=0).values
                        sa = swtab[(swtab.cond == cond) & (swtab.candidate == a)][[scol, wcol]].mean()
                        sb = swtab[(swtab.cond == cond) & (swtab.candidate == b)][[scol, wcol]].mean()
                    else:
                        da = curves[a].loc[sd].values; db = curves[b].loc[sd].values
                        sa = swtab[(swtab.cond == cond) & (swtab.candidate == a) & (swtab.seed == sd)][[scol, wcol]].mean()
                        sb = swtab[(swtab.cond == cond) & (swtab.candidate == b) & (swtab.seed == sd)][[scol, wcol]].mean()
                    fcx, nch, slo, shi = first_crossing(fcs, da - db)
                    dS = sa[scol] - sb[scol]; dW = sb[wcol] - sa[wcol]
                    Lx = dW / dS if (dS > 0 and dW > 0) else np.nan
                    at = {}
                    for fq in (2, 5, 20, 40):
                        i = int(np.argmin(np.abs(np.log(fcs) - np.log(fq))))
                        at[f"diff_at_{fq}Hz"] = float(np.sqrt(da[i]) - np.sqrt(db[i])) if metric != "nmse" else float(da[i] - db[i])
                    rows_c.append({"cond": cond, "a": a, "b": b, "metric": metric, "seed": sd, "fc_cross_measured": fcx,
                                   "n_sign_changes": nch, "sign_low_fc": slo, "sign_high_fc": shi,
                                   "L_cross_pred": Lx, "fc_cross_pred_cont": C.fc_from_leq_cont(Lx),
                                   "fc_cross_pred_exact": C.fc_from_leq_exact(Lx),
                                   "S_a": sa[scol], "W_a": sa[wcol], "S_b": sb[scol], "W_b": sb[wcol], **at})
                    per.append(fcx)
    CR = pd.DataFrame(rows_c)
    CR.to_csv(os.path.join(args.out_dir, "e27_crossing.csv"), index=False)

    # ------------------------------------------------------------------ checks and printout
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    chk = {}
    if os.path.exists(args.e13c):
        ref = pd.read_csv(args.e13c)
        mine = SW[SW.joint == "all"][["seed", "cond", "group", "candidate", "total", "slow_L10", "slow_L25", "slow_L50"]]
        mg = ref.merge(mine, left_on=["seed", "cond", "group", "method"], right_on=["seed", "cond", "group", "candidate"])
        chk = {"n_matched": int(len(mg)),
               "max_abs_diff_total": float((mg.nmse_total - mg.total).abs().max()),
               "max_abs_diff_slow_L25": float((mg.nmse_slow_L25 - mg.slow_L25).abs().max())}
        print("E13c reproduction:", chk)
    tab = F[(F.joint == "all")].groupby(["cond", "candidate", "fc"])["rmse_Nm"].mean().unstack("fc")
    print(tab.round(3).to_string())
    print(CR[(CR.seed == "mean")].round(3).to_string())
    json.dump(chk, open(os.path.join(args.out_dir, "e27_e13c_check.json"), "w"), indent=1)
    print("E27F_DONE", time.time() - t0)


if __name__ == "__main__":
    main()
