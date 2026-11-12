"""Collect the key numbers of E27-E29 into e27_29_summary.json (with the H11 / H12 checks).
python e27_summarize_all.py ../results/sim
"""
import json
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1] if len(sys.argv) > 1 else "../results/sim"
P = lambda f: os.path.join(res, f)
out = {}
REF = "D-S:W-Q5"
ATT = ["D-S+null:W-Q5", "D-M:W-Q5", "D-S|est:W-Q5", "B-N|est", "D-S:e21-W1", "D-S:e21-W3", "D-M:e21-W1", "D-M:e21-W3"]
PLAN = {("D-S+null:W-Q5", "D-S:W-Q5"): 21, ("D-M:W-Q5", "D-S:W-Q5"): 20, ("D-S|est:W-Q5", "D-S:W-Q5"): 13, ("D-S|est:W-Q5", "B-N"): 12}

# ------------------------------------------------------------------ E27 zero-cost
F = pd.read_csv(P("e27_filtered.csv"))
F["fc"] = F.fc.astype(float)
Fa = F[F.joint.astype(str) == "all"]
cell = Fa.groupby(["cond", "candidate", "seed", "fc"], as_index=False)[["rmse_Nm", "nmse", "nmse_pred_sw"]].mean()
tab = cell.groupby(["cond", "candidate", "fc"])[["rmse_Nm", "nmse"]].mean().reset_index()
out["e27_filtered_rms_Nm_O1M"] = {c: {str(int(f)) if np.isfinite(f) else "raw": round(float(v), 4)
                                      for f, v in zip(s.fc, s.rmse_Nm)} for c, s in tab[tab.cond == "O1M"].groupby("candidate")}
h11 = {}
for metric in ("rmse_Nm", "nmse"):
    for fc in (5.0, 2.0, 20.0, 40.0):
        ref = cell[(cell.cond == "O1M") & (cell.candidate == REF) & (cell.fc == fc)].set_index("seed")[metric]
        d = {}
        for a in ATT:
            v = cell[(cell.cond == "O1M") & (cell.candidate == a) & (cell.fc == fc)].set_index("seed")[metric]
            if len(v) == 0:
                continue
            d[a] = {"seeds_above_DS": int((v > ref.loc[v.index]).sum()), "n": int(len(v)),
                    "mean": float(v.mean()), "DS": float(ref.mean())}
        h11[f"{metric}@{int(fc)}Hz"] = d
out["e27_H11_openloop_O1M"] = h11
CR = pd.read_csv(P("e27_crossing.csv"))
cr = CR[(CR.cond == "O1M") & (CR.seed == "mean")]
seeds = CR[(CR.cond == "O1M") & (CR.seed != "mean")]
cross = []
for _, r in cr.iterrows():
    s = seeds[(seeds.a == r.a) & (seeds.b == r.b) & (seeds.metric == r.metric)].fc_cross_measured
    cross.append({"a": r.a, "b": r.b, "metric": r.metric, "fc_measured": r.fc_cross_measured, "fc_seed_min": float(s.min()),
                  "fc_seed_max": float(s.max()), "n_seed_crossings": int(s.notna().sum()), "fc_pred_SW_cont": r.fc_cross_pred_cont,
                  "fc_pred_SW_exact": r.fc_cross_pred_exact, "L_pred": r.L_cross_pred, "plan_A6_Hz": PLAN.get((r.a, r.b)),
                  "worse_at_low_fc": bool(r.sign_low_fc > 0), "better_at_high_fc": bool(r.sign_high_fc < 0)})
out["e27_crossing_O1M"] = cross
nm = [c for c in cross if c["metric"] == "nmse"]
out["e27_H12"] = {"all_attenuating_cross_from_worse_to_better_nmse": all(c["worse_at_low_fc"] and c["better_at_high_fc"] for c in nm),
                  "crossings_nmse_Hz": {f"{c['a']} vs {c['b']}": round(c["fc_measured"], 1) for c in nm},
                  "crossings_rms_Nm_Hz": {f"{c['a']} vs {c['b']}": round(c["fc_measured"], 1) for c in cross if c["metric"] == "rms_Nm"},
                  "within_12_21Hz_nmse_E13": [c["a"] for c in nm if c["plan_A6_Hz"] and 11 <= c["fc_measured"] <= 22]}
SW = pd.read_csv(P("e27_slowwhite.csv"))
swa = SW[SW.joint.astype(str) == "all"].groupby(["cond", "candidate"])[["total", "S", "W", "S_plus_W", "slow_L50", "pred_L50"]].mean()
out["e27_slowwhite_O1M"] = {c: {k: round(float(v), 4) for k, v in r.items()} for c, r in swa.loc["O1M"].iterrows()}
out["e27_e13c_check"] = json.load(open(P("e27_e13c_check.json")))

# ------------------------------------------------------------------ E27 external torque
if os.path.exists(P("e27_ext_summary.csv")):
    X = pd.read_csv(P("e27_ext_summary.csv"))
    xm = X[(X.seed == "mean") & (X.joint.astype(str) == "all")]
    key = ["rmse_err", "auc", "det_rate_cfar", "lat_med_ms_cfar", "far_per_min_cfar", "det_rate_cfar99", "lat_med_ms_cfar99",
           "far_per_min_cfar99", "seeds_rmse_above_DS", "seeds_auc_below_DS"]
    ext = {}
    for (cond, setting, filt), s in xm.groupby(["cond", "setting", "filter"]):
        ext[f"{cond}/{setting}/{filt}"] = {r.candidate: {k: (None if pd.isna(r[k]) else round(float(r[k]), 4)) for k in key}
                                           for _, r in s.iterrows()}
    out["e27_ext_pooled_joints"] = ext
    jj = X[(X.seed == "mean") & (X.joint.astype(str) != "all") & (X["filter"] == "5Hz") & (X.setting == "a") & (X.cond == "O1M")]
    out["e27_ext_O1M_a_5Hz_per_joint"] = {c: {str(r.joint): {"rmse": round(float(r.rmse_err), 3), "auc": round(float(r.auc), 3)}
                                              for _, r in s.iterrows()} for c, s in jj.groupby("candidate")
                                          if c in (REF, "B-N", "D-S+null:W-Q5", "D-S|est:W-Q5", "D-M:W-Q5", "ID-exact")}
    for f in ("e27_ext_meta_a.json",):
        if os.path.exists(P(f)):
            out["e27_ext_accel_filter"] = json.load(open(P(f)))
    if os.path.exists(os.path.join(res, "..", "..", "data_sim", "ext_test_info.json")):
        out["e27_ext_sim"] = json.load(open(os.path.join(res, "..", "..", "data_sim", "ext_test_info.json")))

# ------------------------------------------------------------------ E28
if os.path.exists(P("e28_summary.csv")):
    S = pd.read_csv(P("e28_summary.csv")); PR = pd.read_csv(P("e28_paired.csv"))
    cols = ["free_drift_final_max", "free_drift_final_max_lo", "free_drift_final_max_hi", "free_fail", "free_limit_hit",
            "free_drift_gt_0p5", "free_tau_hat5_rms", "intent_tau_h_rms", "intent_intent_err_rms", "intent_fail", "intent_limit_hit"]
    out["e28_summary"] = {f"c{int(cb)}/{cond}": {r.candidate: {k: round(float(r[k]), 4) for k in cols} for _, r in s.iterrows()}
                          for (cb, cond), s in S.groupby(["c_bd", "cond"])}
    pr = PR[(PR.metric == "drift_final_max")]
    out["e28_H11_drift_vs_DS"] = {f"c{int(cb)}/{cond}": {r.a: {"diff": round(float(r["diff"]), 4), "ci": [round(float(r.ci_lo), 4), round(float(r.ci_hi), 4)],
                                                               "seeds_larger": int(r.seeds_a_larger), "n": int(r.n_seeds)} for _, r in s.iterrows()}
                                  for (cb, cond), s in pr.groupby(["c_bd", "cond"])}
    import glob
    pj = sorted(glob.glob(os.path.join(res, "e28_parts", "*_params.json")))
    out["e28_params"] = {os.path.basename(f): json.load(open(f)) for f in pj[:1] + [f for f in pj if "c50" in f][:1]}

# ------------------------------------------------------------------ E29
if os.path.exists(P("e29_tracking_summary.csv")):
    T = pd.read_csv(P("e29_tracking_summary.csv"))
    out["e29_tracking_mrad"] = {cond: {r.controller: [round(1000 * r.rmse, 2), round(1000 * r.ci_lo, 2), round(1000 * r.ci_hi, 2)]
                                       for _, r in s.iterrows()} for cond, s in T.groupby("cond")}
    TP = pd.read_csv(P("e29_tracking_paired.csv"))
    out["e29_paired"] = [{k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for _, r in TP.iterrows()
                         if ("@ref" in r.a) or ("QG" in r.a)]
if os.path.exists(P("e29_taskweighted.csv")):
    TW = pd.read_csv(P("e29_taskweighted.csv"))
    m = TW[TW.seed.astype(str) == "mean"]
    keep = ["meas_rmse_mean", "pred_agg", "pred_joint", "pred_cg", "S_mean", "W_mean", "S_j2", "W_j2", "share_joint_j2", "share_slow_joint_j2"]
    out["e29_taskweighted"] = {cond: {r.candidate: {k: round(float(r[k]) * (1000 if k.startswith(("meas", "pred")) else 1), 4) for k in keep}
                                      for _, r in s.iterrows()} for cond, s in m.groupby("cond")}
    if os.path.exists(P("e29_taskweighted_meta.json")):
        out["e29_taskweighted_meta"] = json.load(open(P("e29_taskweighted_meta.json")))
json.dump(out, open(P("e27_29_summary.json"), "w"), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o))
print(json.dumps({k: (v if k.startswith(("e27_H", "e28_H")) else "...") for k, v in out.items()}, indent=1)[:6000])
print("SUMMARY_DONE")
