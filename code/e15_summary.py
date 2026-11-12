"""E15 summary: family means, checks against the research-plan table (Sec. 2.2) and the
original manuscript (Table S8 split, S7 forward responses, appendix A4 kappa), flip-rate
rank correlations, stationarity, direction-resolved responses, weights, B-L widths.

$PY e15_summary.py --platform sim [--suffix '']
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e15_common import PLATFORM_DEFAULTS, out_name  # noqa: E402

# research_plan.md Sec. 2.2 (O1-M, back-computed from Table VII of the manuscript)
PLAN = {
    "sarcos": {"D-S": {"R": 0.328, "beta": 0.86, "nu2": 0.30, "R_angle": 0.286, "scale_cost": 0.042},
               "D-M STE": {"R": 0.359, "beta": 0.79, "nu2": 0.31, "R_angle": 0.333, "scale_cost": 0.026},
               "MLP": {"R": 0.497, "beta": 0.93, "nu2": 0.50, "R_angle": 0.367, "scale_cost": 0.130},
               "B-A": {"R": 0.126, "beta": 0.90, "nu2": 0.11, "R_angle": 0.122, "scale_cost": 0.004}},
    "sim": {"D-S": {"R": 0.406, "beta": 0.83, "nu2": 0.39, "R_angle": 0.349, "scale_cost": 0.057},
            "D-M STE": {"R": 0.461, "beta": 0.65, "nu2": 0.35, "R_angle": 0.437, "scale_cost": 0.024},
            "MLP": {"R": 0.446, "beta": 0.95, "nu2": 0.44, "R_angle": 0.318, "scale_cost": 0.128},
            "B-A": {"R": 0.355, "beta": 0.76, "nu2": 0.30, "R_angle": 0.330, "scale_cost": 0.025}},
}
# manuscript tables/s8_split.tex (O1-M; risk, P_n, P_b, cross, slope, output power)
S8 = {
    "sarcos": {"D-S": [0.328, 0.319, 0.016, -0.007, 0.83, 1.03], "D-M window": [0.329, 0.320, 0.017, -0.008, 0.82, 1.03],
               "D-M STE": [0.359, 0.284, 0.153, -0.078, 0.84, 0.94], "D-SA": [0.126, 0.107, 0.029, -0.010, 0.97, 0.92],
               "D-MA window": [0.125, 0.107, 0.029, -0.011, 0.97, 0.92], "D-MA STE": [0.318, 0.201, 0.183, -0.066, 0.91, 0.84],
               "B-A": [0.126, 0.107, 0.029, -0.009, 0.98, 0.91], "B-A+": [0.124, 0.104, 0.028, -0.009, 0.98, 0.91],
               "MLP": [0.497, 0.489, 0.007, 0.001, 0.68, 1.36], "MLP-aug": [0.114, 0.093, 0.018, 0.002, 0.98, 0.91]},
    "sim": {"D-S": [0.406, 0.383, 0.047, -0.025, 0.75, 1.02], "D-M window": [0.407, 0.385, 0.048, -0.026, 0.75, 1.02],
            "D-M STE": [0.461, 0.367, 0.254, -0.160, 0.81, 0.74], "D-SA": [0.350, 0.284, 0.103, -0.037, 0.85, 0.82],
            "D-MA window": [0.354, 0.290, 0.101, -0.036, 0.84, 0.83], "D-MA STE": [0.441, 0.325, 0.249, -0.133, 0.84, 0.71],
            "D-SQ": [0.319, 0.236, 0.125, -0.041, 0.88, 0.80], "D-MQ window": [0.326, 0.251, 0.117, -0.042, 0.86, 0.83],
            "D-MQ STE": [0.428, 0.306, 0.254, -0.131, 0.85, 0.70], "B-A": [0.355, 0.292, 0.103, -0.040, 0.84, 0.84],
            "B-A+": [0.339, 0.262, 0.107, -0.030, 0.88, 0.76], "B-AQ+": [0.308, 0.217, 0.127, -0.037, 0.90, 0.77],
            "MLP": [0.446, 0.440, 0.010, -0.004, 0.69, 1.27], "MLP-aug": [0.324, 0.274, 0.050, -0.000, 0.83, 0.92],
            "MLP-augQ+": [0.292, 0.241, 0.054, -0.002, 0.84, 0.92]},
}
S8_COLS = ["R", "P_n", "P_b", "cross", "slope", "Fnorm2"]
# manuscript supp S7 (forward response, mean ratio); window given as a range
S7 = {"sim": {"D-M STE": 0.566, "D-M window": (0.600, 0.610), "D-S": 0.604, "D-T": 0.598, "B-L": 0.617, "B-A": 0.523, "MLP": 0.643},
      "sarcos": {"D-M STE": 0.540, "D-M window": (0.604, 0.625), "D-S": 0.613, "D-T": 0.607, "B-L": 0.632, "B-A": 0.338, "MLP": 0.769}}
# research_plan.md appendix A4
A4 = {"sim": {"MLP": 1.99, "D-M window": (1.79, 1.80), "D-M STE": 1.66, "B-A": 1.78},
      "sarcos": {"MLP": 1.82, "D-M triangular": 1.58, "D-M STE": 1.54}}
CLEAN = {"dm", "ds", "dt", "bl", "uf", "cap4", "cap8", "cap16", "width"}
AUG = {"dma", "dsa", "dmq", "dsq", "dmas", "dmqs", "ba"}


def clean_json(o):
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def spearman(a, b):
    a = pd.Series(np.asarray(a, float)); b = pd.Series(np.asarray(b, float))
    ok = a.notna() & b.notna()
    if ok.sum() < 3:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    plat = args.platform
    out_dir = args.out or f"../results/{plat}"
    P = PLATFORM_DEFAULTS[plat]
    rM = f"O1:{P['rho_M']}"
    S = {"platform": plat, "suffix": args.suffix}
    m = pd.read_csv(out_name(out_dir, "e15_members.csv", args.suffix))
    t = m[m.role == "test"].copy()
    num = ["R", "beta", "nu2", "slope", "R_angle", "scale_cost", "Fnorm2", "Py", "P_n", "P_b", "cross",
           "r_O1M", "r_cond", "kappa", "excess_per_rho2"]
    num = [c for c in num if c in t.columns]
    S["identity_max_abs_resid"] = {c: float(m[c].abs().max()) for c in ["I1_resid", "I2_resid", "split_resid", "nu_orth_check"] if c in m}
    S["fwd_reproduction_max_abs_diff"] = float(m["fwd_check_maxdiff"].max())
    S["n_members"] = int(m.member.nunique())
    # family means per condition (test role)
    fam = t.groupby(["group", "cond"])[num].mean()
    fam["n"] = t.groupby(["group", "cond"]).size()
    S["family_means_test"] = {f"{g}|{c}": row.to_dict() for (g, c), row in fam.iterrows()}
    # plan check (O1-M)
    chk = {}
    allpass = True
    for g, ref in PLAN.get(plat, {}).items():
        sub = t[(t.group == g) & (t.cond == rM)]
        if sub.empty:
            continue
        e = {}
        for k, v in ref.items():
            val = float(sub[k].mean()); d = val - v
            e[k] = {"plan": v, "measured": val, "diff": d, "pass_0.02": bool(abs(d) <= 0.02)}
            allpass &= abs(d) <= 0.02
        e["n_members"] = int(len(sub))
        chk[g] = e
    S["check_plan_table_O1M"] = {"all_pass": (bool(allpass) if chk else None), "rows": chk,
                                 "tolerance": 0.02, "note": "None = no reference family in this member set"}
    # S8 split check
    s8 = {}
    for g, ref in S8.get(plat, {}).items():
        sub = t[(t.group == g) & (t.cond == rM)]
        if sub.empty:
            continue
        s8[g] = {c: {"manuscript": v, "measured": float(sub[c].mean()), "diff": float(sub[c].mean()) - v}
                 for c, v in zip(S8_COLS, ref)}
        s8[g]["max_abs_diff"] = max(abs(x["diff"]) for x in s8[g].values() if isinstance(x, dict))
        s8[g]["n_members"] = int(len(sub))
    S["check_s8_split_O1M"] = s8
    # S7 r check
    rr = {}
    one = t[t.cond == "N"].drop_duplicates("member")
    for g, ref in S7[plat].items():
        if g == "D-M window":
            sub = one[one.group == g]
            per = sub.groupby("surrogate")["r_O1M"].mean()
            rr[g] = {"manuscript_range": ref, "measured_by_surrogate": per.to_dict(),
                     "measured_range": [float(per.min()), float(per.max())] if len(per) else None}
        else:
            sub = one[one.group == g]
            if len(sub):
                rr[g] = {"manuscript": ref, "measured": float(sub.r_O1M.mean()), "diff": float(sub.r_O1M.mean()) - ref}
    S["check_s7_forward_response"] = rr
    # kappa: mean of member kappas and kappa of the group-mean risk curve
    kap = {}
    lv = P["o1_levels"]
    for g, sub in one.groupby("group"):
        meanR = t[t.group == g].groupby("cond")["R"].mean()
        if "N" in meanR and all(f"O1:{l}" in meanR for l in lv):
            ex = np.array([meanR[f"O1:{l}"] - meanR["N"] for l in lv])
            kc = float(np.polyfit(np.log(lv), np.log(ex), 1)[0]) if np.all(ex > 0) else float("nan")
            kap[g] = {"kappa_member_mean": float(sub.kappa.mean()), "kappa_member_sd": float(sub.kappa.std()),
                      "kappa_of_mean_curve": kc, "excess_per_rho2": {str(l): float(e / l ** 2) for l, e in zip(lv, ex)},
                      "n": int(len(sub))}
    if plat == "sarcos":
        sub = one[(one.group == "D-M window") & (one.surrogate == "triangular")]
        kap["D-M triangular"] = {"kappa_member_mean": float(sub.kappa.mean())}
    S["kappa"] = kap
    S["check_A4_kappa"] = {g: {"plan": v, "measured_member_mean": kap.get(g, {}).get("kappa_member_mean"),
                               "measured_of_mean_curve": kap.get(g, {}).get("kappa_of_mean_curve")} for g, v in A4[plat].items()}
    # plane at O1-M, compact table per group (for the status note)
    S["plane_O1M"] = {g: {k: float(v) for k, v in row.items()} for g, row in
                      t[t.cond == rM].groupby("group")[["R", "beta", "nu2", "slope", "R_angle", "scale_cost", "Fnorm2", "r_O1M", "kappa"]].mean().iterrows()}
    # ---------------------------------------------------------------- flip
    fp = out_name(out_dir, "e15_flip.csv", args.suffix)
    if os.path.exists(fp):
        f = pd.read_csv(fp)
        f1 = f[f.layer.astype(str) == "1"].copy()
        fa = f[f.layer.astype(str) == "all"].set_index(["member", "cond"])["flip_measured"]
        f1["flip_all_layers"] = [fa.get((a, b), np.nan) for a, b in zip(f1.member, f1.cond)]
        snn = f1[f1.model_kind != "mlp"]
        sets = {"all_snn": snn, "clean_snn": snn[snn.prefix.isin(CLEAN)], "dm_only": snn[snn.prefix == "dm"],
                "clean_snn_excl_width": snn[snn.prefix.isin(CLEAN - {"width"})], "all_incl_mlp": f1}
        rc = {}
        for c in sorted(f1.cond.unique()):
            rc[c] = {}
            for nm, df in sets.items():
                d = df[df.cond == c]
                rc[c][nm] = {"n": int(len(d)),
                             "sg2_vs_layer1_all_steps": spearman(d.flip_pred_sg2, d.flip_measured),
                             "biv_vs_layer1_all_steps": spearman(d.flip_pred_biv, d.flip_measured),
                             "emp1_vs_layer1_all_steps": spearman(d.flip_pred_emp1, d.flip_measured),
                             "biv_vs_layer1_step1": spearman(d.flip_pred_biv, d.flip_measured_step1),
                             "emp1_vs_layer1_step1": spearman(d.flip_pred_emp1, d.flip_measured_step1),
                             "sg2_vs_all_layers": spearman(d.flip_pred_sg2, d.flip_all_layers),
                             "biv_vs_all_layers": spearman(d.flip_pred_biv, d.flip_all_layers),
                             "median_snr_vs_layer1_all_steps": spearman(d.snr_q50, -d.flip_measured),
                             "mean_abs_err_emp1_vs_step1": float((d.flip_pred_emp1 - d.flip_measured_step1).abs().mean()),
                             "mean_abs_err_biv_vs_step1": float((d.flip_pred_biv - d.flip_measured_step1).abs().mean())}
        S["flip_rank_correlations"] = rc
        # mechanism links at O1-M: flip / SNR quantities against plane quantities, across members
        tm = t[t.cond == rM].set_index("member")
        d1 = f1[f1.cond == rM].set_index("member")
        j = d1.join(tm[["r_O1M", "beta", "nu2", "R", "slope"]], how="inner")
        links = {}
        for nm, df in {"clean_snn": j[j.prefix.isin(CLEAN)], "all_snn": j[j.model_kind != "mlp"], "all_incl_mlp": j,
                       "dm_only": j[j.prefix == "dm"]}.items():
            links[nm] = {"n": int(len(df))}
            for a in ["flip_measured", "flip_all_layers", "flip_pred_sg2", "flip_pred_biv", "snr_q50", "sigma_in_q50"]:
                for b2 in ["r_O1M", "beta", "nu2", "R"]:
                    links[nm][f"{a}~{b2}"] = spearman(df[a], df[b2])
        S["mechanism_links_O1M"] = links
        S["flip_criterion_note"] = "plan criterion: rank correlation predicted vs measured >= 0.8 across members"
        grp = f.groupby(["group", "cond", "layer"])[["flip_measured", "flip_measured_step1", "flip_pred_sg2", "flip_pred_biv",
                                                      "flip_pred_emp1", "snr_q50", "sigma_in_q50", "spike_rate_clean"]].mean()
        S["flip_family_means"] = {f"{g}|{c}|{l}": row.dropna().to_dict() for (g, c, l), row in grp.iterrows()
                                  if g in ("D-M window", "D-M STE", "D-S", "D-T", "B-L", "B-A", "MLP", "D-MA window", "D-MA STE", "D-SA")}
    # ---------------------------------------------------------------- stationarity
    sp = out_name(out_dir, "e15_stationarity.csv", args.suffix)
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        s = s[s.layer.astype(str).isin(["1", "2", "3"])]
        cols = ["grad_fro", "grad_rel_to_W", "ratio_G", "ratio_K", "floor_G", "floor_K", "cos_GK", "g_mean_rel_peak",
                "frac_in_window", "identity_resid"]
        g = s.groupby(["group", "layer"])[cols].mean()
        S["stationarity_family_means"] = {f"{a}|{b}": row.to_dict() for (a, b), row in g.iterrows()}
        S["stationarity_identity_resid_max"] = float(s.identity_resid.max())
        cmp = {}
        for lay in ["1", "2", "3"]:
            d = s[s.layer.astype(str) == lay]
            w = d[d.group == "D-M window"]; e = d[d.group == "D-M STE"]
            cmp[lay] = {"window_ratio_G": float(w.ratio_G.mean()), "window_ratio_K": float(w.ratio_K.mean()),
                        "window_K_over_G": float((w.ratio_K / w.ratio_G).mean()), "window_cos_GK": float(w.cos_GK.mean()),
                        "ste_ratio_G": float(e.ratio_G.mean()), "ste_ratio_K": float(e.ratio_K.mean()),
                        "floor_mean": float(d.floor_G.mean())}
        S["stationarity_dm_contrast"] = cmp
    # ---------------------------------------------------------------- direction
    dp = out_name(out_dir, "e15_direction.csv", args.suffix)
    if os.path.exists(dp):
        dd = pd.read_csv(dp)
        sub = dd[dd.subspace.isin(["top5", "bottom5", "iso"])]
        pv = sub.pivot_table(index=["member", "group", "variant"], columns="subspace", values="r").reset_index()
        pv["bottom_over_top"] = pv["bottom5"] / pv["top5"]
        g = pv.groupby(["group", "variant"])[["top5", "bottom5", "iso", "bottom_over_top"]].mean()
        S["direction_family_means"] = {f"{a}|{b}": row.to_dict() for (a, b), row in g.iterrows()}
        con = {}
        for v in ["full21", "qqd14"]:
            w = pv[(pv.group == "D-M window") & (pv.variant == v)]; e = pv[(pv.group == "D-M STE") & (pv.variant == v)]
            con[v] = {"ste_minus_window_top5": float(e.top5.mean() - w.top5.mean()),
                      "ste_minus_window_bottom5": float(e.bottom5.mean() - w.bottom5.mean()),
                      "ste_over_window_top5": float(e.top5.mean() / w.top5.mean()),
                      "ste_over_window_bottom5": float(e.bottom5.mean() / w.bottom5.mean())}
        S["direction_ste_vs_window"] = con
    # ---------------------------------------------------------------- weights
    wp = out_name(out_dir, "e15_weights.csv", args.suffix)
    if os.path.exists(wp):
        w = pd.read_csv(wp)
        wm = w[w.group != "traj-init-check"]
        g = wm.groupby(["group", "layer"])[["fro", "dfro", "rel_change", "stable_rank", "stable_rank0", "stable_rank_delta"]].mean()
        S["weights_family_means"] = {f"{a}|{b}": row.to_dict() for (a, b), row in g.iterrows()
                                     if a in ("D-M window", "D-M STE", "D-S", "D-T", "B-L", "B-A", "MLP")}
        lif = wm[wm.layer.isin(["fc1", "fc2", "fc3"]) & wm.group.isin(["D-M window", "D-M STE", "D-S", "MLP"])]
        if "thr_mean" in lif:
            lif = lif.assign(beta_1m_thr=lif.beta_mean * (1 - lif.thr_mean))
            S["lif_params_cross_step_factor"] = {f"{a}|{b}": row.to_dict() for (a, b), row in
                                                 lif.groupby(["group", "layer"])[["beta_mean", "thr_mean", "beta_1m_thr"]].mean().iterrows()}
            S["lif_params_note"] = ("cross-step factor of dmem_pre(t+1)/dmem_pre(t) = beta*(1 - thr*g); for STE (g = 1) "
                                    "beta*(1 - thr); beta_1m_thr uses the layer means (approximation)")
        tc = w[w.group == "traj-init-check"]
        if len(tc):
            S["init_reproduction_check"] = {
                "rel_dist_epoch0_to_seed_init_mean": float(tc.check_rel_dist_epoch0_vs_init_seed.mean()),
                "rel_dist_epoch0_to_seed_init_max": float(tc.check_rel_dist_epoch0_vs_init_seed.max()),
                "rel_dist_epoch0_to_wrong_seed_init_min": float(tc.check_rel_dist_epoch0_vs_init_seed_plus1.min()),
                "n": int(len(tc))}
    bp = out_name(out_dir, "e15_bl_widths.csv", args.suffix)
    if os.path.exists(bp) and os.path.getsize(bp) > 2:
        try:
            b = pd.read_csv(bp)
            agg = b.groupby("layer")[["alpha_learned", "hwhm_learned"]].agg(["mean", "std", "min", "max"])
            agg.columns = [f"{a}_{c}" for a, c in agg.columns]
            S["bl_widths"] = {f"layer{k}": v for k, v in agg.to_dict("index").items()}
            S["bl_widths_init"] = {"alpha": float(b.alpha_init.iloc[0]), "hwhm": float(b.hwhm_init.iloc[0])}
        except pd.errors.EmptyDataError:
            pass
    with open(out_name(out_dir, "e15_summary.json", args.suffix), "w") as fh:
        json.dump(clean_json(S), fh, indent=1)
    # console digest
    print("plan check O1-M:", json.dumps(clean_json(S["check_plan_table_O1M"]), indent=0)[:3000])
    print("identity resid:", S["identity_max_abs_resid"])


if __name__ == "__main__":
    main()
