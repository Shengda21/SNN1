"""Collect the key numbers of E21-E26 into results/<platform>/e2x_summary.json.

python e2x_summary.py --res_root ../results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def r(x, n=4):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return None
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def read(res, name):
    p = os.path.join(res, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def platform_summary(res, platform):
    S = {}
    noise = ["O1-L", "O1-M", "O1-H", "O2B"] if platform == "sim" else ["O1-L", "O1-M", "O1-H", "O1qqd"]
    # ---------------- verification
    v = read(res, "e2x_verify_orig.csv")
    if v is not None:
        exact = v[v.space.isin(["W0", "W5s", "W6"])]
        S["verify_orig_max_abs_diff_exact_spaces"] = r(exact["max_abs_diff"].max(), 18)
        S["verify_orig_W1_vs_manuscript_null_k5_max_abs_diff"] = r(v[v.space == "W1"]["max_abs_diff"].max(), 6)
    vp = read(res, "e2x_verify_paired.csv")
    if vp is not None:
        dm = vp[(vp.method == "D-M") & (vp.space == "W0")]
        S["paired_noise_DM_W0_vs_orig"] = {c: {"paired": r(a), "orig": r(b), "diff": r(a - b)}
                                           for c, a, b in zip(dm.cond, dm.nmse_mean_paired, dm.nmse_mean_orig)}
    # ---------------- E21
    lad = read(res, "e21_ladder.csv")
    if lad is not None:
        key = {}
        for c in ["N"] + noise:
            sub = lad[lad.cond == c]
            key[c] = {f"{m}|{sp}": r(x) for m, sp, x in zip(sub.method, sub.space, sub.nmse_mean)
                      if m in ("D-M", "D-S", "D-T", "B-F", "B-L", "B-A", "B-N", "B-A+", "B-AQ+", "B-NQ+")}
        S["e21_test_nmse"] = key
        sc = lad[(lad.space == "W0") & lad.method.isin(["D-M", "D-S", "D-T", "B-F", "B-L", "B-N"])]
        S["e21_scale_cost_W0"] = {c: {m: r(x) for m, x in zip(sub.method, sub.scale_cost)} for c, sub in sc.groupby("cond")}
        S["e21_slope_W0"] = {c: {m: r(x, 3) for m, x in zip(sub.method, sub.slope)} for c, sub in sc.groupby("cond")}
        b = lad[lad.space.isin(["W0", "W1"]) & lad.method.isin(["D-M", "D-S", "B-F", "B-N", "B-A"])]
        S["e21_geometry_W0_W1"] = {f"{c}|{m}|{sp}": {"beta": r(be, 3), "nu2": r(nu, 3), "slope": r(s, 3), "angle": r(an, 3),
                                                     "scale": r(sk, 4), "null_w": r(nw, 3)}
                                   for c, m, sp, be, nu, s, an, sk, nw in zip(b.cond, b.method, b.space, b.beta, b.nu2, b.slope,
                                                                             b.angle_risk, b.scale_cost, b.null_weight)
                                   if c in ("N", "O1-M", "O1-H")}
    seeds = read(res, "e21_ladder_seeds.csv")
    if seeds is not None:
        cl = seeds[(seeds.space == "W0") & seeds.method.isin(["D-M", "D-S", "D-T", "B-F", "B-L", "B-N"])]
        H6 = {}
        for c, sub in cl.groupby("cond"):
            H6[c] = {"all_seeds_methods_scale_cost_pos": bool((sub.scale_cost > 0).all()),
                     "min_scale_cost": r(sub.scale_cost.min(), 5), "max_scale_cost": r(sub.scale_cost.max(), 5)}
        S["H6_scale_cost_by_cond"] = H6
    att = read(res, "e21_attribution.csv")
    if att is not None:
        S["e21_attribution"] = {row["cond"]: {k: r(row[k]) for k in row.index if k != "cond"} for _, row in att.iterrows()}
    pr = read(res, "e21_paired.csv")
    if pr is not None:
        sub = pr[(pr.a.str.startswith("D-M|")) & (pr.b.str.startswith("D-S|"))]
        S["e21_paired_DM_minus_DS"] = {f"{c}|{a.split('|')[1]}": {"diff": r(d), "ci": [r(lo), r(hi)], "seeds_DM_better": int(k)}
                                       for c, a, d, lo, hi, k in zip(sub.cond, sub.a, sub["diff"], sub.ci_lo, sub.ci_hi, sub.seeds_a_better)}
        cross = pr[~pr.a.str.split("|").str[1].eq(pr.b.str.split("|").str[1])]
        S["e21_paired_cross"] = {f"{c}|{a} - {b_}": {"diff": r(d), "ci": [r(lo), r(hi)]}
                                 for c, a, b_, d, lo, hi in zip(cross.cond, cross.a, cross.b, cross["diff"], cross.ci_lo, cross.ci_hi)
                                 if c in ("O1-M", "O1-H")}
    ms = read(res, "e21_mslice.csv")
    if ms is not None:
        sub = ms[ms.method.isin(["D-M", "D-S", "B-N", "B-F"]) & ms.space.isin(["W0", "W1", "W3", "W4", "W5"]) & ms.cond.isin(["N", "O1-M", "O1-H"])]
        S["e21_mslice"] = {f"{c}|{m}|{sp}": {str(int(mm)): r(x) for mm, x in zip(g.m, g.nmse)}
                           for (c, m, sp), g in sub.groupby(["cond", "method", "space"])}
    # ---------------- E22
    fit = os.path.join(res, "e22_fit.json")
    if os.path.exists(fit):
        S["e22_fit"] = json.load(open(fit))
    cr = read(res, "e22_crossings.csv")
    if cr is not None:
        S["e22_crossings"] = {f"{t}|{s}": r(x, 3) for t, s, x in zip(cr["type"], cr["seed"], cr["rho_cross"])}
    ev = read(res, "e22_critical.csv")
    if ev is not None:
        m = ev.dropna(subset=["rho_measured"])
        S["e22_events_measured"] = int(len(m))
        S["e22_events_matched_within_step"] = int(m["within_one_step"].sum())
        S["e22_events_predicted_unmatched"] = int(ev["rho_measured"].isna().sum())
        firsts = m.sort_values("rho_measured").groupby(["seed", "dict", "space", "member_name", "direction"]).head(1)
        key_ev = firsts[(firsts.direction == "enter") & ((firsts.member_name == "null") | firsts.member_name.str.contains("ste"))]
        S["e22_key_events"] = [{"seed": int(s), "dict": d, "space": sp, "member": mn, "rho_meas": r(a, 3), "rho_pred": r(b, 3),
                                "ok": bool(ok)} for s, d, sp, mn, a, b, ok in zip(key_ev.seed, key_ev["dict"], key_ev.space,
                                                                                    key_ev.member_name, key_ev.rho_measured,
                                                                                    key_ev.rho_predicted, key_ev.within_one_step)]
        S["e22_key_events_ok_frac"] = r(key_ev["within_one_step"].mean(), 3) if len(key_ev) else None
        S["e22_all_measured_events_ok_frac"] = r(m["within_one_step"].mean(), 3) if len(m) else None
    pv = read(res, "e22_predicted_risk.csv")
    if pv is not None:
        nf = pv[~pv.fit_level]
        S["e22_pred_value_abs_err"] = {f"{d}|{sp}": {"mean": r((g.value_pred - g.value_meas).abs().mean(), 5),
                                                     "max": r((g.value_pred - g.value_meas).abs().max(), 5),
                                                     "test_risk_gap_pred_vs_meas_w_max": r((g.risk_test_pred_w - g.risk_test_meas_w).max(), 5)}
                                       for (d, sp), g in nf.groupby(["dict", "space"])}
    kp = read(res, "e22_kappa.csv")
    if kp is not None:
        def grp(t):
            if t == "dm_ste":
                return "STE"
            if t.startswith("dm_") and t != "dm_ste":
                return "window(D-M)"
            if t.startswith("ds_"):
                return "D-S member"
            return t
        kp["grp"] = kp["type"].apply(grp)
        keep = ["STE", "window(D-M)", "D-S member", "bn_mlp", "ba_sigmoid", "ba_triangular", "D-M|W0", "D-S|W0", "D-M|W1", "D-S|W1"]
        kp["grp"] = np.where(kp["name"].str.contains(r"\|"), kp["name"].str.replace(r"_s\d+", "", regex=True), kp["grp"])
        S["e22_kappa"] = {g: {"full": r(sub.kappa_full.mean(), 3), "frozen": r(sub.kappa_frozen.mean(), 3)}
                          for g, sub in kp.groupby("grp") if g in keep}
    # ---------------- E23
    lf = read(res, "e23_labelfree.csv")
    if lf is not None:
        sub = lf[lf.method.isin(["D-M", "D-S"])]
        S["e23"] = {f"{c}|{m}|{sp}|{so}": {"nmse": r(x), "diff_vs_S1": r(d), "ci": [r(lo), r(hi)], "seed_sd_S1": r(sd),
                                           "within_sd": w, "equiv_labeled_traj": (e if isinstance(e, str) else None)}
                    for c, m, sp, so, x, d, lo, hi, sd, w, e in zip(sub.cond, sub.method, sub.space, sub.source, sub.nmse_mean,
                                                                   sub.diff_vs_S1, sub.diff_ci_lo, sub.diff_ci_hi, sub.seed_sd_S1,
                                                                   sub.within_seed_sd,
                                                                   sub["labeled_traj_equiv"] if "labeled_traj_equiv" in sub else [None] * len(sub))}
    # ---------------- E24
    cert = read(res, "e24_certificate.csv")
    if cert is not None:
        sub = cert[cert.space.isin(["W1", "W0"])]
        S["e24"] = {f"{c}|{d}|{sp}|m{int(m)}": {"cert_t": r(a, 3), "cert_boot": r(b, 3), "cov_t": r(ct, 3), "cov_boot": r(cb, 3),
                                                "cov_t_held": r(cth, 3), "cov_boot_held": r(cbh, 3), "lb_t": r(lb, 4),
                                                "true_gain": r(tg, 4)}
                    for c, d, sp, m, a, b, ct, cb, cth, cbh, lb, tg in zip(sub.cond, sub["dict"], sub.space, sub.m, sub.cert_rate_t,
                                                                          sub.cert_rate_boot, sub.coverage_t, sub.coverage_boot,
                                                                          sub.get("coverage_t_held", sub.coverage_t),
                                                                          sub.get("coverage_boot_held", sub.coverage_boot),
                                                                          sub.mean_lb_t, sub.mean_true_gain)}
        if "hoeffding_lb_max" in cert:
            h = cert.dropna(subset=["hoeffding_lb_max"])
            S["e24_hoeffding_max_over_all"] = r(h["hoeffding_lb_max"].max(), 1) if len(h) else None
    mm = read(res, "e24_min_m.csv")
    if mm is not None:
        S["e24_min_m_90"] = {f"{c}|{d}|{sp}|{me}": (None if pd.isna(x) else int(x))
                             for c, d, sp, me, x in zip(mm.cond, mm["dict"], mm.space, mm.method, mm.min_m_90)}
    # ---------------- E25
    bud = read(res, "e25_budget.csv")
    if bud is not None:
        sub = bud[(bud.m == -1) & bud["dict"].isin(["D-M", "D-S"])]
        S["e25_budget_full"] = {f"{c}|{d}|{f}": {str(int(k)): r(x) for k, x in zip(g.k, g.nmse)}
                                for (c, d, f), g in sub.groupby(["cond", "dict", "family"])}
    ret = read(res, "e25_retention.csv")
    if ret is not None:
        sub = ret[(ret.m == -1) & (ret.k == 2) & ret["dict"].isin(["D-M", "D-S"])]
        S["e25_retention_k2"] = {f"{c}|{d}|ref={rf}": {"with_null": r(a, 3), "plain_WQ2_vs_WQ5": r(b, 3)}
                                 for c, d, rf, a, b in zip(sub.cond, sub["dict"], sub.ref, sub.retention_null, sub.retention_plain)}
    pj = read(res, "e25_perjoint.csv")
    if pj is not None:
        sub = pj[pj["dict"] == "D-M"]
        S["e25_perjoint_DM"] = {f"{c}|{sp}": {"union_support": r(g.union_support.mean(), 2),
                                              "mean_joint_support": r(g.mean_joint_support.mean(), 2),
                                              "ste_joints_by_seed": list(g["ste_joints"].fillna(""))}
                                for (c, sp), g in sub.groupby(["cond", "space"])}
    # ---------------- E26
    sel = read(res, "e26_select.csv")
    if sel is not None:
        S["e26"] = {f"{c}|{d}|m{int(m)}": {"freq": {sp: r(sel_row[f'freq_{sp}'], 2) for sp in ["W0", "W1", "W2", "W3", "W4"]},
                                           "test_selected": r(sel_row["test_selected"]), "regret": r(sel_row["regret"], 5)}
                    for (c, d, m), sel_row in ((k, row) for k, row in zip(zip(sel.cond, sel["dict"], sel.m), [x for _, x in sel.iterrows()]))}
    # ---------------- E22 sensitivity: cubic interpolation of C(rho) through the three frozen levels
    ev3 = read(res, "e22_critical_lqc.csv"); pv3 = read(res, "e22_predicted_risk_lqc.csv")
    if ev3 is not None and pv3 is not None:
        m3 = ev3.dropna(subset=["rho_measured"]); en = m3[m3.direction == "enter"]
        nf = pv3[~pv3.fit_level]
        S["e22_sensitivity_cubic"] = {"enter_events": int(len(en)), "enter_within_step": int(en.within_one_step.sum()),
                                      "max_enter_abs_diff": r(en.abs_diff.max(), 3),
                                      "value_abs_err_max": r((nf.value_pred - nf.value_meas).abs().max(), 4),
                                      "test_gap_max": r((nf.risk_test_pred_w - nf.risk_test_meas_w).max(), 4)}
    # ---------------- width dictionary D-W (E16 members)
    lw = read(res, "e21_ladder_w16.csv")
    if lw is not None:
        sub = lw[lw.method.str.startswith("D-") & lw.space.isin(["W0", "W1", "W3"])]
        S["w16_e21_test_nmse"] = {f"{c}|{m}|{sp}": {"nmse": r(x), "seeds": sd} for c, m, sp, x, sd in
                                  zip(sub.cond, sub.method, sub.space, sub.nmse_mean, sub.seeds)}
    aw = read(res, "e21_attribution_w16.csv")
    if aw is not None:
        S["w16_attribution_DS_minus_DW"] = {f"{c}|{d}": {"margin": r(a), "scale": r(b), "angle": r(cc), "angle_ci": [r(lo), r(hi)],
                                                         "share": r(sh, 3)}
                                            for c, d, a, b, cc, lo, hi, sh in zip(aw.cond, aw["dict"], aw.margin_W0, aw.scale_part,
                                                                                 aw.angle_part, aw.angle_part_lo, aw.angle_part_hi, aw.scale_share)}
    cw = read(res, "e22_crossings_w16.csv")
    if cw is not None:
        S["w16_e22_crossings"] = {f"{a}|{b}|{sp}": r(x, 3) for a, b, sp, x in zip(cw.a, cw.b, cw.space, cw.rho_cross_mean_curve)}
    bw = read(res, "e22_bestwidth_w16.csv")
    if bw is not None:
        if "variant" not in bw:
            bw["variant"] = "standard"
        S["w16_best_hwhm_by_rho"] = {f"{v}|{f}": {str(rr): (None if not np.isfinite(h) else h) for rr, h in zip(g.rho, g.best_hwhm)}
                                     for (v, f), g in bw.groupby(["variant", "family"])}
    e18 = read(res, "e21_ladder_e18.csv")
    if e18 is not None:
        sub = e18[e18.cond.isin(["N", "O1-M", "O1-H"])]
        S["e18_single_models"] = {f"{c}|{m}|{sp}": {"nmse": r(x), "beta": r(b, 3), "nu2": r(nu, 3), "slope": r(sl, 3)}
                                  for c, m, sp, x, b, nu, sl in zip(sub.cond, sub.method, sub.space, sub.nmse_mean, sub.beta,
                                                                    sub.nu2, sub.slope)}
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_root", required=True)
    args = ap.parse_args()
    for p in ("sim", "sarcos"):
        res = os.path.join(args.res_root, p)
        S = platform_summary(res, p)
        json.dump(S, open(os.path.join(res, "e2x_summary.json"), "w"), indent=1, default=str)
        print(p, "keys:", list(S.keys()))


if __name__ == "__main__":
    main()
