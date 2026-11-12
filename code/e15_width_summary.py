"""E15 width / detach / layer-wise / noise-injection summary over the E16-E18 members.

python e15_width_summary.py --results ../results [--suffix _new]

Reads <results>/<platform>/e15_{members,flip,direction,weights,stationarity}<suffix>.csv for the new
members and the original tables (no suffix) for the reference blocks dm (incl. dm_ste), bn (ReLU MLP)
and ba (B-A). One row per member with
  r (O1-M forward response), beta / nu2 / slope at O1-M, nMSE at N / O1-M / O1-H, kappa,
  first-layer ||W||_F, stable rank and relative change, layer-3 fraction of |thr - 1| < 0.02,
  median unit SNR and measured flip rate of layer 1 at O1-M, all-layer flip rate,
  bottom-5 / top-5 direction response ratio (full21), SG-1 ratios of layer 1;
then mean and seed standard deviation per (platform, block, family, hwhm, detach, placement,
membrane_noise, aug_scale, activation) -> e15_width_summary.csv (both platforms in one file, and a
copy per platform), plus the hypothesis checks H1, H3, H4 and the kappa of bounded-activation MLPs
-> e15_width_hypotheses.json.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

RHO = {"sim": {"M": "O1:0.7", "H": "O1:1.0", "rho_M": 0.7}, "sarcos": {"M": "O1:0.5", "H": "O1:0.7", "rho_M": 0.5}}
KEYS = ["platform", "block", "family", "hwhm", "detach_reset", "placement", "membrane_noise", "aug_scale", "activation"]
FAM = {"triangular": "tri", "sigmoid": "sig", "atan": "atan", "sre": "sre", "ste": "ste", "none": "mlp"}


def read(path):
    if not os.path.exists(path) or os.path.getsize(path) < 5:
        return pd.DataFrame()
    return pd.read_csv(path)


def member_table(res, plat, suffix, reference):
    d = os.path.join(res, plat)
    m = read(os.path.join(d, f"e15_members{suffix}.csv"))
    f = read(os.path.join(d, f"e15_flip{suffix}.csv"))
    dr = read(os.path.join(d, f"e15_direction{suffix}.csv"))
    w = read(os.path.join(d, f"e15_weights{suffix}.csv"))
    st = read(os.path.join(d, f"e15_stationarity{suffix}.csv"))
    if reference:
        keep = lambda df: df[df.member.str.match(r"^(dm|bn|ba)_")] if len(df) else df
        m, f, dr, w, st = map(keep, (m, f, dr, w, st))
    R = RHO[plat]
    t = m[m.role == "test"]
    one = t[t.cond == "N"].drop_duplicates("member").set_index("member")
    out = pd.DataFrame(index=one.index)
    out["platform"] = plat
    if reference:
        out["block"] = one["prefix"]
        out["family"] = one["surrogate"].map(FAM)
        out["detach_reset"] = False
        out["placement"] = "all"
        out["membrane_noise"] = 0.0
        out["aug_scale"] = (one["aug_sigma"].fillna(0.0) / R["rho_M"]).round(4)
        out["activation"] = np.where(one["surrogate"] == "none", "relu", "spike")
    else:
        for k in ["block", "family", "detach_reset", "placement", "membrane_noise", "aug_scale", "activation"]:
            out[k] = one[k]
    out["hwhm"] = one["hwhm"]
    out["seed"] = one["seed"]
    out["r"] = one["r_O1M"]
    out["kappa"] = one["kappa"]
    for c, lab in [("N", "N"), (R["M"], "M"), (R["H"], "H")]:
        tc = t[t.cond == c].set_index("member")
        out[f"nmse_{lab}"] = tc["R"]
        if lab == "M":
            for q in ["beta", "nu2", "slope", "R_angle", "scale_cost"]:
                out[q] = tc[q]
    if len(w):
        l1 = w[w.layer.isin(["fc1", "net.0"])].set_index("member")
        out["W1_fro"] = l1["fro"]; out["W1_stable_rank"] = l1["stable_rank"]; out["W1_rel_change"] = l1["rel_change"]
        l3 = w[w.layer == "fc3"].set_index("member")
        if "thr_frac_near1" in l3:
            out["thr3_frac_near1"] = l3["thr_frac_near1"]
            out["W3_rel_change"] = l3["rel_change"]
    if len(f):
        f1 = f[(f.cond == R["M"]) & (f.layer.astype(str) == "1")].set_index("member")
        fa = f[(f.cond == R["M"]) & (f.layer.astype(str) == "all")].set_index("member")
        out["snr_median_l1"] = f1["snr_q50"]; out["flip_l1"] = f1["flip_measured"]; out["flip_all"] = fa["flip_measured"]
    if len(dr):
        sub = dr[(dr.variant == "full21") & dr.subspace.isin(["top5", "bottom5"])]
        pv = sub.pivot_table(index="member", columns="subspace", values="r")
        out["r_top5"] = pv["top5"]; out["r_bottom5"] = pv["bottom5"]; out["bottom_over_top"] = pv["bottom5"] / pv["top5"]
    if len(st):
        s1 = st[st.layer.astype(str) == "1"].set_index("member")
        out["sg1_ratio_G_l1"] = s1["ratio_G"]; out["sg1_ratio_K_l1"] = s1["ratio_K"]
        s3 = st[st.layer.astype(str) == "3"].set_index("member")
        out["grad_fro_l3"] = s3["grad_fro"]
    out["source"] = "original" if reference else "new"
    return out.reset_index()


METRICS = ["r", "beta", "nu2", "slope", "R_angle", "scale_cost", "nmse_N", "nmse_M", "nmse_H", "kappa", "W1_fro",
           "W1_stable_rank", "W1_rel_change", "W3_rel_change", "thr3_frac_near1", "snr_median_l1", "flip_l1", "flip_all",
           "r_top5", "r_bottom5", "bottom_over_top", "sg1_ratio_G_l1", "sg1_ratio_K_l1", "grad_fro_l3"]


def spearman(a, b):
    a = pd.Series(np.asarray(a, float)); b = pd.Series(np.asarray(b, float))
    ok = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(a[ok].rank().corr(b[ok].rank()))


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def hypotheses(M, plat):
    H = {}
    P = M[M.platform == plat]
    # ---------------- H1: width-gain law on the peak-normalized, non-detached width sweep
    w16 = P[(P.block == "w16") & (~P.detach_reset.astype(bool))]
    h1 = {}
    for fam, g in w16.groupby("family"):
        grid = g[g.hwhm <= 4]
        h1[fam] = {"n": int(len(g)), "hwhm_max": float(g.hwhm.max()),
                   "spearman_hwhm_r_grid": spearman(grid.hwhm, grid.r), "spearman_hwhm_beta_grid": spearman(grid.hwhm, grid.beta),
                   "spearman_hwhm_nu2_grid": spearman(grid.hwhm, grid.nu2),
                   "spearman_hwhm_r_all": spearman(g.hwhm, g.r), "spearman_hwhm_beta_all": spearman(g.hwhm, g.beta),
                   "r_by_hwhm": g.groupby("hwhm").r.mean().round(4).to_dict(),
                   "beta_by_hwhm": g.groupby("hwhm").beta.mean().round(4).to_dict(),
                   "nmseN_by_hwhm": g.groupby("hwhm").nmse_N.mean().round(4).to_dict()}
    # family spread at equal width vs 2 x seed sd
    spread = {}
    for h, g in w16[w16.hwhm <= 4].groupby("hwhm"):
        fm = g.groupby("family").r.mean(); sd = g.groupby("family").r.std().mean()
        spread[str(h)] = {"family_range_r": float(fm.max() - fm.min()), "two_seed_sd": float(2 * sd)}
    ste = {}
    for lab, sel in [("dm_ste (original, reset not detached)", (P.block == "dm") & (P.family == "ste")),
                     ("w16d_ste (detached reset)", (P.block == "w16d") & (P.family == "ste"))]:
        g = P[sel]
        ste[lab] = {"n": int(len(g)), "r": float(g.r.mean()), "beta": float(g.beta.mean()), "nu2": float(g.nu2.mean()),
                    "nmse_N": float(g.nmse_N.mean())}
    widest = w16[w16.hwhm == w16.hwhm.max()]
    H["H1"] = {"per_family": h1, "family_spread_vs_2sd": spread, "ste_reference": ste,
               "widest": {"hwhm": float(widest.hwhm.max()) if len(widest) else None, "r": float(widest.r.mean()) if len(widest) else None,
                          "beta": float(widest.beta.mean()) if len(widest) else None},
               "criterion": "Spearman(width, r) <= -0.8 within every family, family spread < 2 seed sd, STE as the limit"}
    H["H1"]["supported"] = bool(all((v["spearman_hwhm_r_grid"] <= -0.8) for v in h1.values())) if h1 else None
    # detach comparison at matched width
    det = {}
    for fam in ["tri", "atan"]:
        a = P[(P.block == "w16d") & (P.family == fam)].groupby("hwhm")[["r", "beta", "nu2", "nmse_N"]].mean()
        b = w16[w16.family == fam].groupby("hwhm")[["r", "beta", "nu2", "nmse_N"]].mean()
        j = a.join(b, lsuffix="_detach", rsuffix="_nodetach", how="inner")
        det[fam] = j.round(4).to_dict("index")
    H["detach_vs_nodetach"] = det
    # ---------------- H3: layer-wise (E17), triangular, narrow 0.25
    base = w16[(w16.family == "tri") & (np.isclose(w16.hwhm, 0.25))]
    h3 = {}
    for X in [1.0, 4.0]:
        allw = w16[(w16.family == "tri") & (np.isclose(w16.hwhm, X))]
        e = {}
        for q in ["r", "beta", "nu2"]:
            d_all = allw[q].mean() - base[q].mean()
            row = {"narrow": float(base[q].mean()), "all_wide": float(allw[q].mean()), "delta_all": float(d_all)}
            for plc in ["L1", "L23"]:
                g = P[(P.block == "w17") & (P.placement == plc) & (np.isclose(P.hwhm, X))]
                row[f"{plc}"] = float(g[q].mean()); row[f"frac_{plc}"] = float((g[q].mean() - base[q].mean()) / d_all) if d_all != 0 else float("nan")
                row[f"n_{plc}"] = int(len(g))
            # seed sd of the narrow / all-wide means -> uncertainty of delta_all
            row["se_delta_all"] = float(math.sqrt(base[q].var() / max(len(base), 1) + allw[q].var() / max(len(allw), 1)))
            e[q] = row
        h3[f"hwhm{X:g}"] = e
    H["H3"] = {"by_width": h3, "criterion": "frac_L1 >= 0.7 and frac_L23 <= 0.3 of the r change of the all-wide network"}
    # ---------------- H4: equal-beta comparison of noise knobs (E18)
    # The E18 members use the platform's strong surrogate in the original parametrization:
    # sim sigmoid k = 10, SARCOS triangular d = 0.5 (= peak-normalized tri at HWHM 0.25).
    fam0 = P[P.block == "w18mem"].family.mode().iloc[0] if (P.block == "w18mem").any() else "sig"
    if plat == "sarcos":
        basec = w16[(w16.family == "tri") & np.isclose(w16.hwhm, 0.25)]          # same function, same protocol (300/30)
        base_lab = "w16_tri_h0.25 (identical to triangular d=0.5, E16-E18 protocol)"
        alt = P[(P.block == "dm") & (P.family == fam0) & P.seed.isin([0, 1, 2])]
    else:
        basec = P[(P.block == "dm") & (P.family == fam0) & P.seed.isin([0, 1, 2])]   # sigmoid k = 10, original protocol
        base_lab = "dm_sigmoid s0-s2 (sigmoid k=10, original protocol 200/25)"
        alt = basec
    b0, n0 = float(basec.beta.mean()), float(basec.nu2.mean())
    pts = []
    knobs = [("membrane_noise", (P.block == "w18mem"), "membrane_noise"), ("input_aug", (P.block == "w18aug"), "aug_scale"),
             ("escape_noise", (P.block == "w18esc"), "hwhm"),
             (f"width_orig_param(w16c {fam0})", (P.block == "w16c") & (P.family == fam0), "hwhm"),
             (f"width_peak_norm(w16 {fam0})", (P.block == "w16") & (P.family == fam0) & (P.hwhm <= 4), "hwhm")]
    for name, sel, lev in knobs:
        for L, g in P[sel].groupby(lev):
            b, n = float(g.beta.mean()), float(g.nu2.mean())
            shrink = n0 * (b / b0) ** 2
            pts.append({"knob": name, "level": float(L), "n": int(len(g)), "beta": b, "nu2": n, "nmse_M": float(g.nmse_M.mean()),
                        "nmse_N": float(g.nmse_N.mean()), "nu2_minus_scalar_shrinkage": n - shrink,
                        "exchange_rate": ((n0 - n) / (b0 - b)) if (b0 - b) > 0.01 else float("nan")})
    pts = pd.DataFrame(pts)
    # matched-beta comparison by linear interpolation of nu2(beta) along each knob (baseline included)
    grid = np.round(np.arange(0.60, 0.95, 0.02), 3)
    curves = {}
    for name in pts.knob.unique() if len(pts) else []:
        k = pts[pts.knob == name][["beta", "nu2"]].values.tolist() + [[b0, n0]]
        k = np.array(sorted(k))
        vals = {}
        for gb in grid:
            if k[:, 0].min() <= gb <= k[:, 0].max():
                vals[str(gb)] = float(np.interp(gb, k[:, 0], k[:, 1]))
        vals_shrink = {str(gb): n0 * (gb / b0) ** 2 for gb in grid if gb <= b0}
        curves[name] = vals
    curves["scalar_shrinkage_of_baseline"] = {str(gb): n0 * (gb / b0) ** 2 for gb in grid if gb <= b0}
    H["H4"] = {"baseline": {"label": base_lab, "family": fam0, "beta": b0, "nu2": n0, "nmse_M": float(basec.nmse_M.mean()),
                            "n": int(len(basec)), "alt_dm_beta": float(alt.beta.mean()), "alt_dm_nu2": float(alt.nu2.mean())},
               "points": pts.round(5).to_dict("records") if len(pts) else [], "nu2_at_matched_beta": curves,
               "criterion": "at equal beta, forward noise (membrane / input) gives lower nu2 than the width knob; wide surrogates have the worst exchange rate"}
    # ---------------- kappa of bounded-activation MLPs
    mk = {}
    for act, g in P[P.family == "mlp"].groupby("activation"):
        mk[act] = {"n": int(len(g)), "kappa_mean": float(g.kappa.mean()), "kappa_sd": float(g.kappa.std()),
                   "nmse_N": float(g.nmse_N.mean()), "beta": float(g.beta.mean()), "nu2": float(g.nu2.mean())}
    win = P[(P.block == "dm") & (P.family != "ste") & (P.family != "mlp")]
    mk["snn_window_dm(reference)"] = {"n": int(len(win)), "kappa_mean": float(win.kappa.mean()), "kappa_sd": float(win.kappa.std())}
    H["mlp_bounded_kappa"] = mk
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--suffix", default="_new")
    args = ap.parse_args()
    frames = []
    for plat in ["sim", "sarcos"]:
        if not os.path.exists(os.path.join(args.results, plat, f"e15_members{args.suffix}.csv")):
            continue
        frames.append(member_table(args.results, plat, args.suffix, reference=False))
        frames.append(member_table(args.results, plat, "", reference=True))
    M = pd.concat(frames, ignore_index=True)
    M["detach_reset"] = M["detach_reset"].astype(bool)
    M.to_csv(os.path.join(args.results, "e15_width_members.csv"), index=False)
    g = M.groupby(KEYS, dropna=False)
    agg = g[METRICS].agg(["mean", "std"])
    agg.columns = [f"{a}_{'sd' if b == 'std' else b}" for a, b in agg.columns]
    agg.insert(0, "n_seeds", g.size())
    agg = agg.reset_index()
    agg.to_csv(os.path.join(args.results, "e15_width_summary.csv"), index=False)
    for plat in agg.platform.unique():
        agg[agg.platform == plat].to_csv(os.path.join(args.results, plat, "e15_width_summary.csv"), index=False)
    H = {plat: hypotheses(M, plat) for plat in M.platform.unique()}
    with open(os.path.join(args.results, "e15_width_hypotheses.json"), "w", encoding="utf-8") as fh:
        json.dump(clean(H), fh, indent=1, ensure_ascii=False)
    print(len(M), "members;", len(agg), "groups")


if __name__ == "__main__":
    main()
