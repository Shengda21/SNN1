"""Compact text summary of all frozen results.
python summarize_results.py --root <results dir> --frozen_dir <runs_frozen dir>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 4)
KEY = [("B-F", "D-M"), ("B-L", "-"), ("B-A", "-"), ("B-N", "-"), ("W-E", "D-S"), ("W-Q5", "D-S"), ("W-E", "D-T"), ("W-Q5", "D-T"),
       ("W-S", "D-M"), ("W-E", "D-M"), ("W-B", "D-M"), ("W-D", "D-M"), ("W-Q2", "D-M"), ("W-Q5", "D-M"), ("posthoc-opt", "D-M"), ("posthoc-opt", "D-S"), ("posthoc-opt", "D-T"),
       ("mean-member", "D-M"), ("mean-member", "D-S"), ("mean-member", "D-T")]


def paired(units, seeds, cond, a, b, dn_a, dn_b):
    """paired difference a - b over (seed, unit) with bootstrap over seeds+units."""
    diffs, cl_s, cl_u = [], [], []
    for s in seeds:
        ka = f"{s}|{cond}|{a}|{dn_a}"; kb = f"{s}|{cond}|{b}|{dn_b}"
        if ka in units and kb in units:
            d = units[ka] - units[kb]; diffs.append(d); cl_s.append(np.full(len(d), s)); cl_u.append(np.arange(len(d)))
    if not diffs:
        return None
    d = np.concatenate(diffs); rng = np.random.default_rng(1); S = np.concatenate(cl_s); U = np.concatenate(cl_u)
    us, uu = np.unique(S), np.unique(U)
    boots = []
    for _ in range(500):
        ws = np.bincount(rng.choice(len(us), len(us)), minlength=len(us))[np.searchsorted(us, S)]
        wu = np.bincount(rng.choice(len(uu), len(uu)), minlength=len(uu))[np.searchsorted(uu, U)]
        w = ws * wu
        boots.append(np.sum(w * d) / max(np.sum(w), 1))
    frac_seed_neg = np.mean([np.mean(x) < 0 for x in diffs])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(frac_seed_neg), len(diffs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--frozen_dir", required=True)
    args = ap.parse_args()
    for plat in ["sarcos", "sim"]:
        d = os.path.join(args.root, plat)
        if not os.path.exists(os.path.join(d, "e01_summary.csv")):
            print(f"== {plat}: no results"); continue
        fz = json.load(open(os.path.join(args.frozen_dir, f"frozen_{plat}.json")))
        print(f"\n==================== {plat.upper()}  frozen: strong={fz['strong_surrogate']} widths={fz['widths']} O1_M={fz['O1_M']} O1_H={fz['O1_H']}")
        s = pd.read_csv(os.path.join(d, "e01_summary.csv"))
        for cond in s.cond.unique():
            print(f"\n-- E01 {plat} cond={cond}")
            sub = s[s.cond == cond].set_index(["method", "dictionary"])
            rows = []
            for m, dn in KEY + [(f"single:{x}", "D-M") for x in ["atan", "sigmoid", "ste", "triangular", "sre"]]:
                if (m, dn) in sub.index:
                    r = sub.loc[(m, dn)]
                    rows.append((f"{m:12s} {dn:4s}", f"{r.nmse_mean:.4f} [{r.ci_lo:.4f},{r.ci_hi:.4f}]", f"R2={r.r2_mean:.4f}", f"sd={r.seed_std:.4f}" if not np.isnan(r.seed_std) else ""))
            for row in rows: print("  ", *row)
        # paired differences of interest
        units = dict(np.load(os.path.join(d, "unit_nmse.npz")))
        seeds = sorted({int(k.split("|")[0]) for k in units})
        print(f"\n-- paired differences (a - b) in nMSE, mean [95% CI], fraction of seeds with a<b, n_seeds")
        for cond in s.cond.unique():
            for (a, da, b, db) in [("W-Q5", "D-M", "W-Q5", "D-S"), ("W-Q5", "D-M", "W-Q5", "D-T"), ("W-Q5", "D-M", "W-S", "D-M"), ("W-Q5", "D-M", "B-L", "-"), ("W-Q5", "D-M", "B-A", "-"), ("W-Q5", "D-M", "B-N", "-"),
                                   ("W-E", "D-M", "W-E", "D-S"), ("W-Q2", "D-M", "W-S", "D-M"), ("W-S", "D-M", "B-F", "D-M"), ("W-Q5", "D-M", "W-E", "D-M"), ("W-Q5", "D-M", "W-B", "D-M")]:
                p = paired(units, seeds, cond, a, b, da, db)
                if p: print(f"   {cond:12s} {a}/{da} - {b}/{db}: {p[0]:+.4f} [{p[1]:+.4f},{p[2]:+.4f}]  seeds a<b: {p[3]:.2f}  n={p[4]}")
        # E02 geometry summary
        geo = json.load(open(os.path.join(d, "e02_geometry.json")))
        print("\n-- E02 witness (cal-identified pair) realized on test; value curves (test post-hoc)")
        agg = {}
        for k, g in geo.items():
            cond = k.split("|")[1]
            for dn, e in g.items():
                a = agg.setdefault((cond, dn), {"pred": [], "real": [], "exists": 0, "n": 0, "G2": [], "G5": [], "D_E": [], "weak_E": [], "net_E": [], "D_Q": [], "weak_Q": [], "net_Q": []})
                a["n"] += 1
                if "witness_test" in e:
                    a["exists"] += 1; a["pred"].append(e["witness_test"]["predicted_gain_cal"]); a["real"].append(e["witness_test"]["realized_gain"])
                vc = e["value_curve_test"]; a["G2"].append(vc["G"][1]); a["G5"].append(vc["G"][4])
                a["D_E"].append(e["decomp_equal_test"]["disagreement"]); a["weak_E"].append(e["decomp_equal_test"]["weak_member_cost"]); a["net_E"].append(e["decomp_equal_test"]["net_gain"])
                a["D_Q"].append(e["decomp_WQ5_test"]["disagreement"]); a["weak_Q"].append(e["decomp_WQ5_test"]["weak_member_cost"]); a["net_Q"].append(e["decomp_WQ5_test"]["net_gain"])
        for (cond, dn), a in sorted(agg.items()):
            print(f"   {cond:12s} {dn}: witness exists {a['exists']}/{a['n']}; pred gain {np.mean(a['pred']) if a['pred'] else np.nan:.4f} realized {np.mean(a['real']) if a['real'] else np.nan:.4f} (realized>0: {np.mean(np.array(a['real'])>0) if a['real'] else np.nan:.2f}); "
                  f"G2 {np.mean(a['G2']):.4f} G5 {np.mean(a['G5']):.4f} | equal: D {np.mean(a['D_E']):.4f} weak {np.mean(a['weak_E']):.4f} net {np.mean(a['net_E']):.4f} | WQ5: D {np.mean(a['D_Q']):.4f} weak {np.mean(a['weak_Q']):.4f} net {np.mean(a['net_Q']):.4f}")
        # E03
        e03 = pd.read_csv(os.path.join(d, "e03_m_slice.csv"))
        print("\n-- E03 m-slice (mean over seeds x groups): nmse / g_cal_5 / support")
        g = e03.groupby(["cond", "m", "rule"]).agg(nmse=("nmse", "mean"), sd=("nmse", "std"), gcal=("g_cal_5", "mean"), supp=("support_size", "mean")).reset_index()
        print(g.to_string(index=False))
        e04 = pd.read_csv(os.path.join(d, "e04_k_slice.csv"))
        print("\n-- E04 k-slice (mean over seeds x groups)")
        g = e04.groupby(["cond", "dictionary", "k"]).agg(nmse=("nmse", "mean"), sd=("nmse", "std"), supp=("support_size", "mean"), kkt=("kkt", "max")).reset_index()
        print(g.to_string(index=False))
        e05 = pd.read_csv(os.path.join(d, "e05_boundary.csv"))
        print("\n-- E05 boundary (mean over seeds)")
        print(e05.groupby("cond")[["nmse_WS", "nmse_WQ5", "rel_gain", "posthoc_opt", "best_member_test", "common_residual_nmse", "r2_WS", "r2_WQ5"]].mean().to_string())
        print(e05[["seed", "cond", "rmse_joint_WS_Nm", "rmse_joint_WQ5_Nm"]].to_string(index=False))
        p7 = os.path.join(d, "e07_density.csv")
        if os.path.exists(p7):
            e07 = pd.read_csv(p7); print("\n-- E07"); print(e07.groupby(["m_ind", "sampling", "rule"]).agg(nmse=("nmse", "mean"), sd=("nmse", "std"), frames=("frames", "first"), supp=("support_size", "mean")).to_string())
        pt = os.path.join(d, "theory_quantities.json")
        if os.path.exists(pt): print("\n-- theory", open(pt).read())
        pf = os.path.join(d, "forward_response.json")
        if os.path.exists(pf):
            fr = json.load(open(pf)); agg = {}
            for mid, v in fr["members"].items(): agg.setdefault(mid.split("_s")[0], []).append(v["response_ratio_mean"])
            print("\n-- forward response ratio (rho =", fr["rho"], ")", {k: round(float(np.mean(v)), 4) for k, v in agg.items()})
        p6 = os.path.join(d, "e06_episodes.csv")
        if os.path.exists(p6):
            e06 = pd.read_csv(p6); print("\n-- E06 closed loop (mean over seeds x groups x episodes)")
            print(e06.groupby(["cond", "controller"]).agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), maxerr=("maxerr", "mean"), tau_rms=("tau_rms", "mean"), sat=("saturation", "mean"), fail=("failed", "mean"), n=("rmse", "size"), wall=("wall_s", "mean")).to_string())
    pr = os.path.join(args.root, "runtime", "runtime.json")
    if os.path.exists(pr):
        rt = json.load(open(pr)); print("\n-- runtime", rt["gpu"]); print(pd.DataFrame(rt["rows"]).to_string(index=False)); print("param bytes per SNN", rt["param_bytes_per_snn"])


if __name__ == "__main__":
    main()
