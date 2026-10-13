"""Numbers shown in Fig. 1, read from the frozen result files.

python build_fig1_json.py --sarcos <results/sarcos> --sim <results/sim> \
       --frozen_sarcos <frozen_sarcos.json> --frozen_sim <frozen_sim.json> --out <fig1_results.json>

panel_b  W-Q(5) on D-S and D-M and the noise-augmented network B-A at N and O1-M (e01_summary.csv)
panel_c  straight-through member of D-M: forward response (forward_response.json), risk under N
         (e01_summary.csv), cross-moments at O1-M (e02_geometry.json, test matrices averaged over
         seeds), W-Q(5) weight at N / O1-M / O1-H (e09_weights.csv)
panel_d  paired margin D-M - D-S under W-Q(5) at O1-M for the plain simplex, + null, the input
         estimator and the augmented members (e09_paired.csv / e09_summary.csv, the Table III sources)
panel_e  closed loop E13: tracking RMSE (e13_summary.csv), open-loop nMSE with the closed-loop weights (e13c_lowfreq_matched.csv),
         Spearman rank correlations over the 13 feedforward models (e13_rank_correlations.json)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

MEMBERS = ["atan", "sigmoid", "ste", "triangular", "sre"]   # order of D-M in e02_geometry.json
RUNGS = [("plain", "D-S", "D-M"), ("+ null", "D-S+0", "D-M+0"), ("input estimator", "D-S|wfe", "D-M|wfe"),
         ("augmented members", "D-SA", "D-MA")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarcos", required=True); ap.add_argument("--sim", required=True)
    ap.add_argument("--frozen_sarcos", required=True); ap.add_argument("--frozen_sim", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    DIR = {"sarcos": args.sarcos, "sim": args.sim}
    FZ = {"sarcos": json.load(open(args.frozen_sarcos)), "sim": json.load(open(args.frozen_sim))}
    rel = {"sarcos": "results/sarcos", "sim": "results/sim"}
    sources = defaultdict(list)

    def src(panel, p, f):
        s = f"{rel[p]}/{f}"
        if s not in sources[panel]:
            sources[panel].append(s)

    out = {"panel_b": {}, "panel_c": {}, "panel_d": {}, "panel_e": {}}
    for p in DIR:
        fz = FZ[p]
        cm, ch = f"O1:{fz['O1_M']}", f"O1:{fz['O1_H']}"
        e01 = pd.read_csv(os.path.join(DIR[p], "e01_summary.csv")); src("panel_b", p, "e01_summary.csv")

        def g(c, m, dn):
            r = e01[(e01.cond == c) & (e01.method == m) & (e01.dictionary == dn)].iloc[0]
            return {"mean": float(r.nmse_mean), "ci_lo": float(r.ci_lo), "ci_hi": float(r.ci_hi)}
        out["panel_b"][p] = {k: {"D-S W-Q(5)": g(c, "W-Q5", "D-S"), "D-M W-Q(5)": g(c, "W-Q5", "D-M"), "B-A": g(c, "B-A", "-")}
                             for k, c in [("N", "N"), ("O1-M", cm)]}

        # ---- panel c: straight-through member
        fr = json.load(open(os.path.join(DIR[p], "forward_response.json")))["members"]; src("panel_c", p, "forward_response.json")
        resp = {m: float(np.mean([v["response_ratio_mean"] for k, v in fr.items() if k.rsplit("_s", 1)[0] == f"dm_{m}"])) for m in MEMBERS}
        nom = {m: float(e01[(e01.cond == "N") & (e01.method == f"single:{m}") & (e01.dictionary == "D-M")].iloc[0].nmse_mean) for m in MEMBERS}
        src("panel_c", p, "e01_summary.csv")
        geo = json.load(open(os.path.join(DIR[p], "e02_geometry.json"))); src("panel_c", p, "e02_geometry.json")
        C = np.mean([np.array(v["D-M"]["C_test"]) for k, v in geo.items() if k.endswith("|" + cm)], axis=0)
        j = MEMBERS.index("ste")
        ste_cross = [float(C[j, i]) for i in range(5) if i != j]
        other_cross = [float(C[a, b]) for a in range(5) for b in range(a + 1, 5) if j not in (a, b)]
        w = pd.read_csv(os.path.join(DIR[p], "e09_weights.csv"), keep_default_na=False); src("panel_c", p, "e09_weights.csv")
        w = w[(w.rule == "W-Q5") & (w.dictionary == "D-M") & (w.member == "ste")]
        wt = w.groupby(["cond", "seed"]).weight.sum().groupby("cond").mean()
        others = [m for m in MEMBERS if m != "ste"]
        out["panel_c"][p] = {
            "forward_response": {"ste": resp["ste"], "others_min": min(resp[m] for m in others), "others_max": max(resp[m] for m in others), "per_member": resp},
            "nominal_risk_N": {"ste": nom["ste"], "others_min": min(nom[m] for m in others), "others_max": max(nom[m] for m in others)},
            "cross_moment_O1M": {"ste_min": min(ste_cross), "ste_max": max(ste_cross), "others_min": min(other_cross), "others_max": max(other_cross),
                                 "C_test_mean_over_seeds": C.tolist(), "member_order": MEMBERS},
            "wq5_weight": {"N": float(wt["N"]), "O1-M": float(wt[cm]), "O1-H": float(wt[ch])},
        }

        # ---- panel d: margins at O1-M (Table III sources)
        pr = pd.read_csv(os.path.join(DIR[p], "e09_paired.csv")); src("panel_d", p, "e09_paired.csv")
        sm = pd.read_csv(os.path.join(DIR[p], "e09_summary.csv")); src("panel_d", p, "e09_summary.csv")
        rows = []
        for lab, ds, dm in RUNGS:
            r = pr[(pr.cond == cm) & (pr.a == f"W-Q5|{dm}") & (pr.b == f"W-Q5|{ds}")].iloc[0]
            vm = float(sm[(sm.cond == cm) & (sm.method == "W-Q5") & (sm.dictionary == dm)].iloc[0].nmse_mean)
            vs = float(sm[(sm.cond == cm) & (sm.method == "W-Q5") & (sm.dictionary == ds)].iloc[0].nmse_mean)
            rows.append({"rung": lab, "D-M": dm, "D-S": ds, "diff": float(r["diff"]), "ci_lo": float(r.ci_lo), "ci_hi": float(r.ci_hi),
                         "seeds_DM_better": int(r.seeds_a_better), "n_seeds": int(r.n_seeds), "nmse_DM": vm, "nmse_DS": vs})
        out["panel_d"][p] = rows

    # ---- panel e: closed loop (E13 only)
    s13 = pd.read_csv(os.path.join(args.sim, "e13_summary.csv")); src("panel_e", "sim", "e13_summary.csv")
    lf = pd.read_csv(os.path.join(args.sim, "e13c_lowfreq_matched.csv")); src("panel_e", "sim", "e13c_lowfreq_matched.csv")
    lf["cond"] = lf["cond"].map({"N": "N", "O1M": f"O1:{FZ['sim']['O1_M']}"})
    rc = json.load(open(os.path.join(args.sim, "e13_rank_correlations.json"))); src("panel_e", "sim", "e13_rank_correlations.json")
    o1m = f"O1:{FZ['sim']['O1_M']}"
    L = lf.groupby(["cond", "method"]).nmse_total.mean()

    def t(c, ctl):
        r = s13[(s13.cond == c) & (s13.controller == ctl)].iloc[0]
        return {"rmse": float(r.rmse), "ci_lo": float(r.ci_lo), "ci_hi": float(r.ci_hi)}
    out["panel_e"] = {
        "N": {"D-M single (W-S)": t("N", "D-M:W-S"), "D-M W-Q(5)": t("N", "D-M:W-Q5")},
        "O1-M": {"tracking": {"D-S W-Q(5)": t("O1M", "D-S:W-Q5"), "D-M W-Q(5)": t("O1M", "D-M:W-Q5")},
                 "open_loop_nmse_total": {"D-S W-Q(5)": float(L[(o1m, "D-S:W-Q5")]), "D-M W-Q(5)": float(L[(o1m, "D-M:W-Q5")])}},
        "spearman_O1M": {"n_models": int(rc["O1M"]["n_models"]), "slow_L25": float(rc["O1M"]["slow_L25"]), "total": float(rc["O1M"]["total"])},
    }
    out["sources"] = dict(sources)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    d = out["panel_d"]
    print("fig1 json written", args.out, "| O1-M margins D-M - D-S:",
          {p: [f"{r['rung']} {r['diff']:+.3f}" for r in d[p]] for p in d})


if __name__ == "__main__":
    main()
