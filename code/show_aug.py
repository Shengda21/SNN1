"""Text summary of the E09-E12 results of one platform.
python show_aug.py <results_dir>
"""
import json
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
d = sys.argv[1]
s = pd.read_csv(os.path.join(d, "e09_summary.csv"))
conds = list(dict.fromkeys(pd.read_csv(os.path.join(d, "e09_rows.csv"))["cond"]))
print("== E09/E11/E12 test nMSE (mean over seeds)")
key = s[~s.method.str.startswith("member:")].copy()
key["name"] = key.method + " / " + key.dictionary
piv = key.pivot_table(index="name", columns="cond", values="nmse_mean")[conds]
order = [n for n in piv.index if n.endswith("/ -")] + [n for n in piv.index if not n.endswith("/ -")]
print(piv.loc[order].round(4).to_string())
print("\n== members")
mem = s[s.method.str.startswith("member:")].copy()
mem["name"] = mem.dictionary + " " + mem.method
print(mem[mem.dictionary.isin(["D-MA", "D-MQ", "D-M"])].pivot_table(index="name", columns="cond", values="nmse_mean")[conds].round(4).to_string())
p = pd.read_csv(os.path.join(d, "e09_paired.csv"))
print("\n== paired differences a - b  (mean [CI]  seeds a better / n)")
p["txt"] = p.apply(lambda r: f"{r['diff']:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}] {int(r.seeds_a_better)}/{int(r.n_seeds)}", axis=1)
p["pair"] = p.a + "  -  " + p.b
print(p.pivot_table(index="pair", columns="cond", values="txt", aggfunc="first", sort=False)[conds].to_string())
w = pd.read_csv(os.path.join(d, "e09_weights.csv"))
print("\n== W-Q5 weights (mean over seeds)")
for dn in ["D-M", "D-M+A", "D-M+0", "D-S+0", "D-MA", "D-U", "D-UQ", "D-MQ", "D-M|wfe", "D-M|wfo"]:
    sub = w[(w.dictionary == dn) & (w.rule == "W-Q5")]
    if len(sub):
        print(dn); print(sub.pivot_table(index="member", columns="cond", values="weight", sort=False)[conds].round(3).to_string())
if os.path.exists(os.path.join(d, "e10_dicts.csv")):
    e = pd.read_csv(os.path.join(d, "e10_dicts.csv"))
    print("\n== E10 dictionary-level split (mean over seeds)")
    print(e.groupby(["cond", "dictionary"], sort=False).mean(numeric_only=True).drop(columns=["seed"]).round(4).to_string())
    m = pd.read_csv(os.path.join(d, "e10_members.csv"))
    print("\n== E10 member-level split (mean over seeds)")
    print(m.groupby(["cond", "dictionary", "member"], sort=False).mean(numeric_only=True).drop(columns=["seed"]).round(4).to_string())
g = os.path.join(d, "e11_frontend_geometry.json")
if os.path.exists(g):
    G = json.load(open(g))
    print("\n== input geometry: participation ratio", round(G["participation_ratio"], 2), "n_eig 90%/99%:", G["n_eig_for_90pct"], G["n_eig_for_99pct"])
    print("eigenvalues:", np.round(G["eigenvalues_Sigma_x"], 3).tolist())
    for c, v in G["conditions"].items():
        print(c, "v_est:", np.round(v["v_est"], 3).tolist())
        print("   removed fraction: analytic oracle", round(v.get("removed_fraction_analytic_oracle", 0), 3),
              "| if uncorrelated", round(v.get("removed_fraction_if_uncorrelated", 0), 3),
              "| measured wfe", round(v["wfe"]["removed_fraction_measured"], 3), "wfo", round(v["wfo"]["removed_fraction_measured"], 3))
