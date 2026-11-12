"""E28 summary: merge the per-seed parts (results/sim/e28_parts/e28_episodes*_s*.csv with their
_params.json) into e28_episodes.csv and write e28_summary.csv and e28_paired.csv.

Units: (seed, episode) cells; 95% intervals by the two-way cluster bootstrap over seeds and episodes
(episodes only for the seed-free references 'oracle' and 'ID-exact'). Free episodes give the drift of
the admittance reference (|q_r(10 s) - q_0|, max over joints), intent episodes the human torque RMS
(transparency) and the intention-following error. Paired differences against D-S:W-Q5 per condition
and admittance setting, with the number of seeds in which the candidate drifts more.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1] if len(sys.argv) > 1 else "../results/sim"
REF = "D-S:W-Q5"
parts = []
for f in sorted(glob.glob(os.path.join(res, "e28_parts", "e28_episodes*_s*.csv"))):
    if f.endswith("_params.json"):
        continue
    d = pd.read_csv(f)
    p = json.load(open(f.replace(".csv", "_params.json")))
    d["c_bd"] = p["c_bd"]
    parts.append(d)
E = pd.concat(parts, ignore_index=True)
E.to_csv(os.path.join(res, "e28_episodes.csv"), index=False)


def boot(mat, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    ns, ne = mat.shape
    b = np.empty(n_boot)
    for i in range(n_boot):
        b[i] = mat[rng.integers(0, ns, ns)][:, rng.integers(0, ne, ne)].mean()
    return float(mat.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


METRICS = {"free": ["drift_final_max", "drift_final_rms", "drift_max", "q_drift_max", "tau_hat5_rms", "trk_rmse"],
           "intent": ["tau_h_rms", "intent_err_rms", "tau_err_rms", "trk_rmse"]}
rows, prs = [], []
for (cbd, cond), sub in E.groupby(["c_bd", "cond"]):
    for cand in sub.candidate.unique():
        r = {"c_bd": cbd, "cond": cond, "candidate": cand}
        for et, mets in METRICS.items():
            s = sub[(sub.candidate == cand) & (sub.episode_type == et)]
            for m in mets:
                mat = s.pivot_table(index="seed", columns="episode", values=m).values
                mean, lo, hi = boot(mat)
                r[f"{et}_{m}"] = mean; r[f"{et}_{m}_lo"] = lo; r[f"{et}_{m}_hi"] = hi
            r[f"{et}_fail"] = float(s.failed.mean()); r[f"{et}_limit_hit"] = float(s.limit_hit.mean())
            r[f"{et}_saturation"] = float(s.saturation.mean())
            if et == "free":
                r["free_drift_gt_0p5"] = float((s.drift_final_max > 0.5).mean())
                for l in range(1, 8):
                    r[f"free_absdrift_j{l}"] = float(s[f"drift_final_j{l}"].abs().mean())
                    r[f"free_tau_hat_mean_j{l}"] = float(s[f"tau_hat_mean_j{l}"].mean())
            else:
                for l in range(1, 8):
                    r[f"intent_tau_h_rms_j{l}"] = float(s[f"tau_h_rms_j{l}"].mean())
        r["n_seeds"] = int(sub[sub.candidate == cand].seed.nunique())
        rows.append(r)
        ref = REF + ("@true-fb" if cand.endswith("@true-fb") else "")
        if cand != ref and not cand.startswith(("oracle", "ID-exact")):
            for et, m in [("free", "drift_final_max"), ("free", "tau_hat5_rms"), ("intent", "tau_h_rms"), ("intent", "intent_err_rms")]:
                a = sub[(sub.candidate == cand) & (sub.episode_type == et)].pivot_table(index="seed", columns="episode", values=m)
                b = sub[(sub.candidate == ref) & (sub.episode_type == et)].pivot_table(index="seed", columns="episode", values=m)
                idx = a.index.intersection(b.index)
                if len(idx) == 0:
                    continue
                dm = a.loc[idx].values - b.loc[idx].values
                mean, lo, hi = boot(dm, seed=1)
                prs.append({"c_bd": cbd, "cond": cond, "a": cand, "b": ref, "episode_type": et, "metric": m, "diff": mean,
                            "ci_lo": lo, "ci_hi": hi, "rel": mean / b.loc[idx].values.mean(),
                            "seeds_a_larger": int((dm.mean(1) > 0).sum()), "n_seeds": len(idx)})
S = pd.DataFrame(rows); P = pd.DataFrame(prs)
S.to_csv(os.path.join(res, "e28_summary.csv"), index=False)
P.to_csv(os.path.join(res, "e28_paired.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(S.set_index(["c_bd", "cond", "candidate"])[["free_drift_final_max", "free_drift_final_max_lo", "free_drift_final_max_hi",
                                                    "free_fail", "free_limit_hit", "free_drift_gt_0p5", "free_tau_hat5_rms",
                                                    "intent_tau_h_rms", "intent_intent_err_rms", "intent_fail"]].round(4).to_string())
print(P[P.metric == "drift_final_max"].round(4).to_string())
print("E28SUM_DONE")
