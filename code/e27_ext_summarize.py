"""E27 external-torque simulation: merge the evaluation parts (e27_ext_*_{a,b,c}.csv) and summarize.

e27_ext_episodes.csv  per trajectory rows of every part
e27_ext_summary.csv   per (cond, setting, candidate, filter, joint): group-averaged values per seed and
                      a 'mean' row over seeds (with min/max over seeds and, against D-S:W-Q5, the number
                      of seeds in which the candidate has the larger filtered error / lower AUC)
e27_ext_roc.csv       sample-level ROC (pooled over joints) averaged over groups and seeds
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

res = sys.argv[1] if len(sys.argv) > 1 else "../results/sim"
REF = "D-S:W-Q5"


def cat(pattern):
    fs = sorted(glob.glob(os.path.join(res, pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True), fs


EP, f1 = cat("e27_ext_episodes_[abc].csv")
SU, f2 = cat("e27_ext_summary_rows_[abc].csv")
RO, f3 = cat("e27_ext_roc_rows_[abc].csv")
print("parts", [os.path.basename(f) for f in f1 + f2 + f3])
for D_ in (EP, SU, RO):
    D_.drop_duplicates(subset=[c for c in D_.columns if c not in ("wall_s",)], inplace=True)
EP.to_csv(os.path.join(res, "e27_ext_episodes.csv"), index=False)
key = ["cond", "setting", "candidate", "filter", "joint"]
met = [c for c in SU.columns if c not in key + ["seed", "group"]]
SU["joint"] = SU["joint"].astype(str)
ps = SU.groupby(key + ["seed"], as_index=False)[met].mean()
rows = [ps]
mean = ps.groupby(key, as_index=False)[met].mean()
mn = ps.groupby(key)[["rmse_err", "auc", "det_rate_cfar", "lat_med_ms_cfar"]].min().add_suffix("_min").reset_index()
mx = ps.groupby(key)[["rmse_err", "auc", "det_rate_cfar", "lat_med_ms_cfar"]].max().add_suffix("_max").reset_index()
mean = mean.merge(mn, on=key).merge(mx, on=key)
mean["n_seeds"] = ps.groupby(key).size().values
ref = ps[ps.candidate == REF][key[:2] + ["filter", "joint", "seed", "rmse_err", "auc"]].rename(columns={"rmse_err": "ref_rmse", "auc": "ref_auc"})
cmp_ = ps.merge(ref, on=["cond", "setting", "filter", "joint", "seed"])
cmp_["worse_rmse"] = cmp_.rmse_err > cmp_.ref_rmse
cmp_["worse_auc"] = cmp_.auc < cmp_.ref_auc
cnt = cmp_.groupby(key, as_index=False)[["worse_rmse", "worse_auc"]].sum().rename(
    columns={"worse_rmse": "seeds_rmse_above_DS", "worse_auc": "seeds_auc_below_DS"})
mean = mean.merge(cnt, on=key, how="left")
mean["seed"] = "mean"
out = pd.concat([ps, mean], ignore_index=True)
out.to_csv(os.path.join(res, "e27_ext_summary.csv"), index=False)
r = RO.groupby(["cond", "setting", "candidate", "filter", "fpr"])["tpr"].agg(["mean", "min", "max", "count"]).reset_index()
r = r.rename(columns={"mean": "tpr_mean", "min": "tpr_min", "max": "tpr_max", "count": "n_runs"})
r.to_csv(os.path.join(res, "e27_ext_roc.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
v = mean[(mean.joint == "all")].set_index(["cond", "setting", "filter", "candidate"])
print(v[["rmse_err", "auc", "det_rate_cfar", "lat_med_ms_cfar", "far_per_min_cfar", "det_rate_cfar99", "lat_med_ms_cfar99",
         "far_per_min_cfar99", "seeds_rmse_above_DS", "seeds_auc_below_DS"]].round(3).to_string())
print("E27EXT_SUM_DONE")
