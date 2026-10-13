"""Summary of the closed-loop follow-up E13.
python summarize_e13.py <dir with e13_episodes_*.csv> [out_dir]
Writes e13_summary.csv and e13_paired.csv and prints both.
Units of analysis: (seed, episode) cells, group-dependent controllers averaged over the three
calibration groups first. Intervals: 95% percentile intervals of a two-way cluster bootstrap
over seeds and episodes with shared indices across controllers (paired differences)."""
import glob
import os
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 30)
d = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else d
e = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(d, "e13_episodes_*.csv")))], ignore_index=True)
print("rows", len(e), "seeds", sorted(e.seed.unique()), "conds", list(e.cond.unique()))
ORDER = ["PD", "B-N", "B-N|est", "B-A", "B-A+(q,qd)", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-M+null:W-Q5",
         "D-S|est:W-Q5", "D-M|est:W-Q5", "D-S,est-optional:W-Q5", "D-SQ:W-Q5"]
ORDER += [n + "@true-fb" for n in ["PD", "B-N", "B-N|est", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-S|est:W-Q5"]]

cell = e.groupby(["cond", "controller", "seed", "episode"]).agg(rmse=("rmse", "mean"), maxerr=("maxerr", "mean"), failed=("failed", "mean"),
                                                                 saturation=("saturation", "mean"), tau_rms=("tau_rms", "mean"),
                                                                 tau_fb_rms=("tau_fb_rms", "mean"), tau_ff_rms=("tau_ff_rms", "mean"),
                                                                 n_exec=("n_exec", "mean"), null_weight=("null_weight", "mean"),
                                                                 est_weight=("est_weight", "mean")).reset_index()


def boot(mat, n_boot=2000, seed=0):
    """mat: [n_seed, n_episode] values -> mean and 95% interval, resampling both ways."""
    rng = np.random.default_rng(seed)
    ns, ne = mat.shape
    b = np.empty(n_boot)
    for i in range(n_boot):
        b[i] = mat[rng.integers(0, ns, ns)][:, rng.integers(0, ne, ne)].mean()
    return float(mat.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def mat(cond, ctl, col="rmse"):
    s = cell[(cell.cond == cond) & (cell.controller == ctl)]
    if len(s) == 0:
        return None
    return s.pivot(index="seed", columns="episode", values=col).values


rows, prs = [], []
for cond in e.cond.unique():
    for ctl in ORDER:
        m = mat(cond, ctl)
        if m is None:
            continue
        mean, lo, hi = boot(m)
        s = cell[(cell.cond == cond) & (cell.controller == ctl)]
        rows.append({"cond": cond, "controller": ctl, "rmse": mean, "ci_lo": lo, "ci_hi": hi, "maxerr": s.maxerr.mean(), "fail": s.failed.mean(),
                     "saturation": s.saturation.mean(), "tau_rms": s.tau_rms.mean(), "tau_fb_rms": s.tau_fb_rms.mean(), "tau_ff_rms": s.tau_ff_rms.mean(),
                     "n_exec": s.n_exec.mean(), "null_weight": s.null_weight.mean(), "est_weight": s.est_weight.mean()})
    for base in ["D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "B-N", "D-M:W-S@true-fb", "D-S:W-Q5@true-fb"]:
        mb = mat(cond, base)
        if mb is None:
            continue
        for ctl in ORDER:
            m = mat(cond, ctl)
            if m is None or ctl == base or ctl == "PD" or (("@true-fb" in base) != ("@true-fb" in ctl)):
                continue
            dm = m - mb
            mean, lo, hi = boot(dm, seed=1)
            prs.append({"cond": cond, "a": ctl, "b": base, "diff": mean, "ci_lo": lo, "ci_hi": hi,
                        "rel": mean / mb.mean(), "seeds_a_better": int((dm.mean(1) < 0).sum()), "n_seeds": dm.shape[0]})
S = pd.DataFrame(rows); P = pd.DataFrame(prs)
S.to_csv(os.path.join(out, "e13_summary.csv"), index=False); P.to_csv(os.path.join(out, "e13_paired.csv"), index=False)
for cond in e.cond.unique():
    print("\n=====", cond)
    print(S[S.cond == cond].drop(columns=["cond"]).round(4).to_string(index=False))
    for base in ["D-M:W-S", "D-S:W-Q5", "D-M:W-S@true-fb"]:
        sub = P[(P.cond == cond) & (P.b == base)]
        if len(sub):
            print(f"\n  paired differences against {base} (rmse, rad):")
            for _, r in sub.iterrows():
                print(f"    {r.a:28s} {r['diff']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]  {100 * r.rel:+.1f}%  seeds better {r.seeds_a_better}/{r.n_seeds}")
