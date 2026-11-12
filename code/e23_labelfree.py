"""E23: calibration without target labels (analysis of the e23_dump.py moments).

Sources of the calibration moments (test always on the target condition):
  S1   labeled target calibration role (the manuscript's protocol)
  S0   nominal labeled calibration role, no noise (ignores the shift; reference)
  S2   nominal labeled calibration + simulated diagonal Gaussian noise, variance v_hat by
       eq. (S5) from the unlabeled target calibration inputs
  S2b  (O2B) as S2 but a simulated per-trajectory bias of variance v_hat
  S3   as S2 with the true variance
  S5   as S2 with the variance from a 10 s static-robot recording
  TAB  weight table w*(rho) of E22 (paired grid, W0/W1), interpolated at
       rho_hat = sqrt(mean v_hat over the noisy channels)   (TABs: rho from the static recording,
       TABo: true rho, reference)
Misspecified true noise (t3, ar1) is simulated as diagonal Gaussian by S2/S3/S5/TAB.
Metric: test risk difference to S1 per seed, paired cluster-bootstrap interval (seed x test
unit), seed standard deviation of S1; H9: |mean difference| <= seed sd under observation
noise, and a significantly worse S2 under the payload change P1. The equivalent number
of labeled target trajectories is read off the E21 m-slice (arm).

python e23_labelfree.py --platform sim --res ../results/sim
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2x_common import SEEDS, STRONG, MomFile, member_blocks, solve_space, pj_risk, pj_unit_risk, cboot, jdump


def target_files(platform, tg):
    """label, S1 cal file, test file (relative to res)"""
    if tg == "P1":
        return "P1", "orig/moments_cal_P1_N.npz", "orig/moments_test_P1_N.npz"
    kind, rho = tg.split(":")
    rho = float(rho)
    if kind in ("t3", "ar1"):
        return f"{kind}:{rho}", f"e23/moments_cal_{kind}_rho{rho:.2f}.npz", f"e23/moments_test_{kind}_rho{rho:.2f}.npz"
    names = {"sim": {0.35: "O1-L", 0.7: "O1-M", 1.0: "O1-H"}, "sarcos": {0.25: "O1-L", 0.5: "O1-M", 0.7: "O1-H"}}
    lab = names[platform][rho] if kind == "O1" else ("O2B" if kind == "O2B" else "O1qqd")
    fl = tg.replace(":", "_")
    return lab, f"orig/moments_cal_{fl}.npz", f"orig/moments_test_{fl}.npz"


def table_weights(wt, seed, dn, sp, rho):
    sub = wt[(wt.seed == seed) & (wt.name == dn) & (wt.space == sp)].sort_values("rho")
    r = sub["rho"].values
    W = np.stack([np.array(json.loads(x)) for x in sub["weights"]])
    rho = float(np.clip(rho, r.min(), r.max()))
    k = int(np.searchsorted(r, rho))
    if k == 0:
        return W[0]
    a = (rho - r[k - 1]) / (r[k] - r[k - 1])
    return (1 - a) * W[k - 1] + a * W[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    res = args.res
    info = json.load(open(os.path.join(res, "e23", "e23_variances.json")))
    wt = pd.read_csv(os.path.join(res, "e22_wtable.csv"))
    nom = os.path.join(res, "orig", "moments_cal_N.npz")
    strong = STRONG[args.platform]
    cache = {}

    def F(path):
        p = os.path.join(res, path) if not os.path.isabs(path) else path
        if p not in cache:
            cache[p] = MomFile(p)
        return cache[p]

    rows, units = [], {}
    for tg, ti in info["targets"].items():
        lab, f1, ft = target_files(args.platform, tg)
        if not (os.path.exists(os.path.join(res, f1)) and os.path.exists(os.path.join(res, ft))):
            print("missing", tg, f1, ft, flush=True); continue
        srcs = {"S1": f1, "S0": "orig/moments_cal_N.npz", "S2": ti["S2_file"], "S3": ti["S3_file"], "S5": ti["S5_file"]}
        if "S2b_file" in ti:
            srcs["S2b"] = ti["S2b_file"]
        rho_true = float(np.sqrt(np.max(ti["v_true"]))) if np.any(np.array(ti["v_true"]) > 0) else 0.0
        rho_hat, rho_static = ti["rho_hat_mask_mean"], ti["rho_static_mask_mean"]
        for seed in SEEDS:
            B = member_blocks(args.platform, seed, "core")
            meths = {"D-M": B["dm"], "D-S": B["ds"], "B-F": [f"dm_{strong}_s{seed}"], "B-N": B["bn"], "B-A": B["ba"]}
            Ft = F(ft)
            for mn, mids in meths.items():
                mt = Ft.sel(seed, mids)
                Pt, gt, yt = mt.pooled()
                Pu, gu, yu, nu, _ = mt.per_unit()
                spaces = ("W0", "W1") if len(mids) > 1 else ("W1",)
                for sp in spaces:
                    sols = {}
                    for sn, path in srcs.items():
                        if path is None:
                            continue
                        mc = F(path).sel(seed, mids)
                        W, _ = solve_space(sp, *mc.pooled())
                        sols[sn] = W
                    if mn in ("D-M", "D-S"):
                        for sn, r in (("TAB", rho_hat), ("TABs", rho_static), ("TABo", rho_true)):
                            w = table_weights(wt, seed, mn, sp, r)
                            sols[sn] = np.tile(w[:-1], (7, 1))
                    for sn, W in sols.items():
                        R = pj_risk(W, Pt, gt, yt)[0]
                        rows.append({"cond": lab, "target": tg, "seed": seed, "method": mn, "space": sp, "source": sn,
                                     "nmse": R, "null_weight": float(1 - W[0].sum()), "weights": jdump(W[0], 4),
                                     "rho_hat": rho_hat, "rho_static": rho_static, "rho_true": rho_true})
                        units[(lab, mn, sp, sn, seed)] = (pj_unit_risk(W, Pu, gu, yu), nu)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(res, "e23_rows.csv"), index=False)
    # m-slice equivalence (arm): labeled trajectories needed to match each source
    msl = None
    p = os.path.join(res, "e21_mslice.csv")
    if os.path.exists(p):
        msl = pd.read_csv(p)
    summ = []
    for (lab, mn, sp, sn), sub in df.groupby(["cond", "method", "space", "source"]):
        s1 = df[(df.cond == lab) & (df.method == mn) & (df.space == sp) & (df.source == "S1")]
        vals, wts, cs, cu = [], [], [], []
        for s in SEEDS:
            ka, kb = (lab, mn, sp, sn, s), (lab, mn, sp, "S1", s)
            if ka in units and kb in units:
                v = units[ka][0] - units[kb][0]
                vals.append(v); wts.append(units[ka][1]); cs.append(np.full(len(v), s)); cu.append(np.arange(len(v)))
        if vals:
            pt, lo, hi = cboot(np.concatenate(vals), np.concatenate(wts), {"seed": np.concatenate(cs), "unit": np.concatenate(cu)},
                               n_boot=args.boot, seed=1)
        else:
            pt = lo = hi = np.nan
        sd1 = s1["nmse"].std(ddof=1)
        merged = sub.merge(s1[["seed", "nmse"]], on="seed", suffixes=("", "_s1"))
        rec = {"cond": lab, "method": mn, "space": sp, "source": sn, "nmse_mean": sub["nmse"].mean(),
               "nmse_S1": s1["nmse"].mean(), "diff_vs_S1": pt, "diff_ci_lo": lo, "diff_ci_hi": hi,
               "seed_sd_S1": sd1, "seeds_worse": int(np.sum(merged["nmse"] > merged["nmse_s1"])),
               "within_seed_sd": bool(abs(pt) <= sd1) if np.isfinite(pt) else None,
               "null_weight": sub["null_weight"].mean(), "rho_hat": sub["rho_hat"].iloc[0],
               "rho_static": sub["rho_static"].iloc[0], "rho_true": sub["rho_true"].iloc[0]}
        if msl is not None and lab in set(msl["cond"]):
            ms = msl[(msl.cond == lab) & (msl.method == mn) & (msl.space == sp)].sort_values("m")
            if len(ms):
                mm, rr = ms["m"].values.astype(float), ms["nmse"].values
                x = rec["nmse_mean"]
                if x < rr.min():
                    eq = f">{int(mm.max())}"
                elif x >= rr[0]:
                    eq = f"<={int(mm[0])}"
                else:
                    eq = None
                    for i in range(1, len(mm)):
                        if (rr[i - 1] - x) * (rr[i] - x) <= 0:
                            a = (rr[i - 1] - x) / (rr[i - 1] - rr[i]) if rr[i - 1] != rr[i] else 0.0
                            eq = f"{np.exp(np.log(mm[i - 1]) + a * (np.log(mm[i]) - np.log(mm[i - 1]))):.1f}"
                            break
                rec["labeled_traj_equiv"] = eq
                rec["mslice_m"] = json.dumps(mm.astype(int).tolist()); rec["mslice_nmse"] = jdump(rr, 4)
        summ.append(rec)
    pd.DataFrame(summ).to_csv(os.path.join(res, "e23_labelfree.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s, {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()
