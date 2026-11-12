"""E21/E22 for the E16-E18 members: width dictionaries and knob members.

Dictionaries (seed s):
  D-W_<fam>   E16 members of one family (tri/sig/atan/sre, peak-normalized, E16 protocol) at
              HWHM {0.0625, 0.25, 1, 4} + the manuscript's straight-through member dm_ste_s<s>
              (E16 uses it as the common STE limit). D-W4_<fam>: the four widths only.
  D-Wd_<fam>  detached-reset version (w16d_*, tri/atan) at the same four widths + w16d_ste_s<s>;
              D-Wd4_<fam> without the STE member.
  D-M, D-S    references from the same moment files (tag w16).
Single models: every w16_/w16d_ member (W0/W1) and the E17/E18 members (w17_*, w18_*, w16c_*)
in W0 (as is) and W1 (scalar shrinkage c in [0, 1]) -> e21_ladder_e18.csv (beta, ||nu||^2, s,
angle risk, scale cost: the knob curves of the signal-gain plane).

E21 part (original conditions, files orig/moments_*__{w16,w16d,e18}.npz): W0, W1, W2, W3 for
the dictionaries; paired differences against D-S and D-M and between D-Wd and D-W
(cluster bootstrap seed x test unit); scale/angle split of D-S - D-W(d).
E22 part (paired grid, files paired/moments_*__{w16,w16d}.npz): W0/W1 path of the
dictionaries, risk/beta/slope of every member along rho, the best width at every rho
(three seeds; seed 0 with the ultra-wide 8-64; detached version), and crossings.

python e21_w16.py --platform sim --res ../results/sim
"""
import argparse
import glob
import os
import re
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2x_common import MomFile, cboot, geometry, jdump, member_blocks, C_matrix
from e21_ladder import conditions, fit_eval, stack_units, boot_samples
from e22_path import qp_ext, crossing

WIDTHS = ["0.0625", "0.25", "1", "4"]
FAMS = ["tri", "sig", "atan", "sre"]
TAGS = ("w16", "w16d", "e18")


def dicts_for(platform, names, seed, tag):
    out = {}
    if tag == "w16":
        ste = f"dm_ste_s{seed}"
        for f in FAMS:
            mids = [f"w16_{f}_h{h}_s{seed}" for h in WIDTHS]
            if all(m in names for m in mids) and ste in names:
                out[f"D-W_{f}"] = mids + [ste]
                out[f"D-W4_{f}"] = mids
        B = member_blocks(platform, seed, "full")
        for k, b in (("D-M", "dm"), ("D-S", "ds")):
            if all(m in names for m in B[b]):
                out[k] = B[b]
    elif tag == "w16d":
        ste = f"w16d_ste_s{seed}"
        for f in ("tri", "atan"):
            mids = [f"w16d_{f}_h{h}_s{seed}" for h in WIDTHS]
            if all(m in names for m in mids) and ste in names:
                out[f"D-Wd_{f}"] = mids + [ste]
                out[f"D-Wd4_{f}"] = mids
    return out


def is_single(nm, tag):
    if tag == "w16":
        return nm.startswith("w16_")
    if tag == "w16d":
        return nm.startswith("w16d_")
    return True


def base_name(nm):
    return re.sub(r"_s\d+$", "", nm)


def part_e21(platform, res, boot):
    rows, units = [], {}
    for lab, cr, tr, fl in conditions(platform, "orig"):
        for tag in TAGS:
            fc = os.path.join(res, "orig", f"moments_{cr}_{fl}__{tag}.npz")
            ft = os.path.join(res, "orig", f"moments_{tr}_{fl}__{tag}.npz")
            if not (os.path.exists(fc) and os.path.exists(ft)):
                continue
            Fc, Ft = MomFile(fc), MomFile(ft)
            for seed in Fc.seeds:
                names = Fc.names[seed]
                D = dicts_for(platform, names, seed, tag)
                singles = {m: [m] for m in names if is_single(m, tag)}
                for meth, mids in list(D.items()) + list(singles.items()):
                    cal, test = Fc.sel(seed, mids), Ft.sel(seed, mids)
                    spaces = ("W0", "W1", "W2", "W3") if meth in D else ("W0", "W1")
                    for sp in spaces:
                        r, ur, nu = fit_eval(sp, cal, test, units=True)
                        key = meth if meth in D else base_name(meth)
                        r.update({"platform": platform, "cond": lab, "seed": seed, "method": key, "space": sp,
                                  "tag": tag, "kind": "dict" if meth in D else "single", "members": ",".join(mids)})
                        rows.append(r)
                        units[(lab, key, sp, seed)] = (ur, nu)
        print("e21", lab, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(res, "e21_ladder_w16_seeds.csv"), index=False)

    def summarize(sub_df):
        out = []
        for (lab, meth, sp), sub in sub_df.groupby(["cond", "method", "space"]):
            sds = sorted(sub.seed.unique().tolist())
            st = stack_units(units, [(1.0, (lab, meth, sp))], sds)
            pt, lo, hi = cboot(st[0], st[1], {"seed": st[2], "unit": st[3]}, n_boot=boot) if st else (np.nan,) * 3
            rec = {"cond": lab, "method": meth, "space": sp, "tag": sub["tag"].iloc[0], "kind": sub["kind"].iloc[0],
                   "nmse_mean": sub["nmse"].mean(), "ci_lo": lo, "ci_hi": hi,
                   "seed_sd": sub["nmse"].std(ddof=1) if len(sub) > 1 else np.nan, "n_seeds": len(sub),
                   "seeds": ",".join(map(str, sds))}
            for k in ("beta", "nu2", "slope", "angle_risk", "scale_cost", "out_power", "Py", "n_exec", "null_weight"):
                rec[k] = sub[k].mean()
            out.append(rec)
        return pd.DataFrame(out)

    summarize(df[df.tag.isin(["w16", "w16d"])]).to_csv(os.path.join(res, "e21_ladder_w16.csv"), index=False)
    summarize(df[df.tag == "e18"]).to_csv(os.path.join(res, "e21_ladder_e18.csv"), index=False)
    # paired differences
    prs, att = [], []
    dws = sorted(m for m in df.method.unique() if m.startswith("D-W"))
    for lab in df.cond.unique():
        for dw in dws:
            sds = sorted(df[(df.cond == lab) & (df.method == dw)].seed.unique().tolist())
            refs = ["D-S", "D-M"]
            if dw.startswith("D-Wd_"):
                refs.append("D-W_" + dw.split("_")[1])
            for ref in refs:
                for sp in ("W0", "W1", "W2", "W3"):
                    st = stack_units(units, [(1.0, (lab, dw, sp)), (-1.0, (lab, ref, sp))], sds)
                    if st is None:
                        continue
                    pt, lo, hi = cboot(st[0], st[1], {"seed": st[2], "unit": st[3]}, n_boot=boot, seed=1)
                    ps = [float(np.average(st[0][st[2] == s], weights=st[1][st[2] == s])) for s in np.unique(st[2])]
                    prs.append({"cond": lab, "a": f"{dw}|{sp}", "b": f"{ref}|{sp}", "diff": pt, "ci_lo": lo, "ci_hi": hi,
                                "seeds_a_better": int(sum(x < 0 for x in ps)), "n_seeds": len(ps), "seeds": ",".join(map(str, sds))})
            if dw.startswith(("D-W4", "D-Wd4")):
                continue
            parts = {"margin_W0": [(1, "D-S", "W0"), (-1, dw, "W0")],
                     "scale_part": [(1, "D-S", "W0"), (-1, "D-S", "W1"), (-1, dw, "W0"), (1, dw, "W1")],
                     "angle_part": [(1, "D-S", "W1"), (-1, dw, "W1")],
                     "rem_perjoint_W3": [(1, "D-S", "W3"), (-1, dw, "W3")]}
            cols, names, base = [], [], None
            for pn, terms in parts.items():
                st = stack_units(units, [(c, (lab, m, s)) for c, m, s in terms], sds)
                if st is None:
                    continue
                if base is None:
                    base = st
                cols.append(st[0]); names.append(pn)
            if base is None or len(names) < 3:
                continue
            C = np.stack(cols, 1)
            pts = (base[1] @ C) / base[1].sum()
            bs = boot_samples(C, base[1], {"seed": base[2], "unit": base[3]}, boot)
            rec = {"cond": lab, "dict": dw, "seeds": ",".join(map(str, sds))}
            for i, pn in enumerate(names):
                rec[pn] = pts[i]; rec[pn + "_lo"] = np.percentile(bs[:, i], 2.5); rec[pn + "_hi"] = np.percentile(bs[:, i], 97.5)
            rec["scale_share"] = pts[1] / pts[0] if abs(pts[0]) > 1e-12 else np.nan
            r_ = bs[:, 1] / bs[:, 0]
            rec["scale_share_lo"], rec["scale_share_hi"] = np.percentile(r_, 2.5), np.percentile(r_, 97.5)
            att.append(rec)
    pd.DataFrame(prs).to_csv(os.path.join(res, "e21_paired_w16.csv"), index=False)
    pd.DataFrame(att).to_csv(os.path.join(res, "e21_attribution_w16.csv"), index=False)


def grid_files(res, tag):
    files = {}
    for fn in glob.glob(os.path.join(res, "paired", f"moments_cal_rho*__{tag}.npz")):
        rho = float(re.search(r"rho([0-9.]+)__", fn).group(1))
        ft = fn.replace("moments_cal_", "moments_test_")
        if os.path.exists(ft):
            files[rho] = (fn, ft)
    return dict(sorted(files.items()))


def member_meta(nm):
    m = re.match(r"(w16d?|dm)_([a-z]+)(?:_h([0-9.]+))?_s\d+", nm)
    if not m:
        return None, None, None
    blk, fam, h = m.group(1), m.group(2), m.group(3)
    if fam == "ste":
        return ("w16d" if blk == "w16d" else "w16"), "ste", np.inf
    return blk, fam, float(h)


def part_e22(platform, res):
    rows = []
    for tag in ("w16", "w16d"):
        for rho, (fc, ft) in grid_files(res, tag).items():
            Fc, Ft = MomFile(fc), MomFile(ft)
            for seed in Fc.seeds:
                names = Fc.names[seed]
                for dn, mids in dicts_for(platform, names, seed, tag).items():
                    Pc, gc, yc = Fc.sel(seed, mids).pooled(); Pt, gt, yt = Ft.sel(seed, mids).pooled()
                    Cc = C_matrix(Pc.mean(0), gc.mean(0), float(yc.mean()), null=True)
                    Ct = C_matrix(Pt.mean(0), gt.mean(0), float(yt.mean()), null=True)
                    M = len(mids)
                    for W, null in (("W0", False), ("W1", True)):
                        w, v = qp_ext(Cc, null)
                        FF = float(w[:M] @ Pt.mean(0) @ w[:M]); FY = float(w[:M] @ gt.mean(0)); Py = float(yt.mean())
                        geo = geometry(FF, FY, Py)
                        rows.append({"kind": "dict", "tag": tag, "seed": seed, "name": dn, "space": W, "rho": rho,
                                     "risk_cal": v, "risk_test": float(w @ Ct @ w), "beta": geo["beta"], "nu2": geo["nu2"],
                                     "slope": geo["slope"], "scale_cost": geo["scale_cost"], "null_weight": float(w[M]),
                                     "weights": jdump(w, 4), "members": ",".join(mids)})
                Pt, gt, yt = Ft.sel(seed, names).pooled()
                for j, nm in enumerate(names):
                    if not (nm.startswith(("w16_", "w16d_")) or nm.startswith("dm_ste")):
                        continue
                    geo = geometry(float(Pt[:, j, j].mean()), float(gt[:, j].mean()), float(yt.mean()))
                    rows.append({"kind": "member", "tag": tag, "seed": seed, "name": nm, "space": "-", "rho": rho,
                                 "risk_cal": np.nan, "risk_test": geo["risk"], "beta": geo["beta"], "nu2": geo["nu2"],
                                 "slope": geo["slope"], "scale_cost": geo["scale_cost"], "null_weight": np.nan,
                                 "weights": "", "members": nm})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(res, "e22_path_w16.csv"), index=False)
    mem = df[df.kind == "member"].copy()
    meta = mem["name"].apply(member_meta)
    mem["blk"] = [m[0] for m in meta]; mem["fam"] = [m[1] for m in meta]; mem["hwhm"] = [m[2] for m in meta]
    best = []
    variants = []
    for fam in FAMS:
        variants.append(("standard_3seeds", fam, "w16", [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0], None))
        variants.append(("standard_seed0_ultrawide", fam, "w16", None, [0]))
    for fam in ("tri", "atan"):
        variants.append(("detach_3seeds", fam, "w16d", None, None))
    for vname, fam, blk, hset, seeds_only in variants:
        sub = mem[(mem.blk == blk) & ((mem.fam == fam) | (mem.fam == "ste"))]
        if hset is not None:
            sub = sub[sub.hwhm.isin(hset) | np.isinf(sub.hwhm)]
        if seeds_only is not None:
            sub = sub[sub.seed.isin(seeds_only)]
        fam_seeds = sorted(sub[sub.fam == fam].seed.unique())
        sub = sub[sub.seed.isin(fam_seeds)]
        # keep widths present in every seed used
        cnt = sub.groupby("hwhm")["seed"].nunique()
        sub = sub[sub.hwhm.isin(cnt[cnt == len(fam_seeds)].index)]
        if sub.empty:
            continue
        piv = sub.groupby(["rho", "hwhm"])["risk_test"].mean().unstack(1)
        for rho, row in piv.iterrows():
            best.append({"variant": vname, "family": fam, "rho": rho, "best_hwhm": float(row.idxmin()),
                         "best_risk": float(row.min()), "seeds": ",".join(map(str, fam_seeds)),
                         "widths": ",".join(str(h) for h in row.index),
                         **{f"risk_h{h}": float(v) for h, v in row.items()}})
    pd.DataFrame(best).to_csv(os.path.join(res, "e22_bestwidth_w16.csv"), index=False)
    cr = []
    dd = df[df.kind == "dict"]
    for dn in sorted(n for n in dd.name.unique() if n.startswith("D-W")):
        refs = ["D-S", "D-M"] + (["D-W_" + dn.split("_")[1]] if dn.startswith("D-Wd_") else [])
        for ref in refs:
            for sp in ("W0", "W1"):
                sds = dd[(dd.name == dn)].seed.unique()
                a = dd[(dd.name == dn) & (dd.space == sp)].groupby("rho")["risk_test"].mean()
                b = dd[(dd.name == ref) & (dd.space == sp) & dd.seed.isin(sds)].groupby("rho")["risk_test"].mean()
                j = a.index.intersection(b.index)
                if not len(j):
                    continue
                cr.append({"a": dn, "b": ref, "space": sp, "seeds": ",".join(map(str, sorted(sds))),
                           "rho_cross_mean_curve": crossing(np.array(j), (a[j] - b[j]).values),
                           "diff_at_rho0": float(a[j].iloc[0] - b[j].iloc[0]), "diff_at_max": float(a[j].iloc[-1] - b[j].iloc[-1])})
    pd.DataFrame(cr).to_csv(os.path.join(res, "e22_crossings_w16.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--parts", default="e21,e22")
    args = ap.parse_args()
    t0 = time.time()
    if "e21" in args.parts:
        part_e21(args.platform, args.res, args.boot)
    if "e22" in args.parts:
        part_e22(args.platform, args.res)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
