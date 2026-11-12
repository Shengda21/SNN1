"""Paired-noise re-export of frozen-member predictions as per-unit moments (E21-E26).

Modes
  grid  delta = rho * xi_role * mask with ONE fixed standard-normal array xi per role
        (seed XI_SEED[role], independent of rho), so the path in rho is smooth and
        differences between levels are paired. sim: channels 0-13; sarcos: 21 channels.
        -> <out>/paired/moments_<role>_rho<rho>.npz
  orig  the original conditions (sim/data.py, sarcos/data.py apply_condition, identical
        noisy inputs to the manuscript), optionally behind the label-free (wfe) or
        oracle (wfo) linear input estimator of frontend_preds.py.
        -> <out>/orig/moments_<role>_<cond>[__wfe|__wfo].npz
Nothing but moments is stored (see e2x_common.compute_moments). Predictions are
clipped to the torque limits on the arm exactly as in analysis_openloop.py.

Member sets: --members full | core | glob:<pattern>[,glob:<pattern>...][,block:dm,...]
Members are grouped by seed (token _s<seed>). New members (e.g. w16_*) are added
with --members "glob:w16_*,block:dm" --tag w16, which writes a separate file
(..._<tag>.npz) holding the new members together with the D-M members of the same
seed (cross moments included), so the files can be merged at analysis time.

python e2x_dump_paired.py --platform sim --data_dir ../data_sim --runs ../runs25/sim --out ../results/sim --mode grid
python e2x_dump_paired.py --platform sim --data_dir ../data_sim --runs ../runs25/sim --out ../results/sim --mode orig \
       --conds N,O1:0.35,O1:0.7,O1:1.0,O2B:0.7 --variants raw,wfe,wfo
"""
import argparse
import fnmatch
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2x_common import (SEEDS, XI_SEED, cname, noise_mask, member_blocks, unit_layout,
                        compute_moments, save_moments)
from common.models import predict
from dump_preds import load_member
from run_train import load_platform

torch.set_num_threads(4)


def seed_of(name):
    m = re.search(r"_s(\d+)(?:_|$)", name)
    return int(m.group(1)) if m else -1


def build_groups(args, runs_dirs):
    """seed -> (ordered member names, {block: [indices]})"""
    groups = {}
    if args.members in ("full", "core"):
        for s in SEEDS:
            B = member_blocks(args.platform, s, args.members)
            names, blocks = [], {}
            for b, mids in B.items():
                ok = [m for m in mids if find_run(runs_dirs, m)]
                if len(ok) != len(mids):
                    print("missing members in block", b, s, set(mids) - set(ok), flush=True)
                    continue
                blocks[b] = list(range(len(names), len(names) + len(ok)))
                names += ok
            groups[s] = (names, blocks)
        return groups
    specs = args.members.split(",")
    all_dirs = []
    for rd in runs_dirs:
        all_dirs += [os.path.basename(d) for d in sorted(glob.glob(os.path.join(rd, "*")))
                     if os.path.exists(os.path.join(d, "config.json"))]
    per_seed = {}
    for sp in specs:
        kind, pat = sp.split(":", 1)
        if kind == "glob":
            for mid in all_dirs:
                if fnmatch.fnmatch(mid, pat):
                    per_seed.setdefault(seed_of(mid), {}).setdefault(pat, []).append(mid)
        elif kind == "block":
            glob_seeds = {seed_of(m) for sp2 in specs if sp2.startswith("glob:") for m in all_dirs
                          if fnmatch.fnmatch(m, sp2.split(":", 1)[1])}
            for s in SEEDS:
                if any(sp2.startswith("glob:") for sp2 in specs) and s not in glob_seeds:
                    continue
                mids = member_blocks(args.platform, s, "full")[pat]
                if all(find_run(runs_dirs, m) for m in mids):
                    per_seed.setdefault(s, {})[pat] = mids
    for s, d in per_seed.items():
        names, blocks = [], {}
        for k, mids in d.items():
            mids = sorted(set(mids), key=mids.index)
            bname = k if not any(c in k for c in "*?[") else "group"
            blocks[bname] = list(range(len(names), len(names) + len(mids)))
            names += mids
        groups[s] = (names, blocks)
    return groups


def load_any(rd, mid, n_in, d, device):
    """Original members via dump_preds.load_member; E16-E18 members (model_class SNNX/MLPAct)
    via the E16 builder (read-only use of e16_models.py)."""
    cfg = json.load(open(os.path.join(rd, mid, "config.json")))
    if cfg.get("model_class") in ("SNNX", "MLPAct"):
        from e16_models import build_from_config
        model = build_from_config(cfg, n_in, d)
        model.load_state_dict(torch.load(os.path.join(rd, mid, "model.pt"), map_location=device))
        return model.to(device).eval()
    return load_member(rd, mid, n_in, d, device)


def find_run(runs_dirs, mid):
    for rd in runs_dirs:
        if os.path.exists(os.path.join(rd, mid, "config.json")):
            return rd
    return None


def wiener(Sig, v):
    return Sig @ np.linalg.inv(Sig + np.diag(v))


def oracle_var(platform, cond, n_in):
    v = np.zeros(n_in)
    if cond == "N":
        return v
    kind, rho = cond.split(":")
    rho = float(rho)
    if platform == "sarcos" and kind == "O1":
        v[:] = rho ** 2
    else:
        v[:14] = rho ** 2
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run dirs searched in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["grid", "orig"])
    ap.add_argument("--rhos", default=None, help="grid: comma list; default 0..1.2 (sim) / 0..0.9 (sarcos) step 0.05")
    ap.add_argument("--roles", default=None)
    ap.add_argument("--conds", default=None, help="orig: conditions")
    ap.add_argument("--variants", default="raw", help="orig: raw,wfe,wfo")
    ap.add_argument("--members", default="full")
    ap.add_argument("--xblocks", default="dm,ds,dt,bl,ba,bn", help="blocks with cross-joint moments ('' = none)")
    ap.add_argument("--seg", type=int, default=5, help="sim: segments per trajectory on cal roles (orig mode)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--check", action="store_true", help="compare predictions with stored pred_*.npy")
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, apply_condition = load_platform(args.platform, args.data_dir)
    sim = args.platform == "sim"
    n_in, d = D["n_in"], D["n_out"]
    if sim:
        from sim.panda_env import TORQUE_LIMIT
        lo = (-TORQUE_LIMIT - D["y_mean"]) / D["y_std"]; hi = (TORQUE_LIMIT - D["y_mean"]) / D["y_std"]
        clip = lambda P: np.clip(P, lo, hi)
    else:
        clip = lambda P: P
    runs_dirs = args.runs.split(",")
    groups = build_groups(args, runs_dirs)
    xb_names = [b for b in args.xblocks.split(",") if b]
    # load every model once and keep it on the device (54k parameters each)
    models = {}
    for s, (names, _) in groups.items():
        for mid in names:
            rd = find_run(runs_dirs, mid)
            models[mid] = (load_any(rd, mid, n_in, d, device), rd)
    print(f"loaded {len(models)} models in {time.time() - t0:.1f}s; groups:",
          {s: len(v[0]) for s, v in groups.items()}, flush=True)
    mask = noise_mask(args.platform, n_in)
    sub_dir = "paired" if args.mode == "grid" else "orig"
    os.makedirs(os.path.join(args.out, sub_dir), exist_ok=True)
    checks = []
    tag = f"__{args.tag}" if args.tag else ""

    if args.mode == "grid":
        if args.rhos:
            rhos = [float(x) for x in args.rhos.split(",")]
        else:
            top = 1.2 if sim else 0.9
            rhos = [round(0.05 * i, 2) for i in range(int(round(top / 0.05)) + 1)]
        roles = (args.roles or "cal,test").split(",")
        jobs = []
        for role in roles:
            for rho in rhos:
                jobs.append((role, f"rho{rho:.2f}", {"kind": "grid", "rho": rho, "xi_seed": XI_SEED[role]}))
    else:
        conds = args.conds.split(",")
        variants = args.variants.split(",")
        roles = (args.roles or ("cal,test,cal_P1,test_P1" if sim else "cal,test")).split(",")
        jobs = []
        for role in roles:
            for cond in conds:
                if role.endswith("_P1") and cond != "N":
                    continue
                for var in variants:
                    jobs.append((role, cname(cond) + ("" if var == "raw" else f"__{var}"),
                                 {"kind": "orig", "cond": cond, "variant": var}))
    # input-estimator geometry (frontend_preds.py): training mean and covariance
    Xtr = D["X"]["train"].astype(np.float64)
    mu, Sig, v_train = Xtr.mean(0), np.cov(Xtr.T), Xtr.var(0)
    xi_cache = {}
    for role, label, spec in jobs:
        fn = os.path.join(args.out, sub_dir, f"moments_{role}_{label}{tag}.npz")
        if os.path.exists(fn) and not args.overwrite:
            print("exists", fn, flush=True)
            continue
        t1 = time.time()
        X = D["X"][role]
        Y = D["Y"][role].astype(np.float64)
        seg = args.seg if (sim and args.mode == "orig" and role.startswith("cal")) else 1
        sub, parent = unit_layout(args.platform, D, role, seg=seg)
        meta = {"platform": args.platform, "role": role, "label": label, "spec": spec, "seg": seg,
                "seeds": sorted(groups.keys()), "clip": bool(sim), "units": "sim trajectory segments" if sim else "sarcos 100-row blocks"}
        if spec["kind"] == "grid":
            if role not in xi_cache:
                rng = np.random.default_rng(spec["xi_seed"])
                xi_cache[role] = rng.standard_normal((len(X), n_in))
            Xn = (X.astype(np.float64) + spec["rho"] * xi_cache[role] * mask).astype(np.float32)
            ref = f"pred_{role}_N.npy" if spec["rho"] == 0 else None
        else:
            cond, var = spec["cond"], spec["variant"]
            Xc = apply_condition(X, role, cond)
            if Xc is None:
                continue
            ref = f"pred_{role}_{cname(cond)}" + ("" if var == "raw" else f"__{var}") + ".npy"
            if var == "raw":
                Xn = Xc
            else:
                cal_role = "cal_P1" if role.endswith("_P1") else "cal"
                Xcal = apply_condition(D["X"][cal_role], cal_role, cond).astype(np.float64)
                v = np.maximum(Xcal.var(0) - v_train, 0.0) if var == "wfe" else oracle_var(args.platform, cond, n_in)
                Wm = wiener(Sig, v)
                Xn = (mu + (Xc.astype(np.float64) - mu) @ Wm.T).astype(np.float32)
                meta["v_noise"] = v.tolist()
        out_groups = {}
        for s, (names, blocks) in groups.items():
            P = np.empty((len(names), len(X), d), np.float64)
            for i, mid in enumerate(names):
                model, rd = models[mid]
                Pi = predict(model, Xn, device, batch=args.batch)
                if args.check and ref is not None:
                    fr = os.path.join(rd, mid, ref)
                    if os.path.exists(fr):
                        Pr = np.load(fr)
                        checks.append({"file": label, "role": role, "member": mid,
                                       "max_abs_diff": float(np.max(np.abs(Pr.astype(np.float64) - Pi.astype(np.float64))))})
                P[i] = clip(Pi.astype(np.float64))
            xb = {b: blocks[b] for b in xb_names if b in blocks} if xb_names else {}
            if args.xblocks == "group":
                xb = {"group": list(range(len(names)))}
            mo = compute_moments(P, Y, sub, xb)
            out_groups[s] = {"names": names, "FF": mo["FF"], "FY": mo["FY"],
                             "X": {b: (mo[f"XFF_{b}"], mo[f"XFY_{b}"]) for b in xb}}
            n, YY = mo["n"], mo["YY"]
            del P, mo
        save_moments(fn, meta, n, parent, YY, out_groups)
        print(f"wrote {fn} ({os.path.getsize(fn) / 1e6:.1f} MB) in {time.time() - t1:.1f}s", flush=True)
    if checks:
        cf = os.path.join(args.out, sub_dir, f"check_preds_{args.mode}{tag}.json")
        old = json.load(open(cf)) if os.path.exists(cf) else []
        json.dump(old + checks, open(cf, "w"), indent=0)
        mx = max(c["max_abs_diff"] for c in checks)
        print(f"prediction check: {len(checks)} comparisons, max |diff| = {mx:.3g}", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
