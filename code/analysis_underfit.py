"""E14: is the low response of the straight-through member a property of its surrogate, or of
a member that fits the nominal data poorly?

Part 1 (curve). For every epoch state of the replayed training trajectories (run_train.py,
block traj) and for the final D-M members: nominal test nMSE, measured forward perturbation
response (same inputs and perturbations as forward_response.py), and under O1-M the risk, the
propagated term and the slope of the target on the prediction.
  -> e14_curve.csv

Part 2 (substitution). Dictionaries in which the straight-through member of D-M, or one
member of D-S, is replaced by an early checkpoint of an ordinary member whose development
error brackets that of the straight-through member (uf_*_b: first epoch at or below it,
uf_*_a: last epoch above it). Same rule (W-Q5, full calibration set) and test sets as E01.
  -> e14_rows.csv, e14_summary.csv, e14_paired.csv, e14_weights.csv

python analysis_underfit.py --platform sim --data_dir ../data_sim --runs ../runs/sim \
       --frozen ../runs/sim/frozen.json --out ../results/sim
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models import SNN, predict
from common.surrogates import SURROGATE_NAMES
from common import calibration as cal
from common.metrics import nmse, nmse_rows, cluster_bootstrap
from run_train import load_platform


def cname(c):
    return c.replace(":", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--frozen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=500)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, apply_condition = load_platform(args.platform, args.data_dir)
    frozen = json.load(open(args.frozen))
    sim = args.platform == "sim"
    strong = frozen["strong_surrogate"]; widths = frozen["widths"]
    d = D["n_out"]; one = np.ones(d)
    if sim:
        from sim.panda_env import TORQUE_LIMIT
        lo = (-TORQUE_LIMIT - D["y_mean"]) / D["y_std"]; hi = (TORQUE_LIMIT - D["y_mean"]) / D["y_std"]
        clip = lambda P: np.clip(P, lo, hi)
    else:
        clip = lambda P: P
    O1M = f"O1:{frozen['O1_M']}"
    Ytest = D["Y"]["test"].astype(np.float64)
    Xt_clean = D["X"]["test"]; Xt_noisy = apply_condition(D["X"]["test"], "test", O1M)
    # forward response: identical inputs and perturbations to forward_response.py
    rng = np.random.default_rng(0)
    Xr = D["X"]["dev"][rng.choice(len(D["X"]["dev"]), 2000, replace=False)]
    mask = np.ones(D["n_in"], np.float32)
    if sim:
        mask[14:] = 0
    delta = rng.standard_normal(Xr.shape).astype(np.float32) * frozen["O1_M"] * mask

    def metrics(model):
        Pc = clip(predict(model, Xt_clean, device).astype(np.float64)); Pn = clip(predict(model, Xt_noisy, device).astype(np.float64))
        with torch.no_grad():
            f0 = model(torch.as_tensor(Xr, device=device))[-1].cpu().numpy(); f1 = model(torch.as_tensor(Xr + delta, device=device))[-1].cpu().numpy()
        resp = float(np.mean(np.linalg.norm(f1 - f0, axis=1) / np.linalg.norm(delta, axis=1)))
        return {"test_nmse_N": nmse(Pc, Ytest, one), "risk_O1M": nmse(Pn, Ytest, one),
                "P_n": float(np.mean(np.sum((Pn - Pc) ** 2, 1)) / d), "slope": float(np.sum(Pn * Ytest) / np.sum(Pn ** 2)),
                "output_power": float(np.mean(Pn ** 2)), "response": resp}

    # ---------------- Part 1: curve along the training trajectories
    curve = []
    for tdir in sorted(glob.glob(os.path.join(args.runs, "traj_*_s*"))):
        h = json.load(open(os.path.join(tdir, "history.json")))
        s, seed = h["surrogate"], h["seed"]
        states = torch.load(os.path.join(tdir, "states.pt"), map_location="cpu")
        model = SNN(D["n_in"], d, surrogate=s, width=widths[s]).to(device).eval()
        # check that the replay follows the original run
        orig = json.load(open(os.path.join(args.runs, f"dm_{s}_s{seed}", "history.json")))
        n_cmp = min(len(orig), len(h["history"]))
        dev_gap = max(abs(orig[e][2] - h["history"][e][2]) for e in range(n_cmp))
        for e, st in enumerate(states):
            model.load_state_dict(st)
            r = {"seed": seed, "surrogate": s, "kind": "trajectory", "epoch": e, "dev_nmse": h["history"][e][2], "replay_dev_gap": dev_gap}
            r.update(metrics(model)); curve.append(r)
        print("curve", os.path.basename(tdir), "epochs", len(states), "max dev gap to original run", f"{dev_gap:.2e}", flush=True)
    for seed in range(5):
        for s in SURROGATE_NAMES:
            mid = f"dm_{s}_s{seed}"
            cfg = json.load(open(os.path.join(args.runs, mid, "config.json")))
            model = SNN(D["n_in"], d, surrogate=s, width=widths[s]).to(device).eval()
            model.load_state_dict(torch.load(os.path.join(args.runs, mid, "model.pt"), map_location=device))
            r = {"seed": seed, "surrogate": s, "kind": "final member", "epoch": cfg["best_epoch"], "dev_nmse": cfg["best_dev"], "replay_dev_gap": 0.0}
            r.update(metrics(model)); curve.append(r)
        for tag in ("a", "b"):
            for s in SURROGATE_NAMES:
                mid = f"uf_{s}_s{seed}_{tag}"
                if not os.path.exists(os.path.join(args.runs, mid, "config.json")):
                    continue
                cfg = json.load(open(os.path.join(args.runs, mid, "config.json")))
                model = SNN(D["n_in"], d, surrogate=s, width=widths[s]).to(device).eval()
                model.load_state_dict(torch.load(os.path.join(args.runs, mid, "model.pt"), map_location=device))
                r = {"seed": seed, "surrogate": s, "kind": f"early checkpoint {tag}", "epoch": cfg["epoch"], "dev_nmse": cfg["best_dev"], "replay_dev_gap": 0.0}
                r.update(metrics(model)); curve.append(r)
    for cdir in sorted(glob.glob(os.path.join(args.runs, "cap*_s*"))):
        cfg = json.load(open(os.path.join(cdir, "config.json")))
        model = SNN(D["n_in"], d, hidden=cfg["hidden"], surrogate=cfg["surrogate"], width=cfg["width"]).to(device).eval()
        model.load_state_dict(torch.load(os.path.join(cdir, "model.pt"), map_location=device))
        r = {"seed": cfg["seed"], "surrogate": cfg["surrogate"], "kind": f"hidden {cfg['hidden']}", "epoch": cfg["best_epoch"],
             "dev_nmse": cfg["best_dev"], "replay_dev_gap": 0.0}
        r.update(metrics(model)); curve.append(r)
    pd.DataFrame(curve).to_csv(os.path.join(args.out, "e14_curve.csv"), index=False)

    # ---------------- Part 2: substitution dictionaries
    cases = [("cal", "test", c) for c in frozen["conds"]] + ([("cal_P1", "test_P1", "N")] if sim else [])
    window = [x for x in SURROGATE_NAMES if x != "ste"]

    def pred(mid, role, cond):
        return clip(np.load(os.path.join(args.runs, mid, f"pred_{role}_{cname(cond)}.npy")).astype(np.float64))

    rows, wrows, units = [], [], {}
    for seed in range(5):
        dm = {x: f"dm_{x}_s{seed}" for x in SURROGATE_NAMES}
        ds = [f"ds_{strong}_s{seed}_m{i}" for i in range(5)]
        has = lambda mid: os.path.exists(os.path.join(args.runs, mid, "config.json"))
        for cal_role, test_role, cond in cases:
            lab = "P1" if cal_role.endswith("_P1") else cond
            Yc, Yt = D["Y"][cal_role].astype(np.float64), D["Y"][test_role].astype(np.float64)
            gc = D["groups"][cal_role] if sim else None; gt = D["groups"][test_role] if sim else None
            dicts = {"D-M": ([dm[x] for x in SURROGATE_NAMES], list(SURROGATE_NAMES)),
                     "D-S": (ds, [f"seed{i}" for i in range(5)]),
                     "D-M-STE": ([dm[x] for x in window], window)}
            for tag in ("b", "a"):
                for x in window:
                    uf = f"uf_{x}_s{seed}_{tag}"
                    if has(uf):
                        name = f"D-M[STE->early {x}:{tag}]"
                        dicts[name] = ([dm[y] for y in window] + [uf], window + ["early"])
                uf = f"uf_{strong}_s{seed}_{tag}"
                if tag == "b":
                    for cdir in sorted(glob.glob(os.path.join(args.runs, f"cap*_{strong}_s{seed}"))):
                        cm = os.path.basename(cdir); hname = cm.split("_")[0].replace("cap", "")
                        if has(cm):
                            dicts[f"D-M[STE->hidden {hname}]"] = ([dm[y] for y in window] + [cm], window + ["early"])
                if has(uf):
                    dicts[f"D-S[m0->early:{tag}]"] = (ds[1:] + [uf], [f"seed{i}" for i in range(1, 5)] + ["early"])
                    dicts[f"D-S+early:{tag}"] = (ds + [uf], [f"seed{i}" for i in range(5)] + ["early"])
            for dn, (mids, labels) in dicts.items():
                Pc = [pred(m, cal_role, cond) for m in mids]; Pt = [pred(m, test_role, cond) for m in mids]
                C = cal.residual_matrix(Pc, Yc, one, gc)[0]
                w = cal.budgeted_qp(C, min(5, len(mids)))["w"]
                P = sum(wj * p for wj, p in zip(w, Pt))
                r = nmse_rows(P, Yt, one)
                u = pd.Series(r).groupby(gt).mean().values if gt is not None else r
                rows.append({"seed": seed, "cond": lab, "dictionary": dn, "nmse": float(r.mean()), "support_size": int(np.sum(w > 1e-12))})
                units[f"{seed}|{lab}|{dn}"] = u
                for l, x in zip(labels, w):
                    wrows.append({"seed": seed, "cond": lab, "dictionary": dn, "member": l, "weight": float(x)})
                if "early" in labels:
                    rows.append({"seed": seed, "cond": lab, "dictionary": dn + " | early member alone", "nmse": float(nmse(Pt[-1], Yt, one)), "support_size": 1})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.out, "e14_rows.csv"), index=False)
    pd.DataFrame(wrows).to_csv(os.path.join(args.out, "e14_weights.csv"), index=False)
    summ = []
    for (lab, dn), sub in df.groupby(["cond", "dictionary"], sort=False):
        if "alone" in dn:
            summ.append({"cond": lab, "dictionary": dn, "nmse_mean": sub.nmse.mean(), "ci_lo": np.nan, "ci_hi": np.nan, "n_seeds": len(sub)}); continue
        vals = [units[f"{int(s)}|{lab}|{dn}"] for s in sub.seed]
        v = np.concatenate(vals); cs = np.concatenate([np.full(len(u), s) for u, s in zip(vals, sub.seed)]); cu = np.concatenate([np.arange(len(u)) for u in vals])
        pt, l, h = cluster_bootstrap(v, {"seed": cs, "unit": cu}, n_boot=args.n_boot)
        summ.append({"cond": lab, "dictionary": dn, "nmse_mean": pt, "ci_lo": l, "ci_hi": h, "n_seeds": len(vals)})
    pd.DataFrame(summ).to_csv(os.path.join(args.out, "e14_summary.csv"), index=False)
    prs = []
    names = list(dict.fromkeys(df.dictionary))
    for lab in dict.fromkeys(df.cond):
        for a in names:
            if "alone" in a or a in ("D-M", "D-S", "D-M-STE"):
                continue
            for b in ["D-M", "D-S", "D-M-STE"]:
                diffs = [units[f"{s}|{lab}|{a}"] - units[f"{s}|{lab}|{b}"] for s in range(5) if f"{s}|{lab}|{a}" in units and f"{s}|{lab}|{b}" in units]
                if not diffs:
                    continue
                v = np.concatenate(diffs); cs = np.concatenate([np.full(len(x), i) for i, x in enumerate(diffs)]); cu = np.concatenate([np.arange(len(x)) for x in diffs])
                pt, l, h = cluster_bootstrap(v, {"seed": cs, "unit": cu}, n_boot=args.n_boot, seed=1)
                prs.append({"cond": lab, "a": a, "b": b, "diff": pt, "ci_lo": l, "ci_hi": h, "seeds_a_better": int(sum(np.mean(x) < 0 for x in diffs)), "n_seeds": len(diffs)})
    pd.DataFrame(prs).to_csv(os.path.join(args.out, "e14_paired.csv"), index=False)
    print("E14_DONE", len(curve), "curve rows;", len(rows), "dictionary rows")


if __name__ == "__main__":
    main()
