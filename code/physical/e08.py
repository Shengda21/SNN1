"""E08: re-analysis of the 151-sample physical 2-DOF recording with the
released code. The record stores six joint angles and six
controller torque estimates per sample without timestamps; joints 4 and 5 of the
file are the two active joints. Velocities are first differences of the angle
sequence (angle increments per sample; no sampling interval is stored, so they are
not physical velocities). Chronological 70/30 split (105 / 46 samples); the
held-out partition is shared by weight fitting and reporting, so every ensemble
number is in-sample with respect to its weights.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.models import SNN
from common.surrogates import SURROGATE_NAMES
from common.train import set_seed
from common import calibration as cal
from common.metrics import r2_pooled, nmse

SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]


def load(data_dir, split="chrono"):
    A = np.loadtxt(os.path.join(data_dir, "arr6.txt")); T = np.loadtxt(os.path.join(data_dir, "tau_arr6.txt"))
    q = A[:, 3:5]; dq = np.vstack([np.zeros((1, 2)), np.diff(q, axis=0)])
    X = np.hstack([q, dq]); Y = T[:, 3:5]
    n_tr = int(0.7 * len(X))
    if split == "random":
        perm = np.random.default_rng(0).permutation(len(X))
        tr, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
        return X[tr], Y[tr], X[te], Y[te]
    return X[:n_tr], Y[:n_tr], X[n_tr:], Y[n_tr:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--norm", default="train", choices=["train", "full"],
                    help="normalization statistics: train partition only (leak-free) or the full record (the earlier pipeline, whose statistics include the held-out samples)")
    ap.add_argument("--split", default="chrono", choices=["chrono", "random"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Ytr, Xte, Yte = load(args.data_dir, args.split)
    if args.norm == "train":
        xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8; ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-8
    else:
        Xall, Yall = np.vstack([Xtr, Xte]), np.vstack([Ytr, Yte])
        xm, xs = Xall.mean(0), Xall.std(0) + 1e-8; ym, ys = Yall.mean(0), Yall.std(0) + 1e-8
    Xtr_n, Xte_n = (Xtr - xm) / xs, (Xte - xm) / xs; Ytr_n, Yte_n = (Ytr - ym) / ys, (Yte - ym) / ys
    out = {"n_train": len(Xtr), "n_test": len(Xte), "epochs": args.epochs, "norm": args.norm, "split": args.split, "seeds": SEEDS, "rows": [],
           "test_target_offset_in_train_std": ((Yte.mean(0) - Ytr.mean(0)) / (Ytr.std(0) + 1e-8)).tolist(),
           "test_input_offset_in_train_std": ((Xte.mean(0) - Xtr.mean(0)) / (Xtr.std(0) + 1e-8)).tolist()}
    for seed in SEEDS:
        preds = {}
        for s in SURROGATE_NAMES:
            set_seed(seed)
            model = SNN(4, 2, surrogate=s).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            xb = torch.as_tensor(Xtr_n, dtype=torch.float32, device=device); yb = torch.as_tensor(Ytr_n, dtype=torch.float32, device=device)
            for ep in range(args.epochs):
                model.train(); o = model(xb); loss = ((o - yb.unsqueeze(0)) ** 2).mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                preds[s] = model(torch.as_tensor(Xte_n, dtype=torch.float32, device=device))[-1].cpu().numpy().astype(np.float64)
        P = [preds[s] for s in SURROGATE_NAMES]
        C, _, _ = cal.residual_matrix(P, Yte_n, np.ones(2), None)
        mse = [nmse(p, Yte_n, np.ones(2)) for p in P]
        wB = cal.rule_WB(mse, Yte_n.size); wE = cal.rule_WE(C); wQ = cal.budgeted_qp(C, 5); wit = cal.two_member_witness(C)
        row = {"seed": seed, "single_r2": {s: r2_pooled(preds[s], Yte_n) for s in SURROGATE_NAMES},
               "single_nmse": dict(zip(SURROGATE_NAMES, mse)),
               "best_single_nmse": float(min(mse)), "equal_nmse": cal.risk(wE, C), "bic_nmse": cal.risk(wB, C),
               "wq5_insample_nmse": wQ["value"], "wq5_support": wQ["support"], "bic_w": wB.round(3).tolist(),
               "witness_exists": wit["exists"], "C": C.tolist()}
        out["rows"].append(row)
        print(seed, {k: round(v, 3) for k, v in row["single_r2"].items()}, "best", round(row["best_single_nmse"], 4),
              "equal", round(row["equal_nmse"], 4), "bic", round(row["bic_nmse"], 4), "wq5", round(row["wq5_insample_nmse"], 4), flush=True)
    agg = {}
    for key in ["best_single_nmse", "equal_nmse", "bic_nmse", "wq5_insample_nmse"]:
        v = np.array([r[key] for r in out["rows"]]); agg[key] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
    for s in SURROGATE_NAMES:
        v = np.array([r["single_r2"][s] for r in out["rows"]]); agg[f"r2_{s}"] = {"mean": float(v.mean()), "std": float(v.std(ddof=1))}
    agg["witness_frac"] = float(np.mean([r["witness_exists"] for r in out["rows"]]))
    out["aggregate"] = agg
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
