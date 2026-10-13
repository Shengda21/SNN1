"""Actual forward perturbation response of each frozen member.
For dev inputs x and random perturbations delta on the (q, qd) channels at the
O1-M scale, measure E||f(x + delta) - f(x)||_2 / ||delta||_2 (standardized units)
per member by direct finite differences of the hard-spiking forward pass. This is
the measured sensitivity of the deployed map, not the surrogate Jacobian.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_preds import load_member
from run_train import load_platform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--frozen", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, _ = load_platform(args.platform, args.data_dir)
    frozen = json.load(open(args.frozen))
    rho = frozen["O1_M"]
    rng = np.random.default_rng(0)
    X = D["X"]["dev"][rng.choice(len(D["X"]["dev"]), args.n, replace=False)]
    mask = np.zeros(D["n_in"], np.float32)
    mask[:14] = 1
    if args.platform == "sarcos":
        mask[:] = 1
    delta = rng.standard_normal(X.shape).astype(np.float32) * rho * mask
    out = {}
    for d in sorted(glob.glob(os.path.join(args.runs, "*_s*"))):
        mid = os.path.basename(d)
        if mid.startswith("width") or not os.path.exists(os.path.join(d, "config.json")):
            continue
        model = load_member(args.runs, mid, D["n_in"], D["n_out"], device)
        with torch.no_grad():
            f0 = model(torch.as_tensor(X, device=device))[-1].cpu().numpy()
            f1 = model(torch.as_tensor(X + delta, device=device))[-1].cpu().numpy()
        num = np.linalg.norm(f1 - f0, axis=1); den = np.linalg.norm(delta, axis=1)
        out[mid] = {"response_ratio_mean": float(np.mean(num / den)), "response_ratio_median": float(np.median(num / den)),
                    "output_change_rms": float(np.sqrt(np.mean((f1 - f0) ** 2)))}
    with open(args.out, "w") as f:
        json.dump({"rho": rho, "n": args.n, "members": out}, f, indent=1)
    # summary by block/surrogate
    agg = {}
    for mid, v in out.items():
        key = mid.split("_s")[0]
        agg.setdefault(key, []).append(v["response_ratio_mean"])
    print({k: (round(float(np.mean(v)), 4), len(v)) for k, v in agg.items()})


if __name__ == "__main__":
    main()
