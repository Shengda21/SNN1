"""E11: label-free linear input front end for frozen members.

The members stay frozen. Before a perturbed input reaches them it is passed through
the linear minimum-mean-square-error estimator of the clean input,

    x_hat = mu + W (x_tilde - mu),   W = Sigma_x (Sigma_x + Sigma_delta)^{-1},

where mu and Sigma_x are the mean and covariance of the *training* inputs (standardized
units) and Sigma_delta is a diagonal noise covariance. Two variants are dumped:

  wfe  Sigma_delta estimated without labels from the calibration inputs of the target
       condition: per channel max(0, var(x_tilde_cal) - var(x_train)).
  wfo  oracle Sigma_delta = rho^2 on the perturbed channels (reference only).

Predictions are written next to the ordinary ones as pred_<role>_<cond>__<variant>.npy.
The script also writes the input geometry used in the mechanism analysis: eigenvalues
of Sigma_x, the estimated noise variances, the fraction of input-noise power that the
linear estimator removes (analytic and measured against the clean test inputs).

python frontend_preds.py --platform sarcos --data_dir .. --runs ../runs/sarcos \
       --frozen ../runs/sarcos/frozen.json --out ../results/sarcos/e11_frontend_geometry.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models import predict
from dump_preds import load_member
from run_train import load_platform

PREFIXES = ("dm_", "ds_", "ba_", "bn_", "dma_", "dsa_", "bna_", "dmq_", "dsq_", "bnq_", "dmas_", "dmqs_", "bnas_", "bnqs_")


def cname(c):
    return c.replace(":", "_")


def oracle_var(platform, cond, n_in):
    """Per-channel second moment of the perturbation in standardized units."""
    v = np.zeros(n_in)
    if cond == "N":
        return v
    kind, rho = cond.split(":")
    rho = float(rho)
    if platform == "sarcos" and kind == "O1":
        v[:] = rho ** 2
    else:                       # O1qqd (sarcos), O1 / O2B (sim): q and qdot channels
        v[:14] = rho ** 2
    return v


def wiener(Sig, v):
    return Sig @ np.linalg.inv(Sig + np.diag(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, apply_condition = load_platform(args.platform, args.data_dir)
    frozen = json.load(open(args.frozen))
    sim = args.platform == "sim"
    n_in = D["n_in"]
    Xtr = D["X"]["train"].astype(np.float64)
    mu = Xtr.mean(0)
    Sig = np.cov(Xtr.T)
    v_train = Xtr.var(0)
    ev = np.sort(np.linalg.eigvalsh(Sig))[::-1]
    geo = {"platform": args.platform, "eigenvalues_Sigma_x": ev.tolist(),
           "participation_ratio": float(ev.sum() ** 2 / np.sum(ev ** 2)),
           "n_eig_for_90pct": int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90) + 1),
           "n_eig_for_99pct": int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.99) + 1),
           "conditions": {}}
    cases = [("cal", "test", c) for c in frozen["conds"]]
    if sim:
        cases.append(("cal_P1", "test_P1", "N"))
    mats = {}
    for cal_role, test_role, cond in cases:
        lab = "P1" if cal_role.endswith("_P1") else cond
        Xc_cal = apply_condition(D["X"][cal_role], cal_role, cond).astype(np.float64)
        Xc_test = apply_condition(D["X"][test_role], test_role, cond).astype(np.float64)
        v_est = np.maximum(Xc_cal.var(0) - v_train, 0.0)
        v_orc = oracle_var(args.platform, cond, n_in)
        W = {"wfe": wiener(Sig, v_est), "wfo": wiener(Sig, v_orc)}
        ent = {"v_est": v_est.tolist(), "v_oracle": v_orc.tolist()}
        Xclean = D["X"][test_role].astype(np.float64)
        raw = float(np.mean(np.sum((Xc_test - Xclean) ** 2, 1)))
        for var, Wm in W.items():
            Xh = mu + (Xc_test - mu) @ Wm.T
            err = float(np.mean(np.sum((Xh - Xclean) ** 2, 1)))
            ent[var] = {"input_mse_raw": raw, "input_mse_denoised": err,
                        "removed_fraction_measured": (1.0 - err / raw) if raw > 0 else 0.0}
        if v_orc.sum() > 0:
            mmse = float(np.trace(Sig @ np.linalg.inv(Sig + np.diag(v_orc)) @ np.diag(v_orc)))
            ent["removed_fraction_analytic_oracle"] = 1.0 - mmse / float(v_orc.sum())
            # the same quantity if the inputs were uncorrelated (diagonal Sigma_x)
            dg = np.diag(Sig)
            ent["removed_fraction_if_uncorrelated"] = 1.0 - float(np.sum(dg * v_orc / (dg + v_orc))) / float(v_orc.sum())
        geo["conditions"][lab] = ent
        mats[(cal_role, test_role, cond)] = (W, Xc_cal, Xc_test)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(geo, f, indent=1)
    print("geometry written:", {k: round(v.get("removed_fraction_analytic_oracle", 0.0), 3) for k, v in geo["conditions"].items()}, flush=True)

    dirs = [d for d in sorted(glob.glob(os.path.join(args.runs, "*")))
            if os.path.basename(d).startswith(PREFIXES) and os.path.exists(os.path.join(d, "config.json"))]
    for d in dirs:
        model = None
        for (cal_role, test_role, cond), (W, Xc_cal, Xc_test) in mats.items():
            for var, Wm in W.items():
                for role, Xc in ((cal_role, Xc_cal), (test_role, Xc_test)):
                    fn = os.path.join(d, f"pred_{role}_{cname(cond)}__{var}.npy")
                    if os.path.exists(fn) and not args.overwrite:
                        continue
                    if model is None:
                        model = load_member(args.runs, os.path.basename(d), n_in, D["n_out"], device)
                    Xh = (mu + (Xc - mu) @ Wm.T).astype(np.float32)
                    np.save(fn, predict(model, Xh, device).astype(np.float32))
        print("done", os.path.basename(d), flush=True)


if __name__ == "__main__":
    main()
