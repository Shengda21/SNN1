"""E23 inputs: moments for label-free calibration (no target labels).

For every target condition the script
  1. builds the unlabeled target inputs of the calibration role and estimates the
     per-channel noise variance with eq. (S5): v_hat = max(0, var(x_tilde_cal) - var(x_train));
  2. estimates the variance from a 'static robot' recording: one nominal calibration row
     with the smallest |qdot| held for 1000 steps (10 s at 100 Hz), observed through the
     target noise, v_static = per-channel variance of that recording;
  3. writes calibration moments of the NOMINAL labeled calibration role with simulated
     diagonal Gaussian noise of variance v (sources S2: v_hat, S3: true variance,
     S5: v_static; on O2B also S2b: a per-trajectory bias of variance v_hat);
  4. for the misspecified true noises (t3: Student t, nu = 3, scaled to unit variance;
     ar1: Gaussian with inter-channel correlation 0.5^|i-j| over the noisy channels),
     writes the moments of the labeled target calibration role and of the test role.
Members: the core set (D-M, D-S, D-T, B-L, B-A, B-N). Moments as in e2x_dump_paired.

python e23_dump.py --platform sim --data_dir ../data_sim --runs ../runs25/sim --out ../results/sim
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2x_common import (SEEDS, SIM_SEED, MIS_SEED, STATIC_SEED, noise_mask, member_blocks, flat_members,
                        unit_layout, compute_moments, save_moments, cname)
from common.models import predict
from dump_preds import load_member
from run_train import load_platform

torch.set_num_threads(4)
KIND_CODE = {"O1": 11, "O1qqd": 23, "O2B": 37, "P1": 41, "t3": 53, "ar1": 67}


def targets(platform):
    if platform == "sim":
        return (["O1:0.35", "O1:0.7", "O1:1.0", "O2B:0.7", "P1"]
                + [f"{k}:{r}" for k in ("t3", "ar1") for r in ("0.35", "0.7", "1.0")])
    return (["O1:0.25", "O1:0.5", "O1:0.7", "O1qqd:0.5"]
            + [f"{k}:{r}" for k in ("t3", "ar1") for r in ("0.5", "0.7")])


def corr_chol(mask, r=0.5):
    idx = np.where(mask > 0)[0]
    k = len(idx)
    S = r ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    return idx, np.linalg.cholesky(S)


def mis_noise(kind, shape, mask, seed):
    """Unit-variance misspecified noise on the masked channels."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape)
    idx = np.where(mask > 0)[0]
    if kind == "t3":
        out[:, idx] = rng.standard_t(3, size=(shape[0], len(idx))) / np.sqrt(3.0)
    elif kind == "ar1":
        idx, L = corr_chol(mask)
        out[:, idx] = rng.standard_normal((shape[0], len(idx))) @ L.T
    else:
        raise ValueError(kind)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
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
    groups = {s: flat_members(member_blocks(args.platform, s, "core")) for s in SEEDS}
    models = {m: load_member(args.runs, m, n_in, d, device) for s in groups for m in groups[s]}
    mask = noise_mask(args.platform, n_in)
    v_train = D["X"]["train"].astype(np.float64).var(0)
    odir = os.path.join(args.out, "e23")
    os.makedirs(odir, exist_ok=True)

    def dump(role, Xn, label, meta):
        fn = os.path.join(odir, f"moments_{role}_{label}.npz")
        if os.path.exists(fn) and not args.overwrite:
            return fn
        Y = D["Y"][role].astype(np.float64)
        sub, parent = unit_layout(args.platform, D, role, seg=1)
        out = {}
        for s, names in groups.items():
            P = np.stack([clip(predict(models[m], Xn, device, batch=32768).astype(np.float64)) for m in names])
            mo = compute_moments(P, Y, sub)
            out[s] = {"names": names, "FF": mo["FF"], "FY": mo["FY"]}
            n, YY = mo["n"], mo["YY"]
        meta = dict(meta, platform=args.platform, role=role, label=label, seeds=SEEDS, seg=1)
        save_moments(fn, meta, n, parent, YY, out)
        print("wrote", fn, f"{time.time() - t0:.0f}s", flush=True)
        return fn

    # static recording: the nominal calibration row with the smallest |qdot|
    Xc = D["X"]["cal"].astype(np.float64)
    i_static = int(np.argmin(np.linalg.norm(Xc[:, 7:14] * D["x_std"][7:14], axis=1)))
    x_static = Xc[i_static]
    Xcal_clean = D["X"]["cal"]
    xi_sim = np.random.default_rng(SIM_SEED["cal"]).standard_normal(Xcal_clean.shape)
    bias_sim = np.random.default_rng(SIM_SEED["cal"] + 1).standard_normal((int(np.max(D["groups"]["cal"])) + 1 if sim else 1, n_in))
    info = {"static_row": i_static, "static_qd_norm_phys": float(np.linalg.norm(Xc[i_static, 7:14] * D["x_std"][7:14])),
            "targets": {}}
    sim_cache = {}
    for tg in targets(args.platform):
        if tg == "P1":
            kind, rho = "P1", 0.0
        else:
            kind, rho = tg.split(":"); rho = float(rho)
        # ---- unlabeled target calibration inputs and the true noise variance
        if kind in ("O1", "O2B", "O1qqd"):
            Xt_cal = apply_condition(D["X"]["cal"], "cal", tg).astype(np.float64)
            v_true = np.zeros(n_in)
            if args.platform == "sarcos" and kind == "O1":
                v_true[:] = rho ** 2
            else:
                v_true[:14] = rho ** 2
        elif kind == "P1":
            Xt_cal = D["X"]["cal_P1"].astype(np.float64)
            v_true = np.zeros(n_in)
        else:
            nz = {role: mis_noise(kind, D["X"][role].shape, mask, MIS_SEED[role] + (0 if kind == "t3" else 100) + int(round(rho * 1000)))
                  for role in ("cal", "test")}
            Xt = {role: (D["X"][role].astype(np.float64) + rho * nz[role]).astype(np.float32) for role in ("cal", "test")}
            Xt_cal = Xt["cal"].astype(np.float64)
            v_true = rho ** 2 * mask
            for role in ("cal", "test"):
                dump(role, Xt[role], f"{kind}_rho{rho:.2f}", {"spec": {"kind": kind, "rho": rho}})
        v_hat = np.maximum(Xt_cal.var(0) - v_train, 0.0)
        # ---- static recording through the target noise
        Tst = 1000
        rng = np.random.default_rng(STATIC_SEED + int(round(rho * 1000)) + KIND_CODE[kind])
        if kind in ("O1", "O1qqd"):
            m = mask.copy()
            if kind == "O1qqd":
                m[:] = 0; m[:14] = 1
            st = x_static + rho * rng.standard_normal((Tst, n_in)) * m
        elif kind == "O2B":
            st = x_static + rho * rng.standard_normal((1, n_in)) * mask   # one bias draw for the whole recording
            st = np.repeat(st, Tst, 0)
        elif kind == "P1":
            st = np.repeat(x_static[None], Tst, 0)
        else:
            st = x_static + rho * mis_noise(kind, (Tst, n_in), mask, STATIC_SEED + 7 + int(round(rho * 1000)))
        v_static = st.var(0)
        info["targets"][tg] = {"v_hat": v_hat.tolist(), "v_true": v_true.tolist(), "v_static": v_static.tolist(),
                               "rho_hat_mask_mean": float(np.sqrt(np.mean(v_hat[mask > 0]))),
                               "rho_static_mask_mean": float(np.sqrt(np.mean(v_static[mask > 0])))}
        # ---- simulated calibration sets on the nominal labeled calibration role
        srcs = {"S2": v_hat, "S3": v_true, "S5": v_static}
        for src, v in srcs.items():
            if not np.any(v > 0):
                info["targets"][tg][f"{src}_file"] = "orig/moments_cal_N.npz"
                continue
            key = hashlib.md5(np.round(v, 12).tobytes()).hexdigest()[:10]
            label = f"sim_{key}"
            Xs = (Xcal_clean.astype(np.float64) + np.sqrt(v) * xi_sim).astype(np.float32)
            dump("cal", Xs, label, {"spec": {"kind": "simulated_gaussian", "v": v.tolist()}})
            info["targets"][tg][f"{src}_file"] = f"e23/moments_cal_{label}.npz"
        if kind == "O2B":
            v = v_hat
            key = hashlib.md5(np.round(v, 12).tobytes()).hexdigest()[:10]
            label = f"simbias_{key}"
            g = D["groups"]["cal"]
            Xs = (Xcal_clean.astype(np.float64) + np.sqrt(v) * bias_sim[g]).astype(np.float32)
            dump("cal", Xs, label, {"spec": {"kind": "simulated_bias", "v": v.tolist()}})
            info["targets"][tg]["S2b_file"] = f"e23/moments_cal_{label}.npz"
        json.dump(info, open(os.path.join(odir, "e23_variances.json"), "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
