"""E15 items 1-3, 6, 7, 9: position of every frozen member in the
signal-gain / orthogonal-noise plane (zero training).

$PY e15_position.py --platform sim [--runs ../runs25/sim] [--members '*'] [--suffix '']

Per member x role x condition (from the stored prediction files; predictions are
recomputed with the recorded forward only when a file is missing):
  F = pred/sqrt(d), Y = y/sqrt(d) in standardized units (sim predictions clipped to the
  torque limits exactly as in analysis_aug.py); <U,V> = trajectory average (sim) or row
  average (SARCOS).
  R, beta = <F,Y>/P_y, nu2 = ||F||^2 - beta^2 P_y, slope s = <F,Y>/||F||^2,
  R_angle = P_y sin^2(theta), scale_cost = ||F||^2 (1-s)^2, residuals of identities I1, I2;
  E10 split: P_n = <n,n>, P_b = <b,b>, cross = 2<n,b>, n = F(x~) - F(x), b = F(x) - Y;
  kappa = slope of log(R(rho) - R(0)) on log(rho) over the platform O1 levels.
Per member (label-free):
  r      forward response of forward_response.py (same dev rows, seed and O1-M scale);
  r_cond same perturbation draw scaled to the rho and channel mask of each O1 condition;
  direction-resolved responses (top-5 / bottom-5 principal directions of the training
  input covariance, equal total energy = O1-M energy) -> e15_direction.csv;
  weight statistics against the seed-reproduced initialization -> e15_weights.csv;
  learned ATan widths of B-L -> e15_bl_widths.csv.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e15_common import (PLATFORM_DEFAULTS, load_data, cond_fname, cond_kind, cond_rho, cond_mask,  # noqa: E402
                        discover_conds, InnerProduct, hwhm_of, cfg_fields, member_prefix, member_group, outer_seed,
                        list_members, load_member, build_model, predict_np, dev_subset_and_delta,
                        response_ratio, forward_record, train_input_moments, out_name)
from common.train import set_seed  # noqa: E402


# ----------------------------------------------------------------------------- scalars
def plane_scalars(F, Y, ip):
    """All plane quantities for one predictor (inputs already divided by nothing: the
    InnerProduct divides by d, i.e. works with F = pred/sqrt(d))."""
    Py = ip(Y, Y); Fn2 = ip(F, F); FY = ip(F, Y)
    E = F - Y
    R = ip(E, E)
    beta = FY / Py
    nu2 = Fn2 - beta ** 2 * Py
    nu = F - beta * Y
    nu2_direct = ip(nu, nu)
    slope = FY / Fn2 if Fn2 > 0 else float("nan")
    cos2 = FY ** 2 / (Fn2 * Py) if Fn2 > 0 else float("nan")
    R_angle = Py * (1.0 - cos2)
    scale_cost = Fn2 * (1.0 - slope) ** 2
    return {"Py": Py, "Fnorm2": Fn2, "FY": FY, "R": R, "beta": beta, "nu2": nu2,
            "nu2_direct": nu2_direct, "nu_orth_check": ip(nu, Y), "slope": slope, "cos2": cos2,
            "R_angle": R_angle, "scale_cost": scale_cost,
            "I1_resid": R - ((1 - beta) ** 2 * Py + nu2_direct),
            "I2_resid": R - (R_angle + scale_cost)}


def fit_kappa(levels, excess):
    lv = np.asarray(levels, float); ex = np.asarray(excess, float)
    ok = ex > 0
    if ok.sum() < 2:
        return float("nan"), int(ok.sum())
    k = np.polyfit(np.log(lv[ok]), np.log(ex[ok]), 1)[0]
    return float(k), int(ok.sum())


# ----------------------------------------------------------------------------- directions
def direction_setup(D, platform, rho_M, n_rows, mask_fr):
    mu, Sigma = train_input_moments(D)
    E_total = float(rho_M ** 2 * mask_fr.sum())
    lam21, V21 = np.linalg.eigh(Sigma)
    lam14, V14 = np.linalg.eigh(Sigma[:14, :14])
    V14e = np.zeros((21, 14)); V14e[:14] = V14
    rng = np.random.default_rng(15)
    z_top = rng.standard_normal((n_rows, 5)); z_bot = rng.standard_normal((n_rows, 5))
    z_iso21 = rng.standard_normal((n_rows, 21)); z_iso14 = rng.standard_normal((n_rows, 14))
    zeta = rng.standard_normal(n_rows)
    pert = []   # (variant, subspace, dir_index, mean eigenvalue, delta)
    for variant, lam, V, k, ziso in [("full21", lam21, V21, 21, z_iso21), ("qqd14", lam14, V14e, 14, z_iso14)]:
        pert.append((variant, "top5", -1, float(lam[-5:].mean()), (z_top * math.sqrt(E_total / 5)) @ V[:, -5:].T))
        pert.append((variant, "bottom5", -1, float(lam[:5].mean()), (z_bot * math.sqrt(E_total / 5)) @ V[:, :5].T))
        iso = np.zeros((n_rows, 21))
        if k == 21:
            iso[:] = ziso * math.sqrt(E_total / 21)
        else:
            iso[:, :14] = ziso * math.sqrt(E_total / 14)
        pert.append((variant, "iso", -1, float(lam.mean()), iso))
        for i in range(k):
            pert.append((variant, "dir", i, float(lam[i]), zeta[:, None] * math.sqrt(E_total) * V[:, i][None, :]))
    meta = {"E_total": E_total, "lam21": lam21.tolist(), "lam14": lam14.tolist(),
            "participation_ratio_21": float(lam21.sum() ** 2 / (lam21 ** 2).sum()),
            "participation_ratio_14": float(lam14.sum() ** 2 / (lam14 ** 2).sum())}
    return [(v, s, i, l, d.astype(np.float32)) for v, s, i, l, d in pert], meta


def direction_responses(model, kind, X, pert, device):
    rows = []
    with torch.no_grad():
        f0, _ = forward_record(model, kind, torch.as_tensor(X, device=device), record=False)
        f0 = f0[-1].cpu().numpy()
        for variant, sub, i, lam, delta in pert:
            f1, _ = forward_record(model, kind, torch.as_tensor(X + delta, device=device), record=False)
            f1 = f1[-1].cpu().numpy()
            num = np.linalg.norm(f1 - f0, axis=1); den = np.linalg.norm(delta, axis=1)
            ok = den > 0
            rows.append({"variant": variant, "subspace": sub, "dir_index": i, "eig": lam,
                         "r": float(np.mean(num[ok] / den[ok])), "out_rms": float(np.sqrt(np.mean((f1 - f0) ** 2))),
                         "delta_rms": float(np.sqrt(np.mean(delta ** 2)))})
    return rows


# ----------------------------------------------------------------------------- weights
def _svd_stats(W):
    W = W.astype(np.float64)
    fro = float(np.linalg.norm(W))
    spec = float(np.linalg.norm(W, 2)) if W.size else float("nan")
    return fro, spec, (fro ** 2 / spec ** 2 if spec > 0 else float("nan"))


def weight_rows(mid, group, cfg, model, kind, sd_cpu):
    n_in = sd_cpu["fc1.weight"].shape[1] if "fc1.weight" in sd_cpu else sd_cpu["net.0.weight"].shape[1]
    n_out = sd_cpu["fc5.weight"].shape[0] if "fc5.weight" in sd_cpu else sd_cpu["net.4.weight"].shape[0]
    seed = cfg.get("seed", outer_seed(mid))
    set_seed(int(seed))
    m0, _ = build_model(cfg, sd_cpu, n_in, n_out)
    sd0 = {k: v.detach().cpu().numpy() for k, v in m0.state_dict().items()}
    names = ["fc1", "fc2", "fc3", "fc4", "fc5"] if kind != "mlp" else ["net.0", "net.2", "net.4"]
    rows = []
    for nm in names:
        W = sd_cpu[nm + ".weight"].numpy(); W0 = sd0[nm + ".weight"]
        fro, spec, sr = _svd_stats(W); fro0, spec0, sr0 = _svd_stats(W0); dfro, dspec, srd = _svd_stats(W - W0)
        r = {"member": mid, "group": group, "layer": nm, "init_seed": seed, "surrogate": cfg.get("surrogate"),
             "detach_reset": cfg.get("detach_reset", cfg.get("reset_detach", False)), "fro": fro, "fro0": fro0,
             "dfro": dfro, "rel_change": dfro / fro0, "spec": spec, "spec0": spec0, "stable_rank": sr,
             "stable_rank0": sr0, "stable_rank_delta": srd, "spec_delta": dspec}
        lif = {"fc1": "lif1", "fc2": "lif2", "fc3": "lif3", "fc4": "li4", "fc5": "li5"}.get(nm)
        if lif is not None and kind != "mlp":
            b = sd_cpu[lif + ".beta"].clamp(0, 1).numpy(); th = sd_cpu[lif + ".thr"].numpy()
            r.update({"beta_mean": float(b.mean()), "beta_std": float(b.std()),
                      "thr_mean": float(th.mean()), "thr_std": float(th.std()),
                      "thr_min": float(th.min())})
            if lif.startswith("lif"):
                # non-detached subtractive reset: with g = 1 (STE) the membrane credit is scaled by (1 - thr)
                r.update({"thr_frac_near1": float(np.mean(np.abs(th - 1.0) < 0.02)),
                          "thr_mean_abs_1m": float(np.mean(np.abs(1.0 - th))),
                          "thr_frac_above1": float(np.mean(th > 1.0))})
        rows.append(r)
    return rows


def traj_init_check(runs):
    """traj_*/states.pt[0] is the state after the first epoch (train.py appends after each
    epoch). It should lie close to the init reproduced with the run seed and far from the
    init of seed+1; this verifies the seed-based reproduction of W_0."""
    from common.models import SNN
    rows = []
    for d in sorted(glob.glob(os.path.join(runs, "traj_*"))):
        sp = os.path.join(d, "states.pt"); hp = os.path.join(d, "history.json")
        if not (os.path.exists(sp) and os.path.exists(hp)):
            continue
        h = json.load(open(hp))
        st = torch.load(sp, map_location="cpu")
        s0 = st[0]
        n_in, n_out = s0["fc1.weight"].shape[1], s0["fc5.weight"].shape[0]
        res = {}
        for tag, sd_seed in [("correct", h["seed"]), ("wrong", h["seed"] + 1)]:
            set_seed(int(sd_seed))
            m0 = SNN(n_in, n_out, surrogate=h["surrogate"], width=h["width"])
            res[tag] = {k: v.numpy() for k, v in m0.state_dict().items()}
        for nm in ["fc1", "fc2", "fc3", "fc4", "fc5"]:
            W = s0[nm + ".weight"].numpy()
            rows.append({"member": os.path.basename(d), "group": "traj-init-check", "layer": nm,
                         "init_seed": h["seed"], "fro": float(np.linalg.norm(W)),
                         "check_rel_dist_epoch0_vs_init_seed": float(np.linalg.norm(W - res["correct"][nm + ".weight"]) / np.linalg.norm(res["correct"][nm + ".weight"])),
                         "check_rel_dist_epoch0_vs_init_seed_plus1": float(np.linalg.norm(W - res["wrong"][nm + ".weight"]) / np.linalg.norm(res["wrong"][nm + ".weight"]))})
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--members", default="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--roles", default="test,cal")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--no_traj_check", action="store_true")
    args = ap.parse_args()
    P = PLATFORM_DEFAULTS[args.platform]
    data_dir = args.data_dir or P["data_dir"]
    runs = args.runs or f"../runs25/{args.platform}"
    out_dir = args.out or f"../results/{args.platform}"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    D, apply_condition, clip = load_data(args.platform, data_dir)
    d = D["n_out"]
    sim = args.platform == "sim"
    roles = args.roles.split(",")
    rho_M = P["rho_M"]
    idx, Xr, delta_M, xi, mask_fr = dev_subset_and_delta(D, args.platform, rho_M, args.n)
    pert, dmeta = direction_setup(D, args.platform, rho_M, args.n, mask_fr)
    ips = {r: InnerProduct(D["groups"].get(r), d) for r in D["X"] if r != "train"}
    cond_cache = {}

    def cond_input(role, cond):
        key = (role, cond)
        if key not in cond_cache:
            cond_cache[key] = apply_condition(D["X"][role], role, cond)
        return cond_cache[key]

    members = list_members(runs, args.members)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    log(f"{len(members)} members in {runs} matching {args.members}")
    mrows, drows, wrows, blrows = [], [], [], []
    t0 = time.time()
    for mi, mdir in enumerate(members):
        mid = os.path.basename(mdir)
        try:
            model, cfg, kind = load_member(mdir, device)
        except Exception as e:  # noqa: BLE001
            log(f"skip {mid}: {e}")
            continue
        sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        group = member_group(mid, cfg)
        base = {"platform": args.platform, "member": mid, "prefix": member_prefix(mid), "group": group,
                "surrogate": cfg.get("surrogate"), "width": cfg.get("width"), "hwhm": hwhm_of(cfg),
                "seed": cfg.get("seed"), "outer_seed": outer_seed(mid), "hidden": cfg.get("hidden", 128 if kind != "mlp" else 220),
                "aug_sigma": cfg.get("aug_sigma", 0.0), "best_epoch": cfg.get("best_epoch"), "model_kind": kind}
        cfg.setdefault("member_id", mid)
        base.update(cfg_fields(cfg))
        # forward reproduction check against the stored nominal test predictions
        fchk = float("nan"); fchk_rows = float("nan")
        fp = os.path.join(mdir, "pred_test_N.npy")
        if os.path.exists(fp):
            Pst = np.load(fp)
            Pme = predict_np(model, kind, D["X"]["test"], device)
            dd = np.abs(Pst.astype(np.float64) - Pme).max(1)
            fchk = float(dd.max()); fchk_rows = float(np.mean(dd > 1e-4))
        base.update({"fwd_check_maxdiff": fchk, "fwd_check_frac_rows_gt1e-4": fchk_rows})
        # forward response (forward_response.py definition) and per-condition variant
        r_M, outrms_M = response_ratio(model, kind, Xr, delta_M, device)
        r_cache = {}

        def r_for(cond):
            if cond_kind(cond) not in ("O1", "O1qqd"):
                return float("nan")
            if cond not in r_cache:
                dl = (xi * cond_rho(cond) * cond_mask(args.platform, cond).astype(np.float32)).astype(np.float32)
                r_cache[cond] = response_ratio(model, kind, Xr, dl, device)[0]
            return r_cache[cond]

        # plane scalars from predictions
        for role in roles:
            role_list = [(role, None)]
            if sim and f"{role}_P1" in D["X"]:
                role_list.append((f"{role}_P1", "P1"))
            for rrole, label in role_list:
                Y = D["Y"][rrole].astype(np.float64)
                ip = ips[rrole]
                conds = ["N"] if label == "P1" else discover_conds(mdir, rrole)
                if not conds:
                    conds = ["N"]
                Pc_raw = None
                prs = {}
                for cond in conds:
                    fpath = os.path.join(mdir, f"pred_{rrole}_{cond_fname(cond)}.npy")
                    if os.path.exists(fpath):
                        Praw = np.load(fpath).astype(np.float64); src = "stored"
                    else:
                        Xc = cond_input(rrole, cond)
                        if Xc is None:
                            continue
                        Praw = predict_np(model, kind, Xc, device).astype(np.float64); src = "recomputed"
                    prs[cond] = (Praw, src)
                if "N" not in prs:
                    continue
                Pc_raw = prs["N"][0]; Pc = clip(Pc_raw)
                for cond, (Praw, src) in prs.items():
                    Pt = clip(Praw)
                    row = dict(base)
                    row.update({"role": role, "cond": label or cond, "kind": "P1" if label else cond_kind(cond),
                                "rho": 0.0 if label else cond_rho(cond), "n_units": (ip.n_units if ip.g is not None else len(Y)),
                                "pred_source": src})
                    row.update(plane_scalars(Pt, Y, ip))
                    if label is None and cond != "N":
                        n = Pt - Pc; b = Pc - Y
                        row.update({"P_n": ip(n, n), "P_b": ip(b, b), "cross": 2 * ip(n, b)})
                        row["split_resid"] = row["R"] - (row["P_n"] + row["P_b"] + row["cross"])
                    if sim:
                        Eu = Praw - Y
                        row.update({"R_unclipped": ip(Eu, Eu), "beta_unclipped": ip(Praw, Y) / ip(Y, Y),
                                    "clip_frac": float(np.mean(Praw != Pt))})
                    row["r_O1M"] = r_M
                    row["r_cond"] = r_for(cond) if label is None else float("nan")
                    mrows.append(row)
        # kappa per role (platform O1 levels) and excess per rho^2
        my = [r for r in mrows if r["member"] == mid]
        for role in roles:
            rr = {r["cond"]: r for r in my if r["role"] == role}
            if "N" not in rr:
                continue
            R0 = rr["N"]["R"]
            lv = [l for l in P["o1_levels"] if f"O1:{l}" in rr]
            ex = [rr[f"O1:{l}"]["R"] - R0 for l in lv]
            k, npts = fit_kappa(lv, ex)
            lv_all = sorted(cond_rho(c) for c in rr if cond_kind(c) == "O1")
            k_all, npts_all = fit_kappa(lv_all, [rr[f"O1:{l}"]["R"] - R0 for l in lv_all])
            for c, r in rr.items():
                r["kappa"] = k; r["kappa_npts"] = npts; r["kappa_allO1"] = k_all; r["kappa_allO1_npts"] = npts_all
                if cond_kind(c) == "O1" and r["rho"] > 0:
                    r["excess_per_rho2"] = (r["R"] - R0) / r["rho"] ** 2
        # direction-resolved responses
        for dr in direction_responses(model, kind, Xr, pert, device):
            dr.update({"member": mid, "group": group, "surrogate": cfg.get("surrogate"), "hwhm": hwhm_of(cfg),
                       "outer_seed": outer_seed(mid), "r_O1M": r_M})
            dr.update(cfg_fields(cfg))
            drows.append(dr)
        # weights
        try:
            wrows.extend(weight_rows(mid, group, cfg, model, kind, sd_cpu))
        except Exception as e:  # noqa: BLE001
            log(f"weights {mid}: {e}")
        # learned widths
        if cfg.get("learnable_width") or "widths" in sd_cpu:
            lw = cfg.get("learned_widths") or sd_cpu["widths"].tolist()
            a0 = cfg.get("width")
            for l, a in enumerate(lw):
                blrows.append({"member": mid, "outer_seed": outer_seed(mid), "layer": l + 1,
                               "alpha_init": a0, "alpha_learned": float(a),
                               "alpha_state_dict": float(sd_cpu["widths"][l]) if "widths" in sd_cpu else float("nan"),
                               "hwhm_init": 2 / (math.pi * a0) if a0 else float("nan"),
                               "hwhm_learned": 2 / (math.pi * float(a)),
                               "peak_g_learned": float(a) / 2})
        if (mi + 1) % 10 == 0 or mi == len(members) - 1:
            log(f"{mi + 1}/{len(members)} {mid} r={r_M:.4f} ({time.time() - t0:.0f}s)")
            pd.DataFrame(mrows).to_csv(out_name(out_dir, "e15_members.csv", args.suffix + ".partial"), index=False)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.no_traj_check:
        wrows.extend(traj_init_check(runs))
    pd.DataFrame(mrows).to_csv(out_name(out_dir, "e15_members.csv", args.suffix), index=False)
    pd.DataFrame(drows).to_csv(out_name(out_dir, "e15_direction.csv", args.suffix), index=False)
    pd.DataFrame(wrows).to_csv(out_name(out_dir, "e15_weights.csv", args.suffix), index=False)
    pd.DataFrame(blrows).to_csv(out_name(out_dir, "e15_bl_widths.csv", args.suffix), index=False)
    with open(out_name(out_dir, "e15_position_meta.json", args.suffix), "w") as f:
        json.dump({"platform": args.platform, "runs": runs, "members": args.members, "n_members": len(members),
                   "rho_M": rho_M, "n_dev_rows": args.n, "direction": dmeta,
                   "definitions": {"inner_product": "trajectory average (sim) / row average (sarcos) of sum_l U_l V_l / d",
                                   "clip": "sim predictions clipped to torque limits (as analysis_aug.py)",
                                   "r": "forward_response.py: dev rows rng(0).choice, delta = rng(0) normal * rho_M * mask",
                                   "direction": "rng(15); energy E_total = rho_M^2 * n_masked_channels; full21 = eigvecs of 21x21 train cov; qqd14 = eigvecs of q,qd sub-block"}},
                  f, indent=1)
    part = out_name(out_dir, "e15_members.csv", args.suffix + ".partial")
    if os.path.exists(part):
        os.remove(part)
    log("done")


if __name__ == "__main__":
    main()
