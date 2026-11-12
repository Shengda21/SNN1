"""E15 items 4-5: unit signal-to-noise ratio of the first layer, predicted flip
probabilities (lemma SG-2) and measured spike flips under input noise.

$PY e15_flip.py --platform sim [--runs ../runs25/sim] [--members '*'] [--suffix '']

First layer, neuron k with weight row w_k, bias b_k, threshold thr_k; training inputs
x ~ (mu, Sigma) in standardized units (channel variance 1); noise delta ~ N(0, Sigma_d),
Sigma_d = rho^2 diag(mask) of the condition.
  sigma_in,k = sqrt(w_k' Sigma_d w_k)          induced membrane-noise std
  SNR_k      = w_k' Sigma w_k / w_k' Sigma_d w_k
  P_sg2,k    = (1/pi) arctan(SNR_k^{-1/2})     threshold at the mean of the current
  P_biv,k    = 2 T(h_k, SNR_k^{-1/2}),  h_k = (thr_k - m_k)/sqrt(s_k^2 + sigma_in,k^2),
               m_k = w_k'mu + b_k, s_k^2 = w_k' Sigma w_k, T = Owen's T function.
               This is the exact bivariate-normal quadrant probability
               P[1(u > thr) != 1(u + e > thr)] for u ~ N(m, s^2), e ~ N(0, sigma_in^2);
               it reduces to P_sg2 when thr = m.
  P_emp1,k   = mean over the measured dev rows of Phi(-|w_k'x + b_k - thr_k| / sigma_in,k):
               exact expected flip probability of the FIRST internal step (mem = cur at t = 1),
               no Gaussian assumption on x. Steps t >= 2 depend on the reset history and are
               not predicted in closed form.
Measured: 2,000 dev rows (forward_response.py rows); the noisy copy is apply_condition(...,
'dev', cond) on the full dev set restricted to those rows (identical noise to the stored
dev predictions). Flip rate = fraction of (neuron x internal step) whose spike changes.
MLP members: 'spikes' are ReLU gates (pre-activation > 0), threshold 0, T = 1.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import owens_t, ndtr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e15_common import (PLATFORM_DEFAULTS, load_data, cond_kind, cond_rho, cond_mask, hwhm_of, cfg_fields,  # noqa: E402
                        member_prefix, member_group, outer_seed, list_members, load_member,
                        dev_subset_and_delta, forward_record, train_input_moments, out_name)

QS = [0.05, 0.25, 0.5, 0.75, 0.95]


def layer1_params(model, kind):
    if kind == "mlp":
        W = model.net[0].weight.detach().cpu().numpy().astype(np.float64)
        b = model.net[0].bias.detach().cpu().numpy().astype(np.float64)
        thr = np.zeros(W.shape[0])
    else:
        W = model.fc1.weight.detach().cpu().numpy().astype(np.float64)
        b = model.fc1.bias.detach().cpu().numpy().astype(np.float64)
        thr = model.lif1.thr.detach().cpu().numpy().astype(np.float64)
    return W, b, thr


def predictions(W, b, thr, mu, Sigma, mask, rho, Xdev):
    s2 = np.einsum("ki,ij,kj->k", W, Sigma, W)
    sig2 = rho ** 2 * np.einsum("ki,i,ki->k", W, mask, W)
    sig = np.sqrt(sig2)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = s2 / sig2
        a = np.sqrt(sig2 / s2)                       # SNR^{-1/2}
        p_sg2 = np.arctan(a) / np.pi
        m = W @ mu + b
        h = (thr - m) / np.sqrt(s2 + sig2)
        p_biv = 2.0 * owens_t(h, a)
        cur = Xdev.astype(np.float64) @ W.T + b      # step-1 membrane = current
        p_emp1 = ndtr(-np.abs(cur - thr[None]) / sig[None]).mean(0)
    return {"snr": snr, "sigma_in": sig, "p_sg2": p_sg2, "p_biv": p_biv, "p_emp1": p_emp1,
            "m": m, "s": np.sqrt(s2), "h": h}


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().values; rb = pd.Series(b[ok]).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--members", default="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    P = PLATFORM_DEFAULTS[args.platform]
    data_dir = args.data_dir or P["data_dir"]
    runs = args.runs or f"../runs25/{args.platform}"
    out_dir = args.out or f"../results/{args.platform}"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, apply_condition, _ = load_data(args.platform, data_dir)
    mu, Sigma = train_input_moments(D)
    idx, Xr, _, _, _ = dev_subset_and_delta(D, args.platform, P["rho_M"], args.n)
    conds = P["flip_conds"]
    Xn = {c: apply_condition(D["X"]["dev"], "dev", c)[idx] for c in conds}
    members = list_members(runs, args.members)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    log(f"{len(members)} members")
    rows, neuron = [], {}
    t0 = time.time()
    for mi, mdir in enumerate(members):
        mid = os.path.basename(mdir)
        try:
            model, cfg, kind = load_member(mdir, device)
        except Exception as e:  # noqa: BLE001
            log(f"skip {mid}: {e}")
            continue
        group = member_group(mid, cfg)
        base = {"platform": args.platform, "member": mid, "prefix": member_prefix(mid), "group": group,
                "surrogate": cfg.get("surrogate"), "hwhm": hwhm_of(cfg), "outer_seed": outer_seed(mid),
                "model_kind": kind}
        cfg.setdefault("member_id", mid)
        base.update(cfg_fields(cfg))
        with torch.no_grad():
            _, rc = forward_record(model, kind, torch.as_tensor(Xr, device=device))
        s_clean = [s.cpu().numpy() for s in rc["spk"]]          # [T,B,H] bool per layer
        W, b, thr = layer1_params(model, kind)
        L1 = s_clean[0]
        dead = float(np.mean(~L1.any(axis=(0, 1)))); sat = float(np.mean(L1.all(axis=(0, 1))))
        neuron[mid] = {"thr": thr, "b": b}
        for c in conds:
            rho = cond_rho(c); mask = cond_mask(args.platform, c)
            with torch.no_grad():
                _, rn = forward_record(model, kind, torch.as_tensor(Xn[c], device=device))
            s_noisy = [s.cpu().numpy() for s in rn["spk"]]
            pr = predictions(W, b, thr, mu, Sigma, mask, rho, Xr)
            flips = [(a != z) for a, z in zip(s_clean, s_noisy)]
            tot_f = sum(f.sum() for f in flips); tot_n = sum(f.size for f in flips)
            fl1_step1_k = flips[0][0].mean(0)                     # per neuron, first step
            fl1_all_k = flips[0].mean(axis=(0, 1))
            for l, f in enumerate(flips):
                r = dict(base)
                r.update({"cond": c, "kind": cond_kind(c), "rho": rho, "layer": l + 1, "n_neurons": f.shape[2],
                          "T": f.shape[0], "flip_measured": float(f.mean()), "flip_measured_step1": float(f[0].mean()),
                          "spike_rate_clean": float(s_clean[l].mean()), "spike_rate_noisy": float(s_noisy[l].mean())})
                if l == 0:
                    snr = pr["snr"]
                    r.update({"flip_pred_sg2": float(np.nanmean(pr["p_sg2"])), "flip_pred_biv": float(np.nanmean(pr["p_biv"])),
                              "flip_pred_emp1": float(np.nanmean(pr["p_emp1"])),
                              "frac_dead": dead, "frac_saturated": sat,
                              "neuron_spearman_biv_vs_step1": spearman(pr["p_biv"], fl1_step1_k),
                              "neuron_spearman_emp1_vs_step1": spearman(pr["p_emp1"], fl1_step1_k),
                              "neuron_spearman_biv_vs_all": spearman(pr["p_biv"], fl1_all_k),
                              "neuron_spearman_sg2_vs_all": spearman(pr["p_sg2"], fl1_all_k),
                              "snr_mean_log10": float(np.nanmean(np.log10(snr))),
                              "sigma_in_mean": float(np.mean(pr["sigma_in"])),
                              "s_mean": float(np.mean(pr["s"])), "abs_h_median": float(np.nanmedian(np.abs(pr["h"])))})
                    for q in QS:
                        r[f"snr_q{int(q * 100):02d}"] = float(np.nanquantile(snr, q))
                        r[f"sigma_in_q{int(q * 100):02d}"] = float(np.quantile(pr["sigma_in"], q))
                rows.append(r)
            r = dict(base)
            r.update({"cond": c, "kind": cond_kind(c), "rho": rho, "layer": "all", "flip_measured": float(tot_f / tot_n),
                      "flip_measured_step1": float(sum(f[0].sum() for f in flips) / sum(f[0].size for f in flips)),
                      "spike_rate_clean": float(sum(s.sum() for s in s_clean) / sum(s.size for s in s_clean))})
            rows.append(r)
            neuron[mid].update({f"{c}|snr": pr["snr"], f"{c}|sigma_in": pr["sigma_in"], f"{c}|p_biv": pr["p_biv"],
                                f"{c}|p_emp1": pr["p_emp1"], f"{c}|flip_step1": fl1_step1_k, f"{c}|flip_all": fl1_all_k})
            neuron[mid].update({"m": pr["m"], "s": pr["s"]})
        if (mi + 1) % 20 == 0 or mi == len(members) - 1:
            log(f"{mi + 1}/{len(members)} {mid} ({time.time() - t0:.0f}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out_name(out_dir, "e15_flip.csv", args.suffix), index=False)
    flat = {f"{m}||{k}": np.asarray(v, np.float32) for m, dct in neuron.items() for k, v in dct.items()}
    np.savez_compressed(out_name(out_dir, "e15_flip_neurons.npz", args.suffix), **flat)
    with open(out_name(out_dir, "e15_flip_meta.json", args.suffix), "w") as f:
        json.dump({"conds": conds, "n_dev_rows": args.n, "train_cov_diag_range": [float(np.diag(Sigma).min()), float(np.diag(Sigma).max())],
                   "note": "O2B prediction uses the marginal per-row noise covariance (same as O1 at that rho)"}, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
