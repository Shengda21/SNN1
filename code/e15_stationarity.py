"""E15 item 8: numerical check of lemma SG-1 at the stored checkpoints.

$PY e15_stationarity.py --platform sim [--runs ../runs25/sim] [--members '*'] [--suffix '']

Training loss of common/train.py (MSE over all T internal steps), surrogate backward of
the member (the module's own spike_fn), n = 16,384 training rows (fixed seed), 8 chunks.
For spiking layer l (pre-spike membrane u_l(t) = mem - thr), with
  c_l(t)  = dL/dspk_l(t) from autograd (error arriving at the spikes: next-layer current
            AND the subtractive reset path), g_l(t) = surrogate derivative at u_l(t),
  a(t)    = presynaptic activity (input x for l = 1, spikes of layer l-1 otherwise)
            augmented with a constant 1 (bias),
  a~_i(t) = sum_{s<=t} beta_i^{t-s} a(s)  (leaky presynaptic trace of postsynaptic neuron i),
the exact gradient is dL/d[W_l b_l] = sum_t (g_l(t) * c_l(t) + d_l(t)) a~(t)^T, where d_l is
the smooth tanh path that only layer 3 has (mem3 after reset -> tanh -> fc4). We report
  G_l = E[sum_t (g*c) a~^T]   (lemma SG-1 gated form; equals the gradient for l = 1, 2),
  K_l = E[sum_t c a~^T]       (g-weighting removed: linear correlation of the back-propagated
                               error with the presynaptic trace),
normalized by the no-cancellation magnitude E[sum_t ||v(t) a~(t)^T||_F] (ratio in [0, 1]; 0 =
exactly stationary / decorrelated), the chunk standard-error floor of each ratio, cos(G, K),
the identity residual ||G + D - autograd|| / ||autograd||, and ||dL/dW_l||_F.
For g = 1 (STE) G = K identically.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e15_common import (PLATFORM_DEFAULTS, load_data, hwhm_of, cfg_fields, member_prefix, member_group,  # noqa: E402
                        outer_seed, list_members, load_member, out_name)


def g_functions(model, cfg):
    """Surrogate derivative g(u) of each spiking layer, obtained by autograd through the
    layer's own spike_fn (exactly the backward used in training, for every family:
    common/surrogates.py, learnable ATan, and the E16 peak-normalized / escape families).
    Returns (list of 3 callables, list of peak values g(0))."""
    def make(fn):
        def g(u):
            with torch.enable_grad():
                uu = u.detach().clone().requires_grad_(True)
                fn(uu).sum().backward()
            return uu.grad.detach()
        return g
    fns = [make(l.spike_fn) for l in (model.lif1, model.lif2, model.lif3)]
    dev = model.fc1.weight.device
    peaks = [float(f(torch.zeros(1, device=dev))[0]) for f in fns]
    return fns, peaks


def trace_moment(v, A, beta):
    """sum_b sum_t sum_{s<=t} beta_i^{t-s} v_i(b,t) A_j(b,s) -> [H, P]. v: [T,B,H], A: [T,B,P]."""
    T = v.shape[0]
    out = torch.zeros(v.shape[2], A.shape[2], dtype=torch.float64, device=v.device)
    for t in range(T):
        for s in range(t + 1):
            out += (beta ** (t - s))[:, None] * (v[t].T.double() @ A[s].double())
    return out


def trace_magnitude(v, A, beta):
    """sum_b sum_t ||v(b,t) (x) a~(b,t)||_F with a~_i(b,t) = sum_{s<=t} beta_i^{t-s} A(b,s)."""
    T = v.shape[0]
    Gram = torch.einsum("sbp,rbp->bsr", A.double(), A.double())        # [B,T,T]
    tot = 0.0
    ar = torch.arange(T, device=v.device)
    for t in range(T):
        expo = (t - ar).clamp(min=0).double()
        coef = beta.double()[:, None] ** expo[None, :]                   # [H,T]
        coef = coef * (ar <= t).double()[None, :]
        na2 = torch.einsum("is,bsr,ir->bi", coef, Gram, coef)            # ||a~_i(b,t)||^2
        tot += float(torch.sqrt((v[t].double() ** 2 * na2).sum(1)).sum())
    return tot


def run_member(model, cfg, Xtr, Ytr, device, n, n_chunks, seed):
    gfn, peaks = g_functions(model, cfg)
    detach = bool(getattr(model, "detach_reset", False))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), n, replace=False)
    model.zero_grad(set_to_none=True)
    T, H = model.T, model.hidden
    d = model.n_out
    fcs = [model.fc1, model.fc2, model.fc3]; lifs = [model.lif1, model.lif2, model.lif3]
    betas = [l.beta.detach().clamp(0, 1).double() for l in lifs]
    acc = [{"G": 0, "K": 0, "D": 0, "magG": 0.0, "magK": 0.0, "Gc": [], "Kc": [], "gsum": 0.0, "gwin": 0.0, "cnt": 0}
           for _ in range(3)]
    chunk = n // n_chunks
    for ci in range(n_chunks):
        sel = idx[ci * chunk:(ci + 1) * chunk]
        x = torch.as_tensor(Xtr[sel], dtype=torch.float32, device=device)
        y = torch.as_tensor(Ytr[sel], dtype=torch.float32, device=device)
        B = x.shape[0]
        m = [torch.zeros(B, H, device=device) for _ in range(3)]
        m4 = torch.zeros(B, H, device=device); m5 = torch.zeros(B, d, device=device)
        b4 = model.li4.beta.clamp(0, 1); b5 = model.li5.beta.clamp(0, 1)
        spk = [[], [], []]; us = [[], [], []]; pre = [[], [], []]; h3s = []; outs = []
        for _ in range(T):
            a = x
            for l in range(3):
                beta = lifs[l].beta.clamp(0.0, 1.0)
                m[l] = beta * m[l] + fcs[l](a)
                u = m[l] - lifs[l].thr
                s = lifs[l].spike_fn(u)
                s.retain_grad()
                # reset exactly as in training: detached spike for detach-reset members (w16d)
                m[l] = m[l] - (s.detach() if detach else s) * lifs[l].thr
                spk[l].append(s); us[l].append(u.detach()); pre[l].append(a.detach())
                a = s
            h3 = torch.tanh(m[2]); h3.retain_grad(); h3s.append(h3)
            m4 = b4 * m4 + model.fc4(h3)
            m5 = b5 * m5 + model.fc5(m4)
            outs.append(m5)
        out = torch.stack(outs, 0)
        loss = ((out - y.unsqueeze(0)) ** 2).sum() / (n * T * d)
        loss.backward()
        with torch.no_grad():
            for l in range(3):
                c = torch.stack([(s.grad if s.grad is not None else torch.zeros_like(s)) for s in spk[l]], 0)
                if getattr(model, "widths", None) is not None:
                    model.widths.grad = None                         # [T,B,H]
                u = torch.stack(us[l], 0)
                g = gfn[l](u)
                A = torch.stack(pre[l], 0)
                A = torch.cat([A, torch.ones(T, B, 1, device=device)], 2)           # bias column
                vg = g * c
                Mg = trace_moment(vg, A, betas[l]); Mk = trace_moment(c, A, betas[l])
                ac = acc[l]
                ac["G"] = ac["G"] + Mg; ac["K"] = ac["K"] + Mk
                ac["Gc"].append((Mg / B).cpu().numpy()); ac["Kc"].append((Mk / B).cpu().numpy())
                ac["magG"] += trace_magnitude(vg, A, betas[l]); ac["magK"] += trace_magnitude(c, A, betas[l])
                ac["gsum"] += float(g.double().sum()); ac["gwin"] += float((g > 0.1 * peaks[l]).double().sum())
                ac["cnt"] += g.numel()
                if l == 2:
                    h = torch.stack(h3s, 0)
                    dd = torch.stack([hh.grad for hh in h3s], 0) * (1.0 - h ** 2)
                    ac["D"] = ac["D"] + trace_moment(dd, A, betas[l])
    rows = []
    for l in range(3):
        ac = acc[l]
        gw = fcs[l].weight.grad.double(); gb = fcs[l].bias.grad.double()
        auto = torch.cat([gw, gb[:, None]], 1)
        tot = ac["G"] + (ac["D"] if l == 2 else 0)
        G = ac["G"] / n; K = ac["K"] / n
        Gf = float(torch.linalg.norm(G)); Kf = float(torch.linalg.norm(K))
        magG = ac["magG"] / n; magK = ac["magK"] / n
        Gc = np.stack(ac["Gc"]); Kc = np.stack(ac["Kc"])
        seG = float(np.linalg.norm(Gc.std(0, ddof=1) / math.sqrt(len(Gc))))
        seK = float(np.linalg.norm(Kc.std(0, ddof=1) / math.sqrt(len(Kc))))
        Wn = float(torch.linalg.norm(fcs[l].weight.detach().double()))
        rows.append({"layer": l + 1, "grad_fro": float(torch.linalg.norm(gw)), "grad_fro_with_bias": float(torch.linalg.norm(auto)),
                     "grad_rel_to_W": float(torch.linalg.norm(gw)) / Wn,
                     "identity_resid": float(torch.linalg.norm(tot - auto) / torch.linalg.norm(auto)),
                     "G_fro": Gf, "K_fro": Kf, "D_fro": float(torch.linalg.norm(ac["D"] / n)) if l == 2 else 0.0,
                     "ratio_G": Gf / magG if magG > 0 else float("nan"), "ratio_K": Kf / magK if magK > 0 else float("nan"),
                     "floor_G": seG / magG if magG > 0 else float("nan"), "floor_K": seK / magK if magK > 0 else float("nan"),
                     "cos_GK": float((G * K).sum() / (torch.linalg.norm(G) * torch.linalg.norm(K))),
                     "g_mean_rel_peak": ac["gsum"] / ac["cnt"] / peaks[l], "frac_in_window": ac["gwin"] / ac["cnt"],
                     "g_peak": peaks[l], "loss": None})
    for nm in ["fc4", "fc5"]:
        fc = getattr(model, nm)
        rows.append({"layer": nm, "grad_fro": float(torch.linalg.norm(fc.weight.grad.double())),
                     "grad_rel_to_W": float(torch.linalg.norm(fc.weight.grad.double()) / torch.linalg.norm(fc.weight.detach().double()))})
    model.zero_grad(set_to_none=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--members", default="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=8)
    args = ap.parse_args()
    P = PLATFORM_DEFAULTS[args.platform]
    data_dir = args.data_dir or P["data_dir"]
    runs = args.runs or f"../runs25/{args.platform}"
    out_dir = args.out or f"../results/{args.platform}"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, _, _ = load_data(args.platform, data_dir)
    Xtr, Ytr = D["X"]["train"], D["Y"]["train"]
    members = list_members(runs, args.members)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    log(f"{len(members)} members")
    rows = []
    t0 = time.time()
    for mi, mdir in enumerate(members):
        mid = os.path.basename(mdir)
        model, cfg, kind = load_member(mdir, device)
        if kind == "mlp":
            continue
        base = {"platform": args.platform, "member": mid, "prefix": member_prefix(mid), "group": member_group(mid, cfg),
                "surrogate": cfg.get("surrogate"), "hwhm": hwhm_of(cfg), "outer_seed": outer_seed(mid),
                "best_epoch": cfg.get("best_epoch"), "aug_sigma": cfg.get("aug_sigma", 0.0)}
        cfg.setdefault("member_id", mid)
        base.update(cfg_fields(cfg))
        model.eval()          # deterministic (deployed) forward; no training-time noise
        res = run_member(model, cfg, Xtr, Ytr, device, args.n, args.chunks, args.seed)
        if res is None:
            log(f"{mid}: unknown surrogate {cfg.get('surrogate')}, skipped")
            continue
        for r in res:
            rr = dict(base); rr.update(r); rows.append(rr)
        if (mi + 1) % 20 == 0 or mi == len(members) - 1:
            log(f"{mi + 1}/{len(members)} {mid} ({time.time() - t0:.0f}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out_name(out_dir, "e15_stationarity.csv", args.suffix), index=False)
    log("done")


if __name__ == "__main__":
    main()
