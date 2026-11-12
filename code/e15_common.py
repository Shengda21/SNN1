"""E15 shared utilities (zero-training diagnostics of frozen members).

- data loading without side effects (SARCOS loader is called without a manifest path,
  so nothing in the shared workspace is written);
- member loading from <runs>/<member>/{config.json, model.pt} using the parameter
  names of common/models.py;
- a state-recording forward pass that is a line-by-line copy of LeakyLayer/SNN.forward
  (mem <- clamp(beta,0,1)*mem + cur; spk = H(mem - thr); mem <- mem - spk*thr;
  layer 3 enters fc4 through tanh(mem3 after reset); prediction = mem5 at t = T);
- condition helpers (noise masks, rho) and the trajectory-averaged inner product of
  the research plan (sim: mean within trajectory, then over trajectories; SARCOS: rows).
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from common.models import SNN, MLP  # noqa: E402

PLATFORM_DEFAULTS = {
    "sim": {"data_dir": "../data_sim", "o1_levels": [0.35, 0.7, 1.0], "rho_M": 0.7,
            "flip_conds": ["O1:0.35", "O1:0.7", "O1:1.0", "O2B:0.7"]},
    "sarcos": {"data_dir": "..", "o1_levels": [0.5, 0.7], "rho_M": 0.5,
               "flip_conds": ["O1:0.5", "O1:0.7", "O1qqd:0.5"]},
}
Q_QD = list(range(14))


# ----------------------------------------------------------------------------- data
def load_data(platform, data_dir):
    """Returns (D, apply_condition, clip). clip reproduces the torque-limit clipping that
    analysis_aug.py / analysis_openloop.py apply to sim predictions (identity on SARCOS)."""
    if platform == "sarcos":
        from sarcos.data import load_sarcos, apply_condition
        D = load_sarcos(data_dir)          # no manifest_path -> no file written
        D["groups"] = {r: None for r in D["X"]}
        return D, apply_condition, (lambda P: P)
    if platform == "sim":
        from sim.data import load_sim, apply_condition
        from sim.panda_env import TORQUE_LIMIT
        D = load_sim(data_dir)
        lo = (-TORQUE_LIMIT - D["y_mean"]) / D["y_std"]
        hi = (TORQUE_LIMIT - D["y_mean"]) / D["y_std"]
        return D, apply_condition, (lambda P: np.clip(P, lo, hi))
    raise ValueError(platform)


# ----------------------------------------------------------------------------- conditions
def cond_fname(cond):
    return cond.replace(":", "_")


def fname_to_cond(name):
    """'O1_0.7' -> 'O1:0.7', 'O1qqd_0.5' -> 'O1qqd:0.5', 'N' -> 'N'."""
    if name == "N":
        return "N"
    m = re.match(r"^([A-Za-z0-9]+?)_([0-9.]+)$", name)
    if not m:
        return None
    return f"{m.group(1)}:{m.group(2)}"


def cond_kind(cond):
    return "N" if cond == "N" else cond.split(":")[0]


def cond_rho(cond):
    return 0.0 if cond == "N" else float(cond.split(":")[1])


def cond_mask(platform, cond, n_in=21):
    """Channels that receive the perturbation (1) in standardized units."""
    m = np.zeros(n_in, np.float64)
    if cond == "N":
        return m
    kind = cond_kind(cond)
    if platform == "sim":
        m[Q_QD] = 1.0                       # O1 and O2B act on q, qd only
    elif kind == "O1qqd":
        m[Q_QD] = 1.0
    else:
        m[:] = 1.0                          # SARCOS O1: all 21 channels
    return m


def discover_conds(member_dir, role):
    """Conditions with a stored prediction file for this role (skips __wfe/__wfo variants)."""
    out = []
    for f in sorted(glob.glob(os.path.join(member_dir, f"pred_{role}_*.npy"))):
        name = os.path.basename(f)[len(f"pred_{role}_"):-4]
        if "__" in name or name.startswith("P1_"):
            continue
        c = fname_to_cond(name)
        if c is not None:
            out.append(c)
    order = {"N": 0}
    return sorted(set(out), key=lambda c: (order.get(c, 1), cond_kind(c), cond_rho(c)))


# ----------------------------------------------------------------------------- inner product
class InnerProduct:
    """<U,V> = mean over sampling units of the row mean of sum_l U_l V_l, with
    U = pred/sqrt(d) (standardized units). groups=None: every row is a unit."""
    def __init__(self, groups, d):
        self.d = d
        if groups is None:
            self.g = None
        else:
            g = np.asarray(groups)
            _, self.g = np.unique(g, return_inverse=True)
            self.cnt = np.bincount(self.g).astype(np.float64)
            self.n_units = len(self.cnt)

    def rows(self, U, V):
        return np.einsum("nd,nd->n", U, V) / self.d

    def __call__(self, U, V):
        r = self.rows(U, V)
        if self.g is None:
            return float(np.mean(r))
        return float(np.mean(np.bincount(self.g, weights=r) / self.cnt))


# ----------------------------------------------------------------------------- members
SURR_HWHM = {
    "atan": lambda a: 2.0 / (math.pi * a),
    "sigmoid": lambda k: math.log((1 + 1 / math.sqrt(2)) / (1 - 1 / math.sqrt(2))) / k,
    "triangular": lambda d: d / 2.0,
    "sre": lambda k: math.log(2.0) / k,
    "ste": lambda _: float("inf"),
}


CFG_KEYS = ["block", "experiment", "family", "peak_normalized", "detach_reset", "placement", "hwhm_narrow",
            "membrane_noise", "aug_scale", "activation", "stochastic_forward", "stage", "model_class"]


def cfg_fields(cfg):
    """Config fields used to group members into families (missing keys -> defaults)."""
    out = {k: cfg.get(k) for k in CFG_KEYS}
    s = cfg.get("surrogate")
    if out["family"] is None:
        out["family"] = {"triangular": "tri", "sigmoid": "sig", "atan": "atan", "sre": "sre", "ste": "ste",
                         "none": "mlp"}.get(s, s)
        if out["model_class"] == "MLPAct":
            out["family"] = "mlp"
    out["detach_reset"] = bool(out["detach_reset"]) if out["detach_reset"] is not None else False
    out["membrane_noise"] = float(out["membrane_noise"] or 0.0)
    out["aug_scale"] = float(out["aug_scale"] or 0.0)
    if out["activation"] is None:
        out["activation"] = "relu" if s == "none" else "spike"
    if out["placement"] is None:
        out["placement"] = "all"
    hl = cfg.get("hwhm_layers")
    out["hwhm_layers"] = "|".join(map(str, hl)) if hl else ""
    if out["block"] is None:
        out["block"] = member_prefix(cfg.get("member_id", ""))
    return out


def hwhm_of(cfg):
    s = cfg.get("surrogate")
    w = cfg.get("width")
    if s == "none" or cfg.get("model_class") == "MLPAct":
        return float("nan")
    if "hwhm" in cfg:                       # E16-E18 members store the HWHM directly
        h = cfg["hwhm"]
        return float("inf") if (h is None or s == "ste" or cfg.get("family") == "ste") else float(h)
    if s == "ste":
        return float("inf")
    if s in SURR_HWHM and w is not None:
        return SURR_HWHM[s](float(w))
    return float("nan")


def member_prefix(mid):
    return mid.split("_")[0]


def member_group(mid, cfg):
    """Readable group label used for the family summaries."""
    p = member_prefix(mid)
    s = cfg.get("surrogate")
    win = "STE" if s == "ste" else "window"
    table = {"dm": f"D-M {win}", "dma": f"D-MA {win}", "dmq": f"D-MQ {win}",
             "ds": "D-S", "dt": "D-T", "bl": "B-L", "ba": "B-A", "bn": "MLP",
             "dsa": "D-SA", "dsq": "D-SQ", "bna": "MLP-aug", "bnq": "MLP-augQ",
             "dmas": "B-A+", "dmqs": "B-AQ+", "bnas": "MLP-aug+", "bnqs": "MLP-augQ+",
             "uf": f"E14 early ({'a' if mid.endswith('_a') else 'b'})",
             "cap4": "E14 cap4", "cap8": "E14 cap8", "cap16": "E14 cap16",
             "width": f"width-grid {s}"}
    if p in table:
        return table[p]
    # new members (E16-E18): family, HWHM and training variant from config.json
    lab = f"{p} {cfg.get('family', s)}"
    if cfg.get("hwhm") is not None:
        lab += f" h{cfg['hwhm']}"
    hl = cfg.get("hwhm_layers")
    if hl and len(set(map(str, hl))) > 1:
        lab += " L[" + ",".join(map(str, hl)) + "]"
    if cfg.get("peak_normalized") is False:
        lab += " ctrl"
    if cfg.get("detach_reset"):
        lab += " detach"
    if cfg.get("membrane_noise", 0):
        lab += f" mnoise{cfg['membrane_noise']}"
    if cfg.get("aug_scale", 0):
        lab += f" aug{cfg['aug_scale']}"
    if cfg.get("activation") not in (None, "spike"):
        lab += f" act={cfg['activation']}"
    return lab


def outer_seed(mid):
    m = re.search(r"_s(\d+)", mid)
    return int(m.group(1)) if m else -1


def list_members(runs, pattern):
    out = []
    for d in sorted(glob.glob(os.path.join(runs, pattern))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")) \
                and os.path.exists(os.path.join(d, "model.pt")):
            out.append(d)
    return out


def build_model(cfg, sd, n_in, n_out):
    """Constructs the module with the parameter layout of common/models.py.
    Unknown surrogate names fall back to 'ste' for the module container only: the
    forward pass (Heaviside) does not depend on the surrogate."""
    mc = cfg.get("model_class")
    if mc in ("SNNX", "MLPAct"):
        from e16_models import build_from_config   # read-only use of the E16 module
        return build_from_config(cfg, n_in, n_out), ("mlp" if mc == "MLPAct" else "snn")
    if cfg.get("surrogate") == "none" or any(k.startswith("net.") for k in sd):
        return MLP(n_in, n_out, hidden=sd["net.0.weight"].shape[0]), "mlp"
    hidden = sd["fc1.weight"].shape[0]
    learn = bool(cfg.get("learnable_width", False)) or ("widths" in sd)
    try:
        m = SNN(n_in, n_out, hidden=hidden, surrogate=cfg.get("surrogate", "ste"),
                width=cfg.get("width"), learnable_width=learn)
        return m, "snn"
    except (ValueError, TypeError):
        m = SNN(n_in, n_out, hidden=hidden, surrogate="ste", learnable_width=learn)
        return m, "snn_unknown_surrogate"


def load_member(member_dir, device):
    cfg = json.load(open(os.path.join(member_dir, "config.json")))
    sd = torch.load(os.path.join(member_dir, "model.pt"), map_location="cpu")
    if "fc1.weight" in sd:
        n_in, n_out = sd["fc1.weight"].shape[1], sd["fc5.weight"].shape[0]
    else:
        n_in, n_out = sd["net.0.weight"].shape[1], sd["net.4.weight"].shape[0]
    model, kind = build_model(cfg, sd, n_in, n_out)
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg, kind


# ----------------------------------------------------------------------------- forward
def _snn_params(model):
    out = []
    for fc, lif in [(model.fc1, model.lif1), (model.fc2, model.lif2), (model.fc3, model.lif3)]:
        out.append((fc, lif.beta.clamp(0.0, 1.0), lif.thr))
    return out


@torch.no_grad()
def snn_forward_record(model, x, record=True):
    """Hard-spiking forward pass identical to SNN.forward. Returns (out [T,B,d], rec) with
    rec = {'spk': [3 x bool tensor [T,B,H]], 'u': [3 x tensor [T,B,H]] (pre-spike mem - thr)}."""
    B = x.shape[0]
    H = model.hidden
    dev = x.device
    m = [torch.zeros(B, H, device=dev) for _ in range(3)]
    m4 = torch.zeros(B, H, device=dev)
    m5 = torch.zeros(B, model.n_out, device=dev)
    b4 = model.li4.beta.clamp(0.0, 1.0); b5 = model.li5.beta.clamp(0.0, 1.0)
    P = _snn_params(model)
    outs = []
    spk = [[], [], []]; us = [[], [], []]
    for _ in range(model.T):
        a = x
        for l, (fc, beta, thr) in enumerate(P):
            m[l] = beta * m[l] + fc(a)
            u = m[l] - thr
            s = (u > 0).to(u.dtype)
            m[l] = m[l] - s * thr
            if record:
                spk[l].append(s.bool()); us[l].append(u)
            a = s
        m4 = b4 * m4 + model.fc4(torch.tanh(m[2]))
        m5 = b5 * m5 + model.fc5(m4)
        outs.append(m5)
    out = torch.stack(outs, 0)
    rec = None
    if record:
        rec = {"spk": [torch.stack(v, 0) for v in spk], "u": [torch.stack(v, 0) for v in us]}
    return out, rec


@torch.no_grad()
def mlp_forward_record(model, x, record=True):
    """MLP: 'spikes' are the gates (pre-activation > 0) of the two hidden layers, T = 1.
    The member's own activation module is used (ReLU for B-N; tanh / clipped ReLU / binary
    for the E18 MLPAct members)."""
    z1 = model.net[0](x); h1 = model.net[1](z1)
    z2 = model.net[2](h1); h2 = model.net[3](z2)
    out = model.net[4](h2).unsqueeze(0)
    rec = None
    if record:
        rec = {"spk": [(z1 > 0).unsqueeze(0), (z2 > 0).unsqueeze(0)], "u": [z1.unsqueeze(0), z2.unsqueeze(0)]}
    return out, rec


def forward_record(model, kind, x, record=True):
    if kind == "mlp":
        return mlp_forward_record(model, x, record)
    return snn_forward_record(model, x, record)


@torch.no_grad()
def predict_np(model, kind, X, device, batch=8192):
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        o, _ = forward_record(model, kind, xb, record=False)
        outs.append(o[-1].float().cpu().numpy())
    return np.concatenate(outs, 0)


def dev_subset_and_delta(D, platform, rho, n=2000):
    """Exactly the inputs and perturbations of forward_response.py (seed 0)."""
    rng = np.random.default_rng(0)
    idx = rng.choice(len(D["X"]["dev"]), n, replace=False)
    X = D["X"]["dev"][idx]
    mask = np.zeros(D["n_in"], np.float32)
    mask[:14] = 1
    if platform == "sarcos":
        mask[:] = 1
    xi = rng.standard_normal(X.shape).astype(np.float32)
    delta = xi * rho * mask
    return idx, X, delta, xi, mask


def response_ratio(model, kind, X, delta, device):
    """forward_response.py: mean over rows of ||f(x+delta) - f(x)|| / ||delta||."""
    with torch.no_grad():
        f0, _ = forward_record(model, kind, torch.as_tensor(X, device=device), record=False)
        f1, _ = forward_record(model, kind, torch.as_tensor(X + delta, device=device), record=False)
    f0 = f0[-1].cpu().numpy(); f1 = f1[-1].cpu().numpy()
    num = np.linalg.norm(f1 - f0, axis=1); den = np.linalg.norm(delta, axis=1)
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])), float(np.sqrt(np.mean((f1 - f0) ** 2)))


def train_input_moments(D):
    Xtr = D["X"]["train"].astype(np.float64)
    mu = Xtr.mean(0)
    Sigma = np.cov(Xtr, rowvar=False)
    return mu, Sigma


def out_name(out_dir, base, suffix):
    stem, ext = os.path.splitext(base)
    return os.path.join(out_dir, f"{stem}{suffix}{ext}")
