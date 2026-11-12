"""E16-E18 surrogate functions (new file; common/surrogates.py is left untouched).

Peak-normalized families (g(0) = 1, so the straight-through estimator g == 1 is the
common infinite-width limit of every family). Width is reported as the half width at
half maximum (HWHM) in membrane-potential units (initial threshold = 1):

    tri   g(u) = max(0, 1 - |u|/d)            HWHM = d/2           d = 2 h
    sig   g(u) = 4 s(k u) (1 - s(k u))        HWHM = c/k           k = c/h, c = 2 ln(1+sqrt2) = 1.7627
    atan  g(u) = 1 / (1 + (pi a u / 2)^2)     HWHM = 2/(pi a)      a = 2/(pi h)
    sre   g(u) = exp(-k |u|)                  HWHM = ln2/k         k = ln2/h

Control parametrization (original manuscript, NOT peak-normalized), same HWHM grid:

    atan  g(u) = (a/2) / (1 + (pi a u / 2)^2)    peak a/2
    sig   g(u) = k s(k u) (1 - s(k u))           peak k/4

(tri and sre are already peak-normalized in the original code, so their control
members would coincide with the peak-normalized ones.)

Every forward pass is exactly Heaviside(u); only the backward differs.
"""
import math
import numpy as np
import torch

from common.surrogates import make_surrogate as make_orig_surrogate

SIG_C = 2.0 * math.log(1.0 + math.sqrt(2.0))   # 1.762747...: 4 s(1-s) = 1/2 at k u = SIG_C
FAMILIES = ["tri", "sig", "atan", "sre"]
HWHM_GRID = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
ORIG_NAME = {"tri": "triangular", "sig": "sigmoid", "atan": "atan", "sre": "sre"}


def hwhm_str(h):
    """Canonical string of a HWHM value used in member ids (0.0625, 0.125, ..., 1, 2, 4)."""
    h = float(h)
    return str(int(h)) if h == int(h) else repr(h)


def param_from_hwhm(family, h):
    """Scale parameter of the family (d, k, a or k) that gives half width h."""
    h = float(h)
    if family == "tri":
        return 2.0 * h
    if family == "sig":
        return SIG_C / h
    if family == "atan":
        return 2.0 / (math.pi * h)
    if family == "sre":
        return math.log(2.0) / h
    raise ValueError(family)


def hwhm_from_param(family, p):
    p = float(p)
    if family == "tri":
        return p / 2.0
    if family == "sig":
        return SIG_C / p
    if family == "atan":
        return 2.0 / (math.pi * p)
    if family == "sre":
        return math.log(2.0) / p
    raise ValueError(family)


class _PNTri(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, d):
        ctx.save_for_backward(u)
        ctx.d = d
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * torch.clamp(1.0 - u.abs() / ctx.d, min=0.0), None


class _PNSig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, k):
        ctx.save_for_backward(u)
        ctx.k = k
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        s = torch.sigmoid(ctx.k * u)
        return g * 4.0 * s * (1.0 - s), None


class _PNATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a):
        ctx.save_for_backward(u)
        ctx.a = a
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g / (1.0 + (math.pi / 2.0 * ctx.a * u) ** 2), None


class _PNSRE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, k):
        ctx.save_for_backward(u)
        ctx.k = k
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * torch.exp(-ctx.k * u.abs()), None


_PN = {"tri": _PNTri, "sig": _PNSig, "atan": _PNATan, "sre": _PNSRE}


def make_layer_surrogate(family, hwhm, peak_normalized=True):
    """Return (spike_fn, param, peak) for one spiking layer.

    family: tri | sig | atan | sre | ste.  peak_normalized=False uses the original
    manuscript parametrization (common/surrogates.py) at the same HWHM."""
    if family == "ste":
        fn, _ = make_orig_surrogate("ste")
        return fn, None, 1.0
    p = param_from_hwhm(family, hwhm)
    if peak_normalized:
        F = _PN[family]
        return (lambda u: F.apply(u, p)), p, 1.0
    fn, p2 = make_orig_surrogate(ORIG_NAME[family], p)
    peak = {"atan": p2 / 2.0, "sig": p2 / 4.0, "tri": 1.0, "sre": 1.0}[family]
    return fn, p2, peak


def curve(family, hwhm, u, peak_normalized=True):
    """Numerical g(u) (numpy) for audits and plots."""
    u = np.asarray(u, dtype=float)
    if family == "ste":
        return np.ones_like(u)
    p = param_from_hwhm(family, hwhm)
    if family == "tri":
        return np.clip(1.0 - np.abs(u) / p, 0.0, None)
    if family == "sig":
        s = 1.0 / (1.0 + np.exp(-p * u))
        return (4.0 if peak_normalized else p) * s * (1 - s)
    if family == "atan":
        return (1.0 if peak_normalized else p / 2.0) / (1.0 + (math.pi / 2.0 * p * u) ** 2)
    if family == "sre":
        return np.exp(-p * np.abs(u))
    raise ValueError(family)


def audit(device="cpu"):
    """Backward == documented formula, forward == Heaviside, g(0) and g(HWHM) checks."""
    u = torch.linspace(-6, 6, 4801, device=device, dtype=torch.float64)
    rep = {}
    for pn in (True, False):
        for fam in FAMILIES:
            if not pn and fam in ("tri", "sre"):
                continue
            for h in HWHM_GRID:
                fn, p, peak = make_layer_surrogate(fam, h, pn)
                x = u.clone().requires_grad_(True)
                y = fn(x)
                y.sum().backward()
                ref = curve(fam, h, u.cpu().numpy(), pn)
                g0 = float(curve(fam, h, np.array([0.0]), pn)[0])
                gh = float(curve(fam, h, np.array([h]), pn)[0])
                rep[f"{'pn' if pn else 'orig'}:{fam}:{hwhm_str(h)}"] = {
                    "param": p, "max_abs_err": float(np.max(np.abs(x.grad.cpu().numpy() - ref))),
                    "forward_heaviside": bool(torch.equal(y, (u > 0).to(y.dtype))),
                    "g0": g0, "g_hwhm_over_g0": gh / g0}
    return rep


if __name__ == "__main__":
    import json
    r = audit()
    bad = {k: v for k, v in r.items() if v["max_abs_err"] > 1e-10 or not v["forward_heaviside"]
           or abs(v["g_hwhm_over_g0"] - 0.5) > 1e-9}
    for k, v in r.items():
        print(k, json.dumps(v))
    print("AUDIT", "OK" if not bad else f"FAILED {list(bad)}")
