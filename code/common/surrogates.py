"""Surrogate-gradient functions.

Every surrogate is an explicit torch.autograd.Function. Forward = Heaviside(u),
u = membrane potential minus threshold. Backward = grad_out * g(u), where g is
the formula written below. Nothing is taken from library defaults.

    atan(alpha):    g(u) = (alpha/2) / (1 + (pi*alpha*u/2)^2)
    sigmoid(k):     g(u) = k * s(k u) * (1 - s(k u)),  s = logistic (stable form)
    ste():          g(u) = 1                            (no window)
    triangular(d):  g(u) = max(0, 1 - |u|/d)            (Esser et al. 2016 form)
    sre(k):         g(u) = exp(-k |u|)                  (escape-rate form, beta = 1)
    learnable_atan: atan form whose width a is a learnable scalar per layer; the
                    gradient with respect to a comes from the smooth proxy
                    S_a(u) = (1/pi) arctan(pi a u / 2) + 1/2, i.e.
                    dS/da = (u/2) / (1 + (pi a u/2)^2)   (LSG-style construction)
"""
import math
import torch


class _ATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, alpha):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        a = ctx.alpha
        return g * (a / 2.0) / (1.0 + (math.pi / 2.0 * a * u) ** 2), None


class _Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, k):
        ctx.save_for_backward(u)
        ctx.k = k
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        s = torch.sigmoid(ctx.k * u)
        return g * ctx.k * s * (1.0 - s), None


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        return g


class _Triangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, d):
        ctx.save_for_backward(u)
        ctx.d = d
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * torch.clamp(1.0 - u.abs() / ctx.d, min=0.0), None


class _SRE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, k):
        ctx.save_for_backward(u)
        ctx.k = k
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * torch.exp(-ctx.k * u.abs()), None


class _LearnableATan(torch.autograd.Function):
    """ATan surrogate whose width a is a learnable tensor (one scalar per layer)."""
    @staticmethod
    def forward(ctx, u, a):
        ctx.save_for_backward(u, a)
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        u, a = ctx.saved_tensors
        z = math.pi / 2.0 * a * u
        denom = 1.0 + z ** 2
        gu = g * (a / 2.0) / denom
        ga = (g * (u / 2.0) / denom).sum().reshape(a.shape)
        return gu, ga


def make_surrogate(name, width=None):
    """Return (spike_fn, resolved_width). `width` is the single scale parameter
    of the family (None -> family default). STE has no width."""
    if name == "atan":
        a = 2.0 if width is None else float(width)
        return (lambda u: _ATan.apply(u, a)), a
    if name == "sigmoid":
        k = 25.0 if width is None else float(width)
        return (lambda u: _Sigmoid.apply(u, k)), k
    if name == "ste":
        return (lambda u: _STE.apply(u)), None
    if name == "triangular":
        d = 1.0 if width is None else float(width)
        return (lambda u: _Triangular.apply(u, d)), d
    if name == "sre":
        k = 25.0 if width is None else float(width)
        return (lambda u: _SRE.apply(u, k)), k
    raise ValueError(name)


SURROGATE_NAMES = ["atan", "sigmoid", "ste", "triangular", "sre"]

# Dev-stage width grid: same budget (3 values) for every family that has a width.
WIDTH_GRID = {
    "atan": [1.0, 2.0, 4.0],
    "sigmoid": [5.0, 10.0, 25.0],
    "ste": [None],
    "triangular": [0.5, 1.0, 2.0],
    "sre": [5.0, 10.0, 25.0],
}


def surrogate_curve(name, width, u):
    """Numerical value of g(u) (numpy) for plotting and for the backward audit."""
    import numpy as np
    u = np.asarray(u, dtype=float)
    if name == "atan":
        return (width / 2.0) / (1.0 + (math.pi / 2.0 * width * u) ** 2)
    if name == "sigmoid":
        s = 1.0 / (1.0 + np.exp(-width * u))
        return width * s * (1 - s)
    if name == "ste":
        return np.ones_like(u)
    if name == "triangular":
        return np.clip(1.0 - np.abs(u) / width, 0.0, None)
    if name == "sre":
        return np.exp(-width * np.abs(u))
    raise ValueError(name)


def audit_backward(device="cpu"):
    """Check that each autograd backward equals the documented formula and that
    the forward is exactly Heaviside."""
    import numpy as np
    u = torch.linspace(-3, 3, 601, device=device, dtype=torch.float64)
    report = {}
    for name in SURROGATE_NAMES:
        for w in WIDTH_GRID[name]:
            fn, wr = make_surrogate(name, w)
            x = u.clone().requires_grad_(True)
            y = fn(x)
            y.sum().backward()
            ref = surrogate_curve(name, wr, u.detach().cpu().numpy())
            err = float(np.max(np.abs(x.grad.cpu().numpy() - ref)))
            fwd_ok = bool(torch.equal(y, (u > 0).to(y.dtype)))
            report[f"{name}:{wr}"] = {"max_abs_err": err, "forward_heaviside": fwd_ok}
    return report
