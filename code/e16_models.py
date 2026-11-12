"""E16-E18 model variants (new file; common/models.py is left untouched).

SNNX: same forward architecture, parameters, parameter order and initialization as
      common.models.SNN (it calls SNN.__init__), with
        - one surrogate per spiking layer (E16 uniform width, E17 layer-wise widths);
        - optional training-only Gaussian membrane noise on the first spiking layer
          (E18): m1 <- beta*m1 + fc1(x) + sigma*xi_t, xi_t ~ N(0, I) drawn independently
          at every internal step and neuron, only when model.training is True; the noise
          enters before the threshold comparison and stays in the membrane state (noisy
          input current). Evaluation / predictions are noise-free;
        - optional detach_reset (w16d block): the forward is unchanged, the subtractive
          reset mem <- mem - spk * thr treats spk as a constant in the backward pass
          (mem - spk.detach() * thr), so the reset does not multiply the temporal credit by
          (1 - thr * g(u)); the threshold still receives its gradient through the reset.
MLPAct: B-N MLP (220 hidden units, 55,007 parameters, same init as common.models.MLP)
        with a bounded activation: tanh, clipped ReLU min(max(x,0),1), or a binary
        Heaviside activation H(x) in {0,1} trained with a saturating straight-through
        estimator (backward 1{|x|<=1}, Hubara et al. 2016).

load_member(member_dir, device) rebuilds any member (original or new) from config.json.
"""
import json
import os

import torch
import torch.nn as nn

from common.models import SNN, MLP
from common.surrogates import make_surrogate as make_orig_surrogate
from e16_surrogates import make_layer_surrogate, SIG_C


class _EscapeSpike(torch.autograd.Function):
    """Stochastic spiking with sigmoidal escape noise f(u) = s(beta u) (Gygax & Zenke 2025):
    training forward draws spikes ~ Bernoulli(f(u)); evaluation forward is H(u);
    backward is the exact escape-function derivative beta s(1 - s)."""
    @staticmethod
    def forward(ctx, u, beta, stochastic):
        p = torch.sigmoid(beta * u)
        ctx.save_for_backward(p)
        ctx.beta = beta
        if stochastic:
            return torch.bernoulli(p)
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        return g * ctx.beta * p * (1.0 - p), None, None


class SNNX(SNN):
    def __init__(self, n_in, n_out, layers, mem_noise=0.0, hidden=128, T=5, detach_reset=False):
        """layers: list of three [family, hwhm, peak_normalized] (family 'ste' allowed),
        or of three ['orig', name, width] to use common/surrogates.py directly, or of three
        ['esc', hwhm] for stochastic escape-noise spiking (training only, E18 optional)."""
        super().__init__(n_in, n_out, hidden=hidden, T=T, surrogate="ste")
        fns, params = [], []
        for spec in layers:
            if spec[0] == "orig":
                fn, p = make_orig_surrogate(spec[1], spec[2])
            elif spec[0] == "esc":
                p = SIG_C / float(spec[1])
                fn = self._escape_fn(p)
            else:
                fn, p, _ = make_layer_surrogate(spec[0], spec[1], bool(spec[2]))
            fns.append(fn); params.append(p)
        self.lif1.spike_fn, self.lif2.spike_fn, self.lif3.spike_fn = fns
        self.layer_params = params
        self.layers_spec = [list(s) for s in layers]
        self.mem_noise = float(mem_noise)
        self.detach_reset = bool(detach_reset)
        self.width = params[0]

    def _lif(self, layer, cur, mem):
        """Same update as common.models.LeakyLayer (spiking, subtract reset), with the
        reset term optionally detached."""
        if not self.detach_reset:
            return layer(cur, mem)
        beta = layer.beta.clamp(0.0, 1.0)
        mem = beta * mem + cur
        spk = layer.spike_fn(mem - layer.thr)
        mem = mem - spk.detach() * layer.thr
        return spk, mem

    def _escape_fn(self, beta):
        return lambda u: _EscapeSpike.apply(u, beta, self.training)

    def forward(self, x, return_spikes=False):
        B = x.shape[0]
        dev = x.device
        m1 = torch.zeros(B, self.hidden, device=dev); m2 = torch.zeros_like(m1)
        m3 = torch.zeros_like(m1); m4 = torch.zeros_like(m1)
        m5 = torch.zeros(B, self.n_out, device=dev)
        noisy = self.training and self.mem_noise > 0
        outs, spk_counts = [], []
        for _ in range(self.T):
            c1 = self.fc1(x)
            if noisy:
                c1 = c1 + self.mem_noise * torch.randn_like(c1)
            s1, m1 = self._lif(self.lif1, c1, m1)
            s2, m2 = self._lif(self.lif2, self.fc2(s1), m2)
            s3, m3 = self._lif(self.lif3, self.fc3(s2), m3)
            _, m4 = self.li4(self.fc4(torch.tanh(m3)), m4)
            _, m5 = self.li5(self.fc5(m4), m5)
            outs.append(m5)
            if return_spikes:
                spk_counts.append(torch.stack([s1.sum(1), s2.sum(1), s3.sum(1)], 1))
        out = torch.stack(outs, 0)
        if return_spikes:
            return out, torch.stack(spk_counts, 0)
        return out


class _BinarySTE(torch.autograd.Function):
    """Forward H(x) in {0,1}; backward straight-through, saturated outside |x| <= 1."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1.0).to(g.dtype)


class Binary(nn.Module):
    def forward(self, x):
        return _BinarySTE.apply(x)


class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "crelu": ClippedReLU, "bin": Binary}


class MLPAct(nn.Module):
    """Same layers, sizes and default init order as common.models.MLP (state_dict keys
    net.0 / net.2 / net.4), only the hidden activation changes."""
    def __init__(self, n_in, n_out, hidden=220, act="tanh"):
        super().__init__()
        A = ACTS[act]
        self.net = nn.Sequential(nn.Linear(n_in, hidden), A(),
                                 nn.Linear(hidden, hidden), A(),
                                 nn.Linear(hidden, n_out))
        self.T = 1
        self.act = act

    def forward(self, x):
        return self.net(x).unsqueeze(0)


def build_from_config(cfg, n_in=21, n_out=7):
    mc = cfg.get("model_class")
    if mc == "SNNX":
        return SNNX(n_in, n_out, cfg["surrogate_layers"], mem_noise=cfg.get("membrane_noise", 0.0),
                    hidden=cfg.get("hidden", 128), detach_reset=cfg.get("detach_reset", False))
    if mc == "MLPAct":
        return MLPAct(n_in, n_out, act=cfg["activation"])
    if cfg.get("surrogate") == "none":
        return MLP(n_in, n_out)
    return SNN(n_in, n_out, hidden=cfg.get("hidden", 128), surrogate=cfg["surrogate"], width=cfg["width"],
               learnable_width=cfg.get("learnable_width", False))


def load_member(member_dir, device="cpu", n_in=21, n_out=7):
    """Rebuild a member (E16-E18 or original) and load its weights. Returns (model, cfg)."""
    cfg = json.load(open(os.path.join(member_dir, "config.json")))
    model = build_from_config(cfg, n_in, n_out)
    sd = torch.load(os.path.join(member_dir, "model.pt"), map_location=device)
    model.load_state_dict(sd)
    return model.to(device).eval(), cfg
