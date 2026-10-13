"""Frozen forward architecture shared by every SNN candidate.

Architecture (identical for all surrogates; only the backward function differs):
  fc1 -> LIF1 (spiking, subtract reset)
  fc2 -> LIF2 (spiking, subtract reset)
  fc3 -> LIF3 (spiking, subtract reset)
  fc4(tanh(mem3)) -> LI4 (non-spiking leaky integrator)
  fc5(mem4) -> LI5 readout (non-spiking leaky integrator); prediction = mem5 at t = T
The input vector is repeated for T = 5 internal steps (direct-current encoding) and
the internal state is reset at every query. beta and threshold are learnable per
neuron (initial beta = 0.9, threshold = 1.0). This forward architecture is shared by
every spiking candidate of the paper (54,293 parameters for 21 -> 7).
"""
import torch
import torch.nn as nn
from .surrogates import make_surrogate, _LearnableATan


class LeakyLayer(nn.Module):
    """Explicit LIF / LI update: mem <- beta*mem + cur; spk = H(mem - thr);
    spiking layers subtract thr after a spike."""
    def __init__(self, size, spike_fn, spiking=True, beta=0.9, thr=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.full((size,), float(beta)))
        self.thr = nn.Parameter(torch.full((size,), float(thr)))
        self.spike_fn = spike_fn
        self.spiking = spiking

    def forward(self, cur, mem):
        beta = self.beta.clamp(0.0, 1.0)
        mem = beta * mem + cur
        if self.spiking:
            spk = self.spike_fn(mem - self.thr)
            mem = mem - spk * self.thr
            return spk, mem
        return None, mem


class SNN(nn.Module):
    def __init__(self, n_in, n_out, hidden=128, T=5, surrogate="atan", width=None,
                 learnable_width=False):
        super().__init__()
        self.T = T
        self.hidden = hidden
        self.n_in, self.n_out = n_in, n_out
        self.surrogate, self.learnable_width = surrogate, learnable_width
        if learnable_width:
            a0 = 2.0 if width is None else float(width)
            self.widths = nn.Parameter(torch.full((3,), a0))
            fns = [self._learnable_fn(i) for i in range(3)]
            self.width = a0
        else:
            fn, self.width = make_surrogate(surrogate, width)
            fns = [fn, fn, fn]
        self.fc1 = nn.Linear(n_in, hidden); self.lif1 = LeakyLayer(hidden, fns[0])
        self.fc2 = nn.Linear(hidden, hidden); self.lif2 = LeakyLayer(hidden, fns[1])
        self.fc3 = nn.Linear(hidden, hidden); self.lif3 = LeakyLayer(hidden, fns[2])
        self.fc4 = nn.Linear(hidden, hidden); self.li4 = LeakyLayer(hidden, None, spiking=False)
        self.fc5 = nn.Linear(hidden, n_out); self.li5 = LeakyLayer(n_out, None, spiking=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _learnable_fn(self, i):
        return lambda u: _LearnableATan.apply(u, self.widths[i:i + 1])

    def forward(self, x, return_spikes=False):
        B = x.shape[0]
        dev = x.device
        m1 = torch.zeros(B, self.hidden, device=dev); m2 = torch.zeros_like(m1)
        m3 = torch.zeros_like(m1); m4 = torch.zeros_like(m1)
        m5 = torch.zeros(B, self.n_out, device=dev)
        outs, spk_counts = [], []
        for _ in range(self.T):
            s1, m1 = self.lif1(self.fc1(x), m1)
            s2, m2 = self.lif2(self.fc2(s1), m2)
            s3, m3 = self.lif3(self.fc3(s2), m3)
            _, m4 = self.li4(self.fc4(torch.tanh(m3)), m4)
            _, m5 = self.li5(self.fc5(m4), m5)
            outs.append(m5)
            if return_spikes:
                spk_counts.append(torch.stack([s1.sum(1), s2.sum(1), s3.sum(1)], 1))
        out = torch.stack(outs, 0)   # [T, B, n_out]
        if return_spikes:
            return out, torch.stack(spk_counts, 0)   # [T, B, 3]
        return out


class MLP(nn.Module):
    """Non-spiking baseline B-N; 220 hidden units give 55,007 parameters for 21 -> 7."""
    def __init__(self, n_in, n_out, hidden=220):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))
        self.T = 1

    def forward(self, x):
        return self.net(x).unsqueeze(0)   # [1, B, n_out]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def predict(model, X, device, batch=8192):
    """Prediction at the last internal step; numpy in, numpy out (float32)."""
    import numpy as np
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            outs.append(model(xb)[-1].float().cpu().numpy())
    return np.concatenate(outs, 0)


def spike_stats(model, X, device, batch=8192):
    """Mean spikes per sample per spiking layer over the internal window (for S1 counts)."""
    import numpy as np
    model.eval()
    tot = None; n = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            _, spk = model(xb, return_spikes=True)   # [T,B,3]
            s = spk.sum(0).sum(0).cpu().numpy()
            tot = s if tot is None else tot + s
            n += xb.shape[0]
    return (tot / n).tolist()
