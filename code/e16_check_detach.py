"""Check of the detach-reset variant: forward identical, initialization identical to
common.models.SNN, backward differs (fc1 gradient norm with / without detach)."""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e16_models import SNNX
from common.models import SNN
from common.train import set_seed


def flat_grad(m):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in m.parameters()])


torch.manual_seed(123)
x = torch.randn(256, 21) * 2.0
y = torch.randn(256, 7)
for layers in ([["ste", None, True]] * 3, [["tri", 0.25, True]] * 3, [["tri", 4.0, True]] * 3,
               [["atan", 16.0, True]] * 3):
    outs, grads, fc1 = [], [], []
    for det in (False, True):
        set_seed(0)
        m = SNNX(21, 7, layers, detach_reset=det)
        o = m(x)
        outs.append(o.detach())
        ((o - y) ** 2).mean().backward()
        grads.append(flat_grad(m))
        fc1.append(m.fc1.weight.grad.norm().item())
    set_seed(0)
    ref = SNN(21, 7, surrogate="ste")
    set_seed(0)
    new = SNNX(21, 7, layers, detach_reset=True)
    same_init = all(torch.equal(a, b) for a, b in zip(ref.parameters(), new.parameters()))
    rel = ((grads[0] - grads[1]).norm() / grads[0].norm()).item()
    print(f"{layers[0][0]}:{layers[0][1]} forward_equal={torch.equal(outs[0], outs[1])} init_equal_SNN={same_init} "
          f"grad_rel_diff={rel:.3f} |dL/dW1| no-detach={fc1[0]:.3e} detach={fc1[1]:.3e}")
