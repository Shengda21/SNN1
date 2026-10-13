"""Training with early stopping on D_dev; lambda = 0 (no spike penalty).

Supports plain training, input-noise augmentation (B-A) and a cosine
warm-restart snapshot schedule (D-T). With the same seed, every surrogate sees
the same initialization and batch order (controlled training block).
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batches(n, batch, gen):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n - batch + 1, batch):
        yield perm[i:i + batch]


def train_model(model, Xtr, Ytr, Xdev, Ydev, device, seed, lr=1e-3, batch=512,
                max_epochs=200, patience=25, aug_sigma=0.0, snapshot_cycles=0,
                log=None, tag="", aug_mask=None, dev_aug=False, save_states=False):
    """Returns dict(history, best_epoch, best_dev, train_time, snapshots, epochs_run).

    aug_sigma > 0: each training batch receives N(0, s^2) input noise with
                   s ~ U(0, aug_sigma) drawn per batch (noise-augmented baseline).
    aug_mask: optional [n_in] 0/1 array restricting the augmentation noise to a
                   subset of channels; the random stream is the same with or without it.
    dev_aug: if True (and aug_sigma > 0) early stopping and the learning-rate schedule
                   use the development inputs perturbed once, with a fixed seed, by the
                   same law as the training augmentation (per-row s ~ U(0, aug_sigma));
                   the default uses the clean development inputs.
    snapshot_cycles > 0: cosine annealing with warm restarts, cycle length
                   max_epochs // snapshot_cycles, state saved at the end of each
                   cycle (snapshot dictionary); no early stopping.
    """
    model = model.to(device)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if snapshot_cycles > 0:
        cycle_len = max_epochs // snapshot_cycles
        sched = None
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Ytr_t = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    Xdev_t = torch.as_tensor(Xdev, dtype=torch.float32, device=device)
    Ydev_t = torch.as_tensor(Ydev, dtype=torch.float32, device=device)
    mask_t = None if aug_mask is None else torch.as_tensor(np.asarray(aug_mask), dtype=torch.float32, device=device)
    if dev_aug and aug_sigma > 0:
        gdev = torch.Generator().manual_seed(777)
        s_dev = torch.rand(len(Xdev_t), 1, generator=gdev) * aug_sigma
        nz_dev = (s_dev * torch.randn(Xdev_t.shape, generator=gdev)).to(device)
        if mask_t is not None:
            nz_dev = nz_dev * mask_t
        Xdev_t = Xdev_t + nz_dev
    best, best_state, best_ep, bad = math.inf, None, -1, 0
    hist, snapshots, states = [], [], []
    t0 = time.time()
    mse = nn.MSELoss()
    for ep in range(max_epochs):
        if snapshot_cycles > 0:
            pos = (ep % cycle_len) / cycle_len
            for g in opt.param_groups:
                g["lr"] = lr * 0.5 * (1 + math.cos(math.pi * pos))
        model.train()
        tr_losses = []
        for idx in _batches(len(Xtr_t), batch, gen):
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            if aug_sigma > 0:
                s = float(torch.rand(1, generator=gen)) * aug_sigma
                nz = s * torch.randn(xb.shape, generator=gen).to(device)
                if mask_t is not None:
                    nz = nz * mask_t
                xb = xb + nz
            out = model(xb)                        # [T, B, d]
            loss = mse(out, yb.unsqueeze(0).expand_as(out))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if torch.isfinite(loss):
                tr_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pd_ = torch.cat([model(Xdev_t[i:i + 8192])[-1] for i in range(0, len(Xdev_t), 8192)], 0)
            dev_loss = float(((pd_ - Ydev_t) ** 2).mean())
        hist.append((ep, float(np.mean(tr_losses)) if tr_losses else float("nan"), dev_loss))
        if log is not None and (ep % 10 == 0 or ep == max_epochs - 1):
            log(f"[{tag}] ep {ep} train {hist[-1][1]:.4f} dev {dev_loss:.4f}")
        if snapshot_cycles > 0:
            if (ep + 1) % cycle_len == 0:
                snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            continue
        if save_states:
            states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        sched.step(dev_loss)
        if np.isfinite(dev_loss) and dev_loss < best - 1e-7:
            best, best_ep, bad = dev_loss, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0
    if snapshot_cycles == 0 and best_state is not None:
        model.load_state_dict(best_state)
    return {"history": hist, "best_epoch": best_ep, "best_dev": best, "train_time": train_time,
            "snapshots": snapshots, "epochs_run": len(hist), "states": states}
