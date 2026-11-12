"""E15 item 7 only: regenerate the weight-statistics table (e15_weights<suffix>.csv) without
re-running the plane / direction computations.

$PY e15_weights.py --platform sim [--runs ../runs25/sim] [--members '*'] [--suffix '']
Columns: per layer fc1..fc5 (MLP: net.0/2/4) ||W||_F, ||W - W0||_F, rel_change = ||W - W0||_F / ||W0||_F,
spectral norms and stable ranks; per spiking layer threshold mean / std / min, fraction of neurons
with |thr - 1| < 0.02, mean |1 - thr|, fraction thr > 1, and beta statistics.
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e15_common import PLATFORM_DEFAULTS, member_group, list_members, load_member, out_name  # noqa: E402
from e15_position import weight_rows, traj_init_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--members", default="*")
    ap.add_argument("--out", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--no_traj_check", action="store_true")
    args = ap.parse_args()
    runs = args.runs or f"../runs25/{args.platform}"
    out_dir = args.out or f"../results/{args.platform}"
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    members = list_members(runs, args.members)
    t0 = time.time()
    for mdir in members:
        mid = os.path.basename(mdir)
        model, cfg, kind = load_member(mdir, torch.device("cpu"))
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        rows.extend(weight_rows(mid, member_group(mid, cfg), cfg, model, kind, sd))
    if not args.no_traj_check:
        rows.extend(traj_init_check(runs))
    pd.DataFrame(rows).to_csv(out_name(out_dir, "e15_weights.csv", args.suffix), index=False)
    print(f"{len(members)} members, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
