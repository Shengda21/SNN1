"""Dump frozen-member predictions for any list of conditions (no training).
python dump_preds.py --platform sarcos --data_dir .. --runs ../runs/sarcos --conds N,O1:0.5 [--members dm_*]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models import SNN, MLP, predict
from run_train import load_platform


def load_member(runs, mid, n_in, n_out, device):
    cfg = json.load(open(os.path.join(runs, mid, "config.json")))
    if cfg.get("surrogate") == "none":
        model = MLP(n_in, n_out)
    else:
        model = SNN(n_in, n_out, hidden=cfg.get("hidden", 128), surrogate=cfg["surrogate"], width=cfg["width"], learnable_width=cfg["learnable_width"])
    model.load_state_dict(torch.load(os.path.join(runs, mid, "model.pt"), map_location=device))
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--members", default="*")
    ap.add_argument("--roles", default="dev,cal,test,cal_P1,test_P1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, apply_condition = load_platform(args.platform, args.data_dir)
    conds = args.conds.split(",")
    dirs = sorted(glob.glob(os.path.join(args.runs, args.members)))
    for d in dirs:
        if not os.path.exists(os.path.join(d, "config.json")):
            continue
        model = None
        for role in args.roles.split(","):
            if role not in D["X"]:
                continue
            for cond in conds:
                fn = os.path.join(d, f"pred_{role}_{cond.replace(':', '_')}.npy")
                if os.path.exists(fn) and not args.overwrite:
                    continue
                Xc = apply_condition(D["X"][role], role, cond)
                if Xc is None:
                    continue
                if model is None:
                    model = load_member(args.runs, os.path.basename(d), D["n_in"], D["n_out"], device)
                np.save(fn, predict(model, Xc, device).astype(np.float32))
        print("done", os.path.basename(d), flush=True)


if __name__ == "__main__":
    main()
