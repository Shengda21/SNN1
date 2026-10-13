"""Train one candidate (or a whole dictionary block) and dump frozen predictions.

python run_train.py --platform sarcos --block width  --seed 0 --out runs/sarcos
python run_train.py --platform sarcos --block dm     --seed 3 --out runs/sarcos --widths widths.json
blocks: width (dev-stage width grid, every surrogate), dm (5 surrogates at the
frozen widths), ds (5 seeds of the strong surrogate), dt (snapshot run of the
strong surrogate), bl (learnable-width surrogate), ba (noise-augmented strong
surrogate), bn (MLP); augmented counterparts (E09): dma (5 surrogates, each
trained with the B-A augmentation), dsa (5 seeds of the strong surrogate with
augmentation), bna (augmented MLP). --aug_family all|qqd selects the channels
that receive augmentation noise; the qqd family uses the member prefixes
dmq / dsq / bnq.
Each trained member writes <out>/<member_id>/{config.json, history.json,
model.pt, pred_<role>_<cond>.npy}. Predictions are in standardized units.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models import SNN, MLP, n_params, predict, spike_stats
from common.surrogates import SURROGATE_NAMES, WIDTH_GRID
from common.train import train_model, set_seed


def load_platform(platform, data_dir):
    if platform == "sarcos":
        from sarcos.data import load_sarcos, apply_condition
        D = load_sarcos(data_dir, os.path.join(data_dir, "split_manifest.json"))
        return D, apply_condition
    if platform == "sim":
        from sim.data import load_sim, apply_condition
        D = load_sim(data_dir)
        return D, apply_condition
    raise ValueError(platform)


def dump_predictions(model, D, apply_condition, conds, out_dir, device):
    for role in D["X"]:
        if role == "train":
            continue
        for cond in conds:
            Xc = apply_condition(D["X"][role], role, cond)
            if Xc is None:
                continue
            P = predict(model, Xc, device)
            np.save(os.path.join(out_dir, f"pred_{role}_{cond.replace(':', '_')}.npy"), P.astype(np.float32))


def save_member(model, res, cfg, out_dir, D, apply_condition, conds, device):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    cfg = dict(cfg); cfg["n_params"] = n_params(model)
    cfg["best_epoch"] = res["best_epoch"]; cfg["best_dev"] = res["best_dev"]
    cfg["train_time_s"] = res["train_time"]; cfg["epochs_run"] = res["epochs_run"]
    if isinstance(model, SNN):
        cfg["spikes_per_sample_per_layer_dev"] = spike_stats(model, D["X"]["dev"], device)
        if model.learnable_width:
            cfg["learned_widths"] = model.widths.detach().cpu().tolist()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(res["history"], f)
    dump_predictions(model, D, apply_condition, conds, out_dir, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--block", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--widths", default=None, help="json with frozen widths + strong surrogate + conditions")
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--conds", default="N")
    ap.add_argument("--only", default=None, help="width block: restrict to one surrogate")
    ap.add_argument("--aug_family", default="all", choices=["all", "qqd"],
                    help="augmented blocks: noise on all channels (B-A protocol) or on q, qdot only")
    ap.add_argument("--traj_epochs", type=int, default=25, help="traj block: number of replayed epochs")
    ap.add_argument("--hidden", default="4,8,16", help="cap block: hidden widths of the reduced-capacity members")
    ap.add_argument("--dev_sel", default="clean", choices=["clean", "aug"],
                    help="augmented blocks: early stopping on the clean development set (frozen protocol) "
                         "or on the development set perturbed by the augmentation law (member prefix + s)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, apply_condition = load_platform(args.platform, args.data_dir)
    n_in, n_out = D["n_in"], D["n_out"]
    Xtr, Ytr, Xdev, Ydev = D["X"]["train"], D["Y"]["train"], D["X"]["dev"], D["Y"]["dev"]
    conds = args.conds.split(",")
    frozen = json.load(open(args.widths)) if args.widths else None
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)

    def train_snn(member_id, surrogate, width, seed, learnable=False, aug=0.0, snap=0, max_epochs=None, aug_mask=None, dev_aug=False):
        set_seed(seed)
        model = SNN(n_in, n_out, surrogate=surrogate, width=width, learnable_width=learnable)
        res = train_model(model, Xtr, Ytr, Xdev, Ydev, device, seed, aug_sigma=aug,
                          snapshot_cycles=snap, max_epochs=max_epochs or args.max_epochs,
                          patience=args.patience, log=log, tag=member_id, aug_mask=aug_mask, dev_aug=dev_aug)
        cfg = {"member_id": member_id, "surrogate": surrogate, "width": model.width,
               "learnable_width": learnable, "seed": seed, "aug_sigma": aug,
               "snapshot_cycles": snap, "platform": args.platform, "block": args.block,
               "aug_family": ("none" if aug == 0 else ("all" if aug_mask is None else "qqd")),
               "dev_sel": ("aug" if (dev_aug and aug > 0) else "clean")}
        return model, res, cfg

    if args.block == "width":
        for s in SURROGATE_NAMES:
            if args.only and s != args.only:
                continue
            for w in WIDTH_GRID[s]:
                mid = f"width_{s}_{w}_s{args.seed}"
                if os.path.exists(os.path.join(args.out, mid, "config.json")):
                    continue
                model, res, cfg = train_snn(mid, s, w, args.seed)
                save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
        return

    assert frozen is not None, "formal blocks need --widths"
    widths = frozen["widths"]; strong = frozen["strong_surrogate"]
    seed = args.seed
    if args.block == "dm":
        for s in SURROGATE_NAMES:
            mid = f"dm_{s}_s{seed}"
            if os.path.exists(os.path.join(args.out, mid, "config.json")):
                continue
            model, res, cfg = train_snn(mid, s, widths[s], seed)
            save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
    elif args.block == "ds":
        for i in range(5):
            sd = 1000 + seed * 10 + i
            mid = f"ds_{strong}_s{seed}_m{i}"
            if os.path.exists(os.path.join(args.out, mid, "config.json")):
                continue
            model, res, cfg = train_snn(mid, strong, widths[strong], sd)
            save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
    elif args.block == "dt":
        mid0 = f"dt_{strong}_s{seed}"
        if os.path.exists(os.path.join(args.out, mid0 + "_c4", "config.json")):
            return
        model, res, cfg = train_snn(mid0, strong, widths[strong], 2000 + seed, snap=5, max_epochs=args.max_epochs)
        for ci, st in enumerate(res["snapshots"]):
            model.load_state_dict(st)
            c = dict(cfg); c["member_id"] = f"{mid0}_c{ci}"; c["cycle"] = ci
            c["total_run_time_s"] = res["train_time"]
            r = dict(res); r["best_epoch"] = (ci + 1) * (args.max_epochs // 5) - 1; r["best_dev"] = res["history"][r["best_epoch"]][2]
            save_member(model, r, c, os.path.join(args.out, f"{mid0}_c{ci}"), D, apply_condition, conds, device)
    elif args.block == "bl":
        mid = f"bl_learnable_s{seed}"
        if not os.path.exists(os.path.join(args.out, mid, "config.json")):
            model, res, cfg = train_snn(mid, "atan", widths["atan"], seed, learnable=True)
            save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
    elif args.block == "ba":
        mid = f"ba_{strong}_s{seed}"
        if not os.path.exists(os.path.join(args.out, mid, "config.json")):
            model, res, cfg = train_snn(mid, strong, widths[strong], seed, aug=frozen["aug_sigma"])
            save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
    elif args.block == "bn":
        mid = f"bn_mlp_s{seed}"
        if not os.path.exists(os.path.join(args.out, mid, "config.json")):
            set_seed(seed)
            model = MLP(n_in, n_out)
            res = train_model(model, Xtr, Ytr, Xdev, Ydev, device, seed, max_epochs=args.max_epochs,
                              patience=args.patience, log=log, tag=mid)
            cfg = {"member_id": mid, "surrogate": "none", "width": None, "seed": seed, "platform": args.platform,
                   "block": "bn", "learnable_width": False, "aug_sigma": 0.0, "snapshot_cycles": 0}
            save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
    elif args.block in ("dma", "dsa", "bna"):
        # E09: the same blocks as dm / ds / bn, trained with the B-A augmentation
        fam = args.aug_family
        amask = None
        if fam == "qqd":
            amask = np.zeros(n_in, np.float32); amask[:14] = 1.0
        pre = {"dma": "dma", "dsa": "dsa", "bna": "bna"} if fam == "all" else {"dma": "dmq", "dsa": "dsq", "bna": "bnq"}
        dsel = args.dev_sel == "aug"
        if dsel:
            pre = {k: v + "s" for k, v in pre.items()}
        aug = frozen["aug_sigma"]
        if args.block == "dma":
            for s in SURROGATE_NAMES:
                if args.only and s != (strong if args.only == "strong" else args.only):
                    continue
                mid = f"{pre['dma']}_{s}_s{seed}"
                if os.path.exists(os.path.join(args.out, mid, "config.json")):
                    continue
                model, res, cfg = train_snn(mid, s, widths[s], seed, aug=aug, aug_mask=amask, dev_aug=dsel)
                save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
        elif args.block == "dsa":
            for i in range(5):
                sd = 1000 + seed * 10 + i
                mid = f"{pre['dsa']}_{strong}_s{seed}_m{i}"
                if os.path.exists(os.path.join(args.out, mid, "config.json")):
                    continue
                model, res, cfg = train_snn(mid, strong, widths[strong], sd, aug=aug, aug_mask=amask, dev_aug=dsel)
                save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
        else:
            mid = f"{pre['bna']}_mlp_s{seed}"
            if not os.path.exists(os.path.join(args.out, mid, "config.json")):
                set_seed(seed)
                model = MLP(n_in, n_out)
                res = train_model(model, Xtr, Ytr, Xdev, Ydev, device, seed, max_epochs=args.max_epochs,
                                  patience=args.patience, log=log, tag=mid, aug_sigma=aug, aug_mask=amask, dev_aug=dsel)
                cfg = {"member_id": mid, "surrogate": "none", "width": None, "seed": seed, "platform": args.platform,
                       "block": args.block, "learnable_width": False, "aug_sigma": aug, "snapshot_cycles": 0,
                       "aug_family": fam, "dev_sel": args.dev_sel}
                save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
    elif args.block == "traj":
        # E14: replay the first epochs of each D-M member with the same seed, initialization and
        # batch order, keep the state after every epoch, and save as members the two early
        # checkpoints whose development error brackets that of the straight-through member
        # (uf_<surrogate>_s<seed>_a: last epoch above it, _b: first epoch at or below it).
        K = args.traj_epochs
        for s in SURROGATE_NAMES:
            if args.only and s != (strong if args.only == "strong" else args.only):
                continue
            tdir = os.path.join(args.out, f"traj_{s}_s{seed}")
            if os.path.exists(os.path.join(tdir, "history.json")):
                continue
            set_seed(seed)
            model = SNN(n_in, n_out, surrogate=s, width=widths[s])
            res = train_model(model, Xtr, Ytr, Xdev, Ydev, device, seed, max_epochs=K, patience=10 ** 6,
                              log=log, tag=f"traj_{s}_s{seed}", save_states=True)
            os.makedirs(tdir, exist_ok=True)
            torch.save(res["states"], os.path.join(tdir, "states.pt"))
            with open(os.path.join(tdir, "history.json"), "w") as f:
                json.dump({"history": res["history"], "surrogate": s, "width": model.width, "seed": seed, "epochs": K}, f)
            if s == "ste":
                continue
            target = json.load(open(os.path.join(args.out, f"dm_ste_s{seed}", "config.json")))["best_dev"]
            devs = [h[2] for h in res["history"]]
            below = [e for e, dv in enumerate(devs) if dv <= target]
            if not below:
                continue
            eb = below[0]
            for tag, e in [("b", eb)] + ([("a", eb - 1)] if eb >= 1 else []):
                model.load_state_dict(res["states"][e])
                mid = f"uf_{s}_s{seed}_{tag}"
                cfg = {"member_id": mid, "surrogate": s, "width": model.width, "learnable_width": False, "seed": seed,
                       "aug_sigma": 0.0, "snapshot_cycles": 0, "platform": args.platform, "block": "traj", "epoch": e,
                       "target_dev": target, "aug_family": "none", "dev_sel": "matched"}
                r = {"best_epoch": e, "best_dev": devs[e], "train_time": res["train_time"] * (e + 1) / K, "epochs_run": e + 1,
                     "history": res["history"][: e + 1]}
                save_member(model, r, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
    elif args.block == "cap":
        # E14: the development-selected surrogate with a reduced hidden width, same seed and protocol
        for h in [int(x) for x in args.hidden.split(",")]:
            mid = f"cap{h}_{strong}_s{seed}"
            if os.path.exists(os.path.join(args.out, mid, "config.json")):
                continue
            set_seed(seed)
            model = SNN(n_in, n_out, hidden=h, surrogate=strong, width=widths[strong])
            res = train_model(model, Xtr, Ytr, Xdev, Ydev, device, seed, max_epochs=args.max_epochs,
                              patience=args.patience, log=log, tag=mid)
            cfg = {"member_id": mid, "surrogate": strong, "width": model.width, "learnable_width": False, "seed": seed,
                   "aug_sigma": 0.0, "snapshot_cycles": 0, "platform": args.platform, "block": "cap", "hidden": h,
                   "aug_family": "none", "dev_sel": "clean"}
            save_member(model, res, cfg, os.path.join(args.out, mid), D, apply_condition, conds, device)
    else:
        raise ValueError(args.block)


if __name__ == "__main__":
    main()
