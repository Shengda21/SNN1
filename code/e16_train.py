"""E16-E18 training worker (new file; run_train.py and common/*.py are untouched).

python e16_train.py --tasks ../logs/e16_tasks.jsonl --worker 0

Every worker walks the same priority-ordered task list and trains each member that
(a) has no config.json yet, (b) is not marked as failed, and (c) it can claim
atomically (O_CREAT|O_EXCL on <claims>/<platform>__<member_id>.claim). Several workers
therefore share one queue without duplicates. Training protocol = original formal blocks:
common.train.train_model, Adam 1e-3, batch 512, per-internal-step loss, grad-norm clip 1,
ReduceLROnPlateau(0.5, 8), early stopping on the clean development set (patience 30,
max 300 epochs), set_seed(seed) before model construction (same init and batch order as
the original members with the same seed).

Output: <runs>/<platform>/<member_id>/{model.pt, history.json, pred_<role>_<cond>.npy,
config.json}; config.json is written last and marks completion. Predictions are float32
in standardized units, for the frozen.json conditions on every non-train role (sim: also
cal_P1 / test_P1 under N), exactly as run_train.dump_predictions.
"""
import argparse
import json
import math
import os
import socket
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.models import n_params, predict, spike_stats
from common.train import train_model, set_seed
from e16_models import SNNX, MLPAct

DATA_DIR = {"sarcos": os.path.join(HERE, ".."), "sim": os.path.join(HERE, "..", "data_sim")}
FROZEN = {p: os.path.join(HERE, "..", "runs25", p, "frozen.json") for p in ("sim", "sarcos")}
# E16_RUNS / E16_MAX_EPOCHS overrides exist only for smoke tests into a scratch directory
RUNS = os.environ.get("E16_RUNS", os.path.join(HERE, "..", "runs"))
MAX_EPOCHS, PATIENCE, LR, BATCH = int(os.environ.get("E16_MAX_EPOCHS", 300)), 30, 1e-3, 512


def load_platform(platform):
    # same loaders as run_train.load_platform (copied to stay independent of edits there)
    dd = DATA_DIR[platform]
    if platform == "sarcos":
        from sarcos.data import load_sarcos, apply_condition
        # manifest is written to the scratch path, never over the shared split_manifest.json
        return load_sarcos(dd, None), apply_condition
    from sim.data import load_sim, apply_condition
    return load_sim(dd), apply_condition


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


def build(task, n_in, n_out, frozen):
    if task["kind"] == "mlp":
        return MLPAct(n_in, n_out, act=task["activation"])
    return SNNX(n_in, n_out, task["layers"], mem_noise=task.get("membrane_noise", 0.0),
                detach_reset=bool(task.get("detach_reset", False)))


def run_task(task, cache, device, log):
    plat = task["platform"]
    if plat not in cache:
        D, ac = load_platform(plat)
        cache[plat] = (D, ac, json.load(open(FROZEN[plat])))
    D, apply_condition, frozen = cache[plat]
    n_in, n_out = D["n_in"], D["n_out"]
    mid = task["member_id"]; seed = int(task["seed"])
    out_dir = os.path.join(RUNS, plat, mid)
    aug = float(task.get("aug_scale", 0.0)) * float(frozen["aug_sigma"])
    set_seed(seed)
    model = build(task, n_in, n_out, frozen)
    res = train_model(model, D["X"]["train"], D["Y"]["train"], D["X"]["dev"], D["Y"]["dev"], device, seed,
                      lr=LR, batch=BATCH, max_epochs=MAX_EPOCHS, patience=PATIENCE, aug_sigma=aug,
                      log=log, tag=f"{plat}/{mid}")
    ok = res["best_dev"] is not None and math.isfinite(res["best_dev"])
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(res["history"], f)
    cfg = {k: v for k, v in task.items() if k not in ("layers", "kind")}
    cfg.update({"platform": plat, "seed": seed, "learnable_width": False, "snapshot_cycles": 0,
                "aug_sigma": aug, "aug_family": "none" if aug == 0 else "all", "dev_sel": "clean",
                "lr": LR, "batch": BATCH, "max_epochs": MAX_EPOCHS, "patience": PATIENCE})
    if task["kind"] == "mlp":
        cfg.update({"model_class": "MLPAct", "surrogate": "none", "width": None, "hidden": 220,
                    "activation": task["activation"], "surrogate_layers": None, "membrane_noise": 0.0})
    else:
        cfg.update({"model_class": "SNNX", "hidden": 128, "surrogate_layers": task["layers"],
                    "layer_params": model.layer_params, "membrane_noise": float(task.get("membrane_noise", 0.0)),
                    "activation": "spike", "detach_reset": bool(model.detach_reset)})
        cfg.setdefault("width", model.width)
    cfg["n_params"] = n_params(model)
    cfg["best_epoch"] = res["best_epoch"]; cfg["best_dev"] = res["best_dev"]
    cfg["train_time_s"] = res["train_time"]; cfg["epochs_run"] = res["epochs_run"]
    cfg["train_failed"] = not ok
    if isinstance(model, SNNX):
        cfg["spikes_per_sample_per_layer_dev"] = spike_stats(model, D["X"]["dev"], device)
    if ok:
        dump_predictions(model, D, apply_condition, frozen["conds"], out_dir, device)
        P = np.load(os.path.join(out_dir, "pred_test_N.npy"))
        Y = D["Y"]["test"]
        cfg["test_nmse_N"] = float(np.mean(np.sum((P - Y) ** 2, 1) / Y.shape[1]))
        cfg["preds_finite"] = bool(np.isfinite(P).all())
    cfg["host"] = socket.gethostname(); cfg["torch"] = torch.__version__
    cfg["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # a failed run leaves failed.json (no config.json), so analysis scripts never pick it up
    final = "config.json" if ok else "failed.json"
    with open(os.path.join(out_dir, final + ".tmp"), "w") as f:
        json.dump(cfg, f, indent=1)
    os.replace(os.path.join(out_dir, final + ".tmp"), os.path.join(out_dir, final))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--worker", default="0")
    ap.add_argument("--claims", default=os.path.join(HERE, "..", "logs", "e16_claims"))
    ap.add_argument("--fail_log", default=os.path.join(HERE, "..", "logs", "e16_failures.jsonl"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    os.makedirs(args.claims, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = lambda s: print(time.strftime("%m-%d %H:%M:%S"), f"w{args.worker}", s, flush=True)
    cache = {}
    done = 0
    while True:
        # re-read the task file every pass so that appended stages are picked up
        tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
        picked = None
        for t in tasks:
            key = f"{t['platform']}__{t['member_id']}"
            if os.path.exists(os.path.join(RUNS, t["platform"], t["member_id"], "config.json")):
                continue
            if os.path.exists(os.path.join(args.claims, key + ".failed")):
                continue
            try:
                fd = os.open(os.path.join(args.claims, key + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode()); os.close(fd)
            picked = (t, key)
            break
        if picked is None:
            log(f"queue empty, trained {done}")
            break
        t, key = picked
        t0 = time.time()
        log(f"START {key}")
        try:
            cfg = run_task(t, cache, device, log)
            if cfg["train_failed"]:
                raise RuntimeError(f"non-finite best_dev {cfg['best_dev']}")
            log(f"DONE {key} best_dev {cfg['best_dev']:.5f} ep {cfg['best_epoch']}/{cfg['epochs_run']} "
                f"testN {cfg.get('test_nmse_N', float('nan')):.5f} {time.time() - t0:.0f}s")
            done += 1
        except Exception as e:
            log(f"FAIL {key}: {e!r}")
            traceback.print_exc()
            with open(os.path.join(args.claims, key + ".failed"), "w") as f:
                f.write(repr(e) + "\n")
            with open(args.fail_log, "a") as f:
                f.write(json.dumps({"key": key, "error": repr(e), "time": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
