"""Progress of the E16-E18 queue: completed / failed / running per stage, with the
nominal test nMSE and O1-M test nMSE of completed members (quick look, not the summary).

python e16_progress.py [--detail e16s0]
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
CLAIMS = os.path.join(HERE, "..", "logs", "e16_claims")
O1M = {"sim": "O1_0.7", "sarcos": "O1_0.5"}
_Y = {}


def ytest(p):
    if p not in _Y:
        if p == "sarcos":
            from sarcos.data import load_sarcos
            _Y[p] = load_sarcos(os.path.join(HERE, ".."), None)["Y"]["test"]
        else:
            from sim.data import load_sim
            _Y[p] = load_sim(os.path.join(HERE, "..", "data_sim"))["Y"]["test"]
    return _Y[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(HERE, "..", "logs", "e16_tasks.jsonl"))
    ap.add_argument("--detail", default="")
    args = ap.parse_args()
    tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
    st = defaultdict(lambda: defaultdict(int))
    order = []
    detail = []
    times = defaultdict(list)
    for t in tasks:
        s = t["stage"]
        if s not in order:
            order.append(s)
        key = f"{t['platform']}__{t['member_id']}"
        d = os.path.join(RUNS, t["platform"], t["member_id"])
        st[s]["total"] += 1
        if os.path.exists(os.path.join(d, "config.json")):
            st[s]["done"] += 1
            cfg = json.load(open(os.path.join(d, "config.json")))
            times[t["platform"]].append(cfg["train_time_s"])
            if s in args.detail.split(","):
                Y = ytest(t["platform"])
                n = cfg.get("test_nmse_N")
                P = np.load(os.path.join(d, f"pred_test_{O1M[t['platform']]}.npy"))
                m = float(np.mean(np.sum((P - Y) ** 2, 1) / Y.shape[1]))
                detail.append((t["platform"], t["member_id"], cfg["best_dev"], n, m, cfg["best_epoch"], cfg["epochs_run"],
                               cfg["train_time_s"]))
        elif os.path.exists(os.path.join(CLAIMS, key + ".failed")):
            st[s]["failed"] += 1
        elif os.path.exists(os.path.join(CLAIMS, key + ".claim")):
            st[s]["running"] += 1
    for s in order:
        c = st[s]
        print(f"{s:8s} total {c['total']:3d} done {c['done']:3d} running {c['running']:2d} failed {c['failed']:2d}")
    for p, v in times.items():
        print(f"mean train time {p}: {np.mean(v) / 60:.1f} min over {len(v)}")
    if detail:
        print("platform member best_dev nmse_N nmse_O1M best_ep epochs time_min")
        for r in sorted(detail):
            print(f"{r[0]:6s} {r[1]:24s} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]:3d} {r[6]:3d} {r[7] / 60:.1f}")


if __name__ == "__main__":
    main()
