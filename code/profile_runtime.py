"""E04 cost: measured latency and memory for executing k members.
Includes standardization, the T internal SNN steps, readout and the weighted
sum. Reports batch-1 (control query) and batch-30 latency on GPU and CPU
(single thread), and peak GPU memory. Also the MLP baseline."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models import SNN, MLP, n_params


def bench(models, w, device, batch, n_rep=300):
    x = torch.randn(batch, 21, device=device)
    mean = torch.zeros(21, device=device); std = torch.ones(21, device=device)
    with torch.no_grad():
        for _ in range(20):
            out = sum(wj * m((x - mean) / std)[-1] for m, wj in zip(models, w))
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            out = sum(wj * m((x - mean) / std)[-1] for m, wj in zip(models, w))
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    mem = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else float("nan")
    return float(np.median(ts) * 1000), float(np.percentile(ts, 95) * 1000), mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    res = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpu_threads": 1, "rows": []}
    for dev_name in ["cuda", "cpu"]:
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(dev_name)
        snns = [SNN(21, 7, surrogate=s).to(device).eval() for s in ["atan", "sigmoid", "ste", "triangular", "sre"]]
        for k in range(1, 6):
            for batch in [1, 30]:
                med, p95, mem = bench(snns[:k], [1.0 / k] * k, device, batch)
                res["rows"].append({"device": dev_name, "model": "SNN", "k": k, "batch": batch,
                                    "latency_ms_median": med, "latency_ms_p95": p95, "peak_mem_MiB": mem,
                                    "params_total": k * n_params(snns[0])})
        mlp = MLP(21, 7).to(device).eval()
        for batch in [1, 30]:
            med, p95, mem = bench([mlp], [1.0], device, batch)
            res["rows"].append({"device": dev_name, "model": "MLP", "k": 1, "batch": batch,
                                "latency_ms_median": med, "latency_ms_p95": p95, "peak_mem_MiB": mem,
                                "params_total": n_params(mlp)})
    # parameter memory (float32) per executed member
    res["param_bytes_per_snn"] = 4 * n_params(SNN(21, 7))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    for r in res["rows"]:
        print(r)


if __name__ == "__main__":
    main()
