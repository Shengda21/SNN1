"""Build LaTeX tables and the results manifest from frozen results.
 Table II : core prediction results at N and O1-M (both platforms) + costs
 S2 tables: full E01 per condition (both platforms), seed std
 S3 tables: m-slice and k-slice numbers, E07
 results_manifest.csv : every number -> source file / method / condition
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MAIN_ROWS = [("B-F", "D-M", "Dev-selected single SNN (B-F)"), ("B-L", "-", "Learned ATan-width SNN (B-L)"),
             ("B-A", "-", "Noise-augmented SNN (B-A)"), ("B-N", "-", "MLP, 55k parameters (B-N)"),
             ("W-E", "D-S", "Seed dictionary, equal (D-S, W-E)"), ("W-Q5", "D-S", "Seed dictionary, W-Q(5)"),
             ("W-E", "D-T", "Snapshot dictionary, equal (D-T, W-E)"), ("W-Q5", "D-T", "Snapshot dictionary, W-Q(5)"),
             ("W-S", "D-M", "Multi-surrogate, calibrated single (D-M, W-S)"), ("W-E", "D-M", "Multi-surrogate, equal (D-M, W-E)"),
             ("W-B", "D-M", "Multi-surrogate, BIC-inspired (D-M, W-B)"), ("W-D", "D-M", "Multi-surrogate, direct choice (D-M, W-D)"),
             ("W-Q2", "D-M", "Multi-surrogate, W-Q(2)"), ("W-Q5", "D-M", "Multi-surrogate, W-Q(5)"),
             ("posthoc-opt", "D-M", "D-M post-hoc test optimum (reference only)")]


def fmt(m, lo, hi):
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


def train_cost(runs, block_prefix, seeds=(0, 1, 2, 3, 4)):
    ts = []
    for d in glob.glob(os.path.join(runs, block_prefix + "*")):
        cfg = json.load(open(os.path.join(d, "config.json")))
        ts.append(cfg.get("total_run_time_s", cfg["train_time_s"]))
    return np.mean(ts) if ts else np.nan, len(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarcos", required=True); ap.add_argument("--sim", required=True)
    ap.add_argument("--runs_sarcos", required=True); ap.add_argument("--runs_sim", required=True)
    ap.add_argument("--frozen_sarcos", required=True); ap.add_argument("--frozen_sim", required=True)
    ap.add_argument("--runtime", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fs, fm = json.load(open(args.frozen_sarcos)), json.load(open(args.frozen_sim))
    ss = pd.read_csv(os.path.join(args.sarcos, "e01_summary.csv")); sm = pd.read_csv(os.path.join(args.sim, "e01_summary.csv"))
    rt = json.load(open(args.runtime)); rows_rt = pd.DataFrame(rt["rows"])
    manifest = []

    def get(summ, cond, method, dn):
        r = summ[(summ.cond == cond) & (summ.method == method) & (summ.dictionary == dn)]
        return None if len(r) == 0 else r.iloc[0]

    # ---------------- Table II (main)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Core open-loop results and costs. Test nMSE (standardized units, Eq.~1) with 95\% cluster-bootstrap intervals over five training seeds and test units; calibration uses the full $\Dcal$ of the same condition. Costs are measured on the simulated-arm runs: training time on one NVIDIA RTX 3090 of the deployed model or dictionary (D-S and D-M five runs, D-T one run); selecting the surrogate on $\Ddev$ trained the five D-M members once (36.1 min), a cost that B-F, B-A, D-S, and D-T share; calibration evaluates all five members once on $\Dcal$; execution is the measured batch-1 latency of the executed support. The post-hoc optimum is computed on the test matrix and serves as a reference.}",
             r"\label{tab:cost}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Method & \multicolumn{2}{c}{SARCOS} & \multicolumn{2}{c}{Simulated arm} & Executed & Train (min) \\",
             r" & N & O1-M & N & O1-M & members & deployed \\", r"\midrule"]
    O1s, O1m = f"O1:{fs['O1_M']}", f"O1:{fm['O1_M']}"
    cost_dm = train_cost(args.runs_sim, "dm_")[0] * 5 / 60; cost_ds = train_cost(args.runs_sim, "ds_")[0] * 5 / 60
    cost_dt = train_cost(args.runs_sim, "dt_")[0] / 60; cost_bl = train_cost(args.runs_sim, "bl_")[0] / 60
    cost_ba = train_cost(args.runs_sim, "ba_")[0] / 60; cost_bn = train_cost(args.runs_sim, "bn_")[0] / 60
    cost_bf = train_cost(args.runs_sim, "dm_" + fm.get("strong_surrogate", "sigmoid") + "_")[0] / 60
    cost = {"B-F": cost_bf, "B-L": cost_bl, "B-A": cost_ba, "B-N": cost_bn, "D-S": cost_ds, "D-T": cost_dt, "D-M": cost_dm}
    for method, dn, label in MAIN_ROWS:
        cells = []
        for summ, c1, c2, plat in [(ss, "N", O1s, "sarcos"), (sm, "N", O1m, "sim")]:
            for c in [c1, c2]:
                r = get(summ, c, method, dn)
                if r is None:
                    cells.append("--")
                else:
                    cells.append(f"{r.nmse_mean:.4f}")
                    manifest.append({"table": "II", "platform": plat, "cond": c, "method": method, "dictionary": dn,
                                     "value": r.nmse_mean, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "source": f"results/{plat}/e01_summary.csv"})
        r = get(sm, O1m, method, dn)
        nexec = "--" if r is None else ("1" if method in ("B-F", "B-L", "B-A", "B-N", "W-S") else ("5" if method in ("W-E", "W-B", "posthoc-opt") else f"{r.get('support_mean', np.nan):.1f}" if "support_mean" in r else ("$\\le$5" if method == "W-D" else "$\\le$" + method[-1])))
        key = method if method in cost else dn      # B-F: the one strong-surrogate run per seed, not the D-M dictionary
        lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & {nexec} & {cost.get(key, np.nan):.1f} \\\\")
    lines += [r"\midrule"]
    gpu, cpu = [], []
    for k in range(1, 6):
        r = rows_rt[(rows_rt.device == "cuda") & (rows_rt.model == "SNN") & (rows_rt.batch == 1) & (rows_rt.k == k)]
        rc = rows_rt[(rows_rt.device == "cpu") & (rows_rt.model == "SNN") & (rows_rt.batch == 1) & (rows_rt.k == k)]
        gpu.append(f"{r.iloc[0].latency_ms_median:.2f}"); cpu.append(f"{rc.iloc[0].latency_ms_median:.2f}")
    lines.append("\\multicolumn{7}{l}{Execution of $k=1,\\dots,5$ SNN members on the GPU: " + ", ".join(gpu) + " ms} \\\\")
    lines.append("\\multicolumn{7}{l}{Execution on one CPU thread (Intel Xeon W-2245): " + ", ".join(cpu)
                 + f" ms; parameters: {rt['param_bytes_per_snn'] / 1024:.0f} KiB per member}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(os.path.join(args.out, "table2.tex"), "w").write("\n".join(lines))

    # ---------------- S2 full tables
    for summ, plat, fz in [(ss, "sarcos", fs), (sm, "sim", fm)]:
        conds = list(summ.cond.unique())
        pname = "SARCOS" if plat == "sarcos" else "Simulated arm"
        def cl(c):
            if c in ("N", "P1"): return c
            kind, rho = c.split(":"); rho = float(rho)
            lev = {fz["O1_L"]: "L", fz["O1_M"]: "M", fz["O1_H"]: "H"}.get(rho, f"{rho}")
            return {"O1": f"O1-{lev}", "O1qqd": f"O1-{lev} on $(q,\\dot q)$", "O2B": "O2-B"}.get(kind, c) + f" ($\\rho={rho}$)"
        lines = [r"\begin{table}[h]", r"\centering", f"\\caption{{{pname}: test nMSE of every method under every condition, mean [95\\% cluster-bootstrap CI] (seed standard deviation).}}", f"\\label{{tab:s2{plat}}}", r"\scriptsize",
                 r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * len(conds) + "}", r"\toprule", "Method & " + " & ".join(cl(c) for c in conds) + r" \\", r"\midrule"]
        for method, dn, label in MAIN_ROWS + [(f"single:{s}", "D-M", f"Member {n}") for s, n in [("atan", "ATan"), ("sigmoid", "sigmoid"), ("ste", "STE"), ("triangular", "triangular"), ("sre", "escape rate")]]:
            cells = []
            for c in conds:
                r = get(summ, c, method, dn)
                cells.append("--" if r is None else f"{r.nmse_mean:.4f} [{r.ci_lo:.4f},{r.ci_hi:.4f}] ({r.seed_std:.4f})")
                if r is not None:
                    manifest.append({"table": "S2", "platform": plat, "cond": c, "method": method, "dictionary": dn, "value": r.nmse_mean,
                                     "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "source": f"results/{plat}/e01_summary.csv"})
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
        open(os.path.join(args.out, f"s2_{plat}.tex"), "w").write("\n".join(lines))
    # ---------------- S3: m and k slices (sim)
    e03 = pd.read_csv(os.path.join(args.sim, "e03_m_slice.csv")); e04 = pd.read_csv(os.path.join(args.sim, "e04_k_slice.csv"))
    g = e03.groupby(["cond", "rule", "m"]).agg(nmse=("nmse", "mean"), nmse_sd=("nmse", "std"), g_cal=("g_cal_5", "mean"), support=("support_size", "mean")).reset_index()
    g.to_csv(os.path.join(args.out, "s3_m_slice_summary.csv"), index=False)
    g4 = e04.groupby(["cond", "dictionary", "k"]).agg(nmse=("nmse", "mean"), nmse_sd=("nmse", "std"), support=("support_size", "mean"), kkt=("kkt", "max")).reset_index()
    g4.to_csv(os.path.join(args.out, "s3_k_slice_summary.csv"), index=False)
    for _, r in g.iterrows():
        manifest.append({"table": "S3-m", "platform": "sim", "cond": r.cond, "method": r.rule, "dictionary": "D-M", "m": r.m, "value": r.nmse, "source": "results/sim/e03_m_slice.csv"})
    for _, r in g4.iterrows():
        manifest.append({"table": "S3-k", "platform": "sim", "cond": r.cond, "method": f"W-Q{int(r.k)}", "dictionary": r.dictionary, "value": r.nmse, "source": "results/sim/e04_k_slice.csv"})
    # ---------------- fig1 results json (panel a from sim D-M)
    def pick(c, m, dn):
        r = get(sm, c, m, dn); return None if r is None else float(r.nmse_mean)
    members = {c: min([(pick(c, f"single:{s}", "D-M"), s) for s in ["atan", "sigmoid", "ste", "triangular", "sre"] if pick(c, f"single:{s}", "D-M") is not None]) for c in ["N", O1m]}
    fig1 = {"panel_a": {k: {"best_member": members[c][0], "best_name": members[c][1], "equal": pick(c, "W-E", "D-M"), "wq5": pick(c, "W-Q5", "D-M")}
                        for k, c in [("N", "N"), ("O1M", O1m)]}}
    json.dump(fig1, open(os.path.join(args.out, "fig1_results.json"), "w"), indent=1)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out, "results_manifest.csv"), index=False)
    print("tables written", len(manifest), "manifest rows")


if __name__ == "__main__":
    main()
