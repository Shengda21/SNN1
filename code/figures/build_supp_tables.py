"""Supplementary tables not produced by tables.py: S1 spikes/training, S3 m/k/E07.
python build_supp_tables.py [<repository root>] [<output dir>]   (defaults: the repository, build/locked/tables;
the S3 tables read the slice summaries that tables.py wrote to the output dir)"""
import collections
import glob
import json
import os

import numpy as np
import pandas as pd

import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")   # repository root
OUT = sys.argv[2] if len(sys.argv) > 2 else "build/locked/tables"
os.makedirs(OUT, exist_ok=True)

# ---------------- S1: spikes and training per member family
FAM_LABEL = {"dm_atan": "D-M ATan", "dm_sigmoid": "D-M sigmoid", "dm_ste": "D-M STE", "dm_triangular": "D-M triangular", "dm_sre": "D-M escape rate",
             "ds_triangular": "D-S (triangular)", "ds_sigmoid": "D-S (sigmoid)", "dt_triangular": "D-T (triangular)", "dt_sigmoid": "D-T (sigmoid)",
             "bl_learnable": "B-L learnable width", "ba_triangular": "B-A noise-augmented", "ba_sigmoid": "B-A noise-augmented", "bn_mlp": "B-N MLP"}
rows = []
for p, pname in [("sarcos", "SARCOS"), ("sim", "Simulated arm")]:
    acc = collections.defaultdict(list)
    for d in sorted(glob.glob(f"{R}/runs_frozen/{p}/*/config.json")):
        c = json.load(open(d)); name = c["member_id"]
        fam = name.rsplit("_s", 1)[0]
        if fam.startswith("width_"):
            continue
        sp = c.get("spikes_per_sample_per_layer_dev")
        sp = json.loads(sp) if isinstance(sp, str) else sp
        acc[fam].append((sp, c["best_epoch"], c["epochs_run"], c["train_time_s"], c["best_dev"]))
    for fam in ["dm_atan", "dm_sigmoid", "dm_ste", "dm_triangular", "dm_sre", "ds_triangular", "ds_sigmoid", "dt_triangular", "dt_sigmoid", "bl_learnable", "ba_triangular", "ba_sigmoid", "bn_mlp"]:
        if fam not in acc:
            continue
        v = acc[fam]
        sps = [x[0] for x in v if x[0] is not None]
        sp = np.mean(np.array(sps, float), 0) if sps else None
        be = np.mean([x[1] for x in v]); er = np.mean([x[2] for x in v]); tt = np.mean([x[3] for x in v]) / 60; bd = np.mean([x[4] for x in v])
        spt = "--" if sp is None else " / ".join(f"{s:.0f}" for s in sp[:3])
        rows.append(f"{pname} & {FAM_LABEL[fam]} & {len(v)} & {spt} & {be:.0f} / {er:.0f} & {tt:.1f} & {bd:.4f} \\\\")
lines = [r"\begin{table}[h]", r"\centering",
         r"\caption{Training summary per member family (mean over runs). Spikes: mean spikes per sample over the five internal steps in the three spiking layers on $\Ddev$ (128 neurons each; the maximum is 640 per layer). Epochs: best epoch / epochs run before early stopping (snapshot runs use a fixed 300). Time is wall-clock training time per run on one RTX 3090. Dev nMSE is the development error of the selected checkpoint.}",
         r"\label{tab:s1spikes}", r"\small", r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Platform & Member & Runs & Spikes L1 / L2 / L3 & Epochs & Time (min) & Dev nMSE \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/s1_spikes.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ---------------- S3: m-slice (sim) and k-slice tables, E07
m = pd.read_csv(f"{OUT}/s3_m_slice_summary.csv")
rules = ["W-S", "W-E", "W-B", "W-D", "W-Q5"]
lines = [r"\begin{table}[h]", r"\centering",
         r"\caption{Calibration budget on the simulated arm (E03): test nMSE of each rule on D-M (mean over five seeds and ten groups; seed standard deviation in parentheses), its calibration suboptimality $g_{\mathrm{cal},5}$ on $\widehat C$, and the mean support size. $m=0$ is the development-selected member.}",
         r"\label{tab:s3m}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llccc}", r"\toprule",
         r"Condition & $m$ & Rule & nMSE (sd) & $g_{\mathrm{cal},5}$ / support \\", r"\midrule"]
for cond in m.cond.unique():
    sub = m[m.cond == cond]
    bf = sub[sub.rule == "B-F(m=0)"]
    if len(bf):
        lines.append(f"{cond} & 0 & B-F & {bf.iloc[0].nmse:.4f} ({bf.iloc[0].nmse_sd:.4f}) & -- / 1 \\\\")
    for mm in sorted(sub.m.unique()):
        if mm == 0:
            continue
        for rule in rules:
            r = sub[(sub.m == mm) & (sub.rule == rule)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            lines.append(f"{cond} & {mm} & {rule} & {r.nmse:.4f} ({r.nmse_sd:.4f}) & {r.g_cal:.4f} / {r.support:.2f} \\\\")
    lines.append(r"\midrule")
lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/s3_m.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

k = pd.read_csv(f"{OUT}/s3_k_slice_summary.csv")
kmax = k.kkt.max()
lines = [r"\begin{table}[h]", r"\centering",
         r"\caption{Execution budget on the simulated arm (E04), $m=10$: test nMSE of W-Q($k$) per dictionary (mean over seeds and groups; seed standard deviation in parentheses) and the mean size of the returned support [in brackets]. The largest first-order residual of Eq.~(3) over all solutions is $" + ("%.1e" % kmax).replace("e-", r"\times10^{-") + "}$.}",
         r"\label{tab:s3k}", r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llccccc}", r"\toprule",
         r"Condition & Dictionary & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=5$ \\", r"\midrule"]
for cond in k.cond.unique():
    for dn in ["D-M", "D-S", "D-T"]:
        sub = k[(k.cond == cond) & (k.dictionary == dn)].sort_values("k")
        if len(sub) == 0:
            continue
        cells = [f"{r.nmse:.4f} ({r.nmse_sd:.4f}) [{r.support:.1f}]" for _, r in sub.iterrows()]
        lines.append(f"{cond} & {dn} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
open(f"{OUT}/s3_k.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

e07 = pd.read_csv(f"{R}/results/sim/e07_density.csv")
g = e07.groupby(["m_ind", "sampling", "rule"]).agg(nmse=("nmse", "mean"), sd=("nmse", lambda x: x.groupby(e07.loc[x.index, "seed"]).mean().std()), frames=("frames", "first"), supp=("support_size", "mean")).reset_index()
lines = [r"\begin{table}[h]", r"\centering",
         r"\caption{Equal frame count, different information (E07, O1-M, simulated arm): test nMSE (seed standard deviation) and mean support of each rule when calibrated from $m$ independent trajectories or from ten trajectories subsampled to the same number of frames.}",
         r"\label{tab:s3e07}", r"\small", r"\begin{tabular}{lcccc}", r"\toprule",
         r"Frames & Sampling & W-S & W-B & W-Q(5) \\", r"\midrule"]
for mi in sorted(g.m_ind.unique()):
    for samp in ["independent", "dense"]:
        sub = g[(g.m_ind == mi) & (g.sampling == samp)]
        cells = []
        for rule in ["W-S", "W-B", "W-Q5"]:
            r = sub[sub.rule == rule]
            cells.append("--" if len(r) == 0 else f"{r.iloc[0].nmse:.4f} ({r.iloc[0].sd:.4f}) [{r.iloc[0].supp:.1f}]")
        lab = f"{int(sub.frames.iloc[0])}" if len(sub) else "--"
        lines.append(f"{lab} & {mi} traj." if samp == "independent" else f"{lab} & 10 traj. subsampled")
        lines[-1] += " & " + " & ".join(cells) + r" \\"
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(f"{OUT}/s3_e07.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

print("written: s1_spikes, s3_m, s3_k, s3_e07; kkt max", kmax)
