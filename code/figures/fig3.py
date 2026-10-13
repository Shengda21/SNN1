"""Fig. 3: calibration budget, execution budget, and substitutes for the surrogate dictionary.
 a) arm: test nMSE of W-S, W-E, W-Q(5) on D-M against the number m of independent calibration trajectories
    (1,000 control steps each), under O1-M (top) and N (bottom); dotted: development-selected member (e03_m_slice.csv)
 b) arm, O1-M, m = 10: test nMSE of W-Q(k) on D-M and D-S against k (e04_k_slice.csv), with the measured
    batch-1 latency of k executed members on the GPU and on one CPU thread on the right-hand axis (runtime.json)
 c) both platforms: paired margin D-M - D-S under W-Q(5) at O1-M and O1-H for the plain simplex, + null,
    behind the input estimator, and among augmented members (e09_paired.csv, the Table III sources)
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.titlesize": 7, "axes.labelsize": 6.8,
                     "legend.fontsize": 6.3, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.minor.size": 0, "pdf.fonttype": 42})
COL = {"B-F": "#a6a6a6", "B-L": "#c27c0e", "B-A": "#2e8b57", "B-N": "#595959", "D-S": "#7fb3c8", "D-T": "#9b59b6",
       "D-M": "#1f6f8b", "null": "#c0392b", "est": "#e59866", "PD": "#d9d9d9"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
RULES = [("W-S", "W-S (calibrated single)", "^", (0, (4, 1.5, 1, 1.5)), "white"), ("W-E", "W-E (equal)", "s", (0, (3, 1.5)), "white"),
         ("W-Q5", "W-Q(5)", "o", "-", COL["D-M"])]
RUNGS = [("plain", "D-S", "D-M", COL["D-M"]), ("+ null", "D-S+0", "D-M+0", COL["null"]),
         ("input estimator", "D-S|wfe", "D-M|wfe", COL["est"]), ("augmented\nmembers", "D-SA", "D-MA", COL["B-A"])]


def ci(vals):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(0)
    b = [np.mean(rng.choice(vals, len(vals))) for _ in range(1000)]
    return np.mean(vals), np.percentile(b, 2.5), np.percentile(b, 97.5)


TITLE_Y, LEG_Y = 0.985, 0.915


def letter(fig, ax, l, text, dx=0.0):
    x = ax.get_position().x0 + dx
    fig.text(x, TITLE_Y, l, fontsize=8.5, fontweight="bold", va="top", ha="left")
    fig.text(x + 0.02, TITLE_Y - 0.004, text, fontsize=7, fontweight="bold", va="top", ha="left")


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--frozen_sim", required=True)
    ap.add_argument("--sarcos", default=None, help="default: results/sarcos next to --sim")
    ap.add_argument("--frozen_sarcos", default=None, help="default: frozen_sarcos.json next to --frozen_sim")
    ap.add_argument("--runtime", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sarcos = args.sarcos or os.path.join(os.path.dirname(os.path.normpath(args.sim)), "sarcos")
    fz_s = args.frozen_sarcos or os.path.join(os.path.dirname(args.frozen_sim), "frozen_sarcos.json")
    FZ = {"sarcos": json.load(open(fz_s)), "sim": json.load(open(args.frozen_sim))}
    DIR = {"sarcos": sarcos, "sim": args.sim}
    O1M = f"O1:{FZ['sim']['O1_M']}"
    e03 = pd.read_csv(os.path.join(args.sim, "e03_m_slice.csv")); e04 = pd.read_csv(os.path.join(args.sim, "e04_k_slice.csv"))
    rt = pd.DataFrame(json.load(open(args.runtime))["rows"])
    W, H = 7.16, 2.85
    fig = plt.figure(figsize=(W, H))
    yb, ht = 0.165, 0.565

    # ---------------- a: m slice, O1-M (top) and N (bottom)
    ax_top = fig.add_axes([0.075, yb + ht * 0.53, 0.235, ht * 0.47])
    ax_bot = fig.add_axes([0.075, yb, 0.235, ht * 0.47])
    for ax, cond in [(ax_top, O1M), (ax_bot, "N")]:
        sub = e03[e03.cond == cond]
        dev = sub[sub.rule == "B-F(m=0)"]
        if len(dev):
            ax.axhline(dev.groupby(["seed", "group"]).nmse.mean().mean(), color=COL["B-F"], ls=(0, (1, 1.2)), lw=1.2, zorder=1)
        for rule, lab, mk, ls, fc in RULES:
            r = sub[sub.rule == rule]
            ms = sorted(r.m.unique())
            pts = [ci(r[r.m == m].groupby(["seed", "group"]).nmse.mean().values) for m in ms]
            mean = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
            ax.errorbar(ms, mean, yerr=[mean - lo, hi - mean], fmt=mk, ls=ls, color=COL["D-M"], mfc=fc, mec=COL["D-M"], ms=3.6,
                        lw=0.9, mew=0.8, capsize=0, elinewidth=0.6, zorder=3)
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10]); ax.set_xlim(0.85, 12)
        ax.text(0.03, 0.55, "N" if cond == "N" else "O1-M", transform=ax.transAxes, ha="left", va="center",
                fontsize=6.8, fontweight="bold", color=MUTED)
        ax.set_ylabel("test nMSE", labelpad=1)
        clean(ax)
    ax_top.set_xticklabels([]); ax_bot.set_xticklabels(["1\n(1,000 steps)", "2", "5", "10"])
    ax_bot.set_xlabel("independent calibration trajectories $m$", labelpad=1)
    ax_top.set_ylim(0.35, 0.41); ax_top.set_yticks([0.36, 0.38, 0.40])
    ax_bot.set_ylim(0.028, 0.05); ax_bot.set_yticks([0.03, 0.04, 0.05])
    leg = [Line2D([0], [0], color=COL["D-M"], ls=ls, marker=mk, mfc=fc, mec=COL["D-M"], ms=3.6, lw=0.9) for _, _, mk, ls, fc in RULES]
    leg += [Line2D([0], [0], color=COL["B-F"], ls=(0, (1, 1.2)), lw=1.2)]
    fig.legend(leg, [r[1] for r in RULES] + ["dev-selected member"], loc="upper left", bbox_to_anchor=(0.005, LEG_Y),
               ncol=2, frameon=False, handlelength=2.4, columnspacing=1.0, labelspacing=0.2, fontsize=6.2)
    letter(fig, ax_top, "a", "Calibration budget $m$, arm", dx=-0.068)

    # ---------------- b: k slice with measured latency
    axb = fig.add_axes([0.42, yb, 0.21, ht])
    axr = axb.twinx()
    sub = e04[e04.cond == O1M]
    ks = sorted(sub.k.unique())
    g = rt[(rt.model == "SNN") & (rt.batch == 1)]
    for dev, off, fc, lab in [("cuda", -0.17, "#cfcfcf", "GPU latency"), ("cpu", 0.17, "#ececec", "CPU latency (1 thread)")]:
        r = g[g.device == dev].set_index("k").reindex(ks)
        axr.bar(np.array(ks) + off, r.latency_ms_median.values, 0.32, color=fc, edgecolor="#9a9a9a", lw=0.4, zorder=1)
    for dn in ["D-S", "D-M"]:
        r = sub[sub.dictionary == dn]
        pts = [ci(r[r.k == k].groupby(["seed", "group"]).nmse.mean().values) for k in ks]
        mean = np.array([p[0] for p in pts]); lo = np.array([p[1] for p in pts]); hi = np.array([p[2] for p in pts])
        axb.errorbar(ks, mean, yerr=[mean - lo, hi - mean], fmt="-o", color=COL[dn], ms=3.6, lw=1.0, capsize=0, elinewidth=0.6, zorder=4)
    axb.set_zorder(axr.get_zorder() + 1); axb.patch.set_visible(False)
    axb.set_xticks(ks); axb.set_xlabel("executed members $k$ ($m=10$, O1-M)", labelpad=1)
    axb.set_ylabel("test nMSE of W-Q($k$)", labelpad=1); axb.set_ylim(0.345, 0.415)
    axr.set_ylabel("latency per query, batch 1 (ms)", color=MUTED, labelpad=2); axr.tick_params(axis="y", colors=MUTED)
    axr.set_ylim(0, 14.5)
    axb.spines[["top"]].set_visible(False); axr.spines[["top"]].set_visible(False)
    mlp = rt[(rt.model == "MLP") & (rt.batch == 1)].set_index("device").latency_ms_median
    axb.text(0.97, 0.97, f"MLP: {mlp['cuda']:.2f} ms GPU,\n{mlp['cpu']:.2f} ms CPU", transform=axb.transAxes, ha="right", va="top", fontsize=6, color=MUTED)
    axb.legend([Line2D([0], [0], color=COL["D-M"], marker="o", ms=3.6, lw=1.0), Line2D([0], [0], color=COL["D-S"], marker="o", ms=3.6, lw=1.0),
                Patch(fc="#cfcfcf", ec="#9a9a9a", lw=0.4), Patch(fc="#ececec", ec="#9a9a9a", lw=0.4)],
               ["D-M, W-Q($k$)", "D-S, W-Q($k$)", "GPU latency", "CPU latency (1 thread)"], loc="upper left", bbox_to_anchor=(0.36, LEG_Y),
               bbox_transform=fig.transFigure, ncol=2, frameon=False,
               handlelength=1.4, columnspacing=0.7, handletextpad=0.3, fontsize=6.2)
    letter(fig, axb, "b", "Execution budget $k$, arm", dx=-0.06)

    # ---------------- c: margin D-M - D-S, both platforms, O1-M and O1-H
    axc = fig.add_axes([0.80, yb, 0.185, ht])
    marks = [("sarcos", "O1_M", "o", True, "SARCOS, O1-M"), ("sarcos", "O1_H", "o", False, "SARCOS, O1-H"),
             ("sim", "O1_M", "s", True, "arm, O1-M"), ("sim", "O1_H", "s", False, "arm, O1-H")]
    offs = [-0.27, -0.09, 0.09, 0.27]
    P = {p: pd.read_csv(os.path.join(DIR[p], "e09_paired.csv")) for p in DIR}
    for i, (lab, ds, dm, col) in enumerate(RUNGS):
        for (p, lev, mk, filled, _), off in zip(marks, offs):
            c = f"O1:{FZ[p][lev]}"
            r = P[p][(P[p].cond == c) & (P[p].a == f"W-Q5|{dm}") & (P[p].b == f"W-Q5|{ds}")]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            axc.plot([r.ci_lo, r.ci_hi], [i + off] * 2, color=col, lw=0.8, zorder=2)
            axc.plot(r["diff"], i + off, mk, ms=3.6, mfc=col if filled else "white", mec=col, mew=0.9, zorder=3)
    axc.axvline(0, color=INK, lw=0.6, zorder=1)
    axc.set_yticks(range(len(RUNGS))); axc.set_yticklabels([r[0] for r in RUNGS]); axc.set_ylim(len(RUNGS) - 0.5, -0.5)
    axc.set_xlabel("D-M $-$ D-S, test nMSE of W-Q(5)", labelpad=1)
    axc.set_xlim(-0.11, 0.03); axc.set_xticks([-0.1, -0.05, 0])
    clean(axc, grid="x")
    axc.legend([Line2D([0], [0], marker=mk, ls="", mfc=MUTED if f else "white", mec=MUTED, mew=0.9, ms=3.6) for _, _, mk, f, _ in marks],
               [m[4] for m in marks], loc="upper left", bbox_to_anchor=(0.69, LEG_Y), bbox_transform=fig.transFigure, ncol=2, frameon=False, handletextpad=0.2,
               columnspacing=0.6, labelspacing=0.15, fontsize=6.2)
    letter(fig, axc, "c", "D-M margin under substitutes", dx=-0.11)

    fig.savefig(args.out)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()
