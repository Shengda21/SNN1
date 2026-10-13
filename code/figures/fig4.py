"""Fig. 4: closed loop on the simulated arm, E13 numbers only (identical noise draws across controllers).
 a) tracking RMSE under N, O1-M and P1 for PD, B-N, B-A, D-M W-S, D-S W-Q(5), D-M W-Q(5), D-S + null and
    D-S behind the input estimator: mean over (seed, episode) cells after averaging the calibration groups, 95%
    two-way cluster-bootstrap interval over seeds and episodes (the procedure of summarize_e13.py), from
    e13_episodes_*.csv; thin column: fraction of saturated control steps at O1-M; PD failures written on its bar
 b) total open-loop test nMSE (e13c_lowfreq_matched.csv: weights of the closed-loop run, mean over seeds and groups) against tracking RMSE at O1-M for the 13
    feedforward models, shared feedback law (top, filled) and feedback from the true state (bottom, hollow, 7 models)
 c) the same against the nMSE of the 0.25-s moving average of the residual (slow part)
 Spearman values are read from e13_rank_correlations.json.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.titlesize": 7, "axes.labelsize": 6.8,
                     "legend.fontsize": 6.3, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "hatch.linewidth": 0.5, "pdf.fonttype": 42})
COL = {"B-F": "#a6a6a6", "B-L": "#c27c0e", "B-A": "#2e8b57", "B-N": "#595959", "D-S": "#7fb3c8", "D-T": "#9b59b6",
       "D-M": "#1f6f8b", "null": "#c0392b", "est": "#e59866", "PD": "#d9d9d9"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
# (e13 controller name, paper label, face colour, edge colour, hatch)
CTRL = [("PD", "PD only", COL["PD"], "#8a8a8a", "////"), ("B-N", "B-N MLP", COL["B-N"], COL["B-N"], ""),
        ("B-A", "B-A noise-augmented", COL["B-A"], COL["B-A"], ""), ("D-M:W-S", "D-M single (W-S)", "white", COL["D-M"], "//////"),
        ("D-S:W-Q5", "D-S W-Q(5)", COL["D-S"], COL["D-S"], ""), ("D-M:W-Q5", "D-M W-Q(5)", COL["D-M"], COL["D-M"], ""),
        ("D-S+null:W-Q5", "D-S + null", COL["null"], COL["null"], ""), ("D-S|est:W-Q5", "D-S, input estimator", COL["est"], COL["est"], "")]
CONDS = [("N", "N"), ("O1M", "O1-M"), ("P1", "P1")]
# the 13 feedforward models of E13: (kind colour, marker, fillstyle, short label or None)
FF = {"B-N": (COL["B-N"], "s", "full", "MLP"), "B-N|est": (COL["est"], "^", "full", None),
      "B-A": (COL["B-A"], "D", "full", "B-A"), "B-A+(q,qd)": (COL["B-A"], "D", "full", None), "D-SQ:W-Q5": (COL["B-A"], "D", "full", None),
      "D-M:W-S": (COL["D-M"], "o", "left", "D-M single"), "D-S:W-Q5": (COL["D-S"], "o", "full", "D-S"), "D-M:W-Q5": (COL["D-M"], "o", "full", "D-M"),
      "D-S+null:W-Q5": (COL["null"], "v", "full", None), "D-M+null:W-Q5": (COL["null"], "v", "full", None),
      "D-S|est:W-Q5": (COL["est"], "^", "full", None), "D-M|est:W-Q5": (COL["est"], "^", "full", None),
      "D-S,est-optional:W-Q5": (COL["est"], "^", "full", None)}


# direct labels in panels b (k=0) and c (k=1): offset in points and alignment; D-M and B-A coincide at O1-M
LABEL_POS = {(0, "MLP"): (-5, 2, "right"), (0, "D-S"): (5, -1, "left"), (0, "D-M single"): (5, 0, "left"), (0, "D-M"): (6, 3, "left"),
             (1, "MLP"): (5, 0, "left"), (1, "D-S"): (-5, -3, "right"), (1, "D-M single"): (5, 0, "left"), (1, "D-M"): (-5, 0, "right"),
             (1, "B-A"): (-5, 0, "right")}
LABEL_TXT = {(0, "D-M"): "D-M, B-A"}


def boot(mat, n_boot=2000, seed=0):
    """[n_seed, n_episode] -> mean and 95% interval of a two-way bootstrap over seeds and episodes."""
    rng = np.random.default_rng(seed)
    ns, ne = mat.shape
    b = np.empty(n_boot)
    for i in range(n_boot):
        b[i] = mat[rng.integers(0, ns, ns)][:, rng.integers(0, ne, ne)].mean()
    return float(mat.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def clean(ax, grid="x"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)


def letter(fig, x, y, l, text):
    fig.text(x, y, l, fontsize=8.5, fontweight="bold", va="top", ha="left")
    fig.text(x + 0.02, y - 0.003, text, fontsize=7, fontweight="bold", va="top", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.sim, "e13_episodes_*.csv")))
    e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cell = e.groupby(["cond", "controller", "seed", "episode"]).agg(rmse=("rmse", "mean"), failed=("failed", "mean"),
                                                                     saturation=("saturation", "mean")).reset_index()
    S13 = pd.read_csv(os.path.join(args.sim, "e13_summary.csv")).set_index(["cond", "controller"])
    stats = {}
    for cond, _ in CONDS:
        for name, *_ in CTRL:
            s = cell[(cell.cond == cond) & (cell.controller == name)]
            m, lo, hi = boot(s.pivot(index="seed", columns="episode", values="rmse").values)
            stats[(cond, name)] = {"m": m, "lo": lo, "hi": hi, "fail": s.failed.mean(), "sat": s.saturation.mean(), "n_seeds": s.seed.nunique()}
            ref = S13.loc[(cond, name)]
            assert abs(ref.rmse - m) < 1e-9 and abs(ref.ci_lo - lo) < 1e-9, (cond, name, m, ref.rmse)
    lf = pd.read_csv(os.path.join(args.sim, "e13c_lowfreq_matched.csv"))   # weights of the closed-loop run (E13c)
    o1m = "O1M"
    L = lf[lf.cond == o1m].groupby("method").mean(numeric_only=True)
    L0 = lf[(lf.cond == o1m) & (lf.group == 0)].groupby("method").mean(numeric_only=True)   # true-state diagnostic: group 0
    rc = json.load(open(os.path.join(args.sim, "e13_rank_correlations.json")))

    W, H = 7.16, 5.0
    fig = plt.figure(figsize=(W, H))
    # ---------------- a: tracking RMSE per condition, horizontal bars with direct labels
    top, h1 = 0.885, 0.295
    x0, cw, gap = 0.165, 0.205, 0.02
    xlim = {"N": 18, "O1M": 62, "P1": 36}
    y = np.arange(len(CTRL))
    for j, (cond, cname) in enumerate(CONDS):
        ax = fig.add_axes([x0 + j * (cw + gap), top - h1, cw, h1])
        for i, (name, lab, fc, ec, hatch) in enumerate(CTRL):
            st = stats[(cond, name)]
            v, lo, hi = 1000 * st["m"], 1000 * st["lo"], 1000 * st["hi"]
            if v > xlim[cond]:   # PD: bar cut at the axis edge, value and failure rate written on it
                ax.barh(i, xlim[cond], 0.7, color=fc, edgecolor=ec, hatch=hatch, lw=0.0, zorder=2)
                xb = 0.80 * xlim[cond]
                ax.plot([xb - 0.012 * xlim[cond], xb + 0.012 * xlim[cond]], [i + 0.42, i - 0.42], color="white", lw=2.2, zorder=3, solid_capstyle="butt")
                ax.text(0.03 * xlim[cond], i, f"{v:.0f}; {100 * st['fail']:.0f}% of episodes fail", va="center", ha="left", fontsize=6,
                        color=INK, zorder=4, bbox=dict(fc=COL["PD"], ec="none", pad=0.6))
                continue
            ax.barh(i, v, 0.7, color=fc, edgecolor=ec, hatch=hatch, lw=0.8 if fc == "white" else 0, zorder=2)
            ax.plot([lo, hi], [i, i], color=INK, lw=0.7, zorder=3)
            ax.text(hi + 0.015 * xlim[cond], i, f"{v:.1f}", va="center", ha="left", fontsize=6, color=INK, zorder=4)
        ax.set_xlim(0, xlim[cond]); ax.set_ylim(len(CTRL) - 0.45, -0.55)
        ax.set_yticks(y); ax.set_yticklabels([c[1] for c in CTRL] if j == 0 else [])
        if j:
            ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"tracking RMSE at {cname} (mrad)", labelpad=1)
        clean(ax)
        if j == 0:
            letter(fig, 0.008, 0.985, "a", "Tracking with the shared PD law (E13)")
    # thin column: saturation at O1-M
    axs = fig.add_axes([x0 + 3 * (cw + gap) + 0.01, top - h1, 0.095, h1])
    for i, (name, lab, fc, ec, hatch) in enumerate(CTRL):
        st = stats[("O1M", name)]
        axs.barh(i, 100 * st["sat"], 0.7, color=fc, edgecolor=ec, hatch=hatch, lw=0.8 if fc == "white" else 0, zorder=2)
    axs.set_xlim(0, 6.5); axs.set_xticks([0, 2, 4, 6]); axs.set_ylim(len(CTRL) - 0.45, -0.55)
    axs.set_yticks(y); axs.set_yticklabels([]); axs.tick_params(axis="y", length=0)
    axs.set_xlabel("saturated steps\nat O1-M (%)", labelpad=1)
    nsat = max(stats[(c, n)]["sat"] for c in ("N", "P1") for n, *_ in CTRL)
    axs.text(0.5, 1.02, "none under N, P1" if nsat == 0 else f"N, P1: max {100 * nsat:.1f}%", transform=axs.transAxes, ha="center", va="bottom",
             fontsize=6, color=MUTED)
    clean(axs)
    fig.legend([Line2D([0], [0], color=INK, lw=0.7)], ["95% two-way cluster-bootstrap interval (seeds, episodes)"],
               loc="lower right", bbox_to_anchor=(0.995, top + 0.04), frameon=False, handlelength=1.6)

    # ---------------- b, c: open-loop risk against tracking at O1-M
    C13 = S13.loc["O1M"]
    ms = [m for m in FF if m in L.index and m in C13.index]
    tf = [m for m in ms if m + "@true-fb" in C13.index]
    yb, hb = 0.075, 0.34
    for k, (col, xl, key, lab) in enumerate([("nmse_total", "total open-loop test nMSE at O1-M", "total", "b"),
                                             ("nmse_slow_L25", "open-loop nMSE of the 0.25-s slow part at O1-M", "slow_L25", "c")]):
        xa = 0.075 + k * 0.5
        ax1 = fig.add_axes([xa, yb + hb * 0.52, 0.40, hb * 0.48])
        ax2 = fig.add_axes([xa, yb, 0.40, hb * 0.44])
        for m in ms:
            c, mk, fs, short = FF[m]
            x = L.loc[m, col]
            ax1.plot(x, 1000 * C13.loc[m, "rmse"], mk, ms=4.6 if mk != "D" else 3.9, color=c, mec=c, fillstyle=fs,
                     mfc=c, mfcalt="white", mew=0.8, zorder=3)
            if m in tf:
                ax2.plot(L0.loc[m, col], 1000 * C13.loc[m + "@true-fb", "rmse"], mk, ms=4.6 if mk != "D" else 3.9, mfc="white", mec=c, mew=0.9, zorder=3)
            if short and (k, short) in LABEL_POS:
                dx, dy, ha = LABEL_POS[(k, short)]
                ax1.annotate(LABEL_TXT.get((k, short), short), (x, 1000 * C13.loc[m, "rmse"]), xytext=(dx, dy), textcoords="offset points",
                             fontsize=6, color=INK, va="center", ha=ha)
        r1, r2 = rc["O1M"][key], rc["O1M@true-fb"][key]
        ax1.text(0.02 if k == 0 else 0.98, 0.95, f"shared feedback, {rc['O1M']['n_models']} models: Spearman {r1:+.2f}", transform=ax1.transAxes,
                 ha="left" if k == 0 else "right", va="top", fontsize=6.2, color=INK)
        ax2.text(0.02 if k == 0 else 0.98, 0.95, f"true-state feedback, {rc['O1M@true-fb']['n_models']} models: Spearman {r2:+.2f}", transform=ax2.transAxes,
                 ha="left" if k == 0 else "right", va="top", fontsize=6.2, color=INK)
        ax1.set_ylim(47.8, 56.3); ax2.set_ylim(5, 25)
        xs = L.loc[ms, col]
        pad = 0.06 * (xs.max() - xs.min())
        for a in (ax1, ax2):
            a.set_xlim(xs.min() - pad, xs.max() + pad)
            clean(a, grid="y")
        ax1.set_xticklabels([])
        ax2.set_xlabel(xl, labelpad=1)
        fig.text(xa - 0.052, yb + hb / 2, "tracking RMSE at O1-M (mrad)", rotation=90, ha="center", va="center", fontsize=6.8)
        letter(fig, xa - 0.067, yb + hb + 0.095, lab, "Total open-loop risk vs. tracking" if k == 0 else "Slow part of the risk vs. tracking")
    kinds = [("o", COL["D-S"], "full", "clean-trained: D-S"), ("o", COL["D-M"], "full", "D-M"), ("o", COL["D-M"], "left", "D-M single"),
             ("v", COL["null"], "full", "+ null"), ("^", COL["est"], "full", "input estimator"),
             ("D", COL["B-A"], "full", "augmented"), ("s", COL["B-N"], "full", "MLP"), ("o", MUTED, "none", "hollow: true-state feedback")]
    fig.legend([Line2D([0], [0], marker=mk, ls="", color=c, mec=c if fs != "none" else MUTED, mfc=c if fs != "none" else "white",
                       fillstyle="full" if fs == "none" else fs, mfcalt="white", ms=4.4, mew=0.8) for mk, c, fs, _ in kinds],
               [k[3] for k in kinds], loc="upper center", bbox_to_anchor=(0.5, yb + hb + 0.06), ncol=8, frameon=False,
               handletextpad=0.15, columnspacing=0.9, fontsize=6.2)
    fig.savefig(args.out)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)
    print("saved", args.out, "| episodes from", [os.path.basename(f) for f in files], "| seeds", sorted(e.seed.unique()))


if __name__ == "__main__":
    main()
