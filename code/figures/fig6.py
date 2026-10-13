"""Fig. S2: open-loop risk against closed-loop tracking at O1-M (E13, E13c: weights of the closed-loop run).
 a) tracking RMSE against the total open-loop nMSE of each feedforward model
 b) tracking RMSE against the slow part (0.25-s moving average) of the open-loop residual
 Filled markers: shared feedback law; hollow markers: feedback from the true state (diagnostic).
 c) the same two open-loop quantities per model, sorted by tracking error.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
KIND = {"B-N": ("MLP", "s", "#8c8c8c"), "D-M:W-S": ("clean members", "o", "#1f6f8b"), "D-S:W-Q5": ("clean members", "o", "#1f6f8b"),
        "D-M:W-Q5": ("clean members", "o", "#1f6f8b"), "D-S+null:W-Q5": ("null candidate", "v", "#c0392b"), "D-M+null:W-Q5": ("null candidate", "v", "#c0392b"),
        "D-S|est:W-Q5": ("input estimator", "^", "#e59866"), "D-M|est:W-Q5": ("input estimator", "^", "#e59866"),
        "D-S,est-optional:W-Q5": ("input estimator", "^", "#e59866"), "B-N|est": ("input estimator", "^", "#e59866"),
        "B-A": ("augmented members", "D", "#2e8b57"), "B-A+(q,qd)": ("augmented members", "D", "#2e8b57"), "D-SQ:W-Q5": ("augmented members", "D", "#2e8b57")}
SHORT = {"B-N": "MLP", "D-M:W-S": "D-M single", "D-S:W-Q5": "D-S", "D-M:W-Q5": "D-M", "D-S+null:W-Q5": "D-S + null", "D-M+null:W-Q5": "D-M + null",
         "D-S|est:W-Q5": "D-S, est.", "D-M|est:W-Q5": "D-M, est.", "D-S,est-optional:W-Q5": "D-S, est. optional", "B-N|est": "MLP, est.",
         "B-A": "B-A", "B-A+(q,qd)": "B-A$^{+}$ $(q,\\dot q)$", "D-SQ:W-Q5": "D-SA $(q,\\dot q)$"}


def spearman(a, b):
    return float(np.corrcoef(pd.Series(list(a)).rank(), pd.Series(list(b)).rank())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S = pd.read_csv(os.path.join(args.sim, "e13_summary.csv")); lf = pd.read_csv(os.path.join(args.sim, "e13c_lowfreq_matched.csv"))
    o1m = "O1M"
    L = lf[lf.cond == o1m].groupby("method").mean(numeric_only=True)
    L0 = lf[(lf.cond == o1m) & (lf.group == 0)].groupby("method").mean(numeric_only=True)   # true-state diagnostic: group 0
    C = S[S.cond == "O1M"].set_index("controller")
    ms = [m for m in KIND if m in L.index and m in C.index]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    seen = set()
    for ax, col, xl in [(axes[0], "nmse_total", "open-loop nMSE (total)"), (axes[1], "nmse_slow_L25", "open-loop nMSE, slow part (0.25 s)")]:
        ax2 = ax.twinx()
        for m in ms:
            kind, mk, colr = KIND[m]
            ax.scatter(L.loc[m, col], 1000 * C.loc[m, "rmse"], s=16, marker=mk, color=colr, edgecolor="black", lw=0.3, zorder=3,
                       label=kind if (kind not in seen and ax is axes[0]) else None)
            if ax is axes[0]:
                seen.add(kind)
            if m + "@true-fb" in C.index:
                ax2.scatter(L0.loc[m, col], 1000 * C.loc[m + "@true-fb", "rmse"], s=16, marker=mk, facecolor="none", edgecolor=colr, lw=0.9, zorder=3)
        tf = [m for m in ms if m + "@true-fb" in C.index]
        r1 = spearman(L.loc[ms, col], C.loc[ms, "rmse"]); r2 = spearman(L0.loc[tf, col], C.loc[[m + "@true-fb" for m in tf], "rmse"])
        ax.set_title(f"Spearman {r1:+.2f} (filled), {r2:+.2f} (hollow)", fontsize=6.5)
        ax.set_xlabel(xl); ax.set_ylabel("tracking RMSE ($10^{-3}$ rad)")
        ax2.set_ylabel("with true-state feedback", color="#555555")
        ax2.tick_params(axis="y", colors="#555555")
        ax.set_ylim(48, 55); ax2.set_ylim(6, 22)
        ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    ax = axes[2]
    order = sorted(ms, key=lambda m: C.loc[m, "rmse"])
    y = np.arange(len(order))
    ax.barh(y - 0.2, [L.loc[m, "nmse_total"] for m in order], 0.38, color="#c9c9c9", edgecolor="black", lw=0.3, label="total")
    ax.barh(y + 0.2, [L.loc[m, "nmse_slow_L25"] for m in order], 0.38, color="#1f6f8b", edgecolor="black", lw=0.3, label="slow part")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in order], fontsize=5.6); ax.invert_yaxis()
    ax.set_xlabel("open-loop nMSE at O1-M"); ax.set_title("sorted by tracking error, best on top", fontsize=6.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=5.8)
    ax.spines[["top", "right"]].set_visible(False)
    for a, l in zip(axes, "abc"):
        a.text(-0.3, 1.1, l, transform=a.transAxes, fontsize=9, fontweight="bold")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.36, -0.03), fontsize=5.8, columnspacing=1.0, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.8)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print("fig6 written", args.out)


if __name__ == "__main__":
    main()
