"""Fig. S1: attenuation controls and augmentation (E09-E12).
 Row 1 SARCOS, row 2 simulated arm.
 a, b / d, e) W-Q(5) test nMSE of the seed and the surrogate dictionary at O1-M and O1-H on
        five rungs: plain simplex, + null predictor, behind the label-free input estimator,
        estimator + null, members trained with augmentation. Dashed line: the B-A network.
        The number above each pair is the paired difference D-M minus D-S (an asterisk
        marks a 95% interval that excludes zero).
 c / f) propagated-perturbation term against nominal term of every member at O1-M.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
C_S, C_M, C_A = "#7fb3c8", "#1f6f8b", "#8c8c8c"
RUNGS = [("plain", "D-S", "D-M"), ("+ null", "D-S+0", "D-M+0"), ("input\nest.", "D-S|wfe", "D-M|wfe"),
         ("est.\n+ null", "D-S|wfe+0", "D-M|wfe+0"), ("augm.\nmembers", "D-SA", "D-MA")]


def ladder(ax, summ, pairs, cond, title, legend=False):
    xs, labs = [], []
    ymax = 0
    for i, (lab, ds, dm) in enumerate(RUNGS):
        got = False
        for off, dn, col in [(-0.2, ds, C_S), (0.2, dm, C_M)]:
            r = summ[(summ.cond == cond) & (summ.method == "W-Q5") & (summ.dictionary == dn)]
            if len(r) == 0:
                continue
            r = r.iloc[0]; got = True
            ax.bar(i + off, r.nmse_mean, 0.38, color=col, edgecolor="black", lw=0.3,
                   label={C_S: "seed dictionary", C_M: "surrogate dictionary"}[col] if (legend and i == 0) else None)
            ax.plot([i + off] * 2, [r.ci_lo, r.ci_hi], color="black", lw=0.6)
            ymax = max(ymax, r.ci_hi)
        if got:
            xs.append(i); labs.append(lab)
            p = pairs[(pairs.cond == cond) & (pairs.a == f"W-Q5|{dm}") & (pairs.b == f"W-Q5|{ds}")]
            if len(p):
                p = p.iloc[0]
                star = "*" if (p.ci_lo > 0 or p.ci_hi < 0) else ""
                top = summ[(summ.cond == cond) & (summ.method == "W-Q5") & (summ.dictionary.isin([ds, dm]))].ci_hi.max()
                ax.text(i, top * 1.02, f"{p['diff']:+.3f}{star}", ha="center", va="bottom", fontsize=5.3, zorder=5,
                        bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))
    ba = summ[(summ.cond == cond) & (summ.method == "B-A")]
    if len(ba):
        ax.axhline(ba.iloc[0].nmse_mean, color="black", ls="--", lw=0.7, label="B-A (one augmented network)" if legend else None)
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=5.6)
    ax.set_ylim(0, ymax * 1.16)
    ax.set_title(title); ax.set_ylabel("test nMSE, W-Q(5)")
    ax.spines[["top", "right"]].set_visible(False)


def scatter(ax, mem, cond, title, legend=False):
    m = mem[mem.cond == cond]
    spec = [("D-S", None, "o", C_S, "seed members"), ("D-M", "window", "o", C_M, "window surrogates"), ("D-M", "ste", "^", "#c0392b", "straight-through"),
            ("D-SA", None, "o", "#9ccc9c", "augmented seeds"), ("D-MA", "window", "o", "#2e8b57", "augmented window"), ("D-MA", "ste", "^", "#e59866", "augmented straight-through"),
            ("B-A", None, "*", "black", "B-A"), ("B-N", None, "s", C_A, "MLP"), ("B-NA", None, "D", "#555555", "augmented MLP")]
    for dn, sel, mk, col, lab in spec:
        s = m[m.dictionary == dn]
        if sel == "window":
            s = s[s.member != "ste"]
        elif sel == "ste":
            s = s[s.member == "ste"]
        if len(s):
            ax.scatter(s.P_b, s.P_n, s=(26 if mk == "*" else 9), marker=mk, color=col, edgecolor="black", lw=0.2, label=lab if legend else None, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("nominal term $P_b$"); ax.set_ylabel("propagated term $P_n$")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarcos", required=True); ap.add_argument("--sim", required=True)
    ap.add_argument("--frozen_sarcos", required=True); ap.add_argument("--frozen_sim", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.7), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    for row, (d, fzp, name) in enumerate([(args.sarcos, args.frozen_sarcos, "SARCOS"), (args.sim, args.frozen_sim, "Simulated arm")]):
        fz = json.load(open(fzp))
        summ = pd.read_csv(os.path.join(d, "e09_summary.csv")); pairs = pd.read_csv(os.path.join(d, "e09_paired.csv"))
        mem = pd.read_csv(os.path.join(d, "e10_members.csv"))
        cm, ch = f"O1:{fz['O1_M']}", f"O1:{fz['O1_H']}"
        ladder(axes[row, 0], summ, pairs, cm, f"{name}, O1-M", legend=(row == 0))
        ladder(axes[row, 1], summ, pairs, ch, f"{name}, O1-H")
        scatter(axes[row, 2], mem, cm, f"{name}, O1-M: members", legend=(row == 0))
    for ax, l in zip(axes.ravel(), "abcdef"):
        ax.text(-0.16, 1.06, l, transform=ax.transAxes, fontsize=9, fontweight="bold")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 2].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, ncol=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=5.6, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0, h_pad=1.2)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200, bbox_inches="tight")
    print("fig5 written", args.out)


if __name__ == "__main__":
    main()
