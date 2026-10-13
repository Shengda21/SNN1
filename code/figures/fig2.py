"""Fig. 2: surrogate complementarity.
 a) SARCOS, b) simulated arm: test nMSE at N and O1-M of B-F, B-L, B-A, B-N and of W-Q(5) on D-S, D-T, D-M,
    mean with 95% cluster-bootstrap interval (e01_summary.csv); dashed: post-hoc optimum of D-M on the test matrix
 c) W-Q(5) on D-M, on D-M without the straight-through member, and on D-S at O1-M and O1-H, both platforms
    (e09_summary.csv, the source of the Table III row "D-M without straight-through")
 d) forward perturbation response of every member family, both platforms (forward_response.json), and the
    O1-M test matrix of D-M (seed 0, arm; e02_geometry.json)
 e) gain of the two-member witness identified on D_cal against the gain it realizes on D_test (arm, e02_geometry.json)
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.titlesize": 7, "axes.labelsize": 6.8,
                     "legend.fontsize": 6.3, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "pdf.fonttype": 42})
# method colours shared by Figs. 2-4
COL = {"B-F": "#a6a6a6", "B-L": "#c27c0e", "B-A": "#2e8b57", "B-N": "#595959", "D-S": "#7fb3c8", "D-T": "#9b59b6",
       "D-M": "#1f6f8b", "null": "#c0392b", "est": "#e59866", "PD": "#d9d9d9"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
METHODS = [("B-F", "D-M", "B-F dev-selected single", "B-F"), ("B-L", "-", "B-L learned ATan width", "B-L"),
           ("B-A", "-", "B-A noise-augmented", "B-A"), ("B-N", "-", "B-N MLP", "B-N"),
           ("W-Q5", "D-S", "D-S W-Q(5)", "D-S"), ("W-Q5", "D-T", "D-T W-Q(5)", "D-T"), ("W-Q5", "D-M", "D-M W-Q(5)", "D-M")]
FAMILIES = [("dm_atan", "D-M: ATan", "D-M"), ("dm_sigmoid", "D-M: sigmoid", "D-M"), ("dm_ste", "D-M: straight-through", "D-M"),
            ("dm_triangular", "D-M: triangular", "D-M"), ("dm_sre", "D-M: escape rate", "D-M"),
            ("ds_", "D-S members", "D-S"), ("dt_", "D-T members", "D-T"), ("bl_", "B-L", "B-L"), ("ba_", "B-A", "B-A"), ("bn_", "B-N MLP", "B-N")]


def letter(fig, ax, l, text, dx=-0.0, dy=0.0):
    pos = ax.get_position()
    fig.text(pos.x0 + dx, pos.y1 + 0.012 + dy, l, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    fig.text(pos.x0 + dx + 0.018, pos.y1 + 0.012 + dy, text, fontsize=7, fontweight="bold", va="bottom", ha="left")


def clean(ax, grid="x"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)


def hbars(ax, summ, cond, ylabels):
    y = np.arange(len(METHODS))
    for i, (m, dn, lab, ck) in enumerate(METHODS):
        r = summ[(summ.cond == cond) & (summ.method == m) & (summ.dictionary == dn)]
        if len(r) == 0 and m == "B-F":
            r = summ[(summ.cond == cond) & (summ.method == m)]
        if len(r) == 0:
            continue
        r = r.iloc[0]
        ax.barh(y[i], r.nmse_mean, 0.68, color=COL[ck], edgecolor="none", zorder=2)
        ax.plot([r.ci_lo, r.ci_hi], [y[i]] * 2, color=INK, lw=0.7, zorder=3)
    ph = summ[(summ.cond == cond) & (summ.method == "posthoc-opt") & (summ.dictionary == "D-M")]
    if len(ph):
        ax.axvline(ph.iloc[0].nmse_mean, color=COL["D-M"], ls=(0, (3, 2)), lw=0.8, zorder=4)
    ax.set_yticks(y); ax.set_ylim(len(METHODS) - 0.4, -0.6)
    ax.set_yticklabels([l for _, _, l, _ in METHODS] if ylabels else [])
    if not ylabels:
        ax.tick_params(axis="y", length=0)
    clean(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarcos", required=True); ap.add_argument("--sim", required=True)
    ap.add_argument("--frozen_sarcos", required=True); ap.add_argument("--frozen_sim", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    FZ = {"sarcos": json.load(open(args.frozen_sarcos)), "sim": json.load(open(args.frozen_sim))}
    DIR = {"sarcos": args.sarcos, "sim": args.sim}
    E01 = {p: pd.read_csv(os.path.join(DIR[p], "e01_summary.csv")) for p in DIR}
    E09 = {p: pd.read_csv(os.path.join(DIR[p], "e09_summary.csv")) for p in DIR}
    FR = {p: json.load(open(os.path.join(DIR[p], "forward_response.json")))["members"] for p in DIR}
    geo = json.load(open(os.path.join(args.sim, "e02_geometry.json")))
    W, H = 7.16, 4.45
    fig = plt.figure(figsize=(W, H))

    # ---------------- a, b: four bar columns sharing the method labels
    x0, top, h1 = 0.155, 0.885, 0.30
    cw, gap, pgap = 0.178, 0.022, 0.05
    xs = [x0, x0 + cw + gap, x0 + 2 * cw + gap + pgap, x0 + 3 * cw + 2 * gap + pgap]
    cols = [("sarcos", "N"), ("sarcos", f"O1:{FZ['sarcos']['O1_M']}"), ("sim", "N"), ("sim", f"O1:{FZ['sim']['O1_M']}")]
    xlim = [0.032, 0.56, 0.115, 0.50]
    axs = []
    for k, ((p, c), x, xl) in enumerate(zip(cols, xs, xlim)):
        ax = fig.add_axes([x, top - h1, cw, h1])
        hbars(ax, E01[p], c, ylabels=(k == 0))
        ax.set_xlim(0, xl)
        ax.set_xlabel("test nMSE at " + ("N" if c == "N" else "O1-M"), labelpad=1)
        axs.append(ax)
    letter(fig, axs[0], "a", "SARCOS", dx=-0.145)
    letter(fig, axs[2], "b", "Simulated arm", dx=-0.012)
    fig.legend([Line2D([0], [0], color=COL["D-M"], ls=(0, (3, 2)), lw=0.9), Line2D([0], [0], color=INK, lw=0.7)],
               ["D-M post-hoc optimum on the test matrix", "95% cluster-bootstrap interval"],
               loc="lower right", bbox_to_anchor=(0.995, top + 0.035), ncol=2, frameon=False, handlelength=2.0, columnspacing=1.2)

    # ---------------- row 2 geometry
    y2, h2 = 0.085, 0.33
    # ---------------- c: removing the straight-through member
    axc = fig.add_axes([0.055, y2, 0.205, h2])
    groups = [("sarcos", "O1-M", f"O1:{FZ['sarcos']['O1_M']}"), ("sarcos", "O1-H", f"O1:{FZ['sarcos']['O1_H']}"),
              ("sim", "O1-M", f"O1:{FZ['sim']['O1_M']}"), ("sim", "O1-H", f"O1:{FZ['sim']['O1_H']}")]
    spec = [("D-S", -0.22, "o", COL["D-S"], COL["D-S"], "D-S"), ("D-M-STE", 0.0, "D", "white", COL["D-M"], "D-M without STE"),
            ("D-M", 0.22, "o", COL["D-M"], COL["D-M"], "D-M")]
    for gi, (p, lab, c) in enumerate(groups):
        for dn, off, mk, fc, ec, name in spec:
            r = E09[p][(E09[p].cond == c) & (E09[p].method == "W-Q5") & (E09[p].dictionary == dn)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            axc.plot([gi + off] * 2, [r.ci_lo, r.ci_hi], color=ec, lw=0.8, zorder=2)
            axc.plot(gi + off, r.nmse_mean, mk, ms=4.2 if mk == "o" else 3.6, mfc=fc, mec=ec, mew=0.9, zorder=3,
                     label=name if gi == 0 else None)
    axc.axvline(1.5, color=GRID, lw=0.8)
    axc.set_xticks(range(4)); axc.set_xticklabels([g[1] for g in groups])
    axc.text(0.5, -0.17, "SARCOS", ha="center", va="top", fontsize=6.5, transform=axc.get_xaxis_transform())
    axc.text(2.5, -0.17, "arm", ha="center", va="top", fontsize=6.5, transform=axc.get_xaxis_transform())
    axc.set_xlim(-0.5, 3.5); axc.set_ylim(0.2, 0.8)
    axc.set_ylabel("test nMSE, W-Q(5)", labelpad=1)
    clean(axc, grid="y")
    axc.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.1, labelspacing=0.25, fontsize=6.1)
    letter(fig, axc, "c", "Without the STE member", dx=-0.045)

    # ---------------- d: forward response per member family, both platforms
    axd = fig.add_axes([0.405, y2, 0.13, h2])
    yy = np.arange(len(FAMILIES))
    for i, (pref, lab, ck) in enumerate(FAMILIES):
        for p, mk, off in [("sarcos", "o", -0.17), ("sim", "s", 0.17)]:
            v = [m["response_ratio_mean"] for k, m in FR[p].items() if (k.rsplit("_s", 1)[0] == pref if pref.startswith("dm_") else k.startswith(pref))]
            if not v:
                continue
            axd.plot([min(v), max(v)], [i + off] * 2, color=COL[ck], lw=0.8, zorder=2)
            axd.plot(np.mean(v), i + off, mk, ms=3.6, mfc=COL[ck] if p == "sarcos" else "white", mec=COL[ck], mew=0.9, zorder=3)
    axd.set_yticks(yy); axd.set_yticklabels([f[1] for f in FAMILIES]); axd.set_ylim(len(FAMILIES) - 0.5, -0.5)
    axd.get_yticklabels()[2].set_fontweight("bold")
    axd.set_xlim(0.28, 0.82); axd.set_xticks([0.3, 0.5, 0.7])
    axd.set_xlabel("forward response ratio\n(● SARCOS, □ arm)", labelpad=1)
    clean(axd)
    letter(fig, axd, "d", "Member response; test matrix", dx=-0.125)
    # test matrix of D-M at O1-M, seed 0, arm
    O1M = f"O1:{FZ['sim']['O1_M']}"
    key = f"s0|{O1M}" if f"s0|{O1M}" in geo else next(k for k in geo if k.endswith("|" + O1M))
    C = np.array(geo[key]["D-M"]["C_test"])
    names = ["ATan", "Sig", "STE", "Tri", "Esc"]
    hm = 0.215
    axm = fig.add_axes([0.565, y2 + 0.045, hm * H / W * 1.0, hm])
    axm.imshow(C, cmap="Oranges", vmin=0.2, vmax=0.5)
    for a in range(5):
        for b in range(5):
            axm.text(b, a, f"{C[a, b]:.2f}"[1:], ha="center", va="center", fontsize=6, color="white" if C[a, b] > 0.43 else INK)
    axm.set_xticks(range(5)); axm.set_xticklabels(names, fontsize=6, rotation=90)
    axm.set_yticks(range(5)); axm.set_yticklabels(names, fontsize=6)
    axm.tick_params(length=0)
    for s in axm.spines.values():
        s.set_visible(False)
    axm.set_title(f"D-M test matrix at O1-M\n(arm, seed {key.split('|')[0][1:]})", fontsize=6.3, pad=2)

    # ---------------- e: witness gain (arm, all seeds, conditions, dictionaries)
    axe = fig.add_axes([0.775, y2, 0.21, h2])
    seen = set()
    for k, g in geo.items():
        for dn, mk in [("D-S", "s"), ("D-T", "^"), ("D-M", "o")]:
            if dn in g and "witness_test" in g[dn]:
                wt = g[dn]["witness_test"]
                axe.plot(wt["predicted_gain_cal"], wt["realized_gain"], mk, color=COL[dn], ms=3.2, alpha=0.9, mec="white", mew=0.3,
                         label=dn if dn not in seen else None, zorder=3)
                seen.add(dn)
    lim = max(axe.get_xlim()[1], axe.get_ylim()[1])
    axe.plot([0, lim], [0, lim], color=MUTED, ls=(0, (3, 2)), lw=0.7, zorder=1); axe.axhline(0, color=GRID, lw=0.6)
    axe.set_xlim(0, lim); axe.set_ylim(min(0, axe.get_ylim()[0]), lim)
    axe.set_xlabel("predicted gain (calibration set)", labelpad=1); axe.set_ylabel("realized gain (test set)", labelpad=1)
    axe.legend(frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.1, fontsize=6.1, labelspacing=0.2)
    clean(axe, grid=None)
    letter(fig, axe, "e", "Two-member witness, arm", dx=-0.06)

    fig.savefig(args.out)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()
