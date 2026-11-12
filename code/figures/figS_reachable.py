"""Supplementary figures for S7 (reachable step).

figS1.pdf  E22 noise path: weight on the straight-through member of D-M (W0, W1), weight on the
           null (W1, D-M and D-S), test risk of D-M and D-S on W0 and W1, and the paired difference
           D-M - D-S, against the noise level rho; every seed as a faint line, the seed mean bold.
figS2.pdf  E24 split-sample certificate: certification rate and empirical coverage of the
           one-sided Student-t bound against the calibration budget m, D-M, W0 and W1.

Reads   results/{sim,sarcos}/e22_path.csv, e24_certificate.csv
Writes  build/fig/figS1.pdf, figS2.pdf (vector, fonts embedded, >= 7 pt)

python code/figures/figS_reachable.py
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(os.environ.get("SNN29_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results"
FIG = ROOT / "build" / "fig"

# validated categorical palette (fixed order); ink and grid tokens
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                      "#008300", "#4a3aa7")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dddcd8"
DICT_COL = {"D-M": BLUE, "D-S": ORANGE}
FROZEN = {"sim": (0.35, 0.7, 1.0), "sarcos": (0.25, 0.5, 0.7)}
PNAME = {"sim": "Simulated arm", "sarcos": "SARCOS"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round", "savefig.dpi": 600,
})
LW_MEAN, LW_SEED, A_SEED = 1.3, 0.45, 0.35


def style(ax, plat):
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for r in FROZEN[plat]:
        ax.axvline(r, color=GRID, lw=0.6, zorder=0)


def path_frame(plat):
    d = pd.read_csv(RES / plat / "e22_path.csv")
    d = d[d.kind == "dict"].copy()
    W = np.array([json.loads(x) for x in d.weights])
    d["w_ste"] = W[:, 2]
    d["w_null"] = W[:, -1]
    return d


def curves(d, name, space, col):
    sub = d[(d.name == name) & (d.space == space)]
    piv = sub.pivot_table(index="rho", columns="seed", values=col)
    return piv.index.values, piv.values  # rho, [n_rho, n_seed]


def draw(ax, x, Y, color, ls="-", label=None):
    for j in range(Y.shape[1]):
        ax.plot(x, Y[:, j], color=color, lw=LW_SEED, alpha=A_SEED, ls=ls, zorder=1)
    ax.plot(x, Y.mean(1), color=color, lw=LW_MEAN, ls=ls, label=label, zorder=3)


def fig_s1():
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.75), constrained_layout=True)
    letters = iter("abcdefgh")
    for i, plat in enumerate(("sim", "sarcos")):
        d = path_frame(plat)
        ax = axes[i, 0]
        for sp, ls in (("W0", "-"), ("W1", "--")):
            x, Y = curves(d, "D-M", sp, "w_ste")
            draw(ax, x, Y, BLUE, ls)
        ax.set_ylabel("Weight on STE member of D-M")
        ax.set_ylim(-0.02, 0.75)
        ax = axes[i, 1]
        for dn in ("D-M", "D-S"):
            x, Y = curves(d, dn, "W1", "w_null")
            draw(ax, x, Y, DICT_COL[dn], "--")
        ax.set_ylabel("Weight on null (W1)")
        ax.set_ylim(-0.02, 0.45)
        ax = axes[i, 2]
        for dn in ("D-M", "D-S"):
            for sp, ls in (("W0", "-"), ("W1", "--")):
                x, Y = curves(d, dn, sp, "risk_test")
                draw(ax, x, Y, DICT_COL[dn], ls)
        ax.set_ylabel("Test nMSE")
        ax = axes[i, 3]
        for sp, ls in (("W0", "-"), ("W1", "--")):
            x, Ym = curves(d, "D-M", sp, "risk_test")
            _, Ys = curves(d, "D-S", sp, "risk_test")
            draw(ax, x, Ym - Ys, INK2, ls)
        ax.axhline(0, color=INK2, lw=0.6, zorder=0)
        ax.set_ylabel("D-M $-$ D-S test nMSE")
        for j in range(4):
            a = axes[i, j]
            style(a, plat)
            a.set_xlim(0, 1.2 if plat == "sim" else 0.9)
            a.set_xlabel(r"Noise level $\rho$")
            a.set_title(f"({next(letters)}) {PNAME[plat]}", loc="left", fontsize=7.5, color=INK)
    h_space = [Line2D([], [], color=INK2, lw=LW_MEAN, ls="-", label="W0 (simplex)"),
               Line2D([], [], color=INK2, lw=LW_MEAN, ls="--", label="W1 (simplex + null)")]
    h_dict = [Line2D([], [], color=BLUE, lw=LW_MEAN, label="D-M"),
              Line2D([], [], color=ORANGE, lw=LW_MEAN, label="D-S"),
              Line2D([], [], color=INK2, lw=LW_SEED, alpha=0.6, label="single seed")]
    fig.legend(handles=h_dict + h_space, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.065))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "figS1.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote figS1.pdf")


COND_STYLE = {"N": (BLUE, "o", "N"), "O1-L": (ORANGE, "s", "O1-L"), "O1-M": (AQUA, "^", "O1-M"),
              "O1-H": (YELLOW, "D", "O1-H"), "O2B": (MAGENTA, "v", "O2-B"), "P1": (GREEN, "P", "P1"),
              "O1qqd": (VIOLET, "X", r"O1-M $(q,\dot q)$, SARCOS")}
CONDS = {"sim": ["N", "O1-L", "O1-M", "O1-H", "O2B", "P1"], "sarcos": ["N", "O1-L", "O1-M", "O1-H", "O1qqd"]}


def fig_s2():
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.7), constrained_layout=True)
    letters = iter("abcdefgh")
    for i, plat in enumerate(("sim", "sarcos")):
        d = pd.read_csv(RES / plat / "e24_certificate.csv")
        d = d[d.dict == "D-M"]
        ms = sorted(d.m.unique())
        for j, (col, sp, ref, ylab) in enumerate((("cert_rate_t", "W0", 0.9, "Certification rate"),
                                                  ("cert_rate_t", "W1", 0.9, "Certification rate"),
                                                  ("coverage_t_held", "W0", 0.95, "Empirical coverage"),
                                                  ("coverage_t_held", "W1", 0.95, "Empirical coverage"))):
            ax = axes[i, j]
            ax.axhline(ref, color=INK2, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
            for c in CONDS[plat]:
                color, mk, _ = COND_STYLE[c]
                s = d[(d.cond == c) & (d.space == sp)].sort_values("m")
                ax.plot(s.m, s[col], color=color, lw=1.0, marker=mk, ms=3.2, mew=0.4, mec="white", zorder=3)
            ax.set_xscale("log")
            ax.set_xticks(ms)
            ax.set_xticklabels(["" if m == 8 else str(m) for m in ms])
            ax.minorticks_off()
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.set_xlabel("Trajectories $m$" if plat == "sim" else "100-row blocks $m$")
            ax.set_ylabel(ylab)
            ax.set_title(f"({next(letters)}) {PNAME[plat]}, {sp}", loc="left", fontsize=7.5)
    handles = [Line2D([], [], color=COND_STYLE[c][0], marker=COND_STYLE[c][1], ms=3.6, mew=0.4, mec="white", lw=1.0,
                      label=COND_STYLE[c][2]) for c in ["N", "O1-L", "O1-M", "O1-H", "O2B", "P1", "O1qqd"]]
    handles.append(Line2D([], [], color=INK2, lw=0.6, ls=(0, (1, 1.5)), label="target (0.9) / nominal (0.95)"))
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.065),
               columnspacing=1.0, handlelength=1.8)
    fig.savefig(FIG / "figS2.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote figS2.pdf")


if __name__ == "__main__":
    fig_s1()
    fig_s2()
