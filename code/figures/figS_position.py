"""Supplementary figures for S5 (position step).

figS4.pdf  E16 surrogate width: forward response r and signal gain beta at O1-M, nominal test nMSE, and test nMSE
           at O1-M and O1-H against the HWHM of the four peak-normalized families; original reset (solid, seed
           mean; HWHM 8-64 seed 0) and detached reset (dashed, triangular and ATan); the straight-through member
           with the original and the detached reset at the right end.
figS5.pdf  E15 flips and direction-resolved response: predicted (Lemma 2(iii)) against measured first-layer flip
           rate at O1-M for every spiking member, and response along the five trailing against the five leading
           principal directions of the training inputs, both platforms.

Reads   the E15 member tables through supp_tables_s5.load (results/{sim,sarcos}/e15_*.csv, e15_*_new.csv,
        e16_members.csv)
Writes  build/fig/figS4.pdf, figS5.pdf (vector, fonts embedded, text >= 7 pt); PNG previews in
        $SNN29_PNG if set

python code/figures/figS_position.py
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_style as cs  # noqa: E402
import supp_tables_s5 as s5  # noqa: E402

cs.apply_style()
# family colours: validated all-pairs subset of the common categorical palette (scripts/validate_palette.js,
# light surface): blue, yellow, violet, green; the straight-through member keeps its aqua of the main figures.
FCOL = {"tri": "#2a78d6", "sig": "#eda100", "atan": "#4a3aa7", "sre": "#008300"}
FMK = {"tri": "o", "sig": "s", "atan": "D", "sre": "v"}
STE_C = cs.C["STE"]
X_STE = 256.0  # plotting position of the straight-through member (HWHM = infinity)
XT = [1 / 16, 1, 16, X_STE]
XTL = ["1/16", "1", "16", "STE"]
XT_MINOR = [1 / 4, 4, 64]


def fam_curve(T, fam, key, detach=False):
    d = s5.sel_w16(T, fam, detach=detach)
    g = d.groupby("hwhm")[key].mean()
    return g.index.values.astype(float), g.values


def fig_s4():
    metrics = [("r", r"Forward response $r$"), ("beta_M", r"Signal gain $\beta$"),
               ("R_N", "Nominal test nMSE"), ("R_M", "Test nMSE, O1-M"), ("R_H", "Test nMSE, O1-H")]
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 2.95), constrained_layout=True)
    letters = iter("abcdefghij")
    for i, p in enumerate(s5.PLATS):
        T = s5.mt(p)
        ste_o = s5.sel_locked(T, "D-M STE")
        ste_d = s5.sel_ste_detach(T)
        for j, (key, lab) in enumerate(metrics):
            ax = axes[i, j]
            for fam in s5.FAMS:
                x, y = fam_curve(T, fam, key)
                ax.plot(x, y, color=FCOL[fam], lw=1.0, marker=FMK[fam], ms=2.8, mew=0, zorder=3)
                if fam in ("tri", "atan"):
                    xd, yd = fam_curve(T, fam, key, detach=True)
                    ax.plot(xd, yd, color=FCOL[fam], lw=0.9, ls=(0, (3.0, 1.6)), marker=FMK[fam], ms=3.0,
                            mfc="white", mec=FCOL[fam], mew=0.7, zorder=2)
            ax.plot([X_STE], [ste_o[key].mean()], marker="^", ms=5, color=STE_C, mec="white", mew=0.6, zorder=4,
                    ls="none")
            ax.plot([X_STE], [ste_d[key].mean()], marker="^", ms=5, mfc="white", mec=STE_C, mew=0.9, zorder=4,
                    ls="none")
            ax.set_xscale("log", base=2)
            ax.set_xlim(1 / 22, 400)
            ax.set_xticks(XT)
            ax.set_xticklabels(XTL if i == 1 else [""] * len(XT), rotation=0)
            ax.minorticks_off()
            ax.set_xticks(XT_MINOR, minor=True)
            ax.tick_params(axis="x", which="minor", length=1.5, labelbottom=False)
            for xb in (108, 124):  # axis break between the grid and the STE position
                ax.plot([xb / 1.06, xb * 1.06], [-0.025, 0.025], transform=ax.get_xaxis_transform(), color=cs.AXIS,
                        lw=0.6, clip_on=False, zorder=5)
            if key == "R_N":
                ax.set_yscale("log")
                ax.set_yticks([0.02, 0.05, 0.1, 0.2] if p == "sim" else [0.015, 0.03, 0.06, 0.12])
                ax.set_yticklabels(["0.02", "0.05", "0.1", "0.2"] if p == "sim" else ["0.015", "0.03", "0.06",
                                                                                     "0.12"])
                ax.minorticks_off()
            if i == 0:
                ax.set_title(lab, fontsize=cs.FS)
            if i == 1:
                ax.set_xlabel("HWHM")
            if j == 0:
                ax.set_ylabel(cs.PLATFORM_NAME[p])
            cs.panel_label(ax, next(letters), x=-0.30 if j == 0 else -0.22, y=1.02, ha="left")
    handles = [Line2D([0], [0], color=FCOL[f], marker=FMK[f], ms=3, mew=0, lw=1.0, label=s5.FAMNAME[f])
               for f in s5.FAMS]
    handles += [Line2D([0], [0], color=cs.INK2, lw=1.0, label="original reset"),
                Line2D([0], [0], color=cs.INK2, lw=0.9, ls=(0, (3.0, 1.6)), marker="o", ms=3, mfc="white",
                       mec=cs.INK2, label="detached reset"),
                Line2D([0], [0], color=STE_C, marker="^", ms=5, ls="none", mec="white", label="STE, original"),
                Line2D([0], [0], color=STE_C, marker="^", ms=5, ls="none", mfc="white", mec=STE_C,
                       label="STE, detached")]
    fig.legend(handles=handles, loc="outside upper center", ncol=8, frameon=False, fontsize=cs.FS,
               handlelength=1.8, columnspacing=1.0)
    cs.savefig(fig, "figS4")
    plt.close(fig)


# member classes of Fig. S5 (identity = colour + marker; validated subset: blue, aqua, violet, yellow)
CLS = [("new", "E16–E18 members", "#eda100", "s"),
       ("clean", "Clean-trained window members", "#2a78d6", "o"),
       ("aug", "Augmented members", "#4a3aa7", "h"),
       ("e14", "E14 controls", "#7a7872", "x"),
       ("ste", "STE members", STE_C, "^"),
       ("mlp", "MLP", cs.C["MLP"], "X")]


def member_class(T):
    c = np.full(len(T), "other", dtype=object)
    pre, grp, org = T.prefix.values, T.group.astype(str).values, T.origin.values
    fam = T.family.astype(str).values
    for k in range(len(T)):
        if T.model_kind.values[k] == "mlp" or fam[k] == "mlp":
            c[k] = "mlp"
        elif fam[k] == "ste":
            c[k] = "ste"
        elif org[k] == "new":
            c[k] = "new"
        elif pre[k] in ("cap4", "cap8", "cap16", "uf"):
            c[k] = "e14"
        elif pre[k] in ("dma", "dsa", "dmq", "dsq", "dmas", "dmqs", "ba"):
            c[k] = "aug"
        elif pre[k] in ("dm", "ds", "dt", "bl", "width"):
            c[k] = "clean"
    return c


def fig_s5():
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.1), constrained_layout=True)
    letters = iter("abcd")
    for i, p in enumerate(s5.PLATS):
        T = s5.mt(p)
        cl = member_class(T)
        # flips
        ax = axes[i]
        ok = np.isfinite(T.flip_pred_biv_M.values) & np.isfinite(T.flip1_M.values) & (cl != "mlp")
        lo, hi = 0.04, 0.24
        ax.plot([lo, hi], [lo, hi], color=cs.MUTED, lw=0.6, zorder=1)
        for key, lab, col, mk in CLS:
            m = ok & (cl == key)
            if not m.any():
                continue
            ax.scatter(T.flip_pred_biv_M.values[m], T.flip1_M.values[m], s=7 if key != "new" else 6,
                       marker=mk, c=col, linewidths=0.6 if mk == "x" else 0.25,
                       edgecolors="white" if mk != "x" else None, alpha=0.85, zorder=1.5 if key == "new" else 3)
        rs = s5.rho_s(T.flip_pred_biv_M.values[ok], T.flip1_M.values[ok])
        ax.text(0.97, 0.04, rf"$\rho_s={rs:.2f}$, $n={ok.sum()}$", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=cs.FS, color=cs.INK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
        ax.set_xlim(0.04, 0.17)
        ax.set_ylim(0.09, 0.235)
        ax.set_xlabel("Predicted L1 flip rate, Lemma 2(iii)")
        ax.set_ylabel("Measured L1 flip rate")
        ax.set_title(cs.PLATFORM_NAME[p] + ", O1-M", fontsize=cs.FS)
        cs.panel_label(ax, next(letters), x=-0.30, y=1.02, ha="left")
    for i, p in enumerate(s5.PLATS):
        T = s5.mt(p)
        cl = member_class(T)
        ax = axes[2 + i]
        x, y = T.dir_full21_top5.values, T.dir_full21_bottom5.values
        ok = np.isfinite(x) & np.isfinite(y)
        lo, hi = 0.2, 1.05
        ax.plot([lo, hi], [lo, hi], color=cs.MUTED, lw=0.6, zorder=1)
        for key, lab, col, mk in CLS:
            m = ok & (cl == key)
            if not m.any():
                continue
            ax.scatter(x[m], y[m], s=7 if key != "new" else 6, marker=mk, c=col,
                       linewidths=0.6 if mk == "x" else 0.25, edgecolors="white" if mk != "x" else None,
                       alpha=0.85, zorder=1.5 if key == "new" else 3)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$r$, five leading directions")
        ax.set_ylabel(r"$r$, five trailing directions")
        ax.set_title(cs.PLATFORM_NAME[p] + ", O1-M energy", fontsize=cs.FS)
        cs.panel_label(ax, next(letters), x=-0.30, y=1.02, ha="left")
    handles = [Line2D([0], [0], marker=mk, color=col, ls="none", ms=4, mec="white" if mk != "x" else col,
                      mew=0.3 if mk != "x" else 0.8, label=lab) for _, lab, col, mk in CLS]
    fig.legend(handles=handles, loc="outside upper center", ncol=6, frameon=False, fontsize=cs.FS,
               handletextpad=0.2, columnspacing=1.0)
    cs.savefig(fig, "figS5")
    plt.close(fig)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    fig_s4()
    fig_s5()
