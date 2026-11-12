"""Fig. 4 (fig:ladder): the weight-space ladder.

(a) Test nMSE of B-F, B-L, B-A, B-N, D-S, D-T, D-M in the weight spaces W0-W5, SARCOS and the
    simulated arm at O1-M (O1-H is in Table II) (e21_ladder.csv: nmse_mean, ci_lo, ci_hi; mean over
    five training seeds with the two-level cluster-bootstrap 95% interval).
(b) Attribution of the simplex margin D-S - D-M (W0) into its scale part (removed by W1) and
    angle part (= the W1 margin), and the margin left after the input estimator (W5), per-joint
    weights (W3) and full linear stacking (W4) (e21_attribution.csv: margin_W0, scale_part,
    angle_part, rem_est_W1, rem_perjoint_W3, rem_stack_W4 with *_lo / *_hi).
(c) Calibration budget: test nMSE of D-M in each weight space when the weights are fitted on m
    calibration trajectories, simulated arm O1-M (e21_mslice.csv: nmse, seed_sd; m = 100 is the
    full calibration set).
Run:  python fig4_ladder.py      (writes build/fig/fig4.pdf)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common_style as cs

cs.FS = cs.FS_SMALL = 6.5  # compact version for the 12-page budget (never below 6.5 pt)
cs.FS_PANEL = 7.5

A_PANELS = [("sarcos", "O1-M"), ("sim", "O1-M")]
A_SPACES = ["W0", "W1", "W2", "W3", "W4", "W5"]
SPACE_TICK = {"W0": "W0\nsimplex", "W1": "W1\n+null", "W2": "W2\nper-joint", "W3": "W3\nper-joint\n+null",
              "W4": "W4\nstacking", "W5": "W5\nestimator"}
B_CONDS = [("sarcos", "O1-M"), ("sarcos", "O1-H"), ("sim", "O1-M"), ("sim", "O1-H"), ("sim", "O2B")]
B_REM = [("rem_perjoint_W3", "W3", "after W3"),
         ("rem_stack_W4", "W4", "after W4"),
         ("rem_est_W1", "W5", "after W5")]
C_PLATFORM, C_COND, C_METHOD = "sim", "O1-M", "D-M"
C_SPACES = ["W0", "W1", "W2", "W3", "W4", "W5"]
C_M = [1, 2, 5, 10, 100]
C_GREY = {"W0": "#1a1a1a", "W1": "#3f3e3b", "W2": "#8a8882", "W3": "#5f5d58", "W4": "#a9a7a0", "W5": "#000000"}
BR_Y = 0.17 / (2.0 * (0.895 - 0.155))  # bracket depth below the axes (0.17 in)
FIG_H = 2.0  # in (compact version; 12-page budget)
C_LS = {"W0": "-", "W1": "-", "W2": (0, (1.2, 1.2)), "W3": (0, (3, 1.5)), "W4": (0, (5, 1.5, 1, 1.5)),
        "W5": "-"}


def panel_a(ax, lad, cond, show_ylabel, show_xticks):
    sub = lad[lad.cond == cond]
    n = len(cs.METHOD_ORDER)
    offs = np.linspace(-0.33, 0.33, n)
    x = np.arange(len(A_SPACES))
    for k, m in enumerate(cs.METHOD_ORDER):
        r = sub[sub.method == m].set_index("space").reindex(A_SPACES)
        if r["nmse_mean"].isna().all():
            continue
        y = r["nmse_mean"].values
        lo, hi = r["ci_lo"].values, r["ci_hi"].values
        col, mk = cs.C[m], cs.MK[m]
        ax.vlines(x + offs[k], lo, hi, color=col, lw=0.6, alpha=0.8, zorder=2)
        ax.scatter(x + offs[k], y, marker=mk, s=9, color=col, edgecolors=cs.shade(col, 0.35), linewidths=0.35,
                   zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(A_SPACES if show_xticks else [])
    ax.set_xlim(-0.5, len(A_SPACES) - 0.5)
    ax.grid(axis="x", visible=False)
    for xv in x[:-1] + 0.5:
        ax.axvline(xv, color=cs.GRID, lw=0.4, zorder=0)
    if show_ylabel:
        ax.set_ylabel("Test nMSE")


def panel_b(ax, att):
    xs = np.arange(len(B_CONDS))
    wbar = 0.32
    for i, (p, c) in enumerate(B_CONDS):
        r = att[p].set_index("cond").loc[c]
        sc, an = r["scale_part"], r["angle_part"]
        x0 = xs[i] - 0.22
        ax.bar(x0, sc, width=wbar, color=cs.INK2, edgecolor="white", linewidth=0.5, zorder=2)
        ax.bar(x0, an, width=wbar, bottom=sc if an >= 0 else 0.0, color="#d9d7d0", edgecolor=cs.INK2,
               linewidth=0.4, hatch="////", zorder=2)
        ax.errorbar([x0], [r["margin_W0"]], yerr=[[r["margin_W0"] - r["margin_W0_lo"]],
                                                 [r["margin_W0_hi"] - r["margin_W0"]]],
                    fmt="none", ecolor=cs.INK, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
        for k, (col, sp, _) in enumerate(B_REM):
            xr = xs[i] + 0.06 + 0.13 * k
            v = r[col]
            ax.errorbar([xr], [v], yerr=[[v - r[col + "_lo"]], [r[col + "_hi"] - v]], fmt="none", ecolor=cs.INK,
                        elinewidth=0.6, capsize=0, zorder=3)
            ax.scatter([xr], [v], marker=cs.MK_SPACE[sp], s=16 if sp != "W5" else 30, facecolors="white",
                       edgecolors=cs.INK, linewidths=0.7, zorder=4)
    ax.axhline(0, color=cs.AXIS, lw=0.5, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([c for p, c in B_CONDS])
    ax.set_xlim(-0.55, len(B_CONDS) - 0.45)
    # platform brackets under the condition labels
    for p in dict.fromkeys(q for q, _ in B_CONDS):
        idx = [i for i, (q, _) in enumerate(B_CONDS) if q == p]
        x0, x1 = xs[idx[0]] - 0.42, xs[idx[-1]] + 0.42
        ax.plot([x0, x1], [-BR_Y, -BR_Y], transform=ax.get_xaxis_transform(), color=cs.AXIS, lw=0.5,
                clip_on=False)
        ax.text((x0 + x1) / 2, -(BR_Y + 0.02), cs.PLATFORM_SHORT[p], transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=cs.FS, color=cs.INK2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("D-S $-$ D-M nMSE")


def panel_c(ax, ms):
    sub = ms[(ms.cond == C_COND) & (ms.method == C_METHOD)]
    ends = []
    for sp in C_SPACES:
        r = sub[sub.space == sp].set_index("m").reindex(C_M)
        y = r["nmse"].values
        col = C_GREY[sp]
        ax.errorbar(C_M, y, yerr=r["seed_sd"].values, color=col, lw=0.8, ls=C_LS[sp], elinewidth=0.5, capsize=0,
                    marker=cs.MK_SPACE[sp], ms=3.2 if sp != "W5" else 5, mfc="white", mec=col, mew=0.7, zorder=3)
        ends.append((y[-1], sp))
    # direct labels at the right end, pushed apart vertically
    ys = sorted(ends)
    lo, hi = ax.get_ylim()
    gap = 0.075 * (hi - lo)
    pos = cs.repel([yv for yv, _ in ys], gap)
    for (yv, sp), yp in zip(ys, pos):
        ax.text(C_M[-1] * 1.25, yp, sp, fontsize=cs.FS_SMALL, color=cs.INK, va="center", ha="left")
        if abs(yp - yv) > 1e-9:
            ax.plot([C_M[-1] * 1.05, C_M[-1] * 1.22], [yv, yp], color=cs.MUTED, lw=0.4)
    ax.set_xscale("log")
    ax.set_xticks(C_M)
    ax.set_xticklabels([str(m) for m in C_M])
    ax.minorticks_off()
    ax.set_xlim(0.8, C_M[-1] * 2.2)
    ax.set_xlabel("Calibration trajectories $m$")
    ax.set_ylabel("Test nMSE (D-M)")


def make():
    cs.apply_style()
    lad = {p: cs.read(p, "e21_ladder.csv", ["cond", "method", "space", "nmse_mean", "ci_lo", "ci_hi"])
           for p in cs.PLATFORMS}
    need_b = ["cond", "margin_W0", "margin_W0_lo", "margin_W0_hi", "scale_part", "angle_part"] +              sum([[c, c + "_lo", c + "_hi"] for c, _, _ in B_REM], [])
    att = {p: cs.read(p, "e21_attribution.csv", need_b) for p in cs.PLATFORMS}
    ms = cs.read(C_PLATFORM, "e21_mslice.csv", ["cond", "method", "space", "m", "nmse", "seed_sd"])

    fig = plt.figure(figsize=(cs.DOUBLE_COL, FIG_H))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.08, 0.92], left=0.058, right=0.985, top=0.895,
                          bottom=0.155, wspace=0.36)
    axes_a = []
    for k, (p, c) in enumerate(A_PANELS):
        ax = fig.add_subplot(gs[0, k])
        panel_a(ax, lad[p], c, show_ylabel=(k == 0), show_xticks=True)
        ax.set_title(f"{cs.PLATFORM_NAME[p]}, {c}", loc="left", pad=3)
        axes_a.append(ax)
    cs.panel_label(axes_a[0], "a", x=-0.30, y=1.07, ha="left")
    axb = fig.add_subplot(gs[0, 2])
    panel_b(axb, att)
    axb.set_title("Margin attribution", loc="left", pad=3)
    cs.panel_label(axb, "b", x=-0.30, y=1.07, ha="left")
    axc = fig.add_subplot(gs[0, 3])
    panel_c(axc, ms)
    axc.set_title(f"{cs.PLATFORM_SHORT[C_PLATFORM]}, {C_COND}, {C_METHOD}", loc="left", pad=3)
    axc.set_ylabel("Test nMSE")
    cs.panel_label(axc, "c", x=-0.30, y=1.07, ha="left")

    # legends: methods under (a); attribution parts inside (b)
    hm = [Line2D([], [], color=cs.C[m], ls="none", marker=cs.MK[m], ms=4, mec=cs.shade(cs.C[m], 0.35), mew=0.35)
          for m in cs.METHOD_ORDER]
    x_mid = (axes_a[0].get_position().x0 + axes_a[1].get_position().x1) / 2
    fig.legend(hm, [cs.METHOD_LABEL[m] for m in cs.METHOD_ORDER], loc="lower center", ncol=7,
               bbox_to_anchor=(x_mid, -0.02), frameon=False, columnspacing=0.8, handletextpad=0.2)
    hb = [Patch(facecolor=cs.INK2, edgecolor="white", label="scale"),
          Patch(facecolor="#d9d7d0", edgecolor=cs.INK2, hatch="////", label="angle")]
    hb += [Line2D([], [], ls="none", marker=cs.MK_SPACE[sp], ms=4 if sp != "W5" else 6, mfc="white", mec=cs.INK,
                  mew=0.7, label=lab) for _, sp, lab in B_REM]
    axb.legend(handles=hb, loc="upper left", ncol=2, fontsize=cs.FS_SMALL, handlelength=1.1, labelspacing=0.15,
               columnspacing=0.6, handletextpad=0.3, borderaxespad=0.1)
    ymax = max(att[p].set_index("cond").loc[c, "margin_W0_hi"] for p, c in B_CONDS)
    axb.set_ylim(min(-0.012, axb.get_ylim()[0]), ymax * 1.75)
    cs.savefig(fig, "fig4")
    return fig


if __name__ == "__main__":
    make()
