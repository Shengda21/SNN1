"""Fig. 5 (fig:path): noise path, label-free calibration, certificate.

(a) Simulated arm, D-M, seed 0, paired-noise grid (e22_path.csv):
    left  - optimal W1 weights of the five members and null versus rho (stacked; column
            weights, member order atan, sigmoid, ste, triangular, sre, null), with the predicted
            critical levels (dashed, e22_critical.csv rho_predicted) and the measured support
            changes (solid, rho_measured) of seed 0;
    right - test risk R(rho) of D-M and D-S (W0, W1), the STE member and the MLP, relative to the
            best window member at each rho (pointwise minimum over the four window members, as in
            e22_path.py); circles mark the crossings of seed 0 from e22_crossings.csv.
(b) Simulated arm only (SARCOS in the Supplementary Material). Label-free calibration
    (e23_labelfree.csv, D-M, W1): test-risk difference to the labelled
    target calibration S1 (diff_vs_S1, diff_ci_lo, diff_ci_hi) for the sources S0, S2, S3, S5,
    TAB; grey band = +-1 seed SD of S1 (seed_sd_S1); off-scale values are printed at the edge.
(c) Simulated arm only. Gain certificate (e24_certificate.csv, D-M, W1 gain over the best single
    member, O1-M):
    certification rate (cert_rate_t, cert_rate_boot) and empirical coverage against the
    held-out gain (coverage_t_held, coverage_boot_held) versus the number of calibration units m.
Run:  python fig5_path.py      (writes build/fig/fig5.pdf)
"""
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common_style as cs

cs.FS = cs.FS_SMALL = 6.5  # compact version for the 12-page budget (never below 6.5 pt)
cs.FS_PANEL = 7.5

A_PLATFORM, A_SEED, A_DICT, A_SPACE = "sim", 0, "D-M", "W1"
A_STACK = ["atan", "sigmoid", "triangular", "sre", "ste", "null"]  # bottom -> top
A_FILL = {"atan": cs.shade(cs.C["D-M"], 0.30), "sigmoid": cs.C["D-M"], "triangular": cs.tint(cs.C["D-M"], 0.35),
          "sre": cs.tint(cs.C["D-M"], 0.62), "ste": cs.C["STE"], "null": cs.C["null"]}
A_NAME = {"atan": "ATan", "sigmoid": "Sigmoid", "triangular": "Triangular", "sre": "SRE", "ste": "STE",
          "null": "null"}
# risk curves: (kind, name, space, colour, linestyle, label)
A_CURVES = [("dict", "D-M", "W0", cs.C["D-M"], "-", "D-M, W0"),
            ("dict", "D-M", "W1", cs.C["D-M"], (0, (3, 1.5)), "D-M, W1"),
            ("dict", "D-S", "W0", cs.C["D-S"], "-", "D-S, W0"),
            ("dict", "D-S", "W1", cs.C["D-S"], (0, (3, 1.5)), "D-S, W1"),
            ("member", "dm_ste_s{s}", "-", cs.C["STE"], "-", "STE member"),
            ("member", "bn_mlp_s{s}", "-", cs.C["MLP"], "-", "MLP")]
A_CROSS = {"DM_vs_DS_W0": ("D-M, W0", "D-M = D-S (W0)"), "DM_vs_DS_W1": ("D-M, W1", "D-M = D-S (W1)"),
           "STE_vs_best_window": ("STE member", "STE = best window"), "STE_vs_MLP": ("STE member", "STE = MLP")}

B_CONDS = {"sim": ["O1-L", "O1-M", "O1-H", "O2B", "P1"], "sarcos": ["O1-L", "O1-M", "O1-H", "O1qqd"]}
B_METHOD, B_SPACE = "D-M", "W1"
BC_PLATFORMS = ["sim"]  # (b), (c): simulated arm only; SARCOS is in the Supplementary Material
FIG_H = 2.8  # in (compact version; 12-page budget)
B_SOURCES = [("S0", "x", cs.INK, True, "S0: no recalibration"),
             ("S2", "o", cs.INK, True, r"S2: $\hat v$ from unlabelled inputs"),
             ("S3", "o", cs.INK, False, "S3: true noise variance"),
             ("S5", "^", cs.INK2, False, "S5: static recording"),
             ("TAB", "s", cs.INK2, False, r"TAB: weight table at $\hat\rho$")]
B_YLIM = {"sim": (-0.0062, 0.0082), "sarcos": (-0.0032, 0.0152)}

C_COND, C_DICT, C_SPACE = "O1-M", "D-M", "W1"
C_LINES = [("cert_rate_t", cs.INK, "-", "o", True, "Cert., t"),
           ("cert_rate_boot", "#8a8882", "-", "s", True, "Cert., bootstrap"),
           ("coverage_t_held", cs.INK, (0, (3, 1.5)), "o", False, "Cov., t"),
           ("coverage_boot_held", "#8a8882", (0, (3, 1.5)), "s", False, "Cov., bootstrap")]
C_XLABEL = {"sim": "Calibration trajectories $m$", "sarcos": "Calibration blocks $m$ (100 rows)"}


# ----------------------------------------------------------------------------- (a)
def load_a():
    path = cs.read(A_PLATFORM, "e22_path.csv", ["kind", "seed", "name", "space", "rho", "risk_test", "weights"])
    crit = cs.read(A_PLATFORM, "e22_critical.csv", ["seed", "dict", "space", "member_name", "direction",
                                                    "rho_measured", "rho_predicted", "member_idx"])
    cross = cs.read(A_PLATFORM, "e22_crossings.csv", ["type", "seed", "rho_cross"])
    path = path[path.seed == A_SEED]
    crit = crit[(crit.seed == A_SEED) & (crit.dict == A_DICT) & (crit.space == A_SPACE)]
    cross = cross[cross.seed.astype(str) == str(A_SEED)]
    return path, crit, cross


def panel_a_weights(ax, path, crit):
    d = path[(path.kind == "dict") & (path.name == A_DICT) & (path.space == A_SPACE)].sort_values("rho")
    W = np.array([ast.literal_eval(w) for w in d.weights])  # columns: DM_SURR + null
    cols = cs.DM_SURR + ["null"]
    rho = d.rho.values
    stack = np.vstack([W[:, cols.index(k)] for k in A_STACK])
    polys = ax.stackplot(rho, stack, colors=[A_FILL[k] for k in A_STACK], edgecolor="white", linewidth=0.6,
                         zorder=2)
    polys[A_STACK.index("null")].set_hatch("....")
    polys[A_STACK.index("null")].set_edgecolor("white")
    # direct labels at the right end, centred in each band
    cum = np.cumsum(stack[:, -1])
    mid = cum - stack[:, -1] / 2
    for k, ym in zip(A_STACK, mid):
        ax.text(rho[-1] + 0.02, ym, A_NAME[k], fontsize=cs.FS_SMALL, va="center", ha="left", color=cs.INK)
    ax.set_xlim(rho[0], rho[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Noise level $\rho$")
    ax.set_ylabel("D-M weight (W1)")
    ax.grid(False)
    draw_critical(ax, crit)


def draw_critical(ax, crit, labels=True):
    for _, r in crit.iterrows():
        who = "null" if pd.isna(r.member_name) else r.member_name.split("_")[1]
        col = cs.shade(A_FILL.get(who, cs.INK2), 0.35)
        if np.isfinite(r.rho_predicted):
            ax.axvline(r.rho_predicted, color=col, lw=0.8, ls=(0, (2.5, 1.5)), zorder=4)
    meas = sorted(set(np.round(crit.rho_measured.dropna().values, 4)))
    for m in meas:
        ax.axvline(m, color=cs.INK, lw=0.6, zorder=4)


def panel_a_risk(ax, path, crit, cross):
    mem = path[path.kind == "member"]
    piv = mem.pivot_table(index="rho", columns="name", values="risk_test").sort_index()
    win = [f"dm_{x}_s{A_SEED}" for x in cs.DM_SURR if x != "ste"]
    best = piv[win].min(axis=1)
    rho = piv.index.values
    curves = {}
    for kind, name, sp, col, ls, lab in A_CURVES:
        if kind == "dict":
            r = path[(path.kind == "dict") & (path.name == name) & (path.space == sp)].set_index("rho") \
                .sort_index()["risk_test"].reindex(rho)
        else:
            r = piv[name.format(s=A_SEED)]
        y = r.values - best.values
        curves[lab] = y
        ax.plot(rho, y, color=col, lw=1.0, ls=ls, zorder=3, label=lab)
    ax.axhline(0, color=cs.MUTED, lw=0.8, zorder=2)
    draw_critical(ax, crit)
    ends = [(curves[lab][-1], lab) for *_, lab in A_CURVES] + [(0.0, "Best window")]
    lo, hi = np.nanmin([np.nanmin(v) for v in curves.values()]), np.nanmax([np.nanmax(v) for v in curves.values()])
    # headroom for the legend, then label gap = one text line (7 pt x 1.2) in data units of this axes
    ylo_, yhi_ = ax.get_ylim()
    yhi_ = yhi_ + 0.30 * (yhi_ - ylo_)
    ax.set_ylim(ylo_, yhi_)
    ax_h_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    gap = (yhi_ - ylo_) * cs.FS_SMALL * 1.2 / ax_h_pt
    pos = cs.repel([e[0] for e in ends], gap)
    lift = max(0.0, ylo_ + 0.6 * gap - min(pos))  # keep the lowest label inside the axes
    pos = [y + lift for y in pos]
    for (yv, lab), yp in zip(ends, pos):
        ax.text(rho[-1] + 0.03, yp, lab, fontsize=cs.FS_SMALL, va="center", ha="left", color=cs.INK,
                clip_on=False)
        if abs(yp - yv) > 0.01 * (hi - lo):
            ax.plot([rho[-1] + 0.005, rho[-1] + 0.028], [yv, yp], color=cs.MUTED, lw=0.4, clip_on=False)
    for _, r in cross.iterrows():
        if r.type not in A_CROSS or not np.isfinite(r.rho_cross):
            continue
        lab = A_CROSS[r.type][0]
        yc = 0.0 if r.type == "STE_vs_best_window" else np.interp(r.rho_cross, rho, curves[lab])
        ax.scatter([r.rho_cross], [yc], s=16, facecolors="white", edgecolors=cs.INK, linewidths=0.8, zorder=6)
        ax.annotate(f"{r.rho_cross:.2f}", (r.rho_cross, yc), xytext=(-2 if r.type == "DM_vs_DS_W1" else 0, -9 if yc <= 0 else 5),
                    textcoords="offset points", ha="right" if r.type == "DM_vs_DS_W1" else "center", fontsize=cs.FS_SMALL, color=cs.INK,
                    bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.75), zorder=8)
    ax.set_xlim(rho[0], rho[-1])
    ax.set_xlabel(r"Noise level $\rho$")
    ax.set_ylabel(r"$\Delta R(\rho)$")


# ----------------------------------------------------------------------------- (b)
def panel_b(ax, lf, platform):
    conds = B_CONDS[platform]
    sub = lf[(lf.method == B_METHOD) & (lf.space == B_SPACE)]
    offs = np.linspace(-0.30, 0.30, len(B_SOURCES))
    y0, y1 = B_YLIM[platform]
    for i, c in enumerate(conds):
        sc = sub[sub.cond == c]
        if sc.empty:
            continue
        sd = float(sc.seed_sd_S1.iloc[0])
        ax.fill_between([i - 0.42, i + 0.42], [-sd, -sd], [sd, sd], color="#ecebe6", lw=0, zorder=1)
        for k, (src, mk, col, filled, _) in enumerate(B_SOURCES):
            r = sc[sc.source == src]
            if r.empty:
                continue
            v, lo, hi = float(r.diff_vs_S1.iloc[0]), float(r.diff_ci_lo.iloc[0]), float(r.diff_ci_hi.iloc[0])
            x = i + offs[k]
            if v > y1:  # off-scale: arrow at the top edge with the value
                ax.annotate("", xy=(x, y1), xytext=(x, y1 - 0.12 * (y1 - y0)),
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=5), zorder=5)
                ax.text(x, y1 - 0.135 * (y1 - y0), f"{v:+.3f}".replace("+0.", "+."), rotation=90, ha="center",
                        va="top", fontsize=cs.FS_SMALL, color=cs.INK)
                continue
            ax.vlines(x, max(lo, y0), min(hi, y1), color=col, lw=0.6, zorder=3)
            ms = 14 if mk != "x" else 14
            if mk == "x":
                ax.scatter([x], [v], marker="x", s=ms, color=col, linewidths=0.9, zorder=4)
            else:
                ax.scatter([x], [v], marker=mk, s=ms, facecolors=col if filled else "white", edgecolors=col,
                           linewidths=0.7, zorder=4)
    ax.axhline(0, color=cs.AXIS, lw=0.5, zorder=2)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_xlim(-0.55, len(conds) - 0.45)
    ax.set_ylim(y0, y1)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(r"$\Delta$nMSE vs. S1")


# ----------------------------------------------------------------------------- (c)
def panel_c(ax, cert, platform):
    sub = cert[(cert.cond == C_COND) & (cert.dict == C_DICT) & (cert.space == C_SPACE)].sort_values("m")
    for col, color, ls, mk, filled, lab in C_LINES:
        ax.plot(sub.m, sub[col], color=color, ls=ls, lw=0.9, marker=mk, ms=3.2, mfc=color if filled else "white",
                mec=color, mew=0.7, zorder=3)
    ax.axhline(0.95, color=cs.MUTED, lw=0.5, ls=(0, (1, 1.2)), zorder=1)
    ax.axhline(0.90, color=cs.MUTED, lw=0.5, ls=(0, (1, 1.2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(sub.m.values)
    ax.set_xticklabels([str(int(m)) for m in sub.m.values])
    ax.minorticks_off()
    ax.set_ylim(0.0, 1.04)
    ax.set_xlabel(C_XLABEL[platform])
    ax.set_ylabel("Rate")


def make():
    cs.apply_style()
    path, crit, cross = load_a()
    lf = {p: cs.read(p, "e23_labelfree.csv", ["cond", "method", "space", "source", "diff_vs_S1", "diff_ci_lo",
                                              "diff_ci_hi", "seed_sd_S1"]) for p in BC_PLATFORMS}
    cert = {p: cs.read(p, "e24_certificate.csv", ["cond", "dict", "space", "m"] + [c[0] for c in C_LINES])
            for p in BC_PLATFORMS}

    fig = plt.figure(figsize=(cs.DOUBLE_COL, FIG_H))
    ga = fig.add_gridspec(1, 2, left=0.075, right=0.885, top=0.9286, bottom=0.5893, wspace=0.40)
    gbc = fig.add_gridspec(1, 2, left=0.075, right=0.975, top=0.4321, bottom=0.1464, wspace=0.24,
                           width_ratios=[1.45, 1.0])

    ax_w = fig.add_subplot(ga[0])
    panel_a_weights(ax_w, path, crit)
    ax_w.set_title(f"{cs.PLATFORM_SHORT[A_PLATFORM]}, {A_DICT} {A_SPACE} weights, seed {A_SEED}", loc="left", pad=3)
    cs.panel_label(ax_w, "a", x=-0.20, y=1.05, ha="left")
    ax_r = fig.add_subplot(ga[1])
    panel_a_risk(ax_r, path, crit, cross)
    ax_r.set_title("Test risk minus best window member", loc="left", pad=3)
    hc = [Line2D([], [], color=cs.INK2, lw=0.8, ls=(0, (2.5, 1.5))), Line2D([], [], color=cs.INK, lw=0.6),
          Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec=cs.INK, mew=0.8)]
    ax_r.legend(hc, ["predicted entry", "measured change", "crossing"], loc="upper left",
                bbox_to_anchor=(0.34, 1.0), fontsize=cs.FS_SMALL,
                ncol=1, handlelength=1.5, labelspacing=0.1, borderaxespad=0.2, frameon=True, facecolor="white",
                edgecolor="none", framealpha=1.0, borderpad=0.2).set_zorder(10)

    p = BC_PLATFORMS[0]
    axb = fig.add_subplot(gbc[0, 0])
    panel_b(axb, lf[p], p)
    axb.set_title(f"{cs.PLATFORM_SHORT[p]}: label-free calibration, D-M W1", loc="left", pad=3)
    cs.panel_label(axb, "b", x=-0.115, y=1.07, ha="left")
    axc = fig.add_subplot(gbc[0, 1])
    panel_c(axc, cert[p], p)
    axc.set_title(f"{cs.PLATFORM_SHORT[p]}, {C_COND}: certificate, D-M W1", loc="left", pad=3)
    cs.panel_label(axc, "c", x=-0.17, y=1.07, ha="left")

    hb = []
    for src, mk, col, filled, lab in B_SOURCES:
        if mk == "x":
            hb.append(Line2D([], [], ls="none", marker="x", ms=4, color=col, mew=0.9))
        else:
            hb.append(Line2D([], [], ls="none", marker=mk, ms=4, mfc=col if filled else "white", mec=col, mew=0.7))
    hb.append(Patch(facecolor="#ecebe6", edgecolor="none"))
    lb = [s[0] for s in B_SOURCES] + [r"$\pm$1 SD of S1"]
    xb = (axb.get_position().x0 + axb.get_position().x1) / 2
    fig.legend(hb, lb, loc="lower center", bbox_to_anchor=(xb, -0.012), ncol=len(hb), frameon=False,
               fontsize=cs.FS, columnspacing=0.9, handletextpad=0.25)
    hc = [Line2D([], [], color=c, ls=ls, lw=0.9, marker=mk, ms=3.2, mfc=c if f else "white", mec=c, mew=0.7)
          for _, c, ls, mk, f, _ in C_LINES]
    axc.legend(hc, [c[5] for c in C_LINES], loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=2,
               frameon=False, fontsize=cs.FS, columnspacing=0.8, handletextpad=0.3, handlelength=1.8,
               labelspacing=0.15, borderaxespad=0.2)
    cs.savefig(fig, "fig5")
    return fig


if __name__ == "__main__":
    make()
