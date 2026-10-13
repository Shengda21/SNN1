"""Fig. 1: overview in five panels, flat icons.
 a) problem and intervention (schematic): matched SNNs trained on nominal data, the backward surrogate as the
    only design factor, five frozen members, the deployment conditions, a short labeled recording and the
    choice of executed members
 b) key observation (measured): W-Q(5) on D-S and on D-M and the noise-augmented network B-A at N and O1-M
 c) role of the straight-through member (icons with measured values, arm)
 d) boundary of the substitutes (measured): margin D-M - D-S at O1-M under the plain simplex, + null, the input
    estimator and augmented members
 e) control consequence (measured, E13)
Every number is read from the json written by build_fig1_json.py (its "sources" field names the result files).
Measured parts carry the C_MEAS tag and numbers, schematic parts the C_ANA tag."""
import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Arc, Wedge
import numpy as np

plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.titlesize": 7, "axes.labelsize": 6.5,
                     "legend.fontsize": 6.2, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                     "pdf.fonttype": 42})
C_MEAS, C_ANA, C_SCH = "#1f6f8b", "#b85c00", "#555555"
T, O, R, G, B, Y, K, S = "#1f6f8b", "#e8912d", "#d9534f", "#4aa564", "#4a6fd8", "#f2c230", "#2c3e50", "#9aa5b1"
# method colours shared with Figs. 2-4
COL = {"B-A": "#2e8b57", "B-N": "#595959", "D-S": "#7fb3c8", "D-M": "#1f6f8b", "null": "#c0392b", "est": "#e59866"}
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"


# ---------------------------------------------------------------- layout helpers
def box(ax, x, y, w, h, text="", fc="#f4f4f4", ec="#444444", fs=6, lw=0.7, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fc, ec=ec, lw=lw,
                                transform=ax.transAxes))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, transform=ax.transAxes,
                linespacing=1.15, **kw)


def arrow(ax, p, q, color="#333333", style="-|>", lw=0.8, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, lw=lw, ls=ls,
                                 transform=ax.transAxes, shrinkA=0, shrinkB=0))


def title(ax, letter, text, width=None, kind=None):
    """Bold panel letter + short label above the axes; kind = 'measured' / 'schematic' adds the colour tag."""
    fig = ax.figure
    pos = ax.get_position()
    if width:
        text = "\n".join(textwrap.wrap(text, width))
    off = 0.17 / (pos.width * fig.get_figwidth())
    ax.text(0.0, 1.0, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    t = ax.text(off, 1.0, text, transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left", linespacing=1.1)
    if kind:
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transAxes.inverted())
        tag(ax, bb.x1 + 0.1 / (pos.width * fig.get_figwidth()), 1.0 + 0.004 / (pos.height * fig.get_figheight()), kind)


def tag(ax, x, y, kind):
    c = C_MEAS if kind.startswith("measured") else C_ANA
    ax.text(x, y, kind, transform=ax.transAxes, fontsize=6, color=c, va="bottom", ha="left", style="italic",
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc="white", ec=c, lw=0.6))


def message(ax, text, width, y=0.0):
    ax.text(0.5, y, "\n".join(textwrap.wrap(text, width)), ha="center", va="bottom", fontsize=6.3,
            style="italic", transform=ax.transAxes, linespacing=1.15)


def txt(ax, x, y, s, fs=6.0, color="#222222", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, transform=ax.transAxes, linespacing=1.15, **kw)


def ic(ax, cx, cy, s=0.26):
    """Square icon canvas of side s inches centred at (cx, cy) in axes fractions."""
    fig = ax.figure
    pos = ax.get_position()
    w = s / (pos.width * fig.get_figwidth())
    h = s / (pos.height * fig.get_figheight())
    a = ax.inset_axes([cx - w / 2, cy - h / 2, w, h])
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off"); a.set_zorder(6)
    return a


# ---------------------------------------------------------------- icons (unit square, flat colours)
def i_snn(a):
    xs = [0.15, 0.5, 0.85]
    ys = [[0.3, 0.7], [0.15, 0.5, 0.85], [0.3, 0.7]]
    for l in range(2):
        for y0 in ys[l]:
            for y1 in ys[l + 1]:
                a.plot([xs[l], xs[l + 1]], [y0, y1], color=S, lw=0.5, zorder=1)
    for l in range(3):
        for k, y in enumerate(ys[l]):
            a.add_patch(Circle((xs[l], y), 0.09, fc=O if (l == 1 and k == 1) else T, ec="none", zorder=2))


def i_bump(a, kind):
    a.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.12", fc="#fff1dc", ec="none"))
    x = np.linspace(-1, 1, 200)
    y = {"atan": 1 / (1 + (3 * x) ** 2), "sigmoid": 1 / np.cosh(2.6 * x) ** 2, "ste": np.full_like(x, 0.62),
         "tri": np.clip(1 - np.abs(x) / 0.7, 0, None), "esc": np.exp(-3 * np.abs(x))}[kind]
    X = 0.12 + 0.76 * (x + 1) / 2
    Yv = 0.16 + 0.62 * y
    a.fill_between(X, 0.16, Yv, color=O, alpha=0.35, lw=0)
    a.plot(X, Yv, color=O, lw=1.0)
    a.plot([0.12, 0.88], [0.16, 0.16], color=K, lw=0.5)


def i_flame(a):
    a.add_patch(Polygon([(0.5, 0.97), (0.72, 0.62), (0.86, 0.35), (0.74, 0.12), (0.5, 0.04), (0.26, 0.12),
                         (0.14, 0.35), (0.3, 0.55), (0.38, 0.42)], closed=True, fc=R, ec="none"))
    a.add_patch(Polygon([(0.5, 0.55), (0.65, 0.32), (0.58, 0.14), (0.5, 0.1), (0.42, 0.14), (0.35, 0.32)],
                        closed=True, fc=Y, ec="none"))


def i_lock(a):
    a.add_patch(Arc((0.5, 0.52), 0.46, 0.62, theta1=0, theta2=180, color=K, lw=1.6))
    a.add_patch(FancyBboxPatch((0.16, 0.08), 0.68, 0.5, boxstyle="round,pad=0,rounding_size=0.08", fc=B, ec="none"))
    a.add_patch(Circle((0.5, 0.36), 0.07, fc="white", ec="none"))
    a.add_patch(Rectangle((0.47, 0.18), 0.06, 0.14, fc="white", ec="none"))


def i_traj(a):
    t = np.linspace(0.05, 0.95, 80)
    for ph, c in [(0.0, T), (0.9, T), (1.8, S)]:
        a.plot(t, 0.5 + 0.28 * np.sin(2 * np.pi * 1.1 * t + ph) * (0.6 + 0.4 * t), color=c, lw=0.9)
    a.add_patch(Circle((0.93, 0.5), 0.07, fc=O, ec="none"))


def i_clock(a):
    a.add_patch(Circle((0.5, 0.5), 0.42, fc="white", ec=K, lw=1.0))
    a.plot([0.5, 0.5], [0.5, 0.78], color=K, lw=1.0)
    a.plot([0.5, 0.7], [0.5, 0.4], color=R, lw=1.0)
    a.add_patch(Circle((0.5, 0.5), 0.05, fc=K, ec="none"))


def i_chip(a):
    for v in (0.32, 0.5, 0.68):
        for (x0, y0, x1, y1) in [(v, 0.1, v, 0.9), (0.1, v, 0.9, v)]:
            a.plot([x0, x1], [y0, y1], color=S, lw=0.9, zorder=1)
    a.add_patch(FancyBboxPatch((0.22, 0.22), 0.56, 0.56, boxstyle="round,pad=0,rounding_size=0.05", fc=K, ec="none", zorder=2))
    a.add_patch(Rectangle((0.38, 0.38), 0.24, 0.24, fc=T, ec="none", zorder=3))


def i_matrix(a, C):
    for r in range(3):
        for c in range(3):
            a.add_patch(Rectangle((0.06 + c * 0.3, 0.64 - r * 0.3), 0.27, 0.27, fc=plt.cm.Oranges(min(C[r][c] / 1.15, 1)), ec="none"))


def i_bars(a):
    hs = [0.9, 0.72, 0.6, 0.52, 0.46]
    for i, hh in enumerate(hs):
        a.add_patch(Rectangle((0.08 + i * 0.18, 0.08), 0.14, hh * 0.85, fc=S if i == 0 else (T if i == 4 else "#8fc1d3"), ec="none"))
    a.plot([0.04, 0.96], [0.08, 0.08], color=K, lw=0.6)


def i_sliders(a):
    for y, p, c in [(0.8, 0.3, B), (0.5, 0.65, O), (0.2, 0.45, G)]:
        a.plot([0.08, 0.92], [y, y], color=S, lw=1.2, solid_capstyle="round")
        a.add_patch(Circle((0.08 + 0.84 * p, y), 0.1, fc=c, ec="white", lw=0.5))


def i_target(a):
    for r, c in [(0.45, R), (0.33, "white"), (0.21, R), (0.09, "white")]:
        a.add_patch(Circle((0.5, 0.5), r, fc=c, ec=R, lw=0.5))


def i_relgain(a):
    a.add_patch(Rectangle((0.12, 0.08), 0.25, 0.8, fc=S, ec="none"))
    a.add_patch(Rectangle((0.55, 0.08), 0.25, 0.42, fc=T, ec="none"))
    a.plot([0.5, 0.92], [0.88, 0.88], color=K, lw=0.6, ls="--")
    a.add_patch(FancyArrowPatch((0.88, 0.86), (0.88, 0.52), arrowstyle="-|>", mutation_scale=5, color=G, lw=1.0))


def i_tol(a):
    a.add_patch(Rectangle((0.03, 0.34), 0.94, 0.32, fc=G, ec="none", alpha=0.35))
    t = np.linspace(0.05, 0.95, 80)
    a.plot(t, 0.5 + 0.32 * np.sin(2 * np.pi * 1.2 * t) * np.exp(-1.6 * t), color=T, lw=1.0)
    a.plot([0.03, 0.97], [0.5, 0.5], color=K, lw=0.4, ls=":")


def i_venn(a, d):
    a.add_patch(Circle((0.5 - d / 2, 0.5), 0.22, fc=O, ec="none", alpha=0.6))
    a.add_patch(Circle((0.5 + d / 2, 0.5), 0.22, fc=T, ec="none", alpha=0.6))


def i_arm(a):
    a.add_patch(Rectangle((0.08, 0.05), 0.34, 0.1, fc=K, ec="none"))
    a.plot([0.25, 0.33, 0.78], [0.15, 0.52, 0.78], color=B, lw=2.2, solid_capstyle="round")
    a.add_patch(Circle((0.25, 0.15), 0.06, fc=K, ec="none"))
    a.add_patch(Circle((0.33, 0.52), 0.07, fc=K, ec="none"))
    a.add_patch(Circle((0.78, 0.78), 0.08, fc=O, ec="none"))


def i_scale(a):
    a.add_patch(Polygon([(0.5, 0.1), (0.35, 0.1), (0.5, 0.55), (0.65, 0.1)], closed=True, fc=K, ec="none"))
    a.plot([0.12, 0.88], [0.62, 0.62], color=K, lw=1.4)
    for x, c in [(0.18, T), (0.82, O)]:
        a.add_patch(Polygon([(x, 0.6), (x - 0.13, 0.32), (x + 0.13, 0.32)], closed=True, fc=c, ec="none"))


def i_arrow(a, up):
    a.add_patch(Circle((0.5, 0.5), 0.46, fc=G if up else R, ec="none"))
    if up:
        a.add_patch(Polygon([(0.5, 0.8), (0.24, 0.5), (0.4, 0.5), (0.4, 0.22), (0.6, 0.22), (0.6, 0.5), (0.76, 0.5)], closed=True, fc="white", ec="none"))
    else:
        a.add_patch(Polygon([(0.5, 0.2), (0.24, 0.5), (0.4, 0.5), (0.4, 0.78), (0.6, 0.78), (0.6, 0.5), (0.76, 0.5)], closed=True, fc="white", ec="none"))


def i_bars3(a):
    for i, (hh, c) in enumerate([(0.5, T), (0.78, S), (0.72, S)]):
        a.add_patch(Rectangle((0.1 + i * 0.29, 0.08), 0.22, hh, fc=c, ec="none"))
    a.plot([0.04, 0.96], [0.08, 0.08], color=K, lw=0.6)


def i_sat(a):
    a.plot([0.1, 0.1, 0.92], [0.92, 0.1, 0.1], color=K, lw=0.7)
    t = np.linspace(0, 1, 60)
    a.plot(0.1 + 0.82 * t, 0.1 + 0.75 * (1 - np.exp(-4 * t)), color=T, lw=1.3)
    for x, c in [(0.1 + 0.82 * 0.12, O), (0.1 + 0.82 * 0.3, O)]:
        a.add_patch(Circle((x, 0.1 + 0.75 * (1 - np.exp(-4 * (x - 0.1) / 0.82))), 0.07, fc=c, ec="none"))


def i_track(a):
    t = np.linspace(0.05, 0.95, 90)
    a.plot(t, 0.5 + 0.3 * np.sin(2 * np.pi * 1.3 * t), color=S, lw=1.0, ls="--")
    a.plot(t, 0.5 + 0.3 * np.sin(2 * np.pi * 1.3 * t - 0.35) * 0.9 + 0.04 * np.sin(23 * t), color=T, lw=1.1)




def i_noise(a):
    t = np.linspace(0.06, 0.94, 70)
    base = 0.5 + 0.25 * np.sin(2 * np.pi * 1.1 * t)
    a.plot(t, base, color=S, lw=0.9, ls="--")
    a.plot(t, base + 0.09 * np.random.default_rng(3).standard_normal(len(t)), color=R, lw=0.8)


def i_offset(a):
    t = np.linspace(0.06, 0.94, 70)
    base = 0.42 + 0.22 * np.sin(2 * np.pi * 1.1 * t)
    a.plot(t, base, color=S, lw=0.9, ls="--")
    a.plot(t, base + 0.2, color=O, lw=1.1)


def i_payload(a):
    i_arm(a)
    a.plot([0.78, 0.78], [0.70, 0.52], color=K, lw=0.8)
    a.add_patch(Circle((0.78, 0.42), 0.12, fc=K, ec="none"))


def i_resp(a):
    for x, h, c in [(0.15, 0.80, R), (0.85, 0.42, O)]:
        a.add_patch(FancyArrowPatch((x, 0.5 - h / 2), (x, 0.5 + h / 2), arrowstyle="<|-|>", mutation_scale=5, color=c, lw=1.0))
    a.add_patch(FancyBboxPatch((0.33, 0.3), 0.34, 0.4, boxstyle="round,pad=0,rounding_size=0.06", fc="#fff1dc", ec=O, lw=0.8))
    a.plot([0.24, 0.33], [0.5, 0.5], color=K, lw=0.6); a.plot([0.67, 0.76], [0.5, 0.5], color=K, lw=0.6)


def i_tallbar(a):
    for i, hh in enumerate([0.22, 0.21, 0.86, 0.21, 0.2]):
        a.add_patch(Rectangle((0.08 + i * 0.18, 0.08), 0.14, hh, fc=O if i == 2 else T, ec="none"))
    a.plot([0.04, 0.96], [0.08, 0.08], color=K, lw=0.6)


def i_cmat(a, C, j):
    """5x5 heat map of a measured matrix, member j outlined."""
    C = np.asarray(C)
    lo, hi = C.min(), C.max()
    n = len(C)
    for r in range(n):
        for c in range(n):
            a.add_patch(Rectangle((0.04 + c * 0.185, 0.78 - r * 0.185), 0.17, 0.17, fc=plt.cm.Oranges(0.15 + 0.8 * (C[r, c] - lo) / (hi - lo)), ec="none"))
    a.add_patch(Rectangle((0.03, 0.78 - j * 0.185 - 0.01), 0.935, 0.19, fc="none", ec=K, lw=0.7))


def i_wbars(a, ws):
    """Measured weights as three bars (unit height = weight 0.6)."""
    for i, wv in enumerate(ws):
        a.add_patch(Rectangle((0.12 + i * 0.28, 0.1), 0.2, max(wv, 0.0) / 0.6 * 0.8, fc=C_MEAS, ec="none"))
    a.plot([0.05, 0.95], [0.1, 0.1], color=K, lw=0.6)


# ---------------------------------------------------------------- panels
def minus(v, nd=3):
    """Signed number with nd decimals and a typographic minus (a value that rounds to zero gets '+')."""
    return ("−" if v < 0 and round(abs(v), nd) > 0 else "+") + f"{abs(v):.{nd}f}"


def panel_a(ax):
    title(ax, "a", "Problem and intervention", kind="schematic")
    ax.axis("off")
    hy = 0.972
    for x, s in [(0.083, "trained on\nnominal data"), (0.295, "backward surrogate,\nthe only design factor"), (0.46, "frozen"),
                 (0.635, "deployment\ncondition"), (0.886, "calibration and\nexecution")]:
        txt(ax, x, hy, s, fontweight="bold", va="top")
    i_lock(ic(ax, 0.46, 0.875, 0.13))
    # matched SNNs trained on nominal data
    box(ax, 0.0, 0.18, 0.166, 0.66, fc="#e8eef2")
    i_flame(ic(ax, 0.03, 0.79, 0.15))
    i_snn(ic(ax, 0.083, 0.64, 0.42))
    txt(ax, 0.083, 0.47, "matched SNNs", fontweight="bold")
    txt(ax, 0.083, 0.32, "same net, init.,\nbatch order,\noptimizer and\nstopping rule")
    # five surrogates -> five frozen members
    items = [("ATan", "atan"), ("sigmoid", "sigmoid"), ("STE, no window", "ste"), ("triangular", "tri"), ("escape rate", "esc")]
    for k, (n, kind) in enumerate(items):
        y = 0.77 - k * 0.13
        box(ax, 0.19, y - 0.05, 0.21, 0.1, "", fc="#ffe2bf" if kind == "ste" else "#fff4e6", ec=O if kind == "ste" else "#444444",
            lw=0.9 if kind == "ste" else 0.7)
        i_bump(ic(ax, 0.211, y, 0.17), kind)
        txt(ax, 0.237, y, n, ha="left", fontweight="bold" if kind == "ste" else "normal")
        arrow(ax, (0.166, 0.51), (0.19, y))
        arrow(ax, (0.40, y), (0.428, y))
        box(ax, 0.428, y - 0.05, 0.064, 0.1, f"f{chr(0x2081 + k)}", fc="#eaf5ea", fs=6.5)
    # deployment conditions
    arrow(ax, (0.497, 0.51), (0.522, 0.51), lw=1.0)
    box(ax, 0.522, 0.18, 0.226, 0.66, fc="#fdf0ec")
    for (y, f, s) in [(0.725, i_noise, "O1: white noise\non $q$, $\\dot q$"), (0.51, i_offset, "O2-B: constant\noffset on $q$, $\\dot q$"),
                      (0.295, i_payload, "P1: 1-kg payload,\nnominal sensing")]:
        f(ic(ax, 0.553, y, 0.25))
        txt(ax, 0.585, y, s, ha="left")
    # calibration and execution
    arrow(ax, (0.748, 0.51), (0.772, 0.51), lw=1.0)
    box(ax, 0.772, 0.18, 0.228, 0.66, fc="#eef0ff")
    i_traj(ic(ax, 0.799, 0.725, 0.22)); txt(ax, 0.836, 0.725, "$m$ labeled\ntrajectories,\n1,000 steps each", ha="left")
    i_chip(ic(ax, 0.802, 0.51, 0.22)); txt(ax, 0.834, 0.51, "execute $k\\leq5$\nof the members", ha="left")
    i_sliders(ic(ax, 0.802, 0.295, 0.24)); txt(ax, 0.834, 0.295, "weights on the\nsimplex, frozen\nfor the test", ha="left")
    message(ax, "For predictors trained under nominal sensing: which prediction differences remain useful when measurements "
                "become noisy, and does a lower squared risk give better tracking?", 96)


def panel_b(fig, bg, res):
    title(bg, "b", "Key observation", kind="measured")
    bg.axis("off")
    d = res["panel_b"]
    names = [("D-S W-Q(5)", COL["D-S"]), ("D-M W-Q(5)", COL["D-M"]), ("B-A", COL["B-A"])]
    pos = bg.get_position()
    for cond, (x0, w), ylim, fmt in [("N", (0.16, 0.28), 0.2, "{:.4f}"), ("O1-M", (0.61, 0.385), 0.55, "{:.3f}")]:
        ax = fig.add_axes([pos.x0 + x0 * pos.width, pos.y0 + 0.31 * pos.height, w * pos.width, 0.53 * pos.height])
        for gi, p in enumerate(["sarcos", "sim"]):
            for k, (nm, c) in enumerate(names):
                v = d[p][cond][nm]
                x = gi + (k - 1) * 0.27
                ax.bar(x, v["mean"], 0.25, color=c, zorder=2, label=nm if gi == 0 else None)
                ax.plot([x, x], [v["ci_lo"], v["ci_hi"]], color=INK, lw=0.6, zorder=3)
                ax.text(x, v["ci_hi"] + 0.02 * ylim, fmt.format(v["mean"]), rotation=90, ha="center", va="bottom", fontsize=6, color=INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["SARCOS", "arm"]); ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, ylim)
        ax.set_ylabel(f"test nMSE at {cond}", labelpad=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        if cond == "N":
            ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
            hs, ls = ax.get_legend_handles_labels()
        else:
            ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    bg.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False, handlelength=1.0, handletextpad=0.3, columnspacing=0.9)
    message(bg, "Under O1-M, D-M beats D-S on both platforms; B-A is lower (SARCOS) or equal (arm).", 46)


def panel_c(ax, res):
    title(ax, "c", "STE member of D-M (arm)", kind="measured")
    ax.axis("off")
    d = res["panel_c"]["sim"]
    box(ax, 0.0, 0.33, 0.2, 0.52, fc="#fff1dc", ec=O)
    i_bump(ic(ax, 0.1, 0.65, 0.42), "ste")
    txt(ax, 0.1, 0.43, "STE\nmember", fontweight="bold")
    fr, nm, cm, wq = d["forward_response"], d["nominal_risk_N"], d["cross_moment_O1M"], d["wq5_weight"]
    rows = [(lambda a: i_resp(a), "lowest input response", f"{fr['ste']:.3f} vs {fr['others_min']:.3f}–{fr['others_max']:.3f}"),
            (lambda a: i_tallbar(a), "largest error under N", f"nMSE {nm['ste']:.3f} vs {nm['others_min']:.3f}–{nm['others_max']:.3f}"),
            (lambda a: i_cmat(a, cm["C_test_mean_over_seeds"], cm["member_order"].index("ste")), "smallest cross-moments",
             f"O1-M: {cm['ste_min']:.2f} vs {cm['others_min']:.2f}–{cm['others_max']:.2f}"),
            (lambda a: i_wbars(a, [wq["N"], wq["O1-M"], wq["O1-H"]]), "W-Q(5) weight",
             f"N {wq['N']:.0f} → O1-M {wq['O1-M']:.2f} → O1-H {wq['O1-H']:.2f}")]
    for k, (f, head, val) in enumerate(rows):
        y = 0.875 - k * 0.195
        box(ax, 0.23, y - 0.082, 0.77, 0.164, fc="#f4f4f4")
        f(ic(ax, 0.29, y, 0.22))
        txt(ax, 0.352, y + 0.036, head, ha="left", fontweight="bold")
        txt(ax, 0.352, y - 0.038, val, ha="left", color=C_MEAS)
    message(ax, "Under noise the calibration gives weight to the worst-fitting, least input-sensitive member.", 46)


def panel_d(fig, bg, res):
    title(bg, "d", "Boundary of substitutes", kind="measured")
    bg.axis("off")
    pos = bg.get_position()
    ax = fig.add_axes([pos.x0 + 0.33 * pos.width, pos.y0 + 0.37 * pos.height, 0.64 * pos.width, 0.53 * pos.height])
    cols = [COL["D-M"], COL["null"], COL["est"], COL["B-A"]]
    n = len(res["panel_d"]["sarcos"])
    for p, mk, off in [("sarcos", "o", -0.2), ("sim", "s", 0.2)]:
        for i, r in enumerate(res["panel_d"][p]):
            ax.plot([r["ci_lo"], r["ci_hi"]], [i + off] * 2, color=cols[i], lw=0.8, zorder=2)
            ax.plot(r["diff"], i + off, mk, ms=4.0, color=cols[i], mec=cols[i], zorder=3)
            right = r["diff"] < -0.01
            ax.text(r["ci_hi"] + 0.0015 if right else r["ci_lo"] - 0.0015, i + off, minus(r["diff"]), ha="left" if right else "right",
                    va="center", fontsize=6, color=C_MEAS, zorder=4, bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.9))
    ax.axvline(0, color=INK, lw=0.6, zorder=1)
    ax.set_yticks(range(n)); ax.set_yticklabels(["plain simplex", "+ null", "input estimator", "augmented\nmembers"])
    ax.set_ylim(n - 0.45, -0.55)
    ax.set_xlim(-0.05, 0.02); ax.set_xticks([-0.04, -0.02, 0, 0.02]); ax.set_xticklabels(["−0.04", "−0.02", "0", "+0.02"])
    ax.set_xlabel("D-M − D-S at O1-M, W-Q(5)\n(● SARCOS, ■ arm; 95% interval)", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
    message(bg, "A null candidate shrinks (SARCOS) or removes (arm) the margin; the input estimator and augmented members "
                "remove or reverse it.", 46)


def panel_e(fig, ax, res):
    title(ax, "e", "Control consequence (arm)", kind="measured")
    ax.axis("off")
    d = res["panel_e"]
    n1, n2 = d["N"]["D-M single (W-S)"]["rmse"], d["N"]["D-M W-Q(5)"]["rmse"]
    rel = 100 * (n2 / n1 - 1)
    box(ax, 0.0, 0.795, 1.0, 0.195, fc="#eaf5ea")
    i_track(ic(ax, 0.078, 0.892, 0.25))
    txt(ax, 0.15, 0.94, "N: averaging lowers tracking error", ha="left", fontweight="bold")
    txt(ax, 0.15, 0.85, f"D-M single {1000 * n1:.1f} → D-M W-Q(5) {1000 * n2:.1f} mrad\n"
                        f"({'−' if rel < 0 else '+'}{abs(rel):.0f}%, shared PD law)", ha="left", color=C_MEAS)
    # O1-M: risk and tracking of D-S and D-M
    o = d["O1-M"]
    ol, tr = o["open_loop_nmse_total"], o["tracking"]
    box(ax, 0.0, 0.47, 1.0, 0.3, fc="#fdf0ec")
    txt(ax, 0.03, 0.735, "O1-M: lower risk, higher tracking error", ha="left", fontweight="bold")
    txt(ax, 0.6, 0.655, "open-loop\nnMSE", va="center"); txt(ax, 0.86, 0.655, "tracking\n(mrad)", va="center")
    for k, key in enumerate(["D-S W-Q(5)", "D-M W-Q(5)"]):
        y = 0.575 - k * 0.07
        ax.add_patch(Rectangle((0.03, y - 0.02), 0.035, 0.04, fc=COL["D-S" if k == 0 else "D-M"], ec="none", transform=ax.transAxes))
        txt(ax, 0.08, y, key, ha="left")
        best_ol = ol[key] == min(ol.values())
        best_tr = tr[key]["rmse"] == min(v["rmse"] for v in tr.values())
        txt(ax, 0.6, y, f"{ol[key]:.3f}", color=C_MEAS, fontweight="bold" if best_ol else "normal")
        txt(ax, 0.86, y, f"{1000 * tr[key]['rmse']:.1f}", color=C_MEAS, fontweight="bold" if best_tr else "normal")
    txt(ax, 0.03, 0.655, "bold: lower value", ha="left", color=MUTED)
    # Spearman bars
    sp = d["spearman_O1M"]
    txt(ax, 0.0, 0.41, f"Spearman correlation with tracking\n({sp['n_models']} feedforward models, O1-M)", ha="left", fontweight="bold", va="center")
    pos = ax.get_position()
    a2 = fig.add_axes([pos.x0 + 0.42 * pos.width, pos.y0 + 0.215 * pos.height, 0.52 * pos.width, 0.1 * pos.height])
    vals = [("slow part (0.25 s)", sp["slow_L25"]), ("total risk", sp["total"])]
    for i, (lab, v) in enumerate(vals):
        a2.barh(i, v, 0.62, color=C_MEAS, zorder=2)
        a2.text(v + (0.05 if v > 0 else -0.05), i, minus(v, 2), ha="left" if v > 0 else "right", va="center", fontsize=6, color=INK)
    a2.axvline(0, color=INK, lw=0.6)
    a2.set_xlim(-1, 1); a2.set_ylim(1.5, -0.5); a2.set_yticks([0, 1]); a2.set_yticklabels([v[0] for v in vals])
    a2.set_xticks([-1, 0, 1]); a2.set_xticklabels(["−1", "0", "+1"])
    a2.spines[["top", "right", "left"]].set_visible(False); a2.tick_params(axis="y", length=0)
    message(ax, f"The slow part of the error agrees with the tracking order ({minus(sp['slow_L25'], 2)}); the total risk opposes it "
                f"({minus(sp['total'], 2)}).", 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="json written by build_fig1_json.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = json.load(open(args.results))
    for k in ("panel_b", "panel_c", "panel_d", "panel_e", "sources"):
        if k not in res:
            raise SystemExit(f"{args.results} has no {k}; run build_fig1_json.py")
    fig = plt.figure(figsize=(7.16, 5.25))
    axa = fig.add_axes([0.008, 0.592, 0.645, 0.358])
    bgb = fig.add_axes([0.672, 0.592, 0.323, 0.358])
    axc = fig.add_axes([0.008, 0.02, 0.322, 0.488])
    bgd = fig.add_axes([0.345, 0.02, 0.305, 0.488])
    axe = fig.add_axes([0.668, 0.02, 0.327, 0.488])
    panel_a(axa); panel_b(fig, bgb, res); panel_c(axc, res); panel_d(fig, bgd, res); panel_e(fig, axe, res)
    fig.savefig(args.out)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()
