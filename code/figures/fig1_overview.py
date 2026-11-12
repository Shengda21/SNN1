"""Fig. 1 (fig:overview): one question, one plane, three steps. Schematic.

(a) The question: five spiking members trained under clean sensing that differ only in the backward
    surrogate are frozen and deployed under sensor noise as a weighted average whose weights are fit
    on a short recording; the three questions of the paper (position, reachable, needed).
(b) The plane (signal gain beta, orthogonal noise power ||nu||^2), risk (1-beta)^2 P_y + ||nu||^2:
    conditional-mean curve ||nu||^2 = beta (1-beta) P_y, iso-risk lines through the optima, members,
    the scalar-rescaling path (c beta, c^2 ||nu||^2) of one seed member, the reachable regions of the
    simplex (sum w = 1) and of the simplex with a null candidate (sum w <= 1), the W0 and W1 optima,
    and the input-estimator point.
(c) The task filter: iso-risk lines (1-beta)^2 P_y + ||nu||^2 / L_eq of a fast (L_eq = 1) and a slow
    (L_eq = 16) task and the task optimum on the gain path (c beta, c^2 ||nu||^2), c <= 1, of the same
    member: the fast optimum is interior (on the conditional-mean curve), the slow optimum is the end c = 1
    (no attenuation). The path stops at c = 1 because the paper tests only sum w <= 1 (no amplification).

The points are illustrative. A toy dictionary (four window members and one STE member) is placed at
the arm O1-M coordinates of notes/e15_status.md 4.1 (P_y = 0.923); its noise Gram matrix (window-window
correlation 0.93, STE-window 0.70) is chosen so that the simplex optimum falls on the measured D-M W0
point (beta 0.776, ||nu||^2 0.308, risk 0.355; notes/e21_e26_status.md). The reachable regions and the
W0/W1 optima are computed from this toy matrix; the estimator point is the measured D-M W5 point.
B-A is drawn on the W0 iso-risk line (its arm O1-M risk equals that of D-M W0) at a shifted beta so
that the two markers do not overlap. Measured coordinates are in Fig. 2 (fig:plane).

Colours, markers and line styles follow common_style.py (shared with Figs. 2, 4, 5).
Run:  python fig1_overview.py      (writes build/fig/fig1.pdf; SNN29_PNG=<dir> adds a PNG preview)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy.optimize import minimize

import common_style as cs

# ----------------------------------------------------------------------------- toy dictionary
PY = 0.923
WIN_C = (0.833, 0.382)                               # window / seed members, arm O1-M
WIN_OFF = [(-0.016, 0.0), (0.0, 0.02), (0.016, 0.0), (0.0, -0.02)]  # spread of the four window members
STE_P = (0.649, 0.347)
MLP_P = (0.946, 0.443)
EST_P = (0.728, 0.187)                               # D-M W5 (input estimator + W1), measured
RHO_WW, RHO_SW = 0.93, 0.70
BA_BETA = 0.735                                      # B-A placed on the W0 iso-risk line at this beta
LEQ_SLOW = 16

XLIM = (0.5, 1.05)
YLIM = (0.0, 0.56)
FIG_W, FIG_H = cs.DOUBLE_COL, 3.0

# line styles shared with Fig. 2
LS_L1 = (0, (4, 2))       # iso-risk, L_eq = 1
LS_L16 = (0, (1, 1.3))    # iso-risk, L_eq = 16
C_WIN = cs.C["D-M window"]
C_STE = cs.C["STE"]
C_BA = cs.C["B-A"]
C_MLP = cs.C["MLP"]
C_PATH = cs.C["D-S"]      # rescaling path of a seed member (as in Fig. 2)
C_NULL = cs.C["null"]
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


def toy():
    b = np.array([WIN_C[0] + o[0] for o in WIN_OFF] + [STE_P[0]])
    n = np.array([WIN_C[1] + o[1] for o in WIN_OFF] + [STE_P[1]])
    R = np.full((5, 5), RHO_WW)
    R[4, :] = R[:, 4] = RHO_SW
    np.fill_diagonal(R, 1.0)
    N = R * np.sqrt(np.outer(n, n))
    return b, n, N


def risk(beta, nu2, leq=1.0):
    return (1 - beta) ** 2 * PY + nu2 / leq


def optimum(b, N, null):
    """W0 (sum w = 1) or W1 (sum w <= 1) optimum of (1 - b.w)^2 P_y + w'Nw over w >= 0."""
    k = len(b)
    if null:
        cons = [{"type": "ineq", "fun": lambda w: 1 - w.sum()}]
        w0 = np.ones(k) / (k + 1)
    else:
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        w0 = np.ones(k) / k
    f = lambda w: (1 - b @ w) ** 2 * PY + w @ N @ w
    r = minimize(f, w0, bounds=[(0, 1)] * k, constraints=cons, method="SLSQP",
                 options=dict(ftol=1e-12, maxiter=500))
    w = r.x
    return float(b @ w), float(w @ N @ w), w


def simplex_bounds(b, N, grid):
    """Lower and upper ||nu||^2 of the simplex image at each beta in grid (beta within [min b, max b])."""
    k = len(b)
    lo, hi = [], []
    for g in grid:
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, {"type": "eq", "fun": lambda w, g=g: b @ w - g}]
        best = np.inf
        for start in (np.ones(k) / k, np.eye(k)[np.argmin(np.abs(b - g))] * 0.5 + 0.5 / k):
            r = minimize(lambda w: w @ N @ w, start, bounds=[(0, 1)] * k, constraints=cons, method="SLSQP",
                         options=dict(ftol=1e-13, maxiter=500))
            if r.success:
                best = min(best, r.fun)
        lo.append(best)
        # the maximum of a convex function on the slice is attained on an edge of the simplex
        m = -np.inf
        for i in range(k):
            if abs(b[i] - g) < 1e-12:
                m = max(m, N[i, i])
            for j in range(k):
                if b[j] > b[i] and b[i] <= g <= b[j]:
                    t = (g - b[i]) / (b[j] - b[i])
                    w = np.zeros(k)
                    w[i], w[j] = 1 - t, t
                    m = max(m, w @ N @ w)
        hi.append(m)
    return np.array(lo), np.array(hi)


def regions(b, N):
    gs = np.linspace(b.min(), b.max(), 90)
    lo, hi = simplex_bounds(b, N, gs)
    simplex = np.vstack([np.column_stack([gs, lo]), np.column_stack([gs[::-1], hi[::-1]])])
    # with a null candidate: union of (c beta, c^2 ||nu||^2), c in [0, 1], over the simplex image
    bc = np.linspace(0.3, b.max(), 300)
    rlo, rhi = lo / gs ** 2, hi / gs ** 2
    clo = np.array([x ** 2 * rlo[gs >= x - 1e-12].min() for x in bc])
    chi = np.array([x ** 2 * rhi[gs >= x - 1e-12].max() for x in bc])
    cone = np.vstack([np.column_stack([bc, clo]), np.column_stack([bc[::-1], chi[::-1]])])
    return simplex, cone


# ----------------------------------------------------------------------------- drawing helpers
def rbox(ax, x, y, w, h, fc, ec, lw=0.6, r=0.035, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=cs.INK2, lw=0.7, rad=0.0, ms=6, z=3):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=ms,
                                connectionstyle=f"arc3,rad={rad}"))


def chain(ax, x, y, pieces, transform=None, gap=1.2):
    """Write text pieces one after another on a line. kind: 'n' normal, 'b' bold, 'm' secondary ink,
    'badge' white bold digit in a dark circle."""
    prev = None
    out = []
    for txt, kind in pieces:
        kw = dict(fontsize=cs.FS, color=cs.INK, ha="left", va="baseline")
        g = gap
        if kind == "b":
            kw["fontweight"] = "bold"
        elif kind == "P":     # panel letter
            kw.update(fontweight="bold", fontsize=cs.FS_PANEL)
        elif kind == "m":
            kw["color"] = cs.INK2
        elif kind == "badge":
            kw.update(color="white", fontweight="bold",
                      bbox=dict(boxstyle="circle,pad=0.16", fc=cs.INK, ec="none"))
            g = gap + 2.6
        if prev is not None and prev[1] == "badge":
            g = max(gap, 1.2) + 3.0
        if prev is None:
            a = ax.text(x, y, txt, transform=transform if transform is not None else ax.transData, **kw)
        else:
            kw["va"] = "bottom"
            a = ax.annotate(txt, xy=(1, 0), xycoords=prev[0], xytext=(g, 0), textcoords="offset points", **kw)
        prev = (a, kind)
        out.append(a)
    return out


def surrogate_glyph(ax, x0, y0, w, h, kind, color):
    u = np.linspace(-1, 1, 81)
    if kind == "atan":
        g = 1 / (1 + (4.0 * u) ** 2)
    elif kind == "sig":
        s = 1 / (1 + np.exp(-9.0 * u))
        g = 4 * s * (1 - s)
    elif kind == "tri":
        g = np.clip(1 - np.abs(u) / 0.42, 0, None)
    elif kind == "esc":
        g = np.exp(-np.abs(u) / 0.16)
    else:  # STE: no window
        g = np.ones_like(u)
    ax.plot([x0, x0 + w], [y0, y0], color=cs.MUTED, lw=0.4, zorder=3, solid_capstyle="butt")
    ax.plot(x0 + (u + 1) / 2 * w, y0 + g * h, color=color, lw=0.9, zorder=4, solid_capstyle="round")


# ----------------------------------------------------------------------------- panel (a)
def panel_a(fig, rect, title_y):
    W, H = rect[2], rect[3]
    ax = fig.add_axes([rect[0] / FIG_W, rect[1] / FIG_H, W / FIG_W, H / FIG_H])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    chain(ax, 0.0, title_y - rect[1], [("(a)", "P"), ("The question", "n")], gap=2.5)

    xL = 0.72           # content starts here; stage tags to the left
    xR = W - 0.02
    # stage tags
    for y, tag, sub in ((2.13, "Train", "clean sensing"), (1.58, "Freeze", "members"),
                        (0.86, "Deploy", "sensor noise")):
        ax.text(0.0, y, tag, fontsize=cs.FS, fontweight="bold", color=cs.INK, ha="left", va="baseline")
        ax.text(0.0, y - 0.125, sub, fontsize=cs.FS, color=cs.INK2, ha="left", va="baseline")
    ax.text(0.0, 1.58 - 0.25, r"$f_1$" "–" r"$f_5$", fontsize=cs.FS, color=cs.INK2, ha="left",
            va="baseline")
    ax.text(0.0, 0.86 - 0.25, r"$\delta$", fontsize=cs.FS, color=cs.INK2, ha="left", va="baseline")

    # train row
    ty, th = 1.96, 0.30
    rbox(ax, xL, ty, xR - xL, th, fc="white", ec=cs.AXIS)
    ax.text((xL + xR) / 2, ty + 0.20, "same network, init, data order;", fontsize=cs.FS, ha="center",
            va="center")
    ax.text((xL + xR) / 2, ty + 0.085, r"only the surrogate $\partial S/\partial U$ differs", fontsize=cs.FS,
            ha="center", va="center")

    # member row, framed as one group of frozen members
    names = [("ATan", "atan"), ("sigm.", "sig"), ("STE", "ste"), ("tri.", "tri"), ("esc.", "esc")]
    bw = 0.29
    gap = (xR - xL - 5 * bw) / 4
    by, bh = 1.40, 0.29
    g0, g1 = by - 0.20, by + bh + 0.05            # group frame
    ax.add_patch(FancyBboxPatch((xL - 0.05, g0), xR - xL + 0.07, g1 - g0, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="none", ec=cs.MUTED, lw=0.6, ls=(0, (2.5, 1.5)), zorder=1))
    for i, (nm, kind) in enumerate(names):
        x0 = xL + i * (bw + gap)
        col = C_STE if kind == "ste" else C_WIN
        rbox(ax, x0, by, bw, bh, fc=cs.tint(col, 0.86), ec=col, lw=0.8)
        surrogate_glyph(ax, x0 + 0.04, by + 0.05, bw - 0.08, bh - 0.11, kind, cs.shade(col, 0.25))
        ax.text(x0 + bw / 2, by - 0.035, nm, fontsize=cs.FS, ha="center", va="top", color=cs.INK)
    arrow(ax, ((xL + xR) / 2, ty), ((xL + xR) / 2, g1))

    # deploy row
    yb, hb = 0.66, 0.27
    rbox(ax, xL, yb, 0.40, hb, fc="white", ec=cs.AXIS)
    ax.text(xL + 0.20, yb + hb / 2, r"$x+\delta$", fontsize=cs.FS, ha="center", va="center")
    sx0, sx1 = xL + 0.56, xR - 0.30
    rbox(ax, sx0, yb, sx1 - sx0, hb, fc=cs.tint(C_NULL, 0.80), ec=cs.AXIS)
    ax.text((sx0 + sx1) / 2, yb + hb / 2, r"$\Sigma_j\, w_j\, f_j(x+\delta)$", fontsize=cs.FS, ha="center",
            va="center")
    ax.text(xR - 0.08, yb + hb / 2, r"$\hat{\tau}$", fontsize=cs.FS + 1, ha="center", va="center")
    arrow(ax, (sx1, yb + hb / 2), (xR - 0.16, yb + hb / 2))
    # noisy input into the members, member outputs into the average
    arrow(ax, (xL + 0.20, yb + hb), (xL + 0.20, g0))
    arrow(ax, ((sx0 + sx1) / 2, g0), ((sx0 + sx1) / 2, yb + hb))
    ax.text(xL, 0.50, r"$w\geq 0,\ \Sigma_j w_j=1$, fit on a short recording", fontsize=cs.FS, color=cs.INK2,
            ha="left", va="baseline")

    # the three questions
    ax.plot([0.0, xR], [0.42, 0.42], color=cs.GRID, lw=0.6)
    qs = [("1", "Position", ": where do the members lie?", "(b)"),
          ("2", "Reachable", ": what can averaging reach?", "(b)"),
          ("3", "Needed", ": which point does the task need?", "(c)")]
    for i, (k, head, rest, pan) in enumerate(qs):
        y = 0.27 - i * 0.135
        chain(ax, 0.035, y, [(k, "badge"), (head, "b"), (rest, "n")], gap=0.0)
        ax.text(xR, y, pan, fontsize=cs.FS, color=cs.INK2, ha="right", va="baseline")
    return ax


# ----------------------------------------------------------------------------- panels (b), (c)
def plane_axes(ax):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel(r"Signal gain $\beta$")


def cond_mean(ax):
    bb = np.linspace(0.0, 1.0, 400)
    ax.plot(bb, bb * (1 - bb) * PY, color=cs.MUTED, lw=0.9, zorder=2, solid_capstyle="butt")


def iso(ax, R, leq, lw, ls, color=cs.INK2, z=2):
    xx = np.linspace(XLIM[0] - 0.05, XLIM[1] + 0.05, 800)
    yy = leq * (R - (1 - xx) ** 2 * PY)
    yy[yy < 0] = np.nan
    ax.plot(xx, yy, color=color, lw=lw, ls=ls, zorder=z)


def members(ax, b, n, ba):
    for bi, ni in zip(b[:4], n[:4]):
        ax.scatter([bi], [ni], s=22, marker=cs.MK["D-M window"], color=C_WIN, edgecolors="white", linewidths=0.6,
                   zorder=7)
    ax.scatter([STE_P[0]], [STE_P[1]], s=30, marker=cs.MK["STE"], color=C_STE, edgecolors="white",
               linewidths=0.6, zorder=7)
    ax.scatter([ba[0]], [ba[1]], s=30, marker=cs.MK["B-A"], color=C_BA, edgecolors="white", linewidths=0.6,
               zorder=7)
    ax.scatter([MLP_P[0]], [MLP_P[1]], s=30, marker=cs.MK["MLP"], color=C_MLP, edgecolors="white",
               linewidths=0.5, zorder=7)


def opt_marker(ax, p, mk, s=30):
    ax.scatter([p[0]], [p[1]], s=s, marker=mk, facecolors="white", edgecolors=cs.INK, linewidths=1.0, zorder=8)


def label(ax, x, y, txt, ha="left", va="center", color=cs.INK, **kw):
    return ax.text(x, y, txt, fontsize=cs.FS, ha=ha, va=va, color=color, path_effects=HALO, zorder=9, **kw)


def panel_b(ax, b, n, N):
    simplex, cone = regions(b, N)
    w0 = optimum(b, N, null=False)
    w1 = optimum(b, N, null=True)
    r0, r1, r5 = risk(*w0[:2]), risk(*w1[:2]), risk(*EST_P)
    ba = (BA_BETA, r0 - (1 - BA_BETA) ** 2 * PY)
    print(f"toy W0: beta {w0[0]:.3f} nu2 {w0[1]:.3f} R {r0:.4f} w {np.round(w0[2], 3)}")
    print(f"toy W1: beta {w1[0]:.3f} nu2 {w1[1]:.3f} R {r1:.4f} sum w {w1[2].sum():.3f}")
    print(f"estimator: R {r5:.4f}; B-A at {ba[0]:.3f}, {ba[1]:.3f}")

    ax.add_patch(Polygon(cone, closed=True, fc=cs.tint(C_NULL, 0.72), ec=cs.shade(C_NULL, 0.1), lw=0.5,
                         ls=(0, (2.5, 1.5)), zorder=1))
    ax.add_patch(Polygon(simplex, closed=True, fc=cs.tint(C_WIN, 0.55), ec=C_WIN, lw=0.6, zorder=1.5))
    cond_mean(ax)
    for R in (r0, r1):        # iso-risk lines (L_eq = 1) through the W0 and W1 optima
        iso(ax, R, 1.0, 0.7, LS_L1)
    # scalar rescaling of a seed member, c from 1 down to 0.6, toward the origin
    c = np.linspace(1.0, 0.6, 60)
    ax.plot(c * WIN_C[0], c ** 2 * WIN_C[1], color=C_PATH, lw=1.1, zorder=3, solid_capstyle="butt")
    arrow(ax, (0.63 * WIN_C[0], 0.63 ** 2 * WIN_C[1]), (0.6 * WIN_C[0], 0.36 * WIN_C[1]), color=C_PATH, lw=1.1,
          ms=7, z=3)
    # input-side attenuation
    arrow(ax, (WIN_C[0] + 0.004, WIN_C[1] - 0.035), (EST_P[0] + 0.008, EST_P[1] + 0.012), color=cs.INK, lw=0.8,
          rad=-0.25, ms=7, z=6)
    members(ax, b, n, ba)
    opt_marker(ax, w0, cs.MK_SPACE["W0"])
    opt_marker(ax, w1, cs.MK_SPACE["W1"])
    opt_marker(ax, EST_P, cs.MK_SPACE["W5"], s=60)

    # direct labels
    label(ax, STE_P[0] - 0.012, STE_P[1] + 0.014, "STE", ha="right", va="bottom")
    label(ax, MLP_P[0] + 0.014, MLP_P[1], "MLP", ha="left")
    label(ax, ba[0] + 0.012, ba[1] - 0.045, "B-A", ha="center")
    label(ax, WIN_C[0] + 0.022, WIN_C[1] + 0.036, "window\nmembers", ha="center", va="bottom", linespacing=1.0)
    label(ax, 0.688, 0.432, r"simplex, $\Sigma w=1$", ha="center", va="bottom")
    ax.plot([0.700, 0.718], [0.428, 0.318], color=cs.INK2, lw=0.5, zorder=8)
    label(ax, 0.611, 0.272, "+ null\n" r"$\Sigma w\leq 1$", ha="center", linespacing=1.0)
    label(ax, 0.505, 0.548, "scalar rescaling\n(forbidden by " r"$\Sigma w=1$)", ha="left", va="top",
          linespacing=1.05)
    arrow(ax, (0.53, 0.47), (0.556, 0.178), color=cs.INK2, lw=0.5, ms=5, z=8)
    label(ax, 0.86, 0.205, "input-side\nattenuation", ha="left", linespacing=1.0)
    label(ax, 0.905, 0.10, r"$s=1$", ha="left", color=cs.INK2)
    label(ax, 1.045, r0 + 0.006, f"R = {r0:.2f}", ha="right", va="bottom", color=cs.INK2)
    label(ax, 1.045, r1 - 0.006, f"{r1:.2f}", ha="right", va="top", color=cs.INK2)
    return w0, w1


def panel_c(ax):
    cond_mean(ax)
    bw, nw = WIN_C
    # gain path restricted to c <= 1 (the weight spaces tested in the paper have sum w <= 1, no amplification)
    cf = bw * PY / (bw ** 2 * PY + nw)                          # fast-task optimum, interior (c < 1)
    cslow = min(1.0, bw * PY / (bw ** 2 * PY + nw / LEQ_SLOW))  # slow-task optimum, at the end c = 1
    pf = (cf * bw, cf ** 2 * nw)
    ps = (cslow * bw, cslow ** 2 * nw)
    rf, rs = risk(*pf), risk(*ps, leq=LEQ_SLOW)
    print(f"fast optimum c {cf:.3f} at {pf[0]:.3f}, {pf[1]:.3f} (R {rf:.4f}); "
          f"slow optimum c {cslow:.3f} at {ps[0]:.3f}, {ps[1]:.3f} (R_16 {rs:.4f})")
    # each family: the level through the task optimum (thick) and one higher level (thin)
    iso(ax, rf, 1.0, 0.9, LS_L1)
    iso(ax, rf + 0.07, 1.0, 0.5, LS_L1)
    iso(ax, rs, LEQ_SLOW, 0.9, LS_L16, color=cs.INK)
    iso(ax, rs + 0.008, LEQ_SLOW, 0.5, LS_L16, color=cs.INK)
    c = np.linspace(0.6, 1.0, 60)
    ax.plot(c * bw, c ** 2 * nw, color=C_PATH, lw=1.1, zorder=3, solid_capstyle="butt")
    for p in (pf, ps):
        ax.scatter([p[0]], [p[1]], s=30, marker="D", color=cs.INK, edgecolors="white", linewidths=0.6, zorder=8)
    label(ax, bw + 0.017, nw - 0.012, r"gain path $c\,f$", ha="left", color=cs.INK2)
    # L_eq: 5-Hz intention filter ~6, admittance ~20, PD tracking 18-21; the slow iso-risk lines use LEQ_SLOW
    label(ax, 0.505, 0.548, "slow task, " r"$L_{\rm eq}\approx 6$" "–20:\nno attenuation " r"($c=1$)"
          "\nintention filtering (5 Hz),\nadmittance, tracking", ha="left", va="top", linespacing=1.05)
    label(ax, 0.505, 0.058, "fast task, " r"$L_{\rm eq}\approx 1$" "\ncontact detection", ha="left",
          va="center", linespacing=1.05)
    label(ax, 1.045, rf - 0.008, r"$L_{\rm eq}=1$", ha="right", va="top", color=cs.INK2)
    label(ax, 0.828, 0.021, r"$L_{\rm eq}=16$", ha="left", va="center", color=cs.INK2)
    label(ax, 0.905, 0.10, r"$s=1$", ha="left", color=cs.INK2)
    # leaders from the task labels to their optima
    ax.plot([0.585, pf[0] - 0.006], [0.098, pf[1] - 0.016], color=cs.INK2, lw=0.5, zorder=6)
    ax.plot([0.745, ps[0] - 0.010], [0.428, ps[1] + 0.010], color=cs.INK2, lw=0.5, zorder=6)
    return pf, ps


def make():
    cs.apply_style()
    b, n, N = toy()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    bottom, top = 0.62, 2.74
    title_y = top + 0.10
    panel_a(fig, (0.04, 0.30, 2.50, 2.40), title_y)
    axb = fig.add_axes([2.98 / FIG_W, bottom / FIG_H, 2.12 / FIG_W, (top - bottom) / FIG_H])
    axc = fig.add_axes([5.20 / FIG_W, bottom / FIG_H, 1.86 / FIG_W, (top - bottom) / FIG_H], sharey=axb)
    plane_axes(axb)
    plane_axes(axc)
    axb.set_ylabel(r"Orthogonal noise power $\|\nu\|^2$")
    axc.tick_params(labelleft=False)
    panel_b(axb, b, n, N)
    panel_c(axc)
    tb = fig.transFigure
    chain(axb, 2.62 / FIG_W, title_y / FIG_H, [("(b)", "P"), ("The plane:", "n"), ("1", "badge"),
                                               ("position,", "n"), ("2", "badge"), ("reach", "n")],
          transform=tb, gap=2.5)
    chain(axc, 5.20 / FIG_W, title_y / FIG_H, [("(c)", "P"), ("The task filter:", "n"), ("3", "badge"),
                                               ("need", "n")], transform=tb, gap=2.5)

    def mh(mk, col, s=5, mfc=None, mec="white", mew=0.5):
        return Line2D([], [], ls="none", marker=mk, ms=s, color=col, mfc=mfc if mfc else col, mec=mec, mew=mew)
    handles = [mh(cs.MK["D-M window"], C_WIN), mh(cs.MK["STE"], C_STE), mh(cs.MK["B-A"], C_BA),
               mh(cs.MK["MLP"], C_MLP),
               mh(cs.MK_SPACE["W0"], cs.INK, mfc="white", mec=cs.INK, mew=0.9),
               mh(cs.MK_SPACE["W1"], cs.INK, mfc="white", mec=cs.INK, mew=0.9),
               mh(cs.MK_SPACE["W5"], cs.INK, s=7, mfc="white", mec=cs.INK, mew=0.9),
               mh("D", cs.INK, s=4.5)]
    labels = ["Window members", "STE member", "B-A", "MLP", r"W0 optimum ($\Sigma w=1$)",
              r"W1 optimum ($\Sigma w\leq 1$)", "Estimator + W1", "Task optimum"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False,
               fontsize=cs.FS, handletextpad=0.3, columnspacing=1.1, handlelength=1.0, borderaxespad=0.15)
    cs.savefig(fig, "fig1")
    plt.close(fig)


if __name__ == "__main__":
    make()
