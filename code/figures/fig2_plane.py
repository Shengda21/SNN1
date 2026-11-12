"""Fig. 2 (fig:plane): the signal-gain / orthogonal-noise plane.

Rows: platform (SARCOS, simulated arm); columns: condition (N, O1-M). Each panel shows
  * member families at (beta, ||nu||^2) on the test role (e15_members.csv, role = test):
    large marker = family mean of the per-seed means, small markers = per-seed means
    (D-S seed = outer_seed with five members; D-M window = the four window members of a seed);
  * the scalar-scaling path (c*beta, c^2*||nu||^2), c in [0.5, 1.1], of one seed member
    (ds_<strong>_s0_m0; tick at c = 1);
  * the E16 width sweep (e15_members_w16.csv, HWHM from the hwhm column or parsed from
    w16_<fam>_h<HWHM>_s<seed>): thin lines = per-family (tri/sig/atan/sre) means per HWHM,
    thick line = mean of the four families, dashed = widest window to the STE family mean;
  * weight-space optima of D-S and D-M in W0, W1, W3 and W5 (input estimator + W1),
    e21_ladder.csv columns beta, nu2;
  * O1-M panels only: E18 single models in W0 (e21_ladder_e18.csv, columns beta, nu2): the input-
    augmentation sweep w18_aug<scale> and the membrane-noise sweep w18_mem<sigma>, each joined in
    order of the level, and the bounded-activation MLPs w18_mlp_{tanh,crelu,bin};
  * the conditional-mean curve ||nu||^2 = beta (1 - beta) P_y (P_y from e15 column Py) and the
    iso-risk curves through the D-S W0 optimum for L_eq = 1, (1-beta)^2 P_y + ||nu||^2 = R, and
    L_eq = 16, (1-beta)^2 P_y + ||nu||^2/16 = const (O1-M panels only: the 1/L_eq power reduction of a
    task filter holds for white observation noise, and N has none).
Run:  python fig2_plane.py      (writes build/fig/fig2.pdf)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common_style as cs

cs.FS = cs.FS_SMALL = 6.5  # compact version for the 12-page budget (never below 6.5 pt)
cs.FS_PANEL = 7.5

COND_COLS = ["N", "O1-M"]
ROWS = ["sarcos", "sim"]
INCLUDE_CAP4 = False  # E14 cap4 members (optional; off keeps the N panels readable)
# (key, e15 group, legend label)
FAMILIES = [
    ("D-S", "D-S", "D-S (seed members)"),
    ("D-M window", "D-M window", "D-M window members"),
    ("STE", "D-M STE", "STE members"),
    ("B-L", "B-L", "B-L"),
    ("B-A", "B-A", "B-A"),
    ("MLP", "MLP", "MLP (B-N)"),
] + ([("cap4", "E14 cap4", "E14 cap4")] if INCLUDE_CAP4 else [])
SPACES = [("W0", "W0: simplex"), ("W1", "W1: simplex + null"), ("W3", "W3: per-joint + null"),
          ("W5", "W5: input estimator + W1")]
SCALE_C = np.linspace(0.5, 1.1, 61)
LEQ = (1, 16)
LEQ_LS = {1: (0, (4, 2)), 16: (0, (1, 1.3))}
W2STE_LS = (0, (3, 1.2, 1, 1.2))  # widest window -> STE (dash-dot)
OPT_LS = {"D-S": "-", "D-M": (0, (2, 1))}  # ladder lines joining the optima
W16_FAMS = ["tri", "sig", "atan", "sre"]
DIRECT = {"STE": "STE", "B-A": "B-A", "MLP": "MLP", "cap4": "cap4"}  # direct labels (outside insets)
FIG_H = 3.2  # in (page budget: at most 4.3 in, and figure + caption at most 0.55 page)
SHORT = {"D-S": "D-S", "D-M window": "D-M window", "STE": "STE", "B-L": "B-L", "B-A": "B-A", "MLP": "MLP"}
# inset position (axes fraction) for the nominal panels
INSET = {("sarcos", "N"): [0.15, 0.33, 0.40, 0.62], ("sim", "N"): [0.13, 0.30, 0.40, 0.44]}
# E18 knobs (e21_ladder_e18.csv, single models in W0), drawn in these conditions only
E18_CONDS = ["O1-M"]
E18_SPACE = "W0"
# callout positions (axes fraction) of the two knob labels; the arrow points to the strongest setting
E18_LABEL_POS = {("sarcos", "O1-M"): {"aug": (0.515, 0.13), "mem": (0.92, 0.40)},
                 ("sim", "O1-M"): {"aug": (0.25, 0.60), "mem": (0.62, 0.93)}}
# direct labels placed to the right of the family mean instead of above-left (avoids the E18 lines)
DIRECT_RIGHT = {("sarcos", "O1-M", "B-A")}
# ... or below-right (MLP at O1-M: the bounded-activation MLP markers and the L_eq=16 line sit upper-left)
DIRECT_BELOW_RIGHT = {("sarcos", "O1-M", "MLP"), ("sim", "O1-M", "MLP")}


def load(platform):
    m15 = cs.read(platform, "e15_members.csv", ["member", "group", "role", "cond", "beta", "nu2", "Py", "seed",
                                                  "outer_seed"])
    w16 = cs.read(platform, "e15_members_w16.csv", ["member", "role", "cond", "beta", "nu2"])
    lad = cs.read(platform, "e21_ladder.csv", ["cond", "method", "space", "beta", "nu2", "Py"])
    m15 = m15[m15.role == "test"]
    w16 = w16[w16.role == "test"].copy()
    parsed = w16["member"].apply(cs.parse_w16)
    w16["fam"] = [p[0] for p in parsed]
    w16["H"] = w16["hwhm"] if "hwhm" in w16.columns else [p[1] for p in parsed]
    try:
        e18 = cs.read(platform, "e21_ladder_e18.csv", ["cond", "method", "space", "beta", "nu2"])
    except FileNotFoundError:
        e18 = None
    return m15, w16, lad, e18


def e18_data(e18, cond):
    """Input-augmentation and membrane-noise sweeps (sorted by level) and the bounded MLPs."""
    if e18 is None:
        return None
    sub = e18[(e18.cond == cond) & (e18.space == E18_SPACE)]
    out = {}
    for knob, pat in (("aug", r"^w18_aug([0-9.]+)$"), ("mem", r"^w18_mem([0-9.]+)$")):
        lev = sub["method"].str.extract(pat)[0].astype(float)
        g = sub[lev.notna()].assign(level=lev[lev.notna()]).sort_values("level")
        if len(g):
            out[knob] = g[["level", "beta", "nu2"]].reset_index(drop=True)
    for act in ("tanh", "crelu", "bin"):
        r = sub[sub.method == f"w18_mlp_{act}"]
        if len(r):
            out[act] = (float(r.beta.iloc[0]), float(r.nu2.iloc[0]))
    return out or None


def panel_data(platform, cond, m15, w16, lad, e18=None):
    c15 = cs.E15_COND[platform][cond]
    mc = m15[m15.cond == c15]
    d = {"Py": float(mc["Py"].iloc[0]), "fam": {}, "sol": {}, "width": {}}
    for key, grp, _ in FAMILIES:
        sub = mc[mc.group == grp]
        if sub.empty:
            continue
        scol = "outer_seed" if grp == "D-S" else "seed"
        per = sub.groupby(scol)[["beta", "nu2"]].mean()
        d["fam"][key] = (per, per.mean())
    mem = mc[mc.member == f"ds_{cs.STRONG[platform]}_s0_m0"].iloc[0]
    d["scale_member"] = (float(mem.beta), float(mem.nu2))
    wc = w16[w16.cond == c15]
    for f in W16_FAMS:
        g = wc[wc.fam == f].groupby("H")[["beta", "nu2"]].mean().sort_index()
        if len(g):
            d["width"][f] = g
    if d["width"]:
        allw = wc[wc.fam.isin(W16_FAMS)].groupby(["fam", "H"])[["beta", "nu2"]].mean()
        d["width_mean"] = allw.groupby("H").mean().sort_index()
    lc = lad[lad.cond == cond]
    for dn in ("D-S", "D-M"):
        for sp, _ in SPACES:
            r = lc[(lc.method == dn) & (lc.space == sp)]
            if len(r):
                d["sol"][(dn, sp)] = (float(r.beta.iloc[0]), float(r.nu2.iloc[0]))
    r0 = lc[(lc.method == "D-S") & (lc.space == "W0")].iloc[0]
    d["iso_anchor"] = (float(r0.beta), float(r0.nu2))
    if cond in E18_CONDS:
        k = e18_data(e18, cond)
        if k:
            d["e18"] = k
    return d


def all_points(d):
    pts = []
    for per, _ in d["fam"].values():
        pts += list(zip(per.beta, per.nu2))
    pts += list(d["sol"].values())
    b, n = d["scale_member"]
    pts += [(SCALE_C[0] * b, SCALE_C[0] ** 2 * n), (SCALE_C[-1] * b, SCALE_C[-1] ** 2 * n)]
    for g in d["width"].values():
        pts += list(zip(g.beta, g.nu2))
    for key, v in d.get("e18", {}).items():
        pts += list(zip(v.beta, v.nu2)) if key in ("aug", "mem") else [v]
    return np.array(pts)


def cluster_box(d):
    """Bounding box of the dense cluster near (1, 0): window-like members and the optima."""
    pts = []
    for k in ("D-S", "D-M window", "B-L", "MLP"):
        if k in d["fam"]:
            pts += list(zip(d["fam"][k][0].beta, d["fam"][k][0].nu2))
    pts += list(d["sol"].values())
    pts = np.array(pts)
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    px, py = 0.25 * (x1 - x0), 0.30 * (y1 - y0)  # margin keeps the zoom box visible around the markers
    return x0 - px, x1 + px, max(0.0, y0 - py), y1 + py


def draw(ax, d, xlim, ylim, main=True, main_cond_o1m=True):
    Py = d["Py"]
    bb = np.linspace(0.0, 1.0, 600)
    ax.plot(bb, bb * (1 - bb) * Py, color=cs.MUTED, lw=0.8, zorder=1)
    b0, n0 = d["iso_anchor"]
    xx = np.linspace(min(xlim[0], 0.0), 1.0 + 0.6, 1200)
    for L in LEQ:
        if L > 1 and not main_cond_o1m:  # L_eq > 1 (white-noise filter gain) only where observation noise exists
            continue
        RL = (1 - b0) ** 2 * Py + n0 / L
        yy = L * (RL - (1 - xx) ** 2 * Py)
        yy[yy < 0] = np.nan
        ax.plot(xx, yy, color=cs.INK2, lw=0.7, ls=LEQ_LS[L], zorder=1)
    # scalar-scaling path
    b, n = d["scale_member"]
    ax.plot(SCALE_C * b, SCALE_C ** 2 * n, color=cs.C["D-S"], lw=0.9, zorder=2, solid_capstyle="round")
    ax.plot([b], [n], marker="|", color=cs.C["D-S"], ms=4.5, mew=1.0, zorder=2)
    # width sweep
    for g in d["width"].values():
        ax.plot(g.beta, g.nu2, color=cs.C["width"], lw=0.4, alpha=0.55, zorder=3)
    ste = d["fam"].get("STE", (None, None))[1]
    if "width_mean" in d:
        g = d["width_mean"]
        ax.plot(g.beta, g.nu2, color=cs.C["width"], lw=1.0, marker="o", ms=2.0, mew=0, zorder=3)
        if ste is not None:
            ax.plot([g.beta.iloc[-1], ste.beta], [g.nu2.iloc[-1], ste.nu2], color=cs.C["width"], lw=0.7,
                    ls=W2STE_LS, zorder=2)
    # member families
    for key, _, _ in FAMILIES:
        if key not in d["fam"]:
            continue
        per, mean = d["fam"][key]
        col, mk = cs.C[key], cs.MK[key]
        unfilled = mk in ("x", "+")
        ax.scatter(per.beta, per.nu2, s=5 if main else 8, marker=mk, color=col, alpha=0.45,
                   linewidths=0.8 if unfilled else 0.0, zorder=4)
        kw = dict(color=col, linewidths=1.2) if unfilled else dict(color=col, edgecolors="white", linewidths=0.5)
        ax.scatter([mean.beta], [mean.nu2], s=24 if main else 28, marker=mk, zorder=6, **kw)
    # weight-space optima
    for dn in ("D-S", "D-M"):
        pts = [d["sol"][(dn, sp)] for sp, _ in SPACES if (dn, sp) in d["sol"]]
        if pts:
            p = np.array(pts)
            ax.plot(p[:, 0], p[:, 1], color=cs.C[dn], lw=0.6, alpha=0.8, ls=OPT_LS[dn], zorder=5)
        for sp, _ in SPACES:
            if (dn, sp) in d["sol"]:
                x, y = d["sol"][(dn, sp)]
                ax.scatter([x], [y], s=22 if sp != "W5" else 42, marker=cs.MK_SPACE[sp], facecolors="white",
                           edgecolors=cs.C[dn], linewidths=0.9, zorder=7)
                if dn == "D-M":  # centre dot: dictionary identity survives greyscale printing
                    ax.scatter([x], [y], s=2.5, marker="o", color=cs.shade(cs.C[dn], 0.25), linewidths=0, zorder=8)
    # E18 knobs: input-augmentation and membrane-noise sweeps, bounded-activation MLPs
    if main and "e18" in d:
        k = d["e18"]
        for knob in ("aug", "mem"):
            if knob in k:
                g = k[knob]
                if knob == "mem":  # tri-down stroke marker (not the thin diamond, which reads as the W3 optimum)
                    ax.plot(g.beta, g.nu2, color=cs.C_E18[knob], lw=0.9, ls=cs.LS_E18[knob], marker=cs.MK_MEM,
                            ms=5.0, mec=cs.C_E18[knob], mew=0.9, zorder=5.5)
                else:
                    ax.plot(g.beta, g.nu2, color=cs.C_E18[knob], lw=0.9, ls=cs.LS_E18[knob], marker=cs.MK_E18[knob],
                            ms=3.6, mfc="white", mec=cs.C_E18[knob], mew=0.8, zorder=5.5)
        for act in ("tanh", "crelu", "bin"):
            if act in k:
                ax.scatter([k[act][0]], [k[act][1]], marker=cs.MK_E18[act], s=20, color=cs.C_E18["mlp"],
                           edgecolors="white", linewidths=0.5, zorder=5.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def label_e18(ax, d, pos):
    """Callout labels for the two E18 sweeps, pointing at the strongest setting."""
    if "e18" not in d or pos is None:
        return
    for knob, txt in (("aug", "input\naugmentation"), ("mem", "membrane\nnoise")):
        if knob not in d["e18"] or knob not in pos:
            continue
        g = d["e18"][knob]
        ax.annotate(txt, xy=(g.beta.iloc[-1], g.nu2.iloc[-1]), xytext=pos[knob], textcoords="axes fraction",
                    ha="center", va="center", fontsize=cs.FS_SMALL, color=cs.INK, linespacing=0.95, zorder=9,
                    arrowprops=dict(arrowstyle="-", color=cs.MUTED, lw=0.5, shrinkA=1, shrinkB=2))


def label_families(ax, d, xlim, ylim, avoid=None, right=(), below_right=()):
    dx = 0.012 * (xlim[1] - xlim[0])
    dy = 0.02 * (ylim[1] - ylim[0])
    for key, txt in DIRECT.items():
        if key not in d["fam"]:
            continue
        m = d["fam"][key][1]
        if not (xlim[0] < m.beta < xlim[1] and ylim[0] < m.nu2 < ylim[1]):
            continue
        if avoid is not None and avoid[0] <= m.beta <= avoid[1] and avoid[2] <= m.nu2 <= avoid[3]:
            continue
        if key in below_right:
            ax.text(m.beta + 1.0 * dx, m.nu2 - 1.5 * dy, txt, fontsize=cs.FS_SMALL, color=cs.INK, ha="left",
                    va="top", zorder=8)
        elif key in right:
            ax.text(m.beta + 1.2 * dx, m.nu2, txt, fontsize=cs.FS_SMALL, color=cs.INK, ha="left", va="center",
                    zorder=8)
        else:
            ax.text(m.beta - dx, m.nu2 + dy, txt, fontsize=cs.FS_SMALL, color=cs.INK, ha="right", va="bottom",
                    zorder=8)


def make():
    cs.apply_style()
    data = {}
    e18_seeds, e18_levels = set(), {"aug": set(), "mem": set()}
    for p in ROWS:
        m15, w16, lad, e18 = load(p)
        for c in COND_COLS:
            data[(p, c)] = panel_data(p, c, m15, w16, lad, e18)
        if e18 is not None and "seeds" in e18.columns:
            e18_seeds |= set(e18[e18.method.str.startswith("w18_")]["seeds"].astype(str).unique())
        for c in E18_CONDS:
            for knob in ("aug", "mem"):
                if knob in data[(p, c)].get("e18", {}):
                    e18_levels[knob] |= set(data[(p, c)]["e18"][knob].level)

    fig = plt.figure(figsize=(cs.DOUBLE_COL, FIG_H))
    gs = fig.add_gridspec(2, 2, left=0.066, right=0.988, top=0.935, bottom=0.185, wspace=0.12, hspace=0.30)
    letters = iter("abcd")
    for i, p in enumerate(ROWS):
        for j, c in enumerate(COND_COLS):
            ax = fig.add_subplot(gs[i, j])
            d = data[(p, c)]
            pts = all_points(d)
            xlim = (np.floor((min(pts[:, 0].min(), 1.0) - 0.03) * 20) / 20, 1.03)
            ylim = (0.0, pts[:, 1].max() * 1.08)
            draw(ax, d, xlim, ylim, main_cond_o1m=(c != "N"))
            box = cluster_box(d) if c == "N" else None
            label_families(ax, d, xlim, ylim, avoid=box, right={k for q, cc, k in DIRECT_RIGHT if (q, cc) == (p, c)},
                           below_right={k for q, cc, k in DIRECT_BELOW_RIGHT if (q, cc) == (p, c)})
            label_e18(ax, d, E18_LABEL_POS.get((p, c)))
            ax.set_title(f"{cs.PLATFORM_NAME[p]}, {c}", loc="left", fontsize=cs.FS, pad=2)
            cs.panel_label(ax, next(letters), x=-0.085, y=1.03, ha="left")
            if i == len(ROWS) - 1:
                ax.set_xlabel(r"Signal gain $\beta$", labelpad=1)
            if j == 0:
                ax.set_ylabel(r"Orthogonal noise $\|\nu\|^2$", labelpad=1)
            if c == "N":  # small zoom of the cluster near (1, 0); no tick labels (location shown by the box)
                ins = ax.inset_axes(INSET[(p, c)])
                draw(ins, d, box[:2], box[2:], main=False, main_cond_o1m=False)
                ins.set_xticks([])
                ins.set_yticks([])
                ins.grid(False)
                ins.set_facecolor("white")
                for sp_ in ins.spines.values():
                    sp_.set_visible(True)
                    sp_.set_color(cs.AXIS)
                    sp_.set_linewidth(0.5)
                rect, conns = ax.indicate_inset_zoom(ins, edgecolor=cs.INK2, lw=0.6, alpha=1.0)
                rect.set_zorder(10)
                for cn in conns:  # box only: the connector lines read as data curves
                    cn.set_visible(False)
                for sp_ in ins.spines.values():
                    sp_.set_color(cs.INK2)
                    sp_.set_linewidth(0.6)

    # legend: two rows below the panels (members and optima; curves and E18 MLPs)
    def mh(key):
        mk = cs.MK[key]
        unf = mk in ("x", "+")
        return Line2D([], [], ls="none", marker=mk, ms=4.6, color=cs.C[key], mec=cs.C[key] if unf else "white",
                      mew=1.1 if unf else 0.5)
    leg_kw = dict(frameon=False, fontsize=cs.FS, handletextpad=0.3, columnspacing=0.95, handlelength=1.5,
                  borderaxespad=0.0, borderpad=0.0)
    h1 = [mh(k) for k, _, _ in FAMILIES]
    l1 = [SHORT.get(k, lab) for k, _, lab in FAMILIES]
    h1 += [Line2D([], [], ls="none", marker=cs.MK_SPACE[sp], ms=4.6 if sp != "W5" else 6.4, mfc="white", mec=cs.INK,
                  mew=0.9) for sp, _ in SPACES]
    l1 += [sp for sp, _ in SPACES]
    h1 += [Line2D([], [], color=cs.C["D-S"], lw=0.6, marker="o", ms=4.2, mfc="white", mec=cs.C["D-S"], mew=0.9),
           (Line2D([], [], color=cs.C["D-M"], lw=0.6, ls=OPT_LS["D-M"], marker="o", ms=4.2, mfc="white",
                   mec=cs.C["D-M"], mew=0.9),
            Line2D([], [], ls="none", marker="o", ms=1.5, color=cs.shade(cs.C["D-M"], 0.25)))]
    l1 += ["D-S optima", "D-M optima"]
    lg1 = fig.legend(h1, l1, loc="lower center", bbox_to_anchor=(0.5, 0.047), ncol=len(h1), **leg_kw)
    h2 = [Line2D([0, 0.5, 1], [0, 0, 0], color=cs.C["D-S"], lw=0.9, marker="|", ms=4.5, mew=1.0, markevery=[1]),
          Line2D([], [], color=cs.C["width"], lw=1.0, marker="o", ms=2.0, mew=0),
          Line2D([], [], color=cs.C["width"], lw=0.7, ls=W2STE_LS),
          Line2D([], [], color=cs.MUTED, lw=0.8),
          Line2D([], [], color=cs.INK2, lw=0.7, ls=LEQ_LS[1]),
          Line2D([], [], color=cs.INK2, lw=0.7, ls=LEQ_LS[16])]
    l2 = [r"scaling $cF$", "width sweep", "widest to STE", r"$s$ = 1", r"iso-risk $L_{\mathrm{eq}}$=1",
          r"$L_{\mathrm{eq}}$=16 (O1-M)"]
    if any("e18" in v for v in data.values()):
        h2 += [Line2D([], [], ls="none", marker=cs.MK_E18[a], ms=4.6, color=cs.C_E18["mlp"], mec="white", mew=0.5)
               for a in ("tanh", "crelu", "bin")]
        l2 += ["MLP tanh", "MLP cReLU", "MLP binary"]
    fig.legend(h2, l2, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(h2), **leg_kw)
    fig.add_artist(lg1)
    cs.savefig(fig, "fig2")
    return fig, data


if __name__ == "__main__":
    make()
