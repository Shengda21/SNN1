"""Fig. 3 (fig:width): surrogate width and the straight-through estimator (E15-E18). 7.16 x 3.6 in.

(a) Forward response r (O1-M), signal gain beta (O1-M) and nominal test nMSE (log axis) against the HWHM of
    the peak-normalized surrogate (log2 axis 1/16..64; STE in a separate slot at the right); columns =
    platforms. Families tri/sig/atan/sre of E16 (block w16): three seeds, mean +- seed SD; tri and atan continue
    to HWHM 8-64 with seed 0 only (small markers, dotted). Detached reset (block w16d; tri, atan at HWHM 1/16,
    1/4, 1, 4, 16; three seeds): hollow markers, dashed. STE: original D-M STE members (dm_ste, five seeds;
    filled) and detached-reset STE (w16d_ste, three seeds; hollow). Grey band: HWHM of the original window
    members (0.07-0.25). Ring: lowest mean nMSE among the three-seed tri/ATan configurations and dm_ste.
(b) Test nMSE against HWHM at O1-L, O1-M, O1-H (SARCOS has no O1-L; N is the last row of (a)), tri and atan
    drawn as in (a), with the same ring.
(c) Layer-1 spike-flip rate at O1-M: prediction of Lemma 2 (flip_pred_sg2: threshold at the mean, W1 and the
    input covariance only) against measurement (flip_measured: all internal steps), one point per member of
    E16 (w16 filled, w16d hollow) and of the original D-M (dark edge); Spearman rank correlation over all
    plotted members and over the window members (titles). Inset: first-layer unit SNR_k at O1-M pooled over
    the neurons of the 20 original window members and of the 5 STE members (e15_flip_neurons.npz); boxes
    25-75 %, whiskers 5-95 %.
(d) Left, E17: change in r and in O1-M nMSE relative to the narrow triangular surrogate (HWHM 1/4) when HWHM
    1 or 4 is used in layer 1 only (L1), layers 2-3 only (L2-3) or all layers (all = E16 w16 tri);
    differences paired by training seed, mean +- SD over three seeds. Right, E18 on the simulated arm at
    O1-M in the (beta, ||nu||^2) plane, values from e15_width_summary.csv: first-layer membrane noise sigma in
    {0.25, 0.5, 1, 2} and input augmentation at {0.25, 0.5, 1, 1.4} x rho_M, both drawn from the narrow
    strong-surrogate members (dm_sigmoid, seeds 0-2), the triangular width sweep (HWHM 1/16-4, three seeds;
    8-64, seed 0), the MLP with ReLU (bn), tanh, clipped ReLU and binary activation with straight-through
    backward pass (w18mlp), the 20 original window members (dots; pentagon = mean), STE (filled, original;
    hollow, detached reset) and B-A (= augmentation 1x); grey curves: iso-risk (1 - beta)^2 P_y + ||nu||^2.
    Both platforms' E18 knobs are also drawn in Fig. 2.

Data (results/<platform>/):
  e15_members_new.csv, e15_flip_new.csv, e15_width_summary.csv   E15 pipeline on all E16-E18 members
  e15_members.csv, e15_flip.csv, e15_flip_neurons.npz            original members (dm_*, ba_*, bn_*)
  Fallback when e15_members_new.csv is missing: e16_members.csv for every member (r from e16_summary.py,
  preliminary) and e15_flip_w16.csv (partial); the script prints a note.
  Member-level means are checked against e15_width_summary.csv (printed max |difference|).
Run:  python fig3_width.py      (writes build/fig/fig3.pdf; SNN29_PNG=<dir> also writes a PNG)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
from scipy.stats import spearmanr

import common_style as cs

cs.FS = cs.FS_SMALL = 6.5  # compact version for the 12-page budget (never below 6.5 pt)
cs.FS_PANEL = 7.5

PLATS = ["sim", "sarcos"]
LEVELS = ["N", "O1-L", "O1-M", "O1-H"]
RHO = {"sim": {"O1-L": 0.35, "O1-M": 0.7, "O1-H": 1.0}, "sarcos": {"O1-M": 0.5, "O1-H": 0.7}}
HW_MAIN = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
HW_ULTRA = [8.0, 16.0, 32.0, 64.0]
HW_E17 = (1.0, 4.0)
X_STE, X_BREAK = 7.85, 6.95
DODGE = {"tri": -0.17, "sig": -0.06, "atan": 0.06, "sre": 0.17}
DODGE_D = {"tri": -0.11, "atan": 0.11}
ORIG_BAND = (0.07, 0.25)  # HWHM of the original window members (method.tex / plan appendix C)
BAND = "#f0efec"
EDGE_ORIG = cs.INK      # edge of the original D-M members in (c)
NOTES = []

# (c) inset position in axes fractions (left, bottom, width, height), in the empty upper-left corner
INSET_C = {"sim": (0.26, 0.60, 0.44, 0.21), "sarcos": (0.26, 0.60, 0.44, 0.21)}
# (d) E18 direct labels. ("off", dx, dy, ha, va): offset in points from the anchor;
# ("at", beta, nu2, ha, va): text at data coordinates with a leader line to the anchor.
LAB18 = {
    "sim": {"width": ("off", 4, 2, "left", "bottom"), "membrane": ("at", 0.722, 0.437, "center", "center"),
            "augment": ("off", 4, 0, "left", "center"), "ba": ("off", 5, -2, "left", "top"),
            "ste": ("off", 0, -5, "center", "top"), "win": ("off", 6, -4, "left", "top"),
            "bin": ("off", -5, 0, "right", "center")},
    "sarcos": {"width": ("off", 4, 2, "left", "bottom"), "membrane": ("off", 6, -6, "right", "top"),
               "augment": ("off", -5, 0, "right", "center"), "ba": ("off", -6, 0, "right", "center"),
               "ste": ("off", 0, -5, "center", "top"), "win": ("at", 0.785, 0.215, "left", "center"),
               "relu": ("off", 0, -5, "center", "top"), "tanh": ("off", -5, 0, "right", "center"),
               "crelu": ("off", -5, 0, "right", "center"), "bin": ("at", 0.835, 0.405, "center", "bottom")},
}
AUG_ANCHOR = {"sim": -1, "sarcos": 2}   # index of the augmentation point that carries its label
WIDTH_ANCHOR = {"sim": "top", "sarcos": "top"}  # width label at the HWHM-4 point or at the highest point
TXT18 = {"width": "width", "membrane": "membrane", "augment": "input aug.", "ba": "B-A", "ste": "STE",
         "win": "window", "relu": "ReLU", "tanh": "tanh", "crelu": "clip.", "bin": "bin."}
TXT18_P = {}


def note(msg):
    NOTES.append(msg)
    print("note:", msg)


# ----------------------------------------------------------------------------- data
def _cond(p, lev):
    return cs.E15_COND[p].get(lev)


def _bool(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def _wide_e15(long, p, orig):
    t = long[long.role == "test"]
    base = t[t.cond == "N"].drop_duplicates("member").set_index("member")
    out = pd.DataFrame(index=base.index)
    if orig:
        out["block"] = base["prefix"]
        out["family"] = base["surrogate"].map(cs.FAM_OF_SURR)
        out["detach_reset"] = False
        out["placement"] = "all"
        out["membrane_noise"] = 0.0
        out["aug_scale"] = (base["aug_sigma"].fillna(0.0) / RHO[p]["O1-M"]).round(3)
        out["activation"] = np.where(base["surrogate"] == "none", "relu", "spike")
    else:
        for k in ["block", "family", "placement", "membrane_noise", "aug_scale", "activation"]:
            out[k] = base[k]
        out["detach_reset"] = _bool(base["detach_reset"])
    out["seed"] = base["seed"].astype(int)
    out["hwhm"] = base["hwhm"].astype(float)
    out["r"] = base["r_O1M"]
    for lev in LEVELS:
        c = _cond(p, lev)
        if c is not None:
            out[f"R_{lev}"] = t[t.cond == c].drop_duplicates("member").set_index("member")["R"]
    tm = t[t.cond == _cond(p, "O1-M")].drop_duplicates("member").set_index("member")
    out["beta"], out["nu2"] = tm["beta"], tm["nu2"]
    out["r_src"] = "e15"
    return out


def _wide_e16(p):
    e = cs.read(p, "e16_members.csv").set_index("member_id")
    out = pd.DataFrame(index=e.index)
    for k in ["block", "membrane_noise", "aug_scale", "activation", "seed"]:
        out[k] = e[k]
    out["family"] = [cs.FAM_OF_SURR.get(f, f) if isinstance(f, str) else "mlp" for f in e["family"]]
    out["placement"] = e["placement"].fillna("all")
    out["detach_reset"] = _bool(e["detach_reset"])
    out["hwhm"] = e["hwhm"].astype(float)
    cm = _cond(p, "O1-M")
    out["r"] = e[f"r_{cm}"]
    for lev in LEVELS:
        c = _cond(p, lev)
        if c is not None:
            out[f"R_{lev}"] = e[f"test_nmse_{c}"]
    out["beta"], out["nu2"] = e[f"beta_{cm}"], e[f"nu2_{cm}"]
    out["r_src"] = "e16"
    return out


def members(p):
    """One row per member: metadata, r (O1-M), beta and nu2 (O1-M), test nMSE R_<level>."""
    newf = cs.RES / p / "e15_members_new.csv"
    orig_long = cs.read(p, "e15_members.csv")
    py = float(orig_long.loc[orig_long.role == "test", "Py"].iloc[0])
    if newf.exists():
        new = _wide_e15(pd.read_csv(newf), p, orig=False)
        orig = _wide_e15(orig_long[orig_long.member.str.match(r"^(dm|ba|bn)_")], p, orig=True)
        M = pd.concat([new, orig])
        note(f"{p}: E15 pipeline for all members ({newf.name}: {len(new)}, e15_members.csv: {len(orig)})")
    else:
        M = _wide_e16(p)
        note(f"{p}: e15_members_new.csv missing -> e16_members.csv for all {len(M)} members "
             "(beta, nu2, nMSE identical to E15; r preliminary)")
    M["Py"] = py
    M["placement"] = M["placement"].fillna("all")
    M["seed"] = M["seed"].astype(int)
    return M


def summary(p):
    f = cs.RES / p / "e15_width_summary.csv"
    if not f.exists():
        note(f"{p}: e15_width_summary.csv missing -> (d) plane points from member means")
        return None
    S = pd.read_csv(f)
    S["detach_reset"] = _bool(S["detach_reset"])
    S["placement"] = S["placement"].fillna("all")
    return S


def srow(S, block, family=None, **kw):
    d = S[S.block == block]
    if family is not None:
        d = d[d.family == family]
    for k, v in kw.items():
        d = d[np.isclose(d[k].astype(float), v)] if isinstance(v, float) else d[d[k] == v]
    if len(d) != 1:
        raise KeyError(f"summary row {block} {family} {kw}: {len(d)} rows")
    return d.iloc[0]


def check_summary(M, S, p):
    """Max |member-level mean - e15_width_summary mean| over the rows of the summary."""
    if S is None:
        return np.nan
    worst = 0.0
    q = {"r": "r", "beta": "beta", "nu2": "nu2", "R_N": "nmse_N", "R_O1-M": "nmse_M", "R_O1-H": "nmse_H"}
    for _, row in S.iterrows():
        d = M[(M.block == row.block) & (M.detach_reset == row.detach_reset) & (M.placement == row.placement)]
        d = d[(d.family == row.family) | (pd.isna(row.family) & d.family.isna())]
        d = d[np.isclose(d.membrane_noise.astype(float), row.membrane_noise)]
        d = d[np.isclose(d.aug_scale.astype(float), row.aug_scale)]
        d = d[d.activation == row.activation]
        d = d[np.isclose(d.hwhm.astype(float), row.hwhm, equal_nan=True) | (np.isinf(d.hwhm) & np.isinf(row.hwhm))]
        if len(d) != row.n_seeds:
            note(f"{p}: summary row {row.block} {row.family} {row.hwhm}: {len(d)} members vs n_seeds {row.n_seeds}")
            continue
        for a, b in q.items():
            worst = max(worst, abs(d[a].mean() - row[f"{b}_mean"]))
    print(f"{p}: max |member mean - e15_width_summary| = {worst:.2e}")
    return worst


def flips(p, M):
    cm = _cond(p, "O1-M")
    newf = cs.RES / p / "e15_flip_new.csv"
    if not newf.exists():
        newf = cs.RES / p / "e15_flip_w16.csv"
        note(f"{p}: e15_flip_new.csv missing -> {newf.name} (w16 members only, partial)")
    fn = pd.read_csv(newf)
    fo = cs.read(p, "e15_flip.csv")
    fo = fo[fo.member.str.startswith("dm_")]
    f = pd.concat([fn, fo])
    f = f[(f.layer.astype(str) == "1") & (f.cond == cm)].drop_duplicates("member").set_index("member")
    keep = M.index[M.block.isin(["w16", "w16d", "dm"])]
    f = f.loc[f.index.intersection(keep), ["flip_pred_sg2", "flip_measured"]]
    return f.join(M[["block", "family", "hwhm", "detach_reset", "seed"]])


def snr_pooled(p):
    z = np.load(cs.RES / p / "e15_flip_neurons.npz")
    cm = _cond(p, "O1-M")
    win, ste = [], []
    for fam in ["atan", "sigmoid", "sre", "triangular", "ste"]:
        for s in range(5):
            k = f"dm_{fam}_s{s}||{cm}|snr"
            if k in z:
                (ste if fam == "ste" else win).append(np.asarray(z[k], float))
    return np.concatenate(win), np.concatenate(ste), len(win), len(ste)


def sel(M, block, family=None, detach=None, **kw):
    d = M[M.block == block]
    if family is not None:
        d = d[d.family == family]
    if detach is not None:
        d = d[d.detach_reset == detach]
    for k, v in kw.items():
        d = d[np.isclose(d[k].astype(float), v)] if isinstance(v, float) else d[d[k] == v]
    return d


def by_hwhm(d, col, hw):
    d = d[d.hwhm.round(4).isin([round(h, 4) for h in hw])]
    return d.groupby("hwhm")[col].agg(["mean", "std", "count"]).sort_index()


def ms(d, col):
    v = d[col].astype(float)
    return float(v.mean()), float(v.std()), int(v.count())


# ----------------------------------------------------------------------------- drawing helpers
def xh(h):
    return np.log2(np.asarray(h, float))


def width_axis(ax, show_labels=True):
    ax.set_xlim(-4.55, X_STE + 0.55)
    ax.xaxis.set_major_locator(FixedLocator([-4, -2, 0, 2, 4, 6, X_STE]))
    ax.xaxis.set_minor_locator(FixedLocator([-3, -1, 1, 3, 5]))
    ax.set_xticklabels(["1/16", "1/4", "1", "4", "16", "64", "STE"] if show_labels else [])
    ax.grid(axis="x", which="major", visible=False)
    ax.axvspan(xh(ORIG_BAND[0]), xh(ORIG_BAND[1]), color=BAND, lw=0, zorder=0)
    tr = ax.get_xaxis_transform()  # axis break between HWHM 64 and the STE slot
    ax.add_patch(Rectangle((X_BREAK - 0.2, -0.03), 0.4, 0.06, transform=tr, facecolor="white", edgecolor="none",
                           clip_on=False, zorder=6))
    for dx in (-0.12, 0.12):
        ax.plot([X_BREAK + dx - 0.1, X_BREAK + dx + 0.1], [-0.035, 0.035], transform=tr, color=cs.AXIS, lw=0.6,
                clip_on=False, zorder=7)


def draw_family(ax, M, fam, col, detach_too=True):
    c, mk, size = cs.C_FAM[fam], cs.MK_FAM[fam], cs.MS_FAM[fam]
    g = by_hwhm(sel(M, "w16", fam, detach=False), col, HW_MAIN)
    x = xh(g.index) + DODGE[fam]
    ax.errorbar(x, g["mean"], yerr=g["std"], color=c, lw=1.0, ls=cs.LS_FAM[fam], marker=mk, ms=size, mfc=c,
                mec="white", mew=0.35, elinewidth=0.5, capsize=0, zorder=4)
    u = by_hwhm(sel(M, "w16", fam, detach=False), col, HW_ULTRA)
    if len(u):
        xu = xh(u.index) + DODGE[fam]
        ax.plot(np.r_[x[-1], xu], np.r_[g["mean"].iloc[-1], u["mean"]], color=c, lw=0.7, ls=cs.LS_ULTRA, zorder=3)
        ax.plot(xu, u["mean"], ls="none", marker=mk, ms=cs.MS_SMALL, color=c, mec=c, mew=0.3, zorder=4)
    if detach_too and fam in DODGE_D:
        dd = by_hwhm(sel(M, "w16d", fam, detach=True), col, HW_MAIN + HW_ULTRA)
        if len(dd):
            ax.errorbar(xh(dd.index) + DODGE_D[fam], dd["mean"], yerr=dd["std"], color=c, lw=0.8, ls=cs.LS_DETACH,
                        marker=mk, ms=size * 0.95, mfc="white", mec=c, mew=0.75, elinewidth=0.5, capsize=0, zorder=5)
    return g


def draw_ste(ax, M, col):
    c = cs.C_FAM["ste"]
    out = {}
    for dx, block, hollow in [(-0.2, "dm", False), (0.2, "w16d", True)]:
        d = sel(M, block, "ste")
        if not len(d):
            continue
        m, s, n = ms(d, col)
        ax.errorbar([X_STE + dx], [m], yerr=[s], color=c, ls="none", marker="^", ms=cs.MS_FAM["ste"],
                    mfc="white" if hollow else c, mec=c if hollow else "white", mew=0.75 if hollow else 0.35,
                    elinewidth=0.5, capsize=0, zorder=5)
        out[block] = (m, s, n)
    return out


def corner(ax, text, loc="ul", **kw):
    x, ha = (0.03, "left") if loc[1] == "l" else (0.97, "right")
    y, va = (0.96, "top") if loc[0] == "u" else (0.04, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=cs.FS_SMALL, color=cs.INK2, zorder=9,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85), **kw)


def nice_ylim(ax, vals, pad=0.08, pad_top=None, log=False):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    pt = pad if pad_top is None else pad_top
    if log:
        lo, hi = np.log10(v.min()), np.log10(v.max())
        r = hi - lo
        ax.set_ylim(10 ** (lo - pad * r), 10 ** (hi + pt * r))
    else:
        lo, hi = v.min(), v.max()
        r = hi - lo
        ax.set_ylim(lo - pad * r, hi + pt * r)


def _spread(M, col, fams=None):
    fams = fams or cs.FAMS
    vals = []
    for blk in ["w16", "w16d", "dm"]:
        d = M[(M.block == blk) & (M.family.isin(fams + ["ste"]))]
        if blk == "dm":
            d = d[d.family == "ste"]
        g = d.groupby(["family", "hwhm", "detach_reset"])[col]
        vals += list(g.mean() + g.std().fillna(0)) + list(g.mean() - g.std().fillna(0))
    return vals


# ----------------------------------------------------------------------------- panels
def ring_best(ax, M, col):
    """Ring around the lowest mean nMSE among the three-seed tri/ATan configurations and the STE members."""
    cands = []
    for fam in ["tri", "atan"]:
        g = by_hwhm(sel(M, "w16", fam, detach=False), col, HW_MAIN)
        cands += [(m, xh(h) + DODGE[fam], f"{fam} HWHM {h:g}") for h, m in g["mean"].items()]
    d = sel(M, "dm", "ste")
    if len(d):
        cands.append((float(d[col].mean()), X_STE - 0.2, "STE"))
    best = min(cands)
    ax.plot([best[1]], [best[0]], ls="none", marker="o", ms=7.5, mfc="none", mec=cs.INK, mew=0.75, zorder=8)
    return best


def trim_ticks(ax, frac=0.16):
    """Drop y ticks within `frac` of the axes edges so that stacked strips do not collide."""
    if ax.get_yscale() != "linear":
        return
    lo, hi = ax.get_ylim()
    t = [v for v in ax.get_yticks() if lo + frac * (hi - lo) <= v <= hi - frac * (hi - lo)]
    if t:
        ax.yaxis.set_major_locator(FixedLocator(t))


def panel_a(axes, M, p):
    rows = [("r", r"$r$", False), ("beta", r"$\beta$", False), ("R_N", "nMSE, N", True)]
    best = None
    for ax, (col, lab, log) in zip(axes, rows):
        for fam in cs.FAMS:
            draw_family(ax, M, fam, col)
        draw_ste(ax, M, col)
        if log:
            ax.set_yscale("log")
            best = ring_best(ax, M, col)
        nice_ylim(ax, _spread(M, col), pad=0.10, log=log)
        if p == "sim":
            ax.set_ylabel(lab, labelpad=2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
    a = axes[2]
    lo, hi = a.get_ylim()
    cand = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    ticks = [c for c in cand if lo <= c <= hi]
    a.yaxis.set_major_locator(FixedLocator(ticks[::2] if len(ticks) > 3 else ticks))
    a.yaxis.set_minor_locator(NullLocator())
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_formatter(NullFormatter())
    return best


def panel_b(ax, M, p, lev):
    col = f"R_{lev}"
    for fam in ["tri", "atan"]:
        draw_family(ax, M, fam, col)
    draw_ste(ax, M, col)
    best = ring_best(ax, M, col)
    nice_ylim(ax, _spread(M, col, ["tri", "atan"]), pad=0.14, pad_top=0.55)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune="both"))
    return best


def panel_c(ax, F, p):
    for blk, det in [("w16", False), ("dm", False), ("w16d", True)]:
        d = F[(F.block == blk) & (F.detach_reset == det)]
        for fam in cs.FAMS + ["ste"]:
            e = d[d.family == fam]
            if not len(e):
                continue
            c, mk = cs.C_FAM[fam], cs.MK_FAM[fam]
            size = (cs.MS_FAM[fam] * 1.1) ** 2
            z = 6 if fam == "ste" else 3
            if det:
                ax.scatter(e.flip_pred_sg2, e.flip_measured, s=size, marker=mk, facecolors="white", edgecolors=c,
                           linewidths=0.7, zorder=z)
            elif blk == "dm":
                ax.scatter(e.flip_pred_sg2, e.flip_measured, s=size, marker=mk, color=c, edgecolors=EDGE_ORIG,
                           linewidths=0.6, zorder=z + 1)
            else:
                ax.scatter(e.flip_pred_sg2, e.flip_measured, s=size, marker=mk, color=c, edgecolors="white",
                           linewidths=0.3, zorder=z)
    x, y = F.flip_pred_sg2, F.flip_measured
    xp, yp = 0.06 * (x.max() - x.min()), 0.06 * (y.max() - y.min())
    xlim = (x.min() - xp, x.max() + xp)
    ylim = (y.min() - yp, y.max() + 0.80 * (y.max() - y.min()))
    lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color=cs.MUTED, lw=0.6, ls=cs.LS_DOT, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3, steps=[1, 2, 5, 10]))
    rho_all = spearmanr(x, y)[0]
    Wn = F[F.family != "ste"]
    rho_win = spearmanr(Wn.flip_pred_sg2, Wn.flip_measured)[0]
    ax.set_xlabel("Predicted flip rate (Lemma 2)")
    return rho_all, rho_win, len(F), len(Wn)


def inset_snr(ax, snr):
    win, ste, nw, ns = snr
    lw_, ls_ = np.log10(win), np.log10(ste)
    bp = ax.boxplot([ls_, lw_], positions=[0, 1], widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                    vert=False, medianprops=dict(color=cs.INK, lw=0.8), whiskerprops=dict(lw=0.6, color=cs.INK2),
                    capprops=dict(lw=0.6, color=cs.INK2), boxprops=dict(lw=0.6, color=cs.INK2))
    bp["boxes"][0].set_facecolor(cs.C_FAM["ste"])
    bp["boxes"][1].set_facecolor(cs.tint(cs.INK2, 0.72))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["STE", "win."])
    ax.set_ylim(-0.6, 1.6)
    lo = np.floor(min(np.percentile(lw_, 5), np.percentile(ls_, 5)))
    hi = np.ceil(max(np.percentile(lw_, 95), np.percentile(ls_, 95)))
    ticks = np.arange(lo, hi + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{10 ** t:g}" for t in ticks])
    ax.set_xlim(lo - 0.08, hi + 0.08)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.text(-0.04, 1.0, r"SNR$_k$", transform=ax.transAxes, ha="right", va="bottom", fontsize=cs.FS,
            color=cs.INK2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=1.5, pad=1)
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    return float(np.median(win)), float(np.median(ste)), nw, ns


def e17_values(M, p):
    narrow = sel(M, "w16", "tri", detach=False, hwhm=0.25).set_index("seed")
    out = {}
    for h in HW_E17:
        for pl in ["L1", "L23", "all"]:
            if pl == "all":
                d = sel(M, "w16", "tri", detach=False, hwhm=float(h))
            else:
                d = sel(M, "w17", "tri", hwhm=float(h), placement=pl)
            d = d.set_index("seed")
            idx = d.index.intersection(narrow.index)
            dr = d.loc[idx, "r"] - narrow.loc[idx, "r"]
            dR = d.loc[idx, "R_O1-M"] - narrow.loc[idx, "R_O1-M"]
            out[(h, pl)] = dict(dr=dr.mean(), dr_sd=dr.std(), dR=dR.mean(), dR_sd=dR.std(), n=len(idx))
    return out


def panel_d_e17(ax_r, ax_R, E):
    groups = [(p, h) for p in PLATS for h in HW_E17]
    off = {"L1": -0.26, "L23": 0.0, "all": 0.26}
    for ax, key, lab in [(ax_r, "dr", r"$\Delta r$"), (ax_R, "dR", r"$\Delta$nMSE, O1-M")]:
        for gi, (p, h) in enumerate(groups):
            for pl in ["L1", "L23", "all"]:
                v = E[p][(h, pl)]
                ax.bar(gi + off[pl], v[key], width=0.24, color=cs.C_PLACE[pl], edgecolor="white", linewidth=0.4,
                       zorder=3)
                ax.errorbar(gi + off[pl], v[key], yerr=v[key + "_sd"], color=cs.INK, lw=0, elinewidth=0.6,
                            capsize=0, zorder=4)
        ax.axhline(0, color=cs.AXIS, lw=0.6, zorder=2)
        ax.set_xlim(-0.55, len(groups) - 0.45)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{h:g}" for _, h in groups] if ax is ax_R else [])
        ax.grid(axis="x", visible=False)
        ax.axvline(1.5, color=cs.GRID, lw=0.6, zorder=1)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.35 * (hi - lo))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        corner(ax, lab, loc="ul")
        tr = ax.get_xaxis_transform()
        if ax is ax_r:
            continue
        for gx, p in [(0.5, "sim"), (2.5, "sarcos")]:
            ax.text(gx, -0.40, cs.PLATFORM_SHORT[p], transform=tr, ha="center", va="top", fontsize=cs.FS,
                    color=cs.INK2)
        if ax is ax_R:
            ax.text(-0.75, -0.16, "HWHM", transform=tr, ha="right", va="top", fontsize=cs.FS, color=cs.INK2)


def iso_risk(ax, py, levels, xlim, ylim):
    b = np.linspace(xlim[0], xlim[1], 400)
    for R in levels:
        n = R - (1 - b) ** 2 * py
        ok = (n > ylim[0]) & (n < ylim[1])
        if ok.sum() < 2:
            continue
        ax.plot(b[ok], n[ok], color=cs.GRID, lw=0.8, zorder=1)


def plane_points(M, S, p):
    """(beta, ||nu||^2) at O1-M; knob curves and references from e15_width_summary when present."""
    strong = cs.FAM_OF_SURR[cs.STRONG[p]]
    base = sel(M, "dm", strong)
    base = base[base.seed <= 2]
    P = {"base": (base.beta.mean(), base.nu2.mean())}

    def bn(row):
        return float(row["beta_mean"]), float(row["nu2_mean"])

    if S is not None:
        P["mem"] = np.array([P["base"]] + [bn(srow(S, "w18mem", membrane_noise=s)) for s in (0.25, 0.5, 1.0, 2.0)])
        P["aug"] = np.array([P["base"]] + [bn(srow(S, "w18aug", aug_scale=a)) for a in (0.25, 0.5, 1.0, 1.4)])
        w = S[(S.block == "w16") & (S.family == "tri") & ~S.detach_reset].sort_values("hwhm")
        P["width"] = w[w.hwhm.isin(HW_MAIN)][["beta_mean", "nu2_mean"]].values
        P["width_u"] = w[w.hwhm.isin(HW_ULTRA)][["beta_mean", "nu2_mean"]].values
        P["mlp"] = {"relu": bn(srow(S, "bn"))}
        for a in ["tanh", "crelu", "bin"]:
            P["mlp"][a] = bn(srow(S, "w18mlp", activation=a))
        P["ste"] = bn(srow(S, "dm", "ste"))
        P["ste_d"] = bn(srow(S, "w16d", "ste"))
    else:
        P["mem"] = np.array([P["base"]] + [tuple(sel(M, "w18mem", membrane_noise=s)[["beta", "nu2"]].mean())
                                           for s in (0.25, 0.5, 1.0, 2.0)])
        P["aug"] = np.array([P["base"]] + [tuple(sel(M, "w18aug", aug_scale=a)[["beta", "nu2"]].mean())
                                           for a in (0.25, 0.5, 1.0, 1.4)])
        w = sel(M, "w16", "tri", detach=False)
        P["width"] = w[w.hwhm.isin(HW_MAIN)].groupby("hwhm")[["beta", "nu2"]].mean().sort_index().values
        P["width_u"] = w[w.hwhm.isin(HW_ULTRA)].groupby("hwhm")[["beta", "nu2"]].mean().sort_index().values
        P["mlp"] = {"relu": tuple(sel(M, "bn")[["beta", "nu2"]].mean())}
        for a in ["tanh", "crelu", "bin"]:
            P["mlp"][a] = tuple(sel(M, "w18mlp", activation=a)[["beta", "nu2"]].mean())
        P["ste"] = tuple(sel(M, "dm", "ste")[["beta", "nu2"]].mean())
        P["ste_d"] = tuple(sel(M, "w16d", "ste")[["beta", "nu2"]].mean())
    P["ba"] = tuple(P["aug"][3])  # augmentation 1x = the B-A recipe (seeds 0-2 identical to ba_*_s0-2)
    win = M[(M.block == "dm") & (M.family.isin(cs.FAMS))]
    P["win_members"] = win[["beta", "nu2"]].values
    P["win"] = tuple(win[["beta", "nu2"]].mean())
    return P


def panel_d_e18(ax, M, S, p):
    P = plane_points(M, S, p)
    py = float(M.Py.iloc[0])
    pts = np.vstack([P["mem"], P["aug"], P["width"], P["width_u"], np.array(list(P["mlp"].values())),
                     P["win_members"], [P["ste"], P["ste_d"], P["ba"]]])
    xr = pts[:, 0].max() - pts[:, 0].min()
    yr = pts[:, 1].max() - pts[:, 1].min()
    xlim = (pts[:, 0].min() - 0.10 * xr, pts[:, 0].max() + 0.06 * xr)
    ylim = (pts[:, 1].min() - 0.08 * yr, pts[:, 1].max() + 0.24 * yr)
    levels = {"sim": [0.35, 0.40, 0.45, 0.50], "sarcos": [0.15, 0.25, 0.35, 0.45]}[p]
    iso_risk(ax, py, levels, xlim, ylim)
    wm = P["win_members"]
    ax.scatter(wm[:, 0], wm[:, 1], s=4, color=cs.C["D-M window"], linewidths=0, alpha=0.6, zorder=2)
    kw = dict(lw=1.0, zorder=4, solid_capstyle="round")
    w = P["width"]
    ax.plot(w[:, 0], w[:, 1], color=cs.C["width"], marker="o", ms=2.2, mew=0, **kw)
    if len(P["width_u"]):
        wu = np.vstack([w[-1], P["width_u"]])
        ax.plot(wu[:, 0], wu[:, 1], color=cs.C["width"], lw=0.7, ls=cs.LS_ULTRA, zorder=4)
        ax.plot(P["width_u"][:, 0], P["width_u"][:, 1], ls="none", marker="o", ms=1.5, color=cs.C["width"],
                zorder=4)
    m = P["mem"]
    ax.plot(m[:, 0], m[:, 1], color=cs.C_E18["mem"], ls=cs.LS_E18["mem"], marker=cs.MK_MEM, ms=4.6,
            mec=cs.C_E18["mem"], mew=0.8, **kw)  # stroke marker (the thin diamond read as a W3 optimum)
    a = P["aug"]
    ax.plot(a[:, 0], a[:, 1], color=cs.C_E18["aug"], ls=cs.LS_E18["aug"], marker=cs.MK_E18["aug"], ms=3.4,
            mfc="white", mec=cs.C_E18["aug"], mew=0.7, **kw)
    ax.plot(*P["win"], ls="none", marker=cs.MK["D-M window"], ms=5.5, color=cs.C["D-M window"], mec="white",
            mew=0.5, zorder=6)
    ax.plot(*P["ste"], ls="none", marker="^", ms=5.5, color=cs.C_FAM["ste"], mec="white", mew=0.5, zorder=6)
    ax.plot(*P["ste_d"], ls="none", marker="^", ms=5.0, mfc="white", mec=cs.C_FAM["ste"], mew=0.8, zorder=6)
    ax.plot(*P["ba"], ls="none", marker=cs.MK["B-A"], ms=5.5, color=cs.C["B-A"], mec="white", mew=0.5, zorder=6)
    for k, (b, n) in P["mlp"].items():
        ax.plot(b, n, ls="none", marker=cs.MK["MLP"] if k == "relu" else cs.MK_E18[k], ms=4.2, color=cs.C_E18["mlp"],
                mec="white", mew=0.3, zorder=6)
    allw = np.vstack([w, P["width_u"]]) if len(P["width_u"]) else w
    w_anchor = tuple(w[-1]) if WIDTH_ANCHOR[p] == "hwhm4" else tuple(allw[np.argmax(allw[:, 1])])
    anchors = {"width": w_anchor, "membrane": tuple(m[-1]),
               "augment": tuple(a[AUG_ANCHOR[p]]), "ba": P["ba"], "ste": P["ste"], "ste_d": P["ste_d"],
               "win": P["win"], **{k: v for k, v in P["mlp"].items()}}
    for k, spec in LAB18[p].items():
        txt = TXT18_P.get(p, {}).get(k, TXT18[k])
        kind, u, v, ha, va = spec
        if kind == "off":
            ax.annotate(txt, anchors[k], xytext=(u, v), textcoords="offset points", ha=ha, va=va,
                        fontsize=cs.FS_SMALL, color=cs.INK2, zorder=9)
        else:
            ax.annotate(txt, anchors[k], xytext=(u, v), textcoords="data", ha=ha, va=va, fontsize=cs.FS_SMALL,
                        color=cs.INK2, zorder=9,
                        arrowprops=dict(arrowstyle="-", color=cs.MUTED, lw=0.5, shrinkA=1.5, shrinkB=2.5))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    return P


def legend_top(fig, W, H, y):
    def fam_h(f):
        return Line2D([], [], color=cs.C_FAM[f], lw=1.0, marker=cs.MK_FAM[f], ms=cs.MS_FAM[f], mec="white", mew=0.35)
    ste = (Line2D([], [], ls="none", color=cs.C_FAM["ste"], marker="^", ms=cs.MS_FAM["ste"], mec="white", mew=0.35),
           Line2D([], [], ls="none", color=cs.C_FAM["ste"], marker="^", ms=cs.MS_FAM["ste"], mfc="white", mew=0.75))
    h = [fam_h(f) for f in cs.FAMS] + [ste,
         Line2D([], [], color=cs.INK2, lw=0.8, ls=cs.LS_DETACH, marker="o", ms=3.0, mfc="white", mew=0.75),
         Line2D([], [], color=cs.INK2, lw=0.7, ls=cs.LS_ULTRA, marker="o", ms=cs.MS_SMALL),
         Line2D([], [], ls="none", marker="o", ms=6.5, mfc="none", mec=cs.INK, mew=0.75)]
    lab = [cs.FAM_LABEL[f] for f in cs.FAMS] + ["STE (orig./detached)", "detached reset",
                                                 "HWHM 8-64 (seed 0)", "lowest mean nMSE"]
    from matplotlib.legend_handler import HandlerTuple
    fig.legend(h, lab, ncol=len(h), frameon=False, handlelength=1.4, columnspacing=0.75, handletextpad=0.3,
               fontsize=cs.FS, loc="center", bbox_to_anchor=(0.5, y / H), borderaxespad=0,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})


def legend_mlp(fig, W, H, x, y):
    h = [Line2D([], [], ls="none", marker=cs.MK["MLP"] if k == "relu" else cs.MK_E18[k], ms=4.2,
                color=cs.C_E18["mlp"], mec="white", mew=0.3) for k in ["relu", "tanh", "crelu", "bin"]]
    fig.legend(h, ["MLP: ReLU", "tanh", "clipped", "binary"], ncol=4, frameon=False, bbox_to_anchor=(x / W, y / H),
               loc="lower right", handlelength=0.8, columnspacing=0.7, handletextpad=0.25, fontsize=cs.FS,
               borderaxespad=0)


def legend_e17(ax):
    h = [Patch(facecolor=cs.C_PLACE[k], edgecolor="none") for k in ["L1", "L23", "all"]]
    ax.legend(h, ["L1", "L2-3", "all"], ncol=3, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.04),
              handlelength=0.8, handleheight=0.7, columnspacing=0.6, handletextpad=0.25, fontsize=cs.FS,
              borderaxespad=0)


# ----------------------------------------------------------------------------- figure
def main():
    cs.apply_style()
    data = {}
    for p in PLATS:
        M = members(p)
        S = summary(p)
        data[p] = dict(M=M, S=S, F=flips(p, M), snr=snr_pooled(p), E17=e17_values(M, p),
                       check=check_summary(M, S, p))

    W, H = cs.DOUBLE_COL, 3.0
    fig = plt.figure(figsize=(W, H))

    def axr(x0, y0, w, h):
        return fig.add_axes([x0 / W, y0 / H, w / W, h / H])

    # geometry (inches)
    colx_a, colx_b, wab = [0.42, 2.03], [4.03, 5.66], 1.30
    top, hs, gs = 2.72, 0.325, 0.04            # three strips per column in (a) and (b)
    bot, h2 = 0.27, 0.96                        # bottom row (c, d)
    colx_c, wc = [0.42, 2.07], 1.25
    x17, w17, h17, g17 = 3.98, 1.17, 0.37, 0.12
    x18, w18 = 5.62, 1.43

    legend_top(fig, W, H, 2.925)

    def strip(x, i):
        return axr(x, top - (i + 1) * hs - i * gs, wab, hs)

    # (a)
    best = {}
    for j, p in enumerate(PLATS):
        A = [strip(colx_a[j], i) for i in range(3)]
        best[(p, "N")] = panel_a(A, data[p]["M"], p)
        for i, ax in enumerate(A):
            width_axis(ax, show_labels=(i == 2))
            trim_ticks(ax)
        A[0].set_title(cs.PLATFORM_NAME[p], fontsize=cs.FS, color=cs.INK, pad=2)
        A[2].set_xlabel("HWHM", labelpad=0.5)
    cs.fig_label(fig, "a", 0.0 / W, (top + 0.15) / H)

    # (b)
    for j, p in enumerate(PLATS):
        for i, lev in enumerate(["O1-L", "O1-M", "O1-H"]):
            ax = strip(colx_b[j], i)
            if _cond(p, lev) is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "no O1-L condition on SARCOS", transform=ax.transAxes, ha="center",
                        va="center", fontsize=cs.FS, color=cs.MUTED)
                continue
            best[(p, lev)] = panel_b(ax, data[p]["M"], p, lev)
            width_axis(ax, show_labels=(i == 2))
            trim_ticks(ax)
            corner(ax, rf"{lev}, $\rho$ = {RHO[p][lev]:g}", loc="ul")
            if i == 0:
                ax.set_title(cs.PLATFORM_NAME[p], fontsize=cs.FS, color=cs.INK, pad=2)
            if i == 2:
                ax.set_xlabel("HWHM", labelpad=0.5)
            if j == 0 and i == 1:
                ax.set_ylabel("Test nMSE", labelpad=2)
    cs.fig_label(fig, "b", (colx_b[0] - 0.48) / W, (top + 0.15) / H)

    # (c)
    rhos, meds = {}, {}
    for j, p in enumerate(PLATS):
        ax = axr(colx_c[j], bot, wc, h2)
        rhos[p] = panel_c(ax, data[p]["F"], p)
        ax.set_title(f"{cs.PLATFORM_SHORT[p]}: " + rf"$\rho_s$ = {rhos[p][0]:.2f} all, "
                     f"{rhos[p][1]:.2f} win.", fontsize=cs.FS, color=cs.INK, pad=2)
        ax.set_xlabel("Predicted flip rate (Lemma 2)", labelpad=0.5)
        if j == 0:
            ax.set_ylabel("Measured flip rate", labelpad=2)
        l, b, w, h = INSET_C[p]
        ins = axr(colx_c[j] + l * wc, bot + b * h2, w * wc, h * h2)
        meds[p] = inset_snr(ins, data[p]["snr"])
    cs.fig_label(fig, "c", 0.0 / W, (bot + h2 + 0.19) / H)

    # (d) left: E17 (two stacked strips sharing the x categories)
    E = {p: data[p]["E17"] for p in PLATS}
    ax_r = axr(x17, bot + h2 - h17, w17, h17)
    ax_R = axr(x17, bot + h2 - 2 * h17 - g17 + 0.07, w17, h17)
    panel_d_e17(ax_r, ax_R, E)
    legend_e17(ax_r)
    # (d) right: E18, simulated arm
    P = {}
    p = "sim"
    ax = axr(x18, bot, w18, h2)
    P[p] = panel_d_e18(ax, data[p]["M"], data[p]["S"], p)
    corner(ax, "E18, arm, O1-M", loc="ul")
    legend_mlp(fig, W, H, x18 + w18 + 0.05, bot + h2 + 0.06)
    ax.set_xlabel(r"Signal gain $\beta$", labelpad=0.5)
    ax.set_ylabel(r"$\|\nu\|^2$", labelpad=1)
    cs.fig_label(fig, "d", (x17 - 0.42) / W, (bot + h2 + 0.19) / H)

    cs.savefig(fig, "fig3")
    report(data, best, rhos, meds, E, P)
    return data, best, rhos, meds, E, P


def report(data, best, rhos, meds, E, P):
    print("\n--- values drawn ---")
    for p in PLATS:
        M = data[p]["M"]
        print(f"[{p}] r_src = {sorted(M.r_src.unique())}; summary check {data[p]['check']:.1e}")
        for lev in LEVELS:
            if (p, lev) in best:
                print(f"  (b) {lev}: lowest mean nMSE {best[(p, lev)][0]:.4f} at {best[(p, lev)][2]}")
        ra, rw, n, nw = rhos[p]
        print(f"  (c) Spearman all {ra:.3f} (n={n}), windows {rw:.3f} (n={nw}); "
              f"median SNR window {meds[p][0]:.2f} (n={meds[p][2]} members), STE {meds[p][1]:.2f} (n={meds[p][3]})")
        for (h, pl), v in E[p].items():
            print(f"  (d) E17 HWHM {h:g} {pl:>3}: dr {v['dr']:+.4f}+-{v['dr_sd']:.4f}  "
                  f"dR(O1-M) {v['dR']:+.4f}+-{v['dR_sd']:.4f}  n={v['n']}")
        if p not in P:
            continue
        pp = P[p]
        print(f"  (d) E18 base {np.round(pp['base'], 3)}, mem end {np.round(pp['mem'][-1], 3)}, "
              f"aug end {np.round(pp['aug'][-1], 3)}, B-A {np.round(pp['ba'], 3)}, STE {np.round(pp['ste'], 3)}")
    for n_ in NOTES:
        print("note:", n_)


if __name__ == "__main__":
    main()
