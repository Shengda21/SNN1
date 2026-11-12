"""Shared style, paths, colours and helpers for the main-text figures and tables.

All figure scripts (fig2_plane.py, fig4_ladder.py, fig5_path.py) and tables_main.py import
this module. Data are read from RES (the local results tree); figures are written as vector
PDF to FIG_OUT and tables as LaTeX to TAB_OUT. Set the environment variable SNN29_PNG to a
directory to also write a 300-dpi PNG preview of every figure.

Colour roles (fixed across Figs. 2, 4, 5):
  D-S blue, D-M orange, STE aqua, B-A violet       (these four pass the all-pairs CVD check)
  MLP (B-N) near-black ink, null neutral grey
  B-L green, B-F magenta, D-T yellow                (line charts only, adjacent-order checked)
Every series also carries a distinct marker so that the figures remain readable in greyscale.
"""
import os
import re
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- paths
ROOT = Path(os.environ.get("SNN29_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results"
FIG_OUT = ROOT / "build" / "fig"
TAB_OUT = ROOT / "build" / "tables"
PNG_OUT = os.environ.get("SNN29_PNG")  # optional preview directory

PLATFORMS = ["sarcos", "sim"]
PLATFORM_NAME = {"sarcos": "SARCOS", "sim": "Simulated arm"}
PLATFORM_SHORT = {"sarcos": "SARCOS", "sim": "Arm"}

# condition labels of E21-E26 -> condition labels of the E15 member table
E15_COND = {
    "sim": {"N": "N", "O1-L": "O1:0.35", "O1-M": "O1:0.7", "O1-H": "O1:1.0", "O2B": "O2B:0.7", "P1": "P1"},
    "sarcos": {"N": "N", "O1-M": "O1:0.5", "O1-H": "O1:0.7", "O1qqd": "O1qqd:0.5"},
}
STRONG = {"sim": "sigmoid", "sarcos": "triangular"}  # surrogate of B-F / D-S / D-T (e2x_common.STRONG)
DM_SURR = ["atan", "sigmoid", "ste", "triangular", "sre"]  # order of D-M members in the E22 weight vectors

# ----------------------------------------------------------------------------- figure geometry
DOUBLE_COL = 7.16  # in
SINGLE_COL = 3.5   # in

# ----------------------------------------------------------------------------- colours
INK = "#1a1a1a"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#9a9890"

C = {
    "D-S": "#2a78d6",   # blue
    "D-M": "#eb6834",   # orange
    "STE": "#1baf7a",   # aqua
    "B-A": "#4a3aa7",   # violet
    "MLP": "#262626",   # near-black ink (B-N)
    "null": "#a8a6a0",  # neutral grey
    "B-L": "#008300",   # green
    "B-F": "#e87ba4",   # magenta
    "D-T": "#eda100",   # yellow
    "cap4": "#7a7872",  # grey (E14 small-capacity members)
    "width": "#55534e",  # grey (E16 width sweep)
}
C["B-N"] = C["MLP"]
C["D-M window"] = C["D-M"]

# markers for member families / methods (filled) and for weight spaces (hollow); D-M window members use a
# filled pentagon so that they cannot be mistaken for the hollow W1 square of Fig. 2
MK = {"D-S": "o", "D-M": "s", "D-M window": "p", "STE": "^", "B-A": "h", "MLP": "X", "B-N": "X",
      "B-L": "v", "B-F": "P", "D-T": "D", "cap4": "x", "null": "o"}
MK_SPACE = {"W0": "o", "W1": "s", "W2": "v", "W3": "D", "W4": "P", "W5": "*"}

METHOD_LABEL = {"B-F": "B-F", "B-L": "B-L", "B-A": "B-A", "B-N": "B-N (MLP)", "D-S": "D-S", "D-T": "D-T",
                "D-M": "D-M"}
# legend order of the seven methods (adjacent pairs pass the CVD and normal-vision checks)
METHOD_ORDER = ["B-F", "B-L", "B-A", "B-N", "D-T", "D-S", "D-M"]


def tint(hex_color, frac):
    """Mix a colour with white; frac = 0 keeps it, 1 gives white."""
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    return mpl.colors.to_hex(rgb + (1.0 - rgb) * frac)


def shade(hex_color, frac):
    """Mix a colour with black; frac = 0 keeps it, 1 gives black."""
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    return mpl.colors.to_hex(rgb * (1.0 - frac))


# ----------------------------------------------------------------------------- rcParams
FS = 7.0        # base font size (pt)
FS_SMALL = 7.0  # direct labels inside panels (IEEE: no text below 7 pt)
FS_PANEL = 8.0  # panel letters


def apply_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS,
        "axes.titlesize": FS,
        "axes.labelsize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "it",
        "axes.linewidth": 0.5,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "axes.labelpad": 2,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "patch.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.3,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def panel_label(ax, letter, x=-0.02, y=1.0, ha="right", **kw):
    """Bold panel letter '(a)' at the upper-left outside corner of an axes."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold",
            ha=ha, va="bottom", color=INK, **kw)


def fig_label(fig, letter, x, y, **kw):
    fig.text(x, y, f"({letter})", fontsize=FS_PANEL, fontweight="bold", ha="left", va="top", color=INK, **kw)


def savefig(fig, stem):
    """Write FIG_OUT/<stem>.pdf (vector) and, if SNN29_PNG is set, a PNG preview."""
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    pdf = FIG_OUT / f"{stem}.pdf"
    fig.savefig(pdf, format="pdf")
    out = [str(pdf)]
    if PNG_OUT:
        Path(PNG_OUT).mkdir(parents=True, exist_ok=True)
        png = Path(PNG_OUT) / f"{stem}.png"
        fig.savefig(png, format="png", dpi=300)
        out.append(str(png))
    print("wrote", *out)


# ----------------------------------------------------------------------------- data access
def read(platform, name, required=()):
    """Read RES/<platform>/<name> and check that the required columns exist."""
    path = RES / platform / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing columns {missing}; available: {list(df.columns)}")
    return df


def parse_w16(member):
    """'w16_<fam>_h<HWHM>_s<seed>' -> (fam, hwhm, seed)."""
    m = re.match(r"w16_([a-z]+)_h([0-9.]+)_s(\d+)", member)
    if not m:
        return None, np.nan, None
    return m.group(1), float(m.group(2)), int(m.group(3))


def fmt(x, nd):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}f}"


def repel(values, gap, iters=200):
    """Positions for direct labels: same order as the sorted input, at least `gap` apart, and as
    close as possible to the original values (simple iterative relaxation)."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v)
    p = v[order].copy()
    for _ in range(iters):
        moved = False
        for i in range(1, len(p)):
            d = p[i] - p[i - 1]
            if d < gap - 1e-12:
                push = (gap - d) / 2
                p[i - 1] -= push
                p[i] += push
                moved = True
        if not moved:
            break
    out = np.empty_like(p)
    out[order] = p
    return list(out)


# ----------------------------------------------------------------------------- E18 knobs (Fig. 2, O1-M)
# Appended entries; the colours and markers above are unchanged. Input augmentation shares the
# violet of B-A (aug scale 1 is the B-A recipe) but uses hollow hexagons; membrane noise is a neutral
# grey dashed line; the bounded-activation MLPs keep the MLP ink with their own markers.
C_E18 = {"aug": C["B-A"], "mem": INK2, "mlp": C["MLP"]}
MK_E18 = {"aug": "h", "mem": "d", "tanh": "<", "crelu": ">", "bin": "P"}
LS_E18 = {"aug": "-", "mem": (0, (2.5, 1.2))}
LABEL_E18 = {"aug": "Input augmentation", "mem": "Membrane noise", "tanh": "MLP, tanh",
             "crelu": "MLP, clipped ReLU", "bin": "MLP, binary + STE"}


# ----------------------------------------------------------------------------- task-layer candidates (Fig. 6)
# Candidates of E27-E29, keyed by their ids in results/sim/e27_*.csv, e28_*.csv, e29_*.csv. A derived
# candidate keeps the hue of its base dictionary or network and is told apart by lightness, line style and
# marker; the markers echo the weight-space markers of Fig. 2 (null ~ W1 hollow square, per-joint optimum
# of E21 ~ W3 hollow diamond, input estimator ~ W5 star). Same-hue pairs (D-S / D-S + null, MLP / MLP +
# estimator) are below the normal-vision colour floor on purpose and always carry a distinct line style,
# marker and legend entry. Keys: color, ls (line style), marker, filled (marker face), label (figure text).
LS_SOLID = "-"
LS_DASH = (0, (3.2, 1.6))
LS_DOT = (0, (1.0, 1.3))
LS_DASHDOT = (0, (4.0, 1.4, 1.0, 1.4))

TASK = {
    "D-S:W-Q5":      {"color": C["D-S"], "ls": LS_SOLID, "marker": "o", "filled": True, "label": "D-S"},
    "D-M:W-Q5":      {"color": C["D-M"], "ls": LS_SOLID, "marker": "s", "filled": True, "label": "D-M"},
    "D-M:W-S":       {"color": tint(C["D-M"], 0.35), "ls": LS_DASH, "marker": "p", "filled": True,
                      "label": "D-M, single member"},
    "D-S+null:W-Q5": {"color": tint(C["D-S"], 0.35), "ls": LS_DASH, "marker": "s", "filled": False,
                      "label": "D-S + null"},
    "D-S|est:W-Q5":  {"color": C["D-S"], "ls": LS_DOT, "marker": "*", "filled": True, "label": "D-S + estimator"},
    "D-S:e21-W3":    {"color": tint(C["D-S"], 0.35), "ls": LS_DASHDOT, "marker": "D", "filled": False,
                      "label": "D-S, E21 W3 optimum"},
    "D-M:e21-W3":    {"color": tint(C["D-M"], 0.35), "ls": LS_DASHDOT, "marker": "D", "filled": False,
                      "label": "D-M, E21 W3 optimum"},
    "B-N":           {"color": C["MLP"], "ls": LS_SOLID, "marker": "X", "filled": True, "label": "MLP"},
    "B-N|est":       {"color": tint(C["MLP"], 0.40), "ls": LS_DOT, "marker": "*", "filled": False,
                      "label": "MLP + estimator"},
    "B-A":           {"color": C["B-A"], "ls": LS_SOLID, "marker": "h", "filled": True, "label": "B-A"},
    "ID-exact":      {"color": INK2, "ls": LS_SOLID, "marker": "d", "filled": True, "label": "Exact ID"},
}
TASK_OTHER = {"color": MUTED, "ls": LS_SOLID, "marker": "o", "filled": False, "label": "other E13 models"}


def task_style(cand):
    """Style entry of an E27-E29 candidate id (suffixes such as '@true-fb' are ignored)."""
    return TASK.get(cand.split("@")[0], TASK_OTHER)


# ----------------------------------------------------------------------------- surrogate families (Fig. 3)
# Appended for fig3_width.py; entries above are unchanged. The four peak-normalized surrogate families of
# E16 (tri, sig, atan, sre) and the straight-through estimator keep one colour and one filled marker in
# every panel and are drawn with solid lines; a detached-reset variant (E16 block w16d) keeps colour and
# marker but is drawn hollow with LS_DETACH; the seed-0 extension to HWHM 8-64 uses LS_ULTRA with small
# markers. Family colours plus the STE aqua pass the all-pairs normal-vision floor (worst 15.1, sig/atan)
# with sig/atan in the 6-8 CVD warn band, so the markers are the mandatory second channel (also greyscale).
# The E18 knobs of Fig. 3(d) reuse C_E18 / MK_E18 / LS_E18 (Fig. 2) and C["width"].
FAMS = ["tri", "sig", "atan", "sre"]
FAM_LABEL = {"tri": "Triangular", "sig": "Sigmoid", "atan": "ATan", "sre": "Escape-rate", "ste": "STE"}
FAM_OF_SURR = {"triangular": "tri", "sigmoid": "sig", "atan": "atan", "sre": "sre", "ste": "ste",
               "tri": "tri", "sig": "sig", "triangular_pn": "tri", "sigmoid_pn": "sig", "atan_pn": "atan",
               "sre_pn": "sre", "none": "mlp", "mlp": "mlp"}
C_FAM = {"tri": "#2a78d6", "sig": "#e34948", "atan": "#c98500", "sre": "#008300", "ste": C["STE"]}
MK_FAM = {"tri": "v", "sig": "o", "atan": "D", "sre": "s", "ste": "^"}
MS_FAM = {"tri": 3.6, "sig": 3.2, "atan": 2.7, "sre": 2.9, "ste": 3.8}  # roughly equal marker areas
MS_SMALL = 2.0
LS_FAM = {"tri": "-", "sig": "-", "atan": "-", "sre": "-"}
LS_DETACH = (0, (3.2, 1.6))
LS_ULTRA = (0, (1.0, 1.3))
# E17: number of widened layers (1, 2, 3) as an ordinal ramp of the triangular blue (validated --ordinal)
C_PLACE = {"L1": "#86b6ef", "L23": "#2a78d6", "all": "#104281"}
PLACE_LABEL = {"L1": "layer 1 only", "L23": "layers 2-3 only", "all": "all layers"}


# ----------------------------------------------------------------------------- membrane-noise marker (Figs. 2, 3)
# Appended: the membrane-noise sweep of E18 used the thin diamond MK_E18["mem"] = "d", which reads as the
# W3 optimum (hollow "D") of Fig. 2. The sweep now uses a tri-down stroke marker, which no weight-space
# marker (MK_SPACE), member-family marker (MK) or E18 marker (MK_E18) uses. Drawn in C_E18["mem"] with
# mec = colour (a stroke marker has no face).
MK_MEM = "1"
