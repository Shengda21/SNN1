"""Supplementary tables for S5 (position step: member diagnostics, surrogate width, reset factor,
first-layer geometry and flips, stationarity, layer placement, noise injection).

Reads   results/{sim,sarcos}/e15_{members,flip,weights,direction,stationarity}.csv   (locked and
        post-hoc members of the original protocol)
        results/{sim,sarcos}/e15_{members,flip,weights,direction,stationarity}_new.csv  (the 190 E16-E18
        members per platform, same E15 pipeline)
        results/{sim,sarcos}/e16_members.csv  (best development error and epochs of every new member)
        results/{sim,sarcos}/e21_ladder{,_w16,_e18}.csv (W0/W1 points), e22_bestwidth_w16.csv (best width)
        The directory of the *_new files can be overridden with S5_NEW_DIR=<dir> (expects <dir>/<platform>/).
Writes  build/tables/s5n_{members,width,ctrl,bestwidth,reset,detach,geo,flip,rank,station,e17,e18,
        matched}.tex (labels tab:s5members, tab:s5width, tab:s5ctrl, tab:s5best, tab:s5reset, tab:s5detach,
        tab:s5geo, tab:s5flip, tab:s5rank, tab:s5station, tab:s5e17, tab:s5e18, tab:s5matched = tab:s5kappa)
        and prints the numbers quoted in the text of S5 (also written as JSON to $S5_FACTS if that is set).

python code/figures/supp_tables_s5.py
"""
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as tdist

ROOT = Path(os.environ.get("SNN29_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results"
OUT = ROOT / "build" / "tables"
NEW_DIR = os.environ.get("S5_NEW_DIR")
PLATS = ("sim", "sarcos")
PNAME = {"sim": "Simulated arm", "sarcos": "SARCOS"}
LV = {"sim": {"N": "N", "L": "O1:0.35", "M": "O1:0.7", "H": "O1:1.0"},
      "sarcos": {"N": "N", "M": "O1:0.5", "H": "O1:0.7"}}
RHO = {"sim": {"L": 0.35, "M": 0.7, "H": 1.0}, "sarcos": {"M": 0.5, "H": 0.7}}
LOCKED = ["D-M window", "D-M STE", "D-S", "D-T", "B-L", "B-A", "MLP"]
LOCKED_LABEL = {"D-M window": "D-M, window members", "D-M STE": "D-M, STE member", "D-S": "D-S members",
                "D-T": "D-T members", "B-L": "B-L", "B-A": "B-A", "MLP": "B-N (MLP)"}
CLEAN = {"dm", "ds", "dt", "bl", "uf", "cap4", "cap8", "cap16", "width"}  # as in code/e15_summary.py
FAMS = ["tri", "sig", "atan", "sre"]
FAMNAME = {"tri": "Triangular", "sig": "Sigmoid", "atan": "ATan", "sre": "Escape-rate", "ste": "STE"}
OLD_FAM = {"triangular": "tri", "sigmoid": "sig", "atan": "atan", "sre": "sre", "ste": "ste", "none": "mlp"}
GRID = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
FACTS = {}


# =========================================================================================
# formatting helpers
# =========================================================================================
def num(x, d=3):
    try:
        if x is None or not np.isfinite(float(x)):
            return "--"
    except (TypeError, ValueError):
        return "--"
    s = f"{float(x):.{d}f}"
    if s.startswith("-"):
        if float(s) == 0.0:
            return s[1:]
        return "$-$" + s[1:]
    return s


def sgn(x, d=3):
    """Signed number for differences."""
    s = num(x, d)
    if s == "--" or s.startswith("$-$"):
        return s
    if float(s) == 0.0:
        return s
    return "+" + s


def hw(h):
    if not np.isfinite(h):
        return r"$\infty$"
    return f"{h:g}"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.tex", "w", encoding="utf-8") as fh:
        fh.write(body)
    print("wrote", OUT / f"{name}.tex")


# Captions of the S5 tables (kept short; definitions are in the text of S5 and in Section III of the main paper)
CAPS = {
    "tab:s5members": r"Members of the locked protocol in the plane (E15, test role, mean over the $n$ members). $\mathcal S$: "
                     r"scale cost, $R=R^\circ+\mathcal S$; $P_n$, $P_b$: propagated and nominal terms (S7.3); $\kappa$: "
                     r"slope of $\log(R(\rho)-R(0))$ on $\log\rho$ over the frozen O1 levels.",
    "tab:s5width": r"Surrogate width (E16), peak-normalized families; mean over three seeds ($^\dagger$: seed 0). Dev, N: "
                   r"development and nominal test nMSE $\times10^2$; O1-M, O1-H: test nMSE; $r$, $\beta$, $\|\nu\|^2$ at "
                   r"O1-M; ep: selected epoch. STE: the five straight-through members of D-M, and three seeds with the "
                   r"reset detached.",
    "tab:s5ctrl": r"Width against amplitude (E16 control, seed 0): original parameterization (peak $g(0)$) minus "
                  r"peak-normalized member at equal HWHM. N $\times10^2$; $r$, $\beta$ at O1-M; ep: selected epoch, "
                  r"original/peak-normalized.",
    "tab:s5best": r"Width of lowest seed-mean test nMSE against $\rho$ on the paired-noise path of E22 (step 0.05): best "
                  r"HWHM at $\rho=0$, then each new best with the level from which it is best. Candidates: the grid of "
                  r"the row and the straight-through member (STE) with the same reset. Seed 0 with HWHM 8--64 gives the same "
                  r"switch to STE (arm: triangular 0.5, then 0.25 from 0.50); no ultra-wide member is best.",
    "tab:s5reset": r"Layer thresholds and the reset factor (E15). $f_3$, $f_1$: fraction of thresholds of layer 3 and 1 "
                   r"within 0.02 of their initial value one; $\rho_s$: rank correlation across the $n$ members of the row "
                   r"between $f_3$ and the selected epoch (ep) or the nominal nMSE $R_{\mathrm N}$.",
    "tab:s5detach": r"Detached minus original reset, paired by family, HWHM, and seed (three seeds; $^\dagger$: seed 0). "
                    r"ep: selected epoch; N: nominal nMSE $\times10^2$; $r$, $\beta$ at O1-M; $f_3$ as in "
                    r"Table~\ref{tab:s5reset}.",
    "tab:s5geo": r"First-layer geometry and direction-resolved response (E15). srk: stable rank; $\delta_\ell$: relative "
                 r"change from the initialization; $\bar\vartheta_1$: mean first-layer threshold. Response to the O1-M "
                 r"energy on the five leading (top) or trailing (bottom) principal directions of the training inputs; "
                 r"ratio: bottom/top.",
    "tab:s5flip": r"Unit SNR (quantiles over first-layer units) and spike flips (E15, 2{,}000 inputs). Predicted: "
                  r"Lemma~2(ii) and (iii) for layer 1, first step. Measured: fraction of (neuron, step) pairs whose "
                  r"spike changes.",
    "tab:s5rank": r"Rank correlation across members of predicted and measured flip rates (criterion of H5: $\ge0.8$). "
                  r"(ii), (iii): Lemma~2 against layer-1 flips over all steps; exact $t{=}1$: Lemma~2(iii) averaged over the "
                  r"inputs against first-step flips; $\to$all: (iii) against flips of all layers.",
    "tab:s5station": r"Stationarity residuals of Lemma~1 (E15). $G$: $g$-weighted residual of \eqref{eq:s6-weighted}; $K$: "
                     r"the same without $g$; $\cos$: cosine of the two matrices; floor: block standard error of $G$; ep: "
                     r"selected epoch; --: layer 3 with detached reset.",
    "tab:s5e17": r"Layer placement of the wide triangular surrogate (E17, three seeds; narrow HWHM 0.25). Share: change of "
                 r"$r$ as a fraction of that of the all-wide E16 member (Table~\ref{tab:s5width}; H3: $\ge0.7$ for layer 1, "
                 r"$\le0.3$ for layers 2--3); n.d.: all-wide change within two standard errors.",
    "tab:s5e18": r"Noise injection and bounded activations (E18, three seeds). Locked surrogate: sigmoid (arm) or "
                 r"triangular (SARCOS) at its locked width, D-M seeds 0--2. W1: member with the null predictor, weights fitted on "
                 r"$\Dcal$ (E21); $^\ddagger$: five seeds. Settings: arm/SARCOS.",
    "tab:s5matched": r"Matched-gain comparison of the E18 knobs and growth exponents of bounded activations. (a) Each knob "
                     r"interpolated linearly in $\beta$ through its level means and the locked surrogate (--: gain not "
                     r"reached); Dev.: range of the deviation of the level means from the scalar rescaling "
                     r"$\|\nu_0\|^2(\beta/\beta_0)^2$. (b) $\kappa$ of MLPs with bounded hidden activations (E18; ReLU: "
                     r"B-N) and of the spiking members of D-M.",
    "tab:s5kappa": r"Noise-scaling exponent $\kappa$ (seed standard deviation) of MLPs with bounded hidden activations "
                   r"(E18) and of the spiking members of D-M.",
}


def table(label, caption, colspec, header, rows, size=r"\scriptsize", sep="3pt", place="!htbp"):
    caption = CAPS.get(label, caption)
    lines = [rf"\begin{{table}}[{place}]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
             rf"\setlength{{\tabcolsep}}{{{sep}}}", r"\renewcommand{\arraystretch}{0.9}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += header + [r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def rho_s(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(spearmanr(a[ok], b[ok])[0])


def tci(d):
    """Mean and two-sided 95% Student-t interval of paired differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else np.nan), np.nan, np.nan
    m, s = d.mean(), d.std(ddof=1)
    h = tdist.ppf(0.975, n - 1) * s / math.sqrt(n)
    return float(m), float(m - h), float(m + h)


# =========================================================================================
# data
# =========================================================================================
def _read(path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 5:
        return pd.read_csv(path)
    return None


def _new(plat, name):
    base = Path(NEW_DIR) / plat if NEW_DIR else RES / plat
    df = _read(base / f"{name}_new.csv")
    if df is None:
        raise SystemExit(f"missing {base / (name + '_new.csv')}: run code/e15_run.sh <platform> ../runs/<platform> "
                         f"'*' _new on the server and download the *_new files first")
    return df


def _seed_from_name(m):
    s = re.search(r"_s(\d+)$", m)
    return int(s.group(1)) if s else np.nan


def member_frame(m, plat, origin):
    """One row per member: configuration and the plane coordinates at N, O1-L/M/H (test role)."""
    t = m[m.role == "test"]
    base = t[t.cond == "N"].drop_duplicates("member").set_index("member")
    out = pd.DataFrame(index=base.index)
    out["origin"] = origin
    out["prefix"] = base["prefix"]
    out["model_kind"] = base["model_kind"]
    out["best_epoch"] = base["best_epoch"]
    out["r"] = base["r_O1M"]
    out["kappa"] = base["kappa"]
    out["Py"] = base["Py"]
    out["seed"] = [_seed_from_name(i) for i in out.index]
    if origin == "old":
        out["group"] = base["group"]
        out["block"] = base["prefix"]
        out["family"] = base["surrogate"].map(OLD_FAM)
        out["hwhm"] = np.where(base["surrogate"] == "ste", np.inf, base["hwhm"])
        out["detach"] = False
        out["placement"] = "all"
        out["membrane_noise"] = 0.0
        out["aug_scale"] = 0.0
        out["activation"] = np.where(base["model_kind"] == "mlp", "relu", "spike")
    else:
        out["group"] = base["block"]
        out["block"] = base["block"]
        out["family"] = base["family"]
        out["hwhm"] = base["hwhm"].astype(float)
        out["detach"] = base["detach_reset"].astype(str).str.lower().eq("true")
        out["placement"] = base["placement"]
        out["membrane_noise"] = base["membrane_noise"]
        out["aug_scale"] = base["aug_scale"]
        out["activation"] = base["activation"]
    for key, c in LV[plat].items():
        tc = t[t.cond == c].drop_duplicates("member").set_index("member")
        for q, col in [("R", "R"), ("beta", "beta"), ("nu2", "nu2"), ("slope", "slope"), ("Rang", "R_angle"),
                       ("S", "scale_cost"), ("rc", "r_cond"), ("Pn", "P_n"), ("Pb", "P_b")]:
            out[f"{q}_{key}"] = tc[col]
    return out


def load(plat):
    old = {k: pd.read_csv(RES / plat / f"e15_{k}.csv") for k in
           ["members", "flip", "weights", "direction", "stationarity"]}
    new = {k: _new(plat, f"e15_{k}") for k in ["members", "flip", "weights", "direction", "stationarity"]}
    MT = pd.concat([member_frame(old["members"], plat, "old"), member_frame(new["members"], plat, "new")])
    MT["platform"] = plat
    e16 = pd.read_csv(RES / plat / "e16_members.csv").set_index("member_id")
    MT["best_dev"] = e16["best_dev"].reindex(MT.index)
    MT["epochs_run"] = e16["epochs_run"].reindex(MT.index)
    MT["peak"] = e16["peak"].reindex(MT.index)
    # weights
    W = pd.concat([old["weights"], new["weights"]], ignore_index=True)
    W = W[W.group != "traj-init-check"].drop_duplicates(["member", "layer"])
    for i in (1, 2, 3, 4, 5):
        wl = W[W.layer == f"fc{i}"].set_index("member")
        if i <= 3:
            MT[f"W{i}_fro"] = wl["fro"].reindex(MT.index)
            MT[f"W{i}_srank"] = wl["stable_rank"].reindex(MT.index)
            MT[f"thr{i}_near1"] = wl["thr_frac_near1"].reindex(MT.index)
            MT[f"thr{i}_abs"] = wl["thr_mean_abs_1m"].reindex(MT.index)
            MT[f"thr{i}_mean"] = wl["thr_mean"].reindex(MT.index)
        MT[f"W{i}_rel"] = wl["rel_change"].reindex(MT.index)
    # flips
    F = pd.concat([old["flip"], new["flip"]], ignore_index=True).drop_duplicates(["member", "cond", "layer"])
    F["layer"] = F["layer"].astype(str)
    for key in ("L", "M", "H"):
        if key not in LV[plat]:
            continue
        c = LV[plat][key]
        for lay in ("1", "2", "3", "all"):
            fl = F[(F.cond == c) & (F.layer == lay)].set_index("member")
            MT[f"flip{lay}_{key}"] = fl["flip_measured"].reindex(MT.index)
            MT[f"flip{lay}s1_{key}"] = fl["flip_measured_step1"].reindex(MT.index)
            if lay == "1":
                for q in ["flip_pred_sg2", "flip_pred_biv", "flip_pred_emp1", "snr_q05", "snr_q25", "snr_q50",
                          "snr_q75", "snr_q95", "sigma_in_q50", "abs_h_median"]:
                    MT[f"{q}_{key}"] = fl[q].reindex(MT.index)
    # direction-resolved response
    D = pd.concat([old["direction"], new["direction"]], ignore_index=True)
    for v in ("full21", "qqd14"):
        for sub in ("top5", "bottom5", "iso"):
            dd = D[(D.variant == v) & (D.subspace == sub)].drop_duplicates("member").set_index("member")
            MT[f"dir_{v}_{sub}"] = dd["r"].reindex(MT.index)
    # stationarity
    S = pd.concat([old["stationarity"], new["stationarity"]], ignore_index=True)
    S["layer"] = S["layer"].astype(str)
    for lay in ("1", "2", "3"):
        sl = S[S.layer == lay].drop_duplicates("member").set_index("member")
        for q in ["ratio_G", "ratio_K", "floor_G", "floor_K", "cos_GK", "frac_in_window", "grad_fro"]:
            MT[f"st{lay}_{q}"] = sl[q].reindex(MT.index)
    return MT


DATA = {}


def mt(plat):
    if plat not in DATA:
        DATA[plat] = load(plat)
    return DATA[plat]


# selections ---------------------------------------------------------------------------------
def sel_locked(T, g):
    return T[(T.origin == "old") & (T.group == g)]


def sel_w16(T, fam=None, h=None, detach=False):
    d = T[(T.block == ("w16d" if detach else "w16"))]
    if fam is not None:
        d = d[d.family == fam]
    if h is not None:
        d = d[np.isclose(d.hwhm, h)]
    return d


def sel_ste_detach(T):
    return T[(T.block == "w16d") & (T.family == "ste")]


# =========================================================================================
# S5.1 members of the locked protocol
# =========================================================================================
PLAN = {  # research_plan.md Sec. 2.2: R, beta, nu2, R_angle, scale cost at O1-M
    "sarcos": {"D-S": (0.328, 0.86, 0.30, 0.286, 0.042), "D-M STE": (0.359, 0.79, 0.31, 0.333, 0.026),
               "MLP": (0.497, 0.93, 0.50, 0.367, 0.130), "B-A": (0.126, 0.90, 0.11, 0.122, 0.004)},
    "sim": {"D-S": (0.406, 0.83, 0.39, 0.349, 0.057), "D-M STE": (0.461, 0.65, 0.35, 0.437, 0.024),
            "MLP": (0.446, 0.95, 0.44, 0.318, 0.128), "B-A": (0.355, 0.76, 0.30, 0.330, 0.025)},
}


def t_members():
    rows = []
    for p in PLATS:
        T = mt(p)
        rows.append(rf"\multicolumn{{17}}{{l}}{{\emph{{{PNAME[p]}}} ($P_y={T.Py.iloc[0]:.3f}$; O1-M $\rho={RHO[p]['M']}$, "
                    rf"O1-H $\rho={RHO[p]['H']}$)}} \\")
        for g in LOCKED:
            d = sel_locked(T, g)
            v = d.mean(numeric_only=True)
            cells = [rf"\quad {LOCKED_LABEL[g]}", str(len(d))]
            cells += [num(v.R_N), num(v.beta_N), num(v.nu2_N)]
            cells += [num(v.R_M), num(v.beta_M), num(v.nu2_M), num(v.slope_M), num(v.Rang_M), num(v.S_M),
                      num(v.rc_M), num(v.Pn_M), num(v.Pb_M)]
            cells += [num(v.R_H), num(v.beta_H), num(v.nu2_H), num(v.kappa, 2)]
            rows.append(" & ".join(cells) + r" \\")
    dev = []
    for p in PLATS:
        T = mt(p)
        for g, vals in PLAN[p].items():
            v = sel_locked(T, g).mean(numeric_only=True)
            got = (v["R_M"], v["beta_M"], v["nu2_M"], v["Rang_M"], v["S_M"])
            for q, a, b in zip(["R", "beta", "nu2", "Rang", "S"], got, vals):
                dev.append((abs(a - b), p, g, q, a, b))
    dev.sort(reverse=True)
    FACTS["plan22_max_dev"] = dev[0]
    FACTS["plan22_n_checked"] = len(dev)
    fr = {}
    for p in PLATS:
        T = mt(p)
        old = T[T.origin == "old"]
        win = sel_locked(T, "D-M window").groupby("family").r.mean()
        fr[p] = {g: round(float(sel_locked(T, g).r.mean()), 3) for g in LOCKED}
        fr[p]["window_by_surrogate"] = {k: round(float(x), 3) for k, x in win.items()}
        aug = old[old.prefix.isin(["dma", "dsa", "dmq", "dsq", "dmas", "dmqs", "ba"])].groupby("group").r.mean()
        fr[p]["augmented"] = {k: round(float(x), 3) for k, x in aug.items()}
    FACTS["forward_response"] = fr
    cap = (r"Members of the locked protocol in the plane (E15, test role, mean over the $n$ members of a family). "
           r"$R$: test nMSE; $\beta$, $\|\nu\|^2$, $s$: signal gain, orthogonal-noise power, calibration slope; "
           r"$R^\circ$, $\mathcal S$: angle risk and scale cost, $R=R^\circ+\mathcal S$ \eqref{eq:s6-I2}; $r$: forward "
           r"response; $P_n$, $P_b$: propagated and nominal terms of the residual split (S7.3); $\kappa$: slope of "
           r"$\log(R(\rho)-R(0))$ on $\log\rho$ over the frozen O1 levels (arm $0.35,0.7,1.0$; SARCOS $0.5,0.7$).")
    head = [r" & & \multicolumn{3}{c}{N} & \multicolumn{9}{c}{O1-M} & \multicolumn{3}{c}{O1-H} & \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-14}\cmidrule(lr){15-17}",
            r"Family & $n$ & $R$ & $\beta$ & $\|\nu\|^2$ & $R$ & $\beta$ & $\|\nu\|^2$ & $s$ & $R^\circ$ & "
            r"$\mathcal S$ & $r$ & $P_n$ & $P_b$ & $R$ & $\beta$ & $\|\nu\|^2$ & $\kappa$ \\"]
    rows = [r.replace(r"\multicolumn{17}", r"\multicolumn{18}") for r in rows]
    write("s5n_members", table("tab:s5members", cap, "lc" + "c" * 16, head, rows, sep="2.4pt"))


# =========================================================================================
# S5.2 surrogate width (E16)
# =========================================================================================
def width_row(d, p_first=False):
    v = d.mean(numeric_only=True)
    return [num(100 * v.best_dev, 2), num(100 * v.R_N, 2), num(v.R_M), num(v.R_H), num(v.r), num(v.beta_M),
            num(v.nu2_M), f"{v.best_epoch:.0f}" if np.isfinite(v.best_epoch) else "--"]


def t_width():
    rows = []
    for fam in FAMS:
        hs = sorted(set(sel_w16(mt("sim"), fam).hwhm) | set(sel_w16(mt("sarcos"), fam).hwhm))
        for i, h in enumerate(hs):
            cells = [FAMNAME[fam] if i == 0 else "", hw(h) + (r"$^\dagger$" if h > 4 else "")]
            for p in PLATS:
                cells += width_row(sel_w16(mt(p), fam, h))
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace[2pt]")
    for lab, f in [("STE, original reset", lambda T: sel_locked(T, "D-M STE")),
                   ("STE, detached reset", sel_ste_detach)]:
        cells = [lab, r"$\infty$"]
        for p in PLATS:
            cells += width_row(f(mt(p)))
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Surrogate-width sweep (E16): the four families, peak-normalized to $g(0)=1$, against the half width at "
           r"half maximum (HWHM, membrane units, threshold initialized at one); mean over three seeds, and over seed "
           r"0 only for the rows marked $^\dagger$. Dev and N: development nMSE at the selected epoch and nominal test "
           r"nMSE, both $\times10^{2}$; O1-M and O1-H: test nMSE; $r$, $\beta$, $\|\nu\|^2$ at O1-M (E15 pipeline); "
           r"ep: development-selected epoch. STE, original reset: the five straight-through members of D-M; STE, "
           r"detached reset: three seeds with the spike of the reset detached (S5.3). The triangular member at HWHM "
           r"0.25 is the locked development-selected surrogate of SARCOS and reproduces it exactly.")
    grp = r"\multicolumn{8}{c}{%s}"
    head = [" & & " + " & ".join(grp % PNAME[p] for p in PLATS) + r" \\",
            r"\cmidrule(lr){3-10}\cmidrule(lr){11-18}",
            r"Family & HWHM & " + " & ".join(["Dev & N & O1-M & O1-H & $r$ & $\\beta$ & $\\|\\nu\\|^2$ & ep"] * 2)
            + r" \\"]
    write("s5n_width", table("tab:s5width", cap, "ll" + "c" * 16, head, rows, sep="2.2pt"))
    # hypothesis H1 numbers
    h1 = {}
    for p in PLATS:
        T = mt(p)
        h1[p] = {}
        for fam in FAMS:
            d = sel_w16(T, fam)
            d = d[d.hwhm <= 4]
            h1[p][fam] = {"rs_r": rho_s(d.hwhm, d.r), "rs_beta": rho_s(d.hwhm, d.beta_M),
                          "rs_N": rho_s(d.hwhm, d.R_N), "rs_M": rho_s(d.hwhm, d.R_M), "n": len(d)}
        gaps = []
        for h in GRID:
            ms = [sel_w16(T, f, h).r.mean() for f in FAMS]
            sds = [sel_w16(T, f, h).r.std(ddof=1) for f in FAMS]
            gaps.append((h, max(ms) - min(ms), 2 * float(np.nanmean(sds))))
        h1[p]["gap_vs_2sd"] = gaps
        h1[p]["n_gap_below_2sd"] = int(sum(g < s for _, g, s in gaps))
        # best configuration per noise level among w16 grid + STE
        best = {}
        for key in LV[p]:
            cands = []
            for fam in FAMS:
                for h in GRID:
                    d = sel_w16(T, fam, h)
                    cands.append((d[f"R_{key}"].mean(), fam, h))
            cands.append((sel_locked(T, "D-M STE")[f"R_{key}"].mean(), "ste", np.inf))
            cands.sort()
            best[key] = cands[:3]
        h1[p]["best_by_level"] = best
        # range of r and beta over the grid
        g = T[(T.block == "w16") & (T.hwhm <= 4)].groupby(["family", "hwhm"])[["r", "beta_M", "R_N", "R_M",
                                                                              "R_H"]].mean()
        h1[p]["range_r"] = (float(g.r.min()), float(g.r.max()))
        h1[p]["range_beta"] = (float(g.beta_M.min()), float(g.beta_M.max()))
        h1[p]["ste_r_beta"] = (float(sel_locked(T, "D-M STE").r.mean()), float(sel_locked(T, "D-M STE").beta_M.mean()))
    FACTS["H1_H2"] = h1


def t_ctrl():
    rows = []
    for fam in ["sig", "atan"]:
        for i, h in enumerate([0.0625, 0.25, 1.0, 4.0]):
            cells = [FAMNAME[fam] if i == 0 else "", hw(h)]
            pk = None
            for p in PLATS:
                T = mt(p)
                c = T[(T.block == "w16c") & (T.family == fam) & np.isclose(T.hwhm, h)]
                n = sel_w16(T, fam, h)
                n = n[n.seed == 0]
                if pk is None and len(c):
                    pk = c.peak.iloc[0]
                a, b = c.iloc[0], n.iloc[0]
                cells += [sgn(100 * (a.R_N - b.R_N), 2), sgn(a.R_M - b.R_M), sgn(a.R_H - b.R_H), sgn(a.r - b.r),
                          sgn(a.beta_M - b.beta_M), f"{a.best_epoch:.0f}/{b.best_epoch:.0f}"]
            cells.insert(2, num(pk, 2))
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace[2pt]")
    rows = rows[:-1]
    cap = (r"Width against amplitude (E16 control group, seed 0): members trained with the original "
           r"parameterization of the sigmoid ($g(0)=k/4$) and ATan ($g(0)=\alpha/2$) surrogates at the HWHM of the "
           r"peak-normalized grid, minus the peak-normalized member of the same family, HWHM, and seed. Peak: "
           r"$g(0)$ of the original parameterization. N: nominal test nMSE $\times10^2$; O1-M, O1-H: test nMSE; "
           r"$r$, $\beta$ at O1-M; ep: selected epoch, original/peak-normalized. Negative differences favour the "
           r"original parameterization.")
    grp = r"\multicolumn{6}{c}{%s}"
    head = [" & & & " + " & ".join(grp % PNAME[p] for p in PLATS) + r" \\",
            r"\cmidrule(lr){4-9}\cmidrule(lr){10-15}",
            r"Family & HWHM & Peak & " + " & ".join([r"$\Delta$N & $\Delta$O1-M & $\Delta$O1-H & $\Delta r$ & "
                                                     r"$\Delta\beta$ & ep"] * 2) + r" \\"]
    write("s5n_ctrl", table("tab:s5ctrl", cap, "lll" + "c" * 12, head, rows, sep="2.5pt"))
    # summary numbers: narrow end
    ctrl = {}
    for p in PLATS:
        T = mt(p)
        c = T[T.block == "w16c"]
        ctrl[p] = {}
        for fam in ["sig", "atan"]:
            for h in GRID:
                a = c[(c.family == fam) & np.isclose(c.hwhm, h)].iloc[0]
                n = sel_w16(T, fam, h)
                b = n[n.seed == 0].iloc[0]
                ctrl[p][f"{fam}_{h}"] = {k: (round(float(a[k]), 4), round(float(b[k]), 4)) for k in
                                         ["R_N", "R_M", "R_H", "r", "beta_M", "best_epoch"]}
    FACTS["ctrl"] = ctrl


# =========================================================================================
# S5.3 reset factor and the detach control
# =========================================================================================
def reset_groups(T):
    old = T[T.origin == "old"]
    out = [("D-M, window members", sel_locked(T, "D-M window")), ("D-S members", sel_locked(T, "D-S")),
           ("D-M, STE member", sel_locked(T, "D-M STE")),
           ("E14 early checkpoints", old[old.prefix == "uf"])]
    for fam in FAMS:
        out.append((f"E16 {FAMNAME[fam].lower() if fam != 'atan' else 'ATan'}, original reset", sel_w16(T, fam)))
    for fam in ["tri", "atan"]:
        out.append((f"E16 {FAMNAME[fam].lower() if fam != 'atan' else 'ATan'}, detached reset",
                    sel_w16(T, fam, detach=True)))
    out.append(("STE, detached reset", sel_ste_detach(T)))
    return out


def t_reset():
    names = [n for n, _ in reset_groups(mt("sim"))]
    rows = []
    for i, nm in enumerate(names):
        cells = [nm]
        for p in PLATS:
            d = dict(reset_groups(mt(p)))[nm]
            cells += [str(len(d)), num(d.thr3_near1.mean()), num(d.thr3_abs.mean()), num(d.thr1_near1.mean()),
                      num(rho_s(d.thr3_near1, d.best_epoch), 2), num(rho_s(d.thr3_near1, d.R_N), 2)]
        rows.append(" & ".join(cells) + r" \\")
        if nm in ("E14 early checkpoints", "E16 escape-rate, original reset"):
            rows.append(r"\addlinespace[2pt]")
    cap = (r"Layer thresholds and the reset factor $1-\vartheta_\ell g(u)$ of Lemma~1 (E15 weight statistics). "
           r"$f_3$ and $f_1$: fraction of neurons of layer~3 and layer~1 with $|\vartheta-1|<0.02$, i.e.\ still at "
           r"the initial threshold; $|1-\vartheta_3|$: mean over the neurons of layer 3; mean over the members of a "
           r"row. $\rho_s$: Spearman rank correlation across the members of the row between $f_3$ and the selected "
           r"epoch (ep) and the nominal test nMSE ($R_{\mathrm N}$); -- when fewer than four members or no "
           r"variation. E16 rows pool all widths of a family (0.0625--64 for the original reset, 0.0625--16 detached).")
    grp = r"\multicolumn{6}{c}{%s}"
    head = [" & " + " & ".join(grp % PNAME[p] for p in PLATS) + r" \\",
            r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
            "Members & " + " & ".join([r"$n$ & $f_3$ & $|1-\vartheta_3|$ & $f_1$ & $\rho_s$(ep) & "
                                       r"$\rho_s(R_{\mathrm N})$"] * 2) + r" \\"]
    write("s5n_reset", table("tab:s5reset", cap, "l" + "c" * 12, head, rows, sep="2.6pt"))
    rs = {}
    for p in PLATS:
        T = mt(p)
        ste = sel_locked(T, "D-M STE").sort_values("seed")
        rs[p] = {"ste_seeds": ste[["seed", "thr3_near1", "W1_rel", "best_epoch", "R_N"]].round(4).values.tolist(),
                 "ste_rs_epoch": rho_s(ste.thr3_near1, ste.best_epoch), "ste_rs_RN": rho_s(ste.thr3_near1, ste.R_N),
                 "groups": {nm: (len(d), float(d.thr3_near1.mean())) for nm, d in reset_groups(T)}}
        clean = T[(T.origin == "old") & T.prefix.isin(CLEAN) & (T.model_kind == "snn")]
        rs[p]["clean_rs_epoch"] = rho_s(clean.thr3_near1, clean.best_epoch)
        rs[p]["clean_rs_RN"] = rho_s(clean.thr3_near1, clean.R_N)
        rs[p]["clean_n"] = len(clean)
        w = T[T.block == "w16"]
        rs[p]["w16_by_h"] = w.groupby("hwhm").thr3_near1.mean().round(3).to_dict()
    FACTS["reset"] = rs


def detach_pairs(T):
    out = []
    for fam in ["tri", "atan", "ste"]:
        if fam == "ste":
            dd = sel_ste_detach(T)
            oo = sel_locked(T, "D-M STE")
            hs = [np.inf]
        else:
            dd = sel_w16(T, fam, detach=True)
            oo = sel_w16(T, fam)
            hs = sorted(set(dd.hwhm))
        for h in hs:
            a = dd if fam == "ste" else dd[np.isclose(dd.hwhm, h)]
            b = oo if fam == "ste" else oo[np.isclose(oo.hwhm, h)]
            j = a.set_index("seed").join(b.set_index("seed"), lsuffix="_d", rsuffix="_o", how="inner")
            out.append((fam, h, j))
    return out


def t_detach():
    keys = [("best_epoch", 0, 1), ("R_N", 2, 100), ("R_M", 3, 1), ("R_H", 3, 1), ("r", 3, 1), ("beta_M", 3, 1),
            ("thr3_near1", 2, 1)]
    P = {p: detach_pairs(mt(p)) for p in PLATS}
    rows = []
    for i in range(len(P["sim"])):
        fam, h, _ = P["sim"][i]
        n_s = len(P["sim"][i][2])
        cells = [FAMNAME[fam] if (i == 0 or P["sim"][i - 1][0] != fam) else "",
                 hw(h) + (r"$^\dagger$" if n_s == 1 else "")]
        for p in PLATS:
            j = P[p][i][2]
            for k, d, sc in keys:
                diff = sc * (j[f"{k}_d"] - j[f"{k}_o"]).mean()
                cells.append(sgn(diff, d) if k != "best_epoch" else sgn(diff, 0))
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Detached reset (E16 control): paired differences, detached minus original reset, for the same family, "
           r"HWHM, and seed (three seeds; seed 0 only for $^\dagger$, the width at which the original-reset member "
           r"was trained once). STE: the straight-through members of D-M with the same seed. ep: selected epoch; N: "
           r"nominal test nMSE $\times10^2$; O1-M, O1-H: test nMSE; $r$, $\beta$ at O1-M; $f_3$: fraction of layer-3 "
           r"thresholds within 0.02 of one.")
    grp = r"\multicolumn{7}{c}{%s}"
    head = [" & & " + " & ".join(grp % PNAME[p] for p in PLATS) + r" \\",
            r"\cmidrule(lr){3-9}\cmidrule(lr){10-16}",
            r"Family & HWHM & " + " & ".join([r"ep & N & O1-M & O1-H & $r$ & $\beta$ & $f_3$"] * 2) + r" \\"]
    write("s5n_detach", table("tab:s5detach", cap, "ll" + "c" * 14, head, rows, sep="2.6pt"))
    det = {}
    for p in PLATS:
        det[p] = {}
        for fam, h, j in P[p]:
            det[p][f"{fam}_{h}"] = {k: [round(float(x), 4) for x in (j[f"{k}_o"].mean(), j[f"{k}_d"].mean())]
                                    for k in ["best_epoch", "R_N", "R_M", "R_H", "r", "beta_M", "thr3_near1"]}
            det[p][f"{fam}_{h}"]["n"] = len(j)
    FACTS["detach"] = det


# =========================================================================================
# S5.4 first-layer geometry, unit SNR, flips (E15, Lemma 2)
# =========================================================================================
def geo_groups(T):
    w = T[T.block == "w16"]
    return [("D-M, window members", sel_locked(T, "D-M window")),
            ("D-M, STE member", sel_locked(T, "D-M STE")), ("STE, detached reset", sel_ste_detach(T)),
            ("E16, HWHM $\\le0.25$", w[w.hwhm <= 0.25]), ("E16, HWHM 4", w[np.isclose(w.hwhm, 4)]),
            ("B-A", sel_locked(T, "B-A")), ("B-N (MLP)", sel_locked(T, "MLP"))]


def t_weights():
    rows = []
    dr = {}
    for p in PLATS:
        rows.append(rf"\multicolumn{{15}}{{l}}{{\emph{{{PNAME[p]}}}}} \\")
        dr[p] = {}
        for nm, d in geo_groups(mt(p)):
            v = d.mean(numeric_only=True)
            mlp = nm.startswith("B-N")
            cells = [r"\quad " + nm, str(len(d))]
            for i in (1, 2):
                cells += ["--"] * 3 if mlp else [num(v[f"W{i}_fro"], 1), num(v[f"W{i}_srank"], 1),
                                                 num(v[f"W{i}_rel"], 2)]
            cells += ["--"] * 3 if mlp else [num(v.W3_rel, 2), num(v.W4_rel, 2), num(v.thr1_mean, 2)]
            for var in ("full21", "qqd14"):
                t5, b5 = v[f"dir_{var}_top5"], v[f"dir_{var}_bottom5"]
                cells += [num(t5), num(b5)] if var == "full21" else []
                cells.append(num((d[f"dir_{var}_bottom5"] / d[f"dir_{var}_top5"]).mean(), 2))
                dr[p].setdefault(nm, {})[var] = (round(float(t5), 3), round(float(b5), 3))
            rows.append(" & ".join(cells) + r" \\")
    cap = (r"First-layer geometry and direction-resolved response (E15; mean over the $n$ members of a row). "
           r"$\|W\|_F$, srk: Frobenius norm and stable rank $\|W\|_F^2/\|W\|_2^2$ of layers 1 and 2; $\delta_\ell$: "
           r"relative change from the initialization regenerated from the seed, $\|W_\ell-W_{\ell,0}\|_F/\|W_{\ell,0}\|_F$ "
           r"($\delta_4$: first readout layer); $\bar\vartheta_1$: mean first-layer threshold. Direction-resolved "
           r"response: the O1-M perturbation energy placed on the five leading (top) or trailing (bottom) principal "
           r"directions of the training inputs, all 21 channels; ratio: bottom over top, also for the $(q,\dot q)$ "
           r"block alone. E16 rows pool the four families with the original reset.")
    head = [r" & & \multicolumn{3}{c}{Layer 1} & \multicolumn{3}{c}{Layer 2} & & & & "
            r"\multicolumn{3}{c}{Direction, 21 ch.} & $(q,\dot q)$ \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){12-14}\cmidrule(lr){15-15}",
            r"Members & $n$ & $\|W\|_F$ & srk & $\delta_1$ & $\|W\|_F$ & srk & $\delta_2$ & $\delta_3$ & $\delta_4$ & "
            r"$\bar\vartheta_1$ & top & bottom & ratio & ratio \\"]
    write("s5n_geo", table("tab:s5geo", cap, "lc" + "c" * 13, head, rows, sep="3pt"))
    FACTS["weights"] = {p: {nm: {k: round(float(d[k].mean()), 3) for k in
                                 ["W1_fro", "W1_rel", "W1_srank", "W2_fro", "W2_rel", "W2_srank", "W3_rel", "W4_rel",
                                  "W5_rel", "thr1_mean"]} for nm, d in geo_groups(mt(p))} for p in PLATS}
    FACTS["direction"] = dr


FLIP_GROUPS = ["D-M window", "D-M STE"]


def t_flip():
    rows = []
    for p in PLATS:
        T = mt(p)
        rows.append(rf"\multicolumn{{13}}{{l}}{{\emph{{{PNAME[p]}}}}} \\")
        for key in ("M", "H"):
            for g in FLIP_GROUPS:
                d = sel_locked(T, g)
                v = d.mean(numeric_only=True)
                lab = rf"\quad O1-{key}, $\rho={RHO[p][key]}$" if g == FLIP_GROUPS[0] else ""
                cells = [lab, LOCKED_LABEL[g].replace("D-M, ", "")]
                cells += [num(v[f"snr_q05_{key}"], 2), num(v[f"snr_q50_{key}"], 2), num(v[f"snr_q95_{key}"], 1),
                          num(v[f"sigma_in_q50_{key}"], 3), num(v[f"flip_pred_sg2_{key}"]),
                          num(v[f"flip_pred_biv_{key}"]), num(v[f"flip1s1_{key}"]), num(v[f"flip1_{key}"]),
                          num(v[f"flip2_{key}"]), num(v[f"flip3_{key}"]), num(v[f"flipall_{key}"])]
                rows.append(" & ".join(cells) + r" \\")
    cap = (r"Unit SNR and spike flips (E15, 2{,}000 development inputs, mean over members). SNR quantiles (5, 50, "
           r"95\%) and median induced membrane noise $\sigma_{\mathrm{in}}$ of the first-layer units; first-layer flip "
           r"rate predicted by Lemma~2(ii) (threshold at the mean) and~(iii) (fitted offset), first internal step; "
           r"measured fraction of (neuron, step) pairs whose spike changes: layer 1 at step 1 and over all steps, "
           r"layers 2 and 3, and all layers.")
    head = [r" & & \multicolumn{4}{c}{Layer-1 units} & \multicolumn{2}{c}{Predicted, L1} & "
            r"\multicolumn{5}{c}{Measured flip rate} \\",
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-13}",
            r"Level & Members & SNR$_{5}$ & SNR$_{50}$ & SNR$_{95}$ & $\sigma_{\mathrm{in}}$ & (ii) & (iii) & "
            r"L1, $t{=}1$ & L1 & L2 & L3 & all \\"]
    write("s5n_flip", table("tab:s5flip", cap, "ll" + "c" * 11, head, rows, sep="3pt"))
    FACTS["flip_means"] = {p: {g: {k: round(float(sel_locked(mt(p), g)[k].mean()), 4) for k in
                                   ["flip1_M", "flipall_M", "snr_q50_M", "flip_pred_sg2_M", "flip_pred_biv_M",
                                    "flip1s1_M"]} for g in FLIP_GROUPS} for p in PLATS}


def rank_sets(T):
    snn_old = T[(T.origin == "old") & (T.model_kind == "snn")]
    snn_new = T[(T.origin == "new") & (T.model_kind == "snn") & (T.activation == "spike")]
    return [("Original members, all spiking", snn_old), ("Original, clean-trained", snn_old[snn_old.prefix.isin(CLEAN)]),
            ("Original, D-M only", snn_old[snn_old.prefix == "dm"]), ("E16--E18 members", snn_new),
            ("All spiking members", pd.concat([snn_old, snn_new]))]


def t_rank():
    rows = []
    rk = {}
    for p in PLATS:
        T = mt(p)
        rows.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{PNAME[p]}}}}} \\")
        rk[p] = {}
        for nm, d in rank_sets(T):
            cells = [r"\quad " + nm, str(len(d))]
            rk[p][nm] = {}
            for key in ("M", "H"):
                d2 = d[np.isfinite(d[f"flip1_{key}"])]
                a = rho_s(d2[f"flip_pred_sg2_{key}"], d2[f"flip1_{key}"])
                b = rho_s(d2[f"flip_pred_biv_{key}"], d2[f"flip1_{key}"])
                c = rho_s(d2[f"flip_pred_emp1_{key}"], d2[f"flip1s1_{key}"])
                e = rho_s(d2[f"flip_pred_biv_{key}"], d2[f"flipall_{key}"])
                cells += [num(a, 2), num(b, 2), num(c, 3), num(e, 2)]
                rk[p][nm][key] = (a, b, c, e, len(d2))
            rows.append(" & ".join(cells[:2] + cells[2:]) + r" \\")
    cap = (r"Rank correlation across members between predicted and measured flip rates (E15; preregistered criterion "
           r"$\ge0.8$ for H5). (ii), (iii): first-layer prediction of Lemma~2(ii) and~(iii) against the measured "
           r"first-layer flip rate over all internal steps; exact $t{=}1$: the conditional formula of Lemma~2(iii) "
           r"averaged over the 2{,}000 inputs, against the measured first-step flip rate; (iii)$\to$all: Lemma~2(iii) "
           r"against the flip rate of all spiking layers. Clean-trained: D-M, D-S, D-T, B-L, the development width grid, "
           r"and the E14 controls; the E16--E18 rows exclude the MLP variants.")
    head = [r" & & \multicolumn{4}{c}{O1-M} & \multicolumn{4}{c}{O1-H} \\",
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
            r"Members & $n$ & (ii) & (iii) & exact $t{=}1$ & (iii)$\to$all & (ii) & (iii) & exact $t{=}1$ & "
            r"(iii)$\to$all \\"]
    rows = [r.replace(r"\multicolumn{9}", r"\multicolumn{10}") for r in rows]
    write("s5n_rank", table("tab:s5rank", cap, "lc" + "c" * 8, head, rows, sep="3.5pt"))
    FACTS["rank"] = rk


def t_station():
    rows = []
    stn = {}
    for p in PLATS:
        T = mt(p)
        w, wd = T[T.block == "w16"], T[T.block == "w16d"]
        groups = [("D-M, window members", sel_locked(T, "D-M window")),
                  ("D-M, STE member", sel_locked(T, "D-M STE")), ("STE, detached reset", sel_ste_detach(T)),
                  ("E16, HWHM 4", w[np.isclose(w.hwhm, 4)]),
                  ("E16 detached, HWHM 4--16", wd[(wd.family != "ste") & (wd.hwhm >= 4)])]
        rows.append(rf"\multicolumn{{13}}{{l}}{{\emph{{{PNAME[p]}}}}} \\")
        stn[p] = {}
        for nm, d in groups:
            v = d.mean(numeric_only=True)
            cells = [r"\quad " + nm, str(len(d))]
            for lay in ("1", "2", "3"):
                cells += [num(v[f"st{lay}_ratio_G"], 3), num(v[f"st{lay}_ratio_K"], 3), num(v[f"st{lay}_cos_GK"], 2)]
            fl = np.nanmean([v[f"st{l}_floor_G"] for l in "123"])
            cells += [num(fl, 3), f"{v.best_epoch:.0f}"]
            rows.append(" & ".join(cells) + r" \\")
            stn[p][nm] = {f"L{l}": (round(float(v[f'st{l}_ratio_G']), 4), round(float(v[f'st{l}_ratio_K']), 4),
                                    round(float(v[f'st{l}_floor_G']), 4), round(float(v[f'st{l}_cos_GK']), 3),
                                    round(float(v[f'st{l}_frac_in_window']), 3) if np.isfinite(
                                        v[f'st{l}_frac_in_window']) else None) for l in "123"}
            stn[p][nm]["best_epoch"] = float(v.best_epoch)
    cap = (r"Stationarity residuals of Lemma~1 at the selected checkpoints (E15, 16{,}384 training rows in eight "
           r"blocks). $G$: $\|\E\sum_tg\,c\,\bar a^\top\|_F/\E\sum_t\|g\,c\,\bar a^\top\|_F$, the $g$-weighted condition "
           r"\eqref{eq:s6-weighted}; $K$: the same without $g$; $\cos$: cosine between the two matrices; zero is exact "
           r"stationarity; floor: block standard error of $G$, mean over layers; ep: selected epoch. Under the "
           r"straight-through surrogate $G\equiv K$; --: layer 3 with the detached reset, where $g$ does not enter.")
    head = [r" & & \multicolumn{3}{c}{Layer 1} & \multicolumn{3}{c}{Layer 2} & \multicolumn{3}{c}{Layer 3} & & \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
            r"Members & $n$ & $G$ & $K$ & $\cos$ & $G$ & $K$ & $\cos$ & $G$ & $K$ & $\cos$ & floor & ep \\"]
    write("s5n_station", table("tab:s5station", cap, "lc" + "c" * 11, head, rows, sep="3pt"))
    FACTS["station"] = stn


# =========================================================================================
# S5.6 layer placement (E17) and noise injection (E18)
# =========================================================================================
def t_e17():
    rows = []
    e17 = {}
    defs = [("All layers narrow", None, 0.25), ("Layer 1 wide", "L1", 1.0), ("Layers 2--3 wide", "L23", 1.0),
            ("Layer 1 wide", "L1", 4.0), ("Layers 2--3 wide", "L23", 4.0)]
    for nm, pl, h in defs:
        cells = [nm, hw(h)]
        for p in PLATS:
            T = mt(p)
            base = sel_w16(T, "tri", 0.25)
            if pl is None:
                d = base
            elif pl == "all":
                d = sel_w16(T, "tri", h)
            else:
                d = T[(T.block == "w17") & (T.placement == pl) & np.isclose(T.hwhm, h)]
            full = sel_w16(T, "tri", h)
            dr_full = full.r.mean() - base.r.mean()
            se = math.sqrt(full.r.var(ddof=1) / len(full) + base.r.var(ddof=1) / len(base))
            share = (d.r.mean() - base.r.mean()) / dr_full if pl is not None else np.nan
            defined = abs(dr_full) > 2 * se
            v = d.mean(numeric_only=True)
            cells += [num(100 * v.R_N, 2), num(v.R_M), num(v.R_H), num(v.r), num(v.beta_M), num(v.nu2_M),
                      (num(share, 2) if defined else "n.d.") if pl is not None else "--"]
            e17.setdefault(p, {})[f"{nm}|{h}"] = {"R_N": float(v.R_N), "R_M": float(v.R_M), "R_H": float(v.R_H),
                                                  "r": float(v.r), "share": float(share), "dr_full": float(dr_full),
                                                  "se_full": se, "beta": float(v.beta_M), "nu2": float(v.nu2_M)}
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Layer placement of the wide surrogate (E17, triangular family, three seeds). The narrow surrogate has "
           r"HWHM 0.25 (the locked SARCOS surrogate); the wide one has the HWHM given. N: nominal test nMSE "
           r"$\times10^2$; O1-M, O1-H: test nMSE; $r$, $\beta$, $\|\nu\|^2$ at O1-M; share: change of $r$ from the "
           r"all-narrow member as a fraction of the change when all layers are wide (preregistered criterion of H3: "
           r"$\ge0.7$ for layer 1 alone and $\le0.3$ for layers 2--3 alone); n.d.: not defined because the all-wide "
           r"change is within two standard errors. The all-wide rows are the E16 members.")
    grp = r"\multicolumn{7}{c}{%s}"
    head = [" & & " + " & ".join(grp % PNAME[p] for p in PLATS) + r" \\",
            r"\cmidrule(lr){3-9}\cmidrule(lr){10-16}",
            r"Placement & HWHM & " + " & ".join([r"N & O1-M & O1-H & $r$ & $\beta$ & $\|\nu\|^2$ & share"] * 2)
            + r" \\"]
    write("s5n_e17", table("tab:s5e17", cap, "ll" + "c" * 14, head, rows, sep="2.6pt"))
    FACTS["e17"] = e17


STRONG = {"sim": "sig", "sarcos": "tri"}


def e18_groups(T, p):
    fam = STRONG[p]
    base = T[(T.origin == "old") & (T.prefix == "dm") & (T.family == fam) & (T.seed <= 2)]
    g = [("Locked surrogate", "", base)]
    for s in sorted(set(T[T.block == "w18mem"].membrane_noise)):
        g.append(("Membrane noise", rf"$\sigma={s:g}$", T[(T.block == "w18mem") & (T.membrane_noise == s)]))
    for a in sorted(set(T[T.block == "w18aug"].aug_scale)):
        g.append(("Input augmentation", rf"${a:g}\rho_{{\mathrm M}}$", T[(T.block == "w18aug") & (T.aug_scale == a)]))
    for h in sorted(set(T[T.block == "w18esc"].hwhm)):
        g.append(("Escape noise", rf"HWHM ${h:g}$", T[(T.block == "w18esc") & np.isclose(T.hwhm, h)]))
    # backward-only width: the E16 grid member of the locked family with the lowest mean angle risk at O1-M
    best = None
    for h in GRID:
        d = sel_w16(T, fam, h)
        m = d.Rang_M.mean()
        if best is None or m < best[0]:
            best = (m, h, d)
    g.append(("Peak-normalized width", rf"${best[1]:g}$", best[2]))  # H4 reference only (not printed)
    g.append(("MLP, ReLU (B-N)", "", sel_locked(T, "MLP")))
    for act, lab in [("tanh", "MLP, tanh"), ("crelu", "MLP, clipped ReLU"), ("bin", "MLP, binary + STE")]:
        g.append((lab, "", T[(T.block == "w18mlp") & (T.activation == act)]))
    return g


def w1_lookup(p):
    """Seed-mean W0/W1 risk, beta, nu2 at O1-M of single members from the E21 ladders."""
    out = {}
    for f in ("e21_ladder_e18.csv", "e21_ladder_w16.csv", "e21_ladder.csv"):
        d = pd.read_csv(RES / p / f)
        d = d[d.cond == "O1-M"]
        if "kind" in d:
            d = d[d.kind == "single"]
        for _, r in d.iterrows():
            if r.space in ("W0", "W1"):
                out.setdefault(r.method, {})[r.space] = (r.nmse_mean, r.beta, r.nu2, r.n_seeds)
    return out


def e18_w1_key(nm, d):
    if nm.startswith("Locked"):
        return "B-F"
    if nm.startswith("MLP, ReLU"):
        return "B-N"
    blk = d.block.iloc[0] if len(d) else ""
    if blk == "w18mem":
        return f"w18_mem{d.membrane_noise.iloc[0]:g}"
    if blk == "w18aug":
        return f"w18_aug{d.aug_scale.iloc[0]:g}"
    if blk == "w18esc":
        return f"w18_esc_h{d.hwhm.iloc[0]:g}"
    if blk == "w18mlp":
        return f"w18_mlp_{d.activation.iloc[0]}"
    if blk == "w16":
        return f"w16_{d.family.iloc[0]}_h{d.hwhm.iloc[0]:g}"
    return None


def t_e18():
    G = {p: e18_groups(mt(p), p) for p in PLATS}
    W1 = {p: w1_lookup(p) for p in PLATS}
    rows = []
    for i in range(len(G["sim"])):
        nm, par_sim, _ = G["sim"][i]
        if nm == "Peak-normalized width":
            continue
        par_sar = G["sarcos"][i][1]
        par = par_sim if par_sim == par_sar else f"{par_sim}/{par_sar}".replace("HWHM $", "$")
        cells = [nm, par]
        for p in PLATS:
            d = G[p][i][2]
            v = d.mean(numeric_only=True)
            w = W1[p].get(e18_w1_key(nm, d), {})
            w1 = w.get("W1")
            flag = r"$^\ddagger$" if (w1 is not None and w1[3] != len(d)) else ""
            cells += [num(100 * v.R_N, 2), num(v.R_M), (num(w1[0]) + flag) if w1 else "--", num(v.beta_M),
                      num(w1[1]) if w1 else "--", num(v.nu2_M), num(w1[2]) if w1 else "--", num(v.kappa, 2)]
        rows.append(" & ".join(cells) + r" \\")
        nxt = G["sim"][i + 1][0] if i + 1 < len(G["sim"]) else None
        if nxt is not None and nxt != nm and not (nm.startswith("MLP") and nxt.startswith("MLP")):
            rows.append(r"\addlinespace[1pt]")
    cap = (r"Noise injection and bounded activations (E18, three seeds each). Locked surrogate: the "
           r"development-selected surrogate at its locked width (sigmoid on the arm, triangular on SARCOS), seeds "
           r"0--2 of D-M. The E18 spiking members train it with Gaussian noise of standard deviation $\sigma$ on the "
           r"first-layer membrane, with input-noise augmentation at a per-batch level drawn up to the stated multiple "
           r"of $\rho_{\mathrm M}$ ($1\rho_{\mathrm M}$ reproduces B-A), or with an escape-noise stochastic forward "
           r"pass, and are tested with the deterministic forward pass. Peak-normalized width (backward only): the E16 "
           r"members of the same family at the HWHM with the lowest angle risk $R^\circ$ at O1-M on the grid, and at "
           r"the widest grid HWHM. "
           r"N: nominal test nMSE $\times10^2$; W0: test nMSE of the member at O1-M; W1: test nMSE at O1-M of the "
           r"member with the null predictor, weights fitted on $\Dcal$ (E21); $\beta$ and $\|\nu\|^2$ of the member "
           r"(W0) and of the W1 average; $\kappa$ as in Table~\ref{tab:s5members}. $^\ddagger$: W1 over the five "
           r"seeds of B-F or B-N. Two settings separated by a slash: arm/SARCOS.")
    grp = r"\multicolumn{8}{c}{%s}"
    head = [" & & " + " & ".join(grp % PNAME[p] for p in PLATS) + r" \\",
            r"\cmidrule(lr){3-10}\cmidrule(lr){11-18}",
            " & & " + " & ".join([r" & \multicolumn{2}{c}{O1-M} & \multicolumn{2}{c}{$\beta$} & "
                                  r"\multicolumn{2}{c}{$\|\nu\|^2$} & "] * 2) + r" \\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){12-13}\cmidrule(lr){14-15}"
            r"\cmidrule(lr){16-17}",
            r"Training & Setting & " + " & ".join([r"N & W0 & W1 & W0 & W1 & W0 & W1 & $\kappa$"] * 2) + r" \\"]
    write("s5n_e18", table("tab:s5e18", cap, "ll" + "c" * 16, head, rows, sep="2.1pt"))
    # H4: paired angle-risk differences against the backward-only width member, same seed
    h4 = {}
    for p in PLATS:
        g = G[p]
        refrow = [x for x in g if x[0] == "Peak-normalized width"][0]
        ref = refrow[2].set_index("seed")
        h4[p] = {"ref": refrow[1], "ref_Rang": round(float(ref.Rang_M.mean()), 4),
                 "ref_beta": round(float(ref.beta_M.mean()), 3), "ref_nu2": round(float(ref.nu2_M.mean()), 3)}
        for nm, par, d in g:
            rec = {"Rang": round(float(d.Rang_M.mean()), 4), "beta": round(float(d.beta_M.mean()), 3),
                   "nu2": round(float(d.nu2_M.mean()), 3), "kappa": round(float(d.kappa.mean()), 3),
                   "R_M": round(float(d.R_M.mean()), 4), "R_H": round(float(d.R_H.mean()), 4),
                   "R_N": round(float(d.R_N.mean()), 4)}
            w = W1[p].get(e18_w1_key(nm, d), {})
            if "W1" in w:
                rec["W1"] = [round(float(x), 4) for x in w["W1"][:3]]
            if nm in ("Membrane noise", "Input augmentation", "Escape noise"):
                j = d.set_index("seed").join(ref, lsuffix="_a", rsuffix="_b", how="inner")
                m, lo, hi = tci(j.Rang_M_a - j.Rang_M_b)
                rec["dRang_vs_width"] = (round(m, 4), round(lo, 4), round(hi, 4), len(j))
            h4[p][f"{nm} {par}"] = rec
    FACTS["H4"] = h4


def t_bestwidth():
    def seq(g):
        g = g.sort_values("rho")
        out, prev = [], None
        for _, r in g.iterrows():
            h = r.best_hwhm
            if prev is None or h != prev:
                lab = "STE" if not np.isfinite(h) else f"{h:g}"
                out.append(lab if prev is None else f"{lab} ({r.rho:.2f})")
                prev = h
        return r", ".join(out)
    B = {p: pd.read_csv(RES / p / "e22_bestwidth_w16.csv") for p in PLATS}
    rows = []
    vlab = {"standard_3seeds": "Original reset, 3 seeds", "standard_seed0_ultrawide": "Original reset, seed 0",
            "detach_3seeds": "Detached reset, 3 seeds"}
    bw = {}
    for v in ["standard_3seeds", "detach_3seeds"]:
        fams = [f for f in FAMS if f in set(B["sim"][B["sim"].variant == v].family)]
        for i, fam in enumerate(fams):
            cells = [vlab[v] if i == 0 else "", FAMNAME[fam]]
            for p in PLATS:
                g = B[p][(B[p].variant == v) & (B[p].family == fam)]
                cells.append(seq(g))
                bw.setdefault(p, {})[f"{v}|{fam}"] = seq(g)
            rows.append(" & ".join(cells) + r" \\")
        if v != "detach_3seeds":
            rows.append(r"\addlinespace[2pt]")
    cap = (r"Width with the lowest seed-mean test nMSE against the noise level $\rho$ (E16 members on the paired "
           r"noise $\delta=\rho\xi$ of E22, grid step 0.05, $0\le\rho\le1.2$ on the arm and $\le0.9$ on SARCOS). "
           r"Each entry gives the best HWHM at $\rho=0$ and each later best HWHM with, in parentheses, the level from "
           r"which it is best. Candidates: the width grid of the row (0.0625--4 with three seeds; 0.0625--64 with seed "
           r"0; 0.0625, 0.25, 1, 4, 16 detached) and the straight-through member (STE) with the same reset.")
    head = [r"Candidates & Family & Simulated arm & SARCOS \\"]
    write("s5n_bestwidth", table("tab:s5best", cap, "llp{5.6cm}p{5.6cm}", head, rows, sep="4pt"))
    FACTS["bestwidth"] = bw


def h4_knobs(T, p):
    """(name, member frame grouped by level) for the matched-gain comparison; baseline first."""
    fam = STRONG[p]
    if p == "sarcos":
        base = sel_w16(T, "tri", 0.25)  # identical to the locked triangular surrogate, same protocol as E16-E18
    else:
        base = T[(T.origin == "old") & (T.prefix == "dm") & (T.family == fam) & (T.seed <= 2)]
    knobs = [("Membrane noise", T[T.block == "w18mem"], "membrane_noise"),
             ("Input augmentation", T[T.block == "w18aug"], "aug_scale"),
             ("Escape noise", T[T.block == "w18esc"], "hwhm"),
             ("Width, original parameterization", T[(T.block == "w16c") & (T.family == fam)], "hwhm"),
             ("Width, peak-normalized", sel_w16(T, fam)[sel_w16(T, fam).hwhm <= 4], "hwhm")]
    return base, knobs


BGRID = {"sim": [0.72, 0.76, 0.80, 0.82, 0.84, 0.86], "sarcos": [0.84, 0.86, 0.88, 0.90]}


def t_matched():
    res = {}
    for p in PLATS:
        T = mt(p)
        base, knobs = h4_knobs(T, p)
        b0, n0 = float(base.beta_M.mean()), float(base.nu2_M.mean())
        res[p] = {"base": (b0, n0, len(base)), "curves": {}, "dev": {}}
        res[p]["curves"]["Scalar rescaling of the locked surrogate"] = {
            b: n0 * (b / b0) ** 2 for b in BGRID[p] if b <= b0 + 1e-9}
        for nm, d, lev in knobs:
            pts = d.groupby(lev)[["beta_M", "nu2_M"]].mean().values.tolist() + [[b0, n0]]
            k = np.array(sorted(pts))
            res[p]["curves"][nm] = {b: float(np.interp(b, k[:, 0], k[:, 1])) for b in BGRID[p]
                                    if k[:, 0].min() - 1e-9 <= b <= k[:, 0].max() + 1e-9}
            g = d.groupby(lev)[["beta_M", "nu2_M"]].mean()
            res[p]["dev"][nm] = [float(n - n0 * (b / b0) ** 2) for b, n in g.values]
    names = list(res["sim"]["curves"].keys())
    rows = []
    for nm in names:
        cells = [nm]
        for p in PLATS:
            c = res[p]["curves"][nm]
            cells += [num(c[b]) if b in c else "--" for b in BGRID[p]]
            dv = res[p]["dev"].get(nm)
            cells.append(f"{sgn(min(dv))}, {sgn(max(dv))}" if dv else "--")
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Orthogonal noise at matched signal gain, O1-M (E18 and E16 members, three seeds; the original "
           r"parameterization, seed 0). Each knob is the locked surrogate (arm: sigmoid at its locked width, D-M seeds "
           r"0--2, $\beta_0=%.3f$, $\|\nu_0\|^2=%.3f$; SARCOS: triangular at HWHM 0.25, $\beta_0=%.3f$, "
           r"$\|\nu_0\|^2=%.3f$) followed by its levels; $\|\nu\|^2$ is interpolated linearly in $\beta$ through the "
           r"level means and the baseline, and -- marks gains the knob does not reach. Scalar rescaling: "
           r"$\|\nu_0\|^2(\beta/\beta_0)^2$. Dev.: smallest and largest deviation of the level means from the "
           r"scalar-rescaling curve at their own $\beta$; negative is better than rescaling the output.") % (
        res["sim"]["base"][0], res["sim"]["base"][1], res["sarcos"]["base"][0], res["sarcos"]["base"][1])
    head = [r" & \multicolumn{%d}{c}{Simulated arm, $\beta=$} & & \multicolumn{%d}{c}{SARCOS, $\beta=$} & \\"
            % (len(BGRID["sim"]), len(BGRID["sarcos"])),
            r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1 + len(BGRID["sim"]), 3 + len(BGRID["sim"]),
                                                         2 + len(BGRID["sim"]) + len(BGRID["sarcos"])),
            "Knob & " + " & ".join(f"{b:.2f}" for b in BGRID["sim"]) + " & Dev. & "
            + " & ".join(f"{b:.2f}" for b in BGRID["sarcos"]) + r" & Dev. \\"]
    ncol = len(BGRID["sim"]) + len(BGRID["sarcos"]) + 2
    matched_tab = [rf"\begin{{tabular}}{{l{'c' * ncol}}}", r"\toprule"] + head + [r"\midrule"] + rows + \
                  [r"\bottomrule", r"\end{tabular}"]
    FACTS["matched"] = res
    # kappa of bounded activations (rows: platforms)
    kap = {}
    defs = [("ReLU", lambda T: sel_locked(T, "MLP")),
            ("tanh", lambda T: T[(T.block == "w18mlp") & (T.activation == "tanh")]),
            ("clipped ReLU", lambda T: T[(T.block == "w18mlp") & (T.activation == "crelu")]),
            ("binary", lambda T: T[(T.block == "w18mlp") & (T.activation == "bin")]),
            ("window", lambda T: sel_locked(T, "D-M window")),
            ("STE", lambda T: sel_locked(T, "D-M STE")),
            ("STE, det.", sel_ste_detach)]
    rows = []
    for p in PLATS:
        cells = [PNAME[p]]
        for nm, f in defs:
            d = f(mt(p))
            cells.append(f"{num(d.kappa.mean(), 2)} ({num(d.kappa.std(ddof=1), 2)})")
            kap.setdefault(p, {})[nm] = (round(float(d.kappa.mean()), 3), round(float(d.kappa.std(ddof=1)), 3))
        rows.append(" & ".join(cells) + r" \\")
    cap = (r"Noise-scaling exponent $\kappa$ (seed standard deviation) of MLPs with bounded hidden activations "
           r"(E18; ReLU: B-N, five seeds; others three seeds) and of the spiking members of D-M; fit over the frozen "
           r"O1 levels as in Table~\ref{tab:s5members}. $\kappa=2$: quadratic growth of a smooth map.")
    head = [r" & \multicolumn{4}{c}{MLP} & \multicolumn{3}{c}{Spiking, D-M} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
            r"$\kappa$ & ReLU & tanh & clipped ReLU & binary + STE & window & STE & STE, detached \\"]
    kappa_tab = [r"\begin{tabular}{l" + "c" * 7 + "}", r"\toprule"] + head + [r"\midrule"] + rows + \
                [r"\bottomrule", r"\end{tabular}"]
    lines = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{CAPS['tab:s5matched']}}}", r"\label{tab:s5matched}",
             r"\label{tab:s5kappa}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{0.9}",
             r"(a) Orthogonal noise $\|\nu\|^2$ at matched gain, O1-M\\[2pt]"] + matched_tab + \
            [r"\\[3pt] (b) Noise-scaling exponent $\kappa$ (seed standard deviation)\\[2pt]"] + kappa_tab + [r"\end{table}"]
    write("s5n_matched", "\n".join(lines) + "\n")
    FACTS["kappa_bounded"] = kap


# =========================================================================================
def main():
    for p in PLATS:
        mt(p)
    t_members()
    t_width()
    t_ctrl()
    t_reset()
    t_detach()
    t_weights()
    t_flip()
    t_rank()
    t_station()
    t_e17()
    t_e18()
    t_bestwidth()
    t_matched()

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 4)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    txt = json.dumps(clean(FACTS), indent=1)
    if os.environ.get("S5_FACTS"):
        Path(os.environ["S5_FACTS"]).write_text(txt, encoding="utf-8")
        print("facts ->", os.environ["S5_FACTS"])
    else:
        print(txt)


if __name__ == "__main__":
    main()
