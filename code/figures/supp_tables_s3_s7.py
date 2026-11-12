"""Supplementary tables for S3 (weight spaces x calibration budget) and S7 (reachable step:
E21 ladder, E22 noise path, E23 label-free calibration, E24 certificate, E25 budget, E26
weight-space selection).

Reads   results/{sim,sarcos}/e2*_*.csv
Writes  build/tables/s3_*.tex and s7_*.tex (new names; the older s3_m, s3_k,
        s3_e07 and s8_* tables are left untouched)

python code/figures/supp_tables_s3_s7.py
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SNN29_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results"
OUT = ROOT / "build" / "tables"
PLATS = ("sim", "sarcos")
PNAME = {"sim": "Simulated arm", "sarcos": "SARCOS"}
CONDS = {"sim": ["N", "O1-L", "O1-M", "O1-H", "O2B", "P1"],
         "sarcos": ["N", "O1-L", "O1-M", "O1-H", "O1qqd"]}
NOISY = {"sim": ["O1-L", "O1-M", "O1-H", "O2B"], "sarcos": ["O1-L", "O1-M", "O1-H", "O1qqd"]}
SPACES = ["W0", "W1", "W2", "W3", "W4", "W5", "W6"]


def rd(p, f):
    return pd.read_csv(RES / p / f)


def cl(c):
    """Condition label."""
    if c == "O2B":
        return "O2-B"
    if c == "O1qqd":
        return r"O1-M $(q,\dot q)$"
    if c.startswith("t3:"):
        return rf"$t_3$, $\rho={c[3:]}$"
    if c.startswith("ar1:"):
        return rf"AR(1), $\rho={c[4:]}$"
    return c


def ml(m):
    """Method label."""
    return {"B-A+": r"B-A$^{+}$", "B-AQ+": r"B-A$^{+}_{q}$", "B-NQ+": r"B-N$^{+}_{q}$"}.get(m, m)


def num(x, d=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    try:
        if not np.isfinite(x):
            return "--"
    except TypeError:
        pass
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        if float(s) == 0.0:
            return s[1:]
        return "$-$" + s[1:]
    return s


def iv(lo, hi, d=3):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"[{num(lo, d)}, {num(hi, d)}]"


def mc(top, bot):
    if not bot:
        return top
    return r"\makecell{" + top + r"\\{\tiny " + bot + "}}"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.tex", "w", encoding="utf-8") as fh:
        fh.write(body)
    print("wrote", name)


def table(label, caption, colspec, header, rows, size=r"\scriptsize", sep="3pt", place="!htbp", resize=False,
          extra=""):
    lines = [rf"\begin{{table}}[{place}]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
             rf"\setlength{{\tabcolsep}}{{{sep}}}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + header + [r"\midrule"] + rows + [r"\bottomrule",
                                                                                              r"\end{tabular}"]
    if resize:
        lines.append("}")
    if extra:
        lines.append(extra)
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# =========================================================================================
# S3: weight spaces x m (E21 m slice, simulated arm)
# =========================================================================================
def s3_wspace_m():
    d = rd("sim", "e21_mslice.csv")
    d = d[d.method == "D-M"]
    rr = rd("sim", "e21_mslice_rows.csv")
    rr = rr[rr.method == "D-M"]
    full = rd("sim", "e21_ladder_seeds.csv")
    full = full[full.method == "D-M"]
    ms = [1, 2, 5, 10, 100]

    def sd(c, sp, m):
        if m == 100:
            return full[(full.cond == c) & (full.space == sp)].nmse.std(ddof=1)
        return rr[(rr.cond == c) & (rr.space == sp) & (rr.m == m)].nmse.std(ddof=1)
    rows = []
    for c in CONDS["sim"]:
        for i, sp in enumerate(SPACES):
            cells = []
            for m in ms:
                r = d[(d.cond == c) & (d.space == sp) & (d.m == m)]
                cells.append("--" if r.empty else f"{num(r.nmse.iloc[0])} ({num(sd(c, sp, m))})")
            rows.append(f"{cl(c) if i == 0 else ''} & {sp} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule" if c != CONDS["sim"][-1] else "")
    rows = [r for r in rows if r]
    cap = (r"Weight spaces and calibration budget on the simulated arm (E21): test nMSE of D-M when the weights are "
           r"fitted on the first $m$ trajectories of a calibration group (mean over five seeds and ten groups; standard "
           r"deviation over the 50 seed--group runs in parentheses); $m=100$ is the full calibration role (five "
           r"seeds; standard deviation over seeds). W0 is W-Q(5) of Table~\ref{tab:s3m}. W4 chooses its ridge parameter by leave-one-trajectory-out "
           r"cross-fitting, and by leaving out one 200-step segment when $m=1$. W5 estimates the noise variances of "
           r"the input estimator from all unlabeled calibration inputs of the condition, so only its weights depend "
           r"on $m$. W6 uses the noise-augmented members D-MA.")
    header = [r"Condition & Space & $m=1$ & $m=2$ & $m=5$ & $m=10$ & $m=100$ \\"]
    write("s3_wspace_m", table("tab:s3wm", cap, "llccccc", header, rows, sep="4pt"))


# =========================================================================================
# S7.2: E21 ladder
# =========================================================================================
METHODS = {"sim": ["D-M", "D-S", "D-T", "B-F", "B-L", "B-N", "B-A", "B-A+", "B-AQ+", "B-NQ+"],
           "sarcos": ["D-M", "D-S", "D-T", "B-F", "B-L", "B-N", "B-A", "B-A+"]}


def s7_ladder(plat, conds, part):
    d = rd(plat, "e21_ladder.csv")
    rows = []
    for c in conds:
        for i, m in enumerate(METHODS[plat]):
            cells = []
            for sp in SPACES:
                r = d[(d.cond == c) & (d.method == m) & (d.space == sp)]
                if r.empty:
                    cells.append("--")
                else:
                    r = r.iloc[0]
                    cells.append(mc(num(r.nmse_mean), iv(r.ci_lo, r.ci_hi)))
            rows.append(f"{cl(c) if i == 0 else ''} & {ml(m)} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows = rows[:-1]
    cl_txt = ", ".join(cl(c) for c in conds[:-1]) + f" and {cl(conds[-1])}"
    cap = (rf"{PNAME[plat]}, weight-space ladder (E21) under {cl_txt}: test nMSE (top: mean over five seeds; "
           r"bottom: 95\% two-level cluster-bootstrap interval over seeds and test units). Weights are fitted on the "
           r"full calibration role of the condition. For a single model W0 and W2 return the model, W1 is a scalar "
           r"gain in $[0,1]$ and W3 a per-joint gain. W5: label-free input estimator, then W1. W6: the dictionary "
           r"replaced by its noise-augmented counterpart (D-MA, D-SA). B-A$^{+}$: B-A stopped on perturbed "
           r"development inputs; subscript $q$: augmentation noise on $(q,\dot q)$ only. Dashes: not defined (W4 "
           r"needs cross-joint moments, which were stored for the core models only).")
    header = [r"Condition & Method & W0 & W1 & W2 & W3 & W4 & W5 & W6 \\"]
    lab = f"tab:s7lad{plat}{part}"
    write(f"s7_ladder_{plat}_{part}", table(lab, cap, "llccccccc", header, rows, size=r"\scriptsize", sep="2.5pt"))


def s7_geometry():
    blocks = []
    mins = []
    clean = ["D-M", "D-S", "D-T", "B-F", "B-L", "B-N"]
    for plat in PLATS:
        d = rd(plat, "e21_ladder_seeds.csv")
        d = d[d.space == "W0"]
        conds = CONDS[plat]
        g = d.groupby(["cond", "method"]).agg(s=("slope", "mean"), S=("scale_cost", "mean"),
                                              Smin=("scale_cost", "min"), beta=("beta", "mean")).reset_index()
        noisy = [c for c in conds if c not in ("N", "P1")]
        sub = g[g.method.isin(clean) & g.cond.isin(noisy)]
        k = sub.Smin.idxmin()
        mins.append((plat, sub.loc[k, "cond"], sub.loc[k, "method"], sub.loc[k, "Smin"]))
        nmax = g[g.method.isin(clean) & (g.cond == "N")].S.max()
        mins[-1] = mins[-1] + (nmax,)
        rows = [rf"\multicolumn{{{1 + 3 * len(conds)}}}{{l}}{{\emph{{{PNAME[plat]}}}}} \\"]
        for m in METHODS[plat]:
            cells = []
            for c in conds:
                r = g[(g.cond == c) & (g.method == m)]
                if r.empty:
                    cells += ["--"] * 3
                else:
                    r = r.iloc[0]
                    cells += [num(r.beta, 2), num(r.s, 2), num(r.S, 4)]
            rows.append(f"{ml(m)} & " + " & ".join(cells) + r" \\")
        blocks.append((conds, rows))
    cap_min = "; ".join(
        rf"{PNAME[p]}: smallest per-seed value under noise {num(v, 4)} ({ml(m)}, {cl(c)}), largest mean under N "
        rf"{num(nm, 4)}" for p, c, m, v, nm in mins)
    cap = (r"Geometry of the simplex solution (W0, E21): signal gain $\beta$, calibration slope $s$, and scale cost "
           r"$\Scost=\|F\|^2(1-s)^2$ of the W0 average of every method on the test role (mean over five seeds). "
           r"$s<1$: the average responds too strongly and a scalar below one lowers the risk; $s>1$: it is "
           r"attenuated too much. Clean-trained methods are D-M, D-S, D-T, B-F, B-L, and B-N. " + cap_min + ".")
    out = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{cap}}}", r"\label{tab:s7geo}", r"\scriptsize",
           r"\setlength{\tabcolsep}{2pt}"]
    for i, (conds, rows) in enumerate(blocks):
        ncol = 3 * len(conds)
        out.append(r"\begin{tabular}{l" + "ccc" * len(conds) + "}")
        out.append(r"\toprule")
        out.append(" & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{cl(c)}}}" for c in conds) + r" \\")
        out.append("".join(rf"\cmidrule(lr){{{2 + 3 * j}-{4 + 3 * j}}}" for j in range(len(conds))))
        out.append("Method & " + " & ".join([r"$\beta$ & $s$ & $\Scost$"] * len(conds)) + r" \\")
        out.append(r"\midrule")
        out += rows
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        if i == 0:
            out.append(r"\vspace{4pt}")
            out.append("")
    out.append(r"\end{table}")
    write("s7_geometry", "\n".join(out) + "\n")


def s7_paired():
    rows = []
    for plat in PLATS:
        d = rd(plat, "e21_paired.csv")
        rows.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{PNAME[plat]}}}}} \\")
        for c in CONDS[plat]:
            cells = []
            for sp in SPACES:
                r = d[(d.cond == c) & (d.a == f"D-M|{sp}") & (d.b == f"D-S|{sp}")]
                if r.empty:
                    cells.append("--")
                else:
                    r = r.iloc[0]
                    cells.append(mc(f"{num(r['diff'])} ({int(r.seeds_a_better)}/{int(r.n_seeds)})",
                                    iv(r.ci_lo, r.ci_hi, 4)))
            rows.append(f"{cl(c)} & " + " & ".join(cells) + r" \\")
        if plat == "sim":
            rows.append(r"\midrule")
    cap = (r"Paired difference of test nMSE, D-M minus D-S, in each weight space (E21): mean over seeds and test "
           r"units, number of seeds (of five) in which D-M is better in parentheses, and 95\% cluster-bootstrap "
           r"interval below. Negative values favor D-M. The intervals share the bootstrap indices of the two "
           r"dictionaries.")
    header = [r"Condition & W0 & W1 & W2 & W3 & W4 & W5 & W6 \\"]
    write("s7_paired", table("tab:s7pairs", cap, "lccccccc", header, rows, sep="2pt"))


ATTR_ROWS = [
    ("margin_W0", r"Margin on W0", 4),
    ("scale_part", r"Scale part", 4),
    ("angle_part", r"Angle part (margin on W1)", 4),
    ("scale_share", r"Scale share", 2),
    ("rem_perjoint_W2", r"Margin on W2 (per joint)", 4),
    ("rem_perjoint_W3", r"Margin on W3 (per joint, null)", 4),
    ("rem_stack_W4", r"Margin on W4 (stacking)", 4),
    ("rem_est_W1", r"Margin on W5 (estimator, W1)", 4),
    ("rem_est_W0", r"Margin, estimator and W0", 4),
    ("rem_est_oracle_W1", r"Margin, oracle estimator and W1", 4),
    ("rem_aug_W6", r"Margin on W6 (augmented)", 4),
    ("scale_cost_DS", r"Relaxation gain of D-S", 4),
    ("scale_cost_DM", r"Relaxation gain of D-M", 4),
]


def s7_attribution():
    out = [r"\begin{table}[!htbp]", r"\centering",
           r"\caption{Attribution of the margin of D-M over D-S (E21), margin $=R(\text{D-S})-R(\text{D-M})$ on the "
           r"test role, so positive values favor D-M: mean over seeds and test units with the 95\% cluster-bootstrap "
           r"interval below. Scale part: difference of the two relaxation gains $R(\mathrm{W0})-R(\mathrm{W1})$ "
           r"(D-S minus D-M); angle part: the margin left on W1; scale share: scale part over margin (meaningful "
           r"only where the margin is clearly positive). The rows below give the margin left in each further "
           r"weight space. Removed: fraction of the angle part removed, $1-$(margin left)/(angle part), shown "
           r"where the angle part has an interval above zero.}",
           r"\label{tab:s7attr}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}"]
    for bi, plat in enumerate(PLATS):
        d = rd(plat, "e21_attribution.csv").set_index("cond")
        conds = CONDS[plat]
        out.append(r"\begin{tabular}{l" + "c" * len(conds) + "}")
        out.append(r"\toprule")
        out.append(rf"\emph{{{PNAME[plat]}}} & " + " & ".join(cl(c) for c in conds) + r" \\")
        out.append(r"\midrule")
        for key, lab, dd in ATTR_ROWS:
            cells = []
            for c in conds:
                if c not in d.index or key not in d.columns or not np.isfinite(d.loc[c, key]):
                    cells.append("--")
                    continue
                v, lo, hi = d.loc[c, key], d.loc[c, key + "_lo"], d.loc[c, key + "_hi"]
                if key == "scale_share" and not (d.loc[c, "margin_W0_lo"] > 0):
                    cells.append("--")
                    continue
                cells.append(mc(num(v, dd), iv(lo, hi, dd)))
            out.append(f"{lab} & " + " & ".join(cells) + r" \\")
            if key in ("scale_share", "rem_aug_W6"):
                out.append(r"\midrule")
        # removed fractions
        for key, lab in (("rem_perjoint_W3", "W3"), ("rem_stack_W4", "W4"), ("rem_est_W1", "W5")):
            cells = []
            for c in conds:
                ok = c in d.index and d.loc[c, "angle_part_lo"] > 0
                cells.append(num(100 * (1 - d.loc[c, key] / d.loc[c, "angle_part"]), 0) + r"\%" if ok else "--")
            out.append(f"Angle part removed by {lab} & " + " & ".join(cells) + r" \\")
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        if bi == 0:
            out.append(r"\vspace{4pt}")
            out.append("")
    out.append(r"\end{table}")
    write("s7_attribution", "\n".join(out) + "\n")


def s7_width():
    """Width dictionaries D-W (standard), D-W4 (without STE), D-Wd / D-Wd4 (detached reset)."""
    rows = []
    fams = ["tri", "sig", "atan", "sre"]
    fname = {"tri": "triangular", "sig": "sigmoid", "atan": "ATan", "sre": "escape-rate"}
    show = ["O1-M", "O1-H"]
    for plat in PLATS:
        L = rd(plat, "e21_ladder_w16.csv")
        A = rd(plat, "e21_attribution_w16.csv")
        P = rd(plat, "e21_paired_w16.csv")

        def nm(m, c, sp):
            r = L[(L.cond == c) & (L.method == m) & (L.space == sp)]
            return "--" if r.empty else num(r.nmse_mean.iloc[0])

        def pair(c, a, b, sp, sign=1.0):
            r = P[(P.cond == c) & (P.a == f"{a}|{sp}") & (P.b == f"{b}|{sp}")]
            if r.empty:
                return None
            r = r.iloc[0]
            v, lo, hi = sign * r["diff"], sign * r.ci_lo, sign * r.ci_hi
            return v, min(lo, hi), max(lo, hi)

        def cellp(t):
            return "--" if t is None else mc(num(t[0]), iv(t[1], t[2], 4))

        rows.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{PNAME[plat]}}}}} \\")
        for c in show:
            first = True
            entries = [("D-M", "D-M", None), ("D-S", "D-S", None)]
            entries += [(f"D-W_{f}", f"D-W, {fname[f]}", "attr") for f in fams]
            entries += [(f"D-W4_{f}", f"D-W$_4$, {fname[f]}", "pair") for f in fams]
            entries += [(f"D-Wd_{f}", f"D-Wd, {fname[f]}", "attr") for f in ("tri", "atan")]
            entries += [(f"D-Wd4_{f}", f"D-Wd$_4$, {fname[f]}", "pair") for f in ("tri", "atan")]
            for dn, lab, kind in entries:
                cells = [nm(dn, c, sp) for sp in ("W0", "W1", "W3")]
                if kind is None:
                    cells += ["--"] * 5
                else:
                    if kind == "attr":
                        a = A[(A.cond == c) & (A.dict == dn)].iloc[0]
                        m0 = (a.margin_W0, a.margin_W0_lo, a.margin_W0_hi)
                        m1 = (a.angle_part, a.angle_part_lo, a.angle_part_hi)
                        share = a.scale_share
                    else:
                        m0 = pair(c, dn, "D-S", "W0", -1.0)
                        m1 = pair(c, dn, "D-S", "W1", -1.0)
                        share = (m0[0] - m1[0]) / m0[0] if m0 is not None and m1 is not None else np.nan
                    cells.append(cellp(m0))
                    cells.append(num(share, 2) if m0 is not None and m0[1] > 0 else "--")
                    cells.append(cellp(m1))
                    cells.append(cellp(pair(c, dn, "D-M", "W1")))
                    if dn.startswith("D-Wd_"):
                        cells.append(cellp(pair(c, dn, "D-W_" + dn.split("_")[1], "W1")))
                    else:
                        cells.append("--")
                rows.append(f"{cl(c) if first else ''} & {lab} & " + " & ".join(cells) + r" \\")
                first = False
            rows.append(r"\midrule")
    rows = rows[:-1]
    cap = (r"Width dictionaries (E21 with the E16 members, seeds 0--2 for every dictionary). D-W of one surrogate "
           r"family holds the members of half width at half maximum 0.0625, 0.25, 1, and 4 and the straight-through "
           r"member of D-M of the same seed; D-W$_4$ drops the straight-through member. D-Wd and D-Wd$_4$ repeat D-W "
           r"and D-W$_4$ with members trained with the reset term detached from the gradient (and a detached "
           r"straight-through member). Columns: test nMSE (mean over seeds); the margin over D-S on W0, "
           r"$R(\text{D-S})-R(\cdot)$, its scale share, and the angle part (the margin left on W1) as in "
           r"Table~\ref{tab:s7attr}; the paired difference to D-M on W1 and, for D-Wd, to D-W of the same family on "
           r"W1 (positive: worse than the reference). Intervals are 95\% cluster-bootstrap intervals over seeds and "
           r"test units. The conditions N, O1-L, O2-B, P1, and O1-M $(q,\dot q)$ are in the code release.")
    header = [r" & & \multicolumn{3}{c}{Test nMSE} & \multicolumn{3}{c}{Margin over D-S} & "
              r"\multicolumn{2}{c}{Difference on W1} \\",
              r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
              r"Condition & Dictionary & W0 & W1 & W3 & on W0 & share & angle & $-$ D-M & $-$ D-W \\"]
    write("s7_width", table("tab:s7width", cap, "llcccccccc", header, rows, size=r"\scriptsize", sep="2pt"))


def s7_bestwidth():
    """Risk-minimizing single width along the noise path (E22 with the E16 members)."""
    rhos = [0.0, 0.05, 0.1, 0.15, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.8, 0.85, 0.9, 1.0, 1.2]
    cols = [("standard_3seeds", f) for f in ("tri", "sig", "atan", "sre")] + \
           [("detach_3seeds", f) for f in ("tri", "atan")]
    B = {p: rd(p, "e22_bestwidth_w16.csv") for p in PLATS}

    def fmt(h):
        if not np.isfinite(h):
            return "STE"
        return f"{h:g}"

    rows = []
    for r in rhos:
        cells = []
        for p in PLATS:
            for var, f in cols:
                x = B[p][(B[p].variant == var) & (B[p].family == f) & (np.isclose(B[p].rho, r))]
                cells.append("--" if x.empty else fmt(float(x.best_hwhm.iloc[0])))
        rows.append(f"{r:g} & " + " & ".join(cells) + r" \\")
    cap = (r"Risk-minimizing single width along the noise path (E22 with the E16 members): half width at half "
           r"maximum of the member with the lowest seed-mean test nMSE (seeds 0--2) at each level. Standard "
           r"surrogates: grid 0.0625--4 and the straight-through member (STE); detached versions (subscript d): "
           r"0.0625, 0.25, 1, 4, 16 and a detached straight-through member. Omitted levels repeat the value of the "
           r"level above them. The run of seed 0 with the ultra-wide members 8--64 is in the code release; no "
           r"ultra-wide member is best at any level. Dash: beyond the SARCOS grid.")
    header = [r" & \multicolumn{6}{c}{Simulated arm} & \multicolumn{6}{c}{SARCOS} \\",
              r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
              r"$\rho$ & tri. & sig. & ATan & esc. & tri.$_d$ & ATan$_d$ & tri. & sig. & ATan & esc. & tri.$_d$ & "
              r"ATan$_d$ \\"]
    write("s7_bestwidth", table("tab:s7bw", cap, "l" + "c" * 12, header, rows, sep="3pt"))


E18_ROWS = [("w18_aug0.25", r"input augmentation, $0.25\times$"), ("w18_aug0.5", r"input augmentation, $0.5\times$"),
            ("w18_aug1", r"input augmentation, $1\times$"), ("w18_aug1.4", r"input augmentation, $1.4\times$"),
            ("w18_mem0.25", r"membrane noise, $\sigma=0.25$"), ("w18_mem0.5", r"membrane noise, $\sigma=0.5$"),
            ("w18_mem1", r"membrane noise, $\sigma=1$"), ("w18_mem2", r"membrane noise, $\sigma=2$"),
            ("w18_esc_h0.25", r"escape noise, HWHM 0.25"), ("w18_esc_h1", r"escape noise, HWHM 1"),
            ("w18_mlp_tanh", r"MLP, tanh"), ("w18_mlp_crelu", r"MLP, clipped ReLU"), ("w18_mlp_bin", r"MLP, binary"),
            ("w17_tri_L1h1", r"E17: layer 1 wide, HWHM 1"), ("w17_tri_L1h4", r"E17: layer 1 wide, HWHM 4"),
            ("w17_tri_L23h1", r"E17: layers 2--3 wide, HWHM 1"), ("w17_tri_L23h4", r"E17: layers 2--3 wide, HWHM 4")]


def s7_e18():
    E = {p: rd(p, "e21_ladder_e18.csv") for p in PLATS}
    R = {p: rd(p, "e21_ladder.csv") for p in PLATS}
    rows = []

    def block(df, name):
        g = lambda c, sp: df[(df.cond == c) & (df.method == name) & (df.space == sp)]
        n0, m0, m1 = g("N", "W0"), g("O1-M", "W0"), g("O1-M", "W1")
        if m0.empty:
            return ["--"] * 5
        return [num(n0.nmse_mean.iloc[0], 3) if not n0.empty else "--", num(m0.beta.iloc[0], 2),
                num(m0.nu2.iloc[0], 3), num(m0.nmse_mean.iloc[0], 3), num(m1.nmse_mean.iloc[0], 3)]

    for key, lab in E18_ROWS:
        cells = []
        for p in PLATS:
            cells += block(E[p], key)
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if key in ("w18_aug1.4", "w18_mem2", "w18_esc_h1", "w18_mlp_bin"):
            rows.append(r"\midrule")
    rows.append(r"\midrule")
    for key, lab in (("B-N", "B-N (ReLU MLP, reference)"), ("B-A", "B-A (reference)"), ("D-S", "D-S (reference)"),
                     ("D-M", "D-M (reference)")):
        cells = []
        for p in PLATS:
            cells += block(R[p], key)
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    cap = (r"E18 and E17 single models in the weight-space ladder (three seeds each; the references are the five-seed "
           r"results of Tables~\ref{tab:s7ladsima}--\ref{tab:s7ladsarcosb}): test nMSE under N, and at O1-M the "
           r"signal gain $\beta$ and orthogonal noise power $\|\nu\|^2$ of the model (W0) with its test nMSE on W0 "
           r"and with a scalar gain (W1). Input augmentation is given as a multiple of the B-A augmentation level, "
           r"membrane noise as the standard deviation of the first-layer membrane noise during training, and escape "
           r"noise as the half width of the stochastic forward pass used in training (evaluated deterministically); "
           r"the MLPs have bounded activations (binary with a saturating straight-through gradient). E17 rows: "
           r"triangular surrogate of the stated width in the named layers and of width 0.25 elsewhere. Definitions "
           r"and the E16 amplitude controls are in S5.")
    header = [r" & \multicolumn{5}{c}{Simulated arm} & \multicolumn{5}{c}{SARCOS} \\",
              r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
              r" & N & \multicolumn{4}{c}{O1-M} & N & \multicolumn{4}{c}{O1-M} \\",
              r"\cmidrule(lr){2-2}\cmidrule(lr){3-6}\cmidrule(lr){7-7}\cmidrule(lr){8-11}",
              r"Model & W0 & $\beta$ & $\|\nu\|^2$ & W0 & W1 & W0 & $\beta$ & $\|\nu\|^2$ & W0 & W1 \\"]
    write("s7_e18", table("tab:s7e18", cap, "l" + "c" * 10, header, rows, sep="3pt"))


# =========================================================================================
# S7.5: E22 noise path
# =========================================================================================
def ev_name(row):
    mn = row.get("member_name")
    if not isinstance(mn, str):
        return "null"
    if "_ste_" in mn:
        return "straight-through"
    for k, v in (("atan", "ATan"), ("sigmoid", "sigmoid"), ("triangular", "triangular"), ("sre", "escape-rate")):
        if f"_{k}_" in mn:
            return "a seed member" if mn.startswith("ds_") else v
    return mn


def s7_critical():
    rows = []
    summary = []
    for plat in PLATS:
        a = rd(plat, "e22_critical.csv")
        b = rd(plat, "e22_critical_lqc.csv")
        for df in (a, b):
            df["ev"] = df.apply(ev_name, axis=1)
            df["key"] = df.dict + "|" + df.space + "|" + df.ev + "|" + df.direction
        keys = list(dict.fromkeys(list(a.sort_values(["direction", "dict", "space"]).key) + list(b.key)))
        order = lambda k: (k.split("|")[3] != "enter", k.split("|")[2] != "straight-through", k)
        keys = sorted(keys, key=order)
        rows.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{PNAME[plat]}}}}} \\")
        for k in keys:
            dn, sp, ev, dr = k.split("|")
            cells = []
            for s in range(5):
                ra = a[(a.key == k) & (a.seed == s)]
                rb = b[(b.key == k) & (b.seed == s)]
                if ra.empty and rb.empty:
                    cells.append("")
                    continue
                meas = ra.rho_measured.iloc[0] if not ra.empty else (rb.rho_measured.iloc[0] if not rb.empty else np.nan)
                pa = ra.rho_predicted.iloc[0] if not ra.empty else np.nan
                pb = rb.rho_predicted.iloc[0] if not rb.empty else np.nan
                cells.append(f"{num(meas, 4)}/{num(pa, 4)}/{num(pb, 4)}")
            sa, sb = a[(a.key == k) & a.rho_measured.notna()], b[(b.key == k) & b.rho_measured.notna()]
            fa = f"{int(sa.within_one_step.sum())}/{len(sa)}" if len(sa) else "--"
            fb = f"{int(sb.within_one_step.sum())}/{len(sb)}" if len(sb) else "--"
            mxa = num(sa.abs_diff.max(), 3) if sa.abs_diff.notna().any() else "--"
            mxb = num(sb.abs_diff.max(), 3) if sb.abs_diff.notna().any() else "--"
            verb = "enters" if dr == "enter" else "leaves"
            lab = f"{dn}, {sp}: {ev} {verb}"
            rows.append(f"{lab} & " + " & ".join(cells) + f" & {mxa} & {fa} & {mxb} & {fb}" + r" \\")
        for fam, df in (("lq", a), ("lqc", b)):
            mm = df[df.rho_measured.notna()]
            key = mm[(mm.direction == "enter")]
            summary.append((plat, fam, int(key.within_one_step.sum()), len(key), int(mm.within_one_step.sum()), len(mm)))
        if plat == "sim":
            rows.append(r"\midrule")
    sm = "; ".join(f"{PNAME[p]}, {'linear-plus-quadratic' if f == 'lq' else 'cubic'}: entries {e1}/{n1}, all {e2}/{n2}"
                   for p, f, e1, n1, e2, n2 in summary)
    fit = {p: json.load(open(RES / p / "e22_fit.json")) for p in PLATS}
    cap = (r"Critical noise levels of the calibrated weights (E22). Each cell gives, for one seed, the level at "
           r"which the event occurs on the measured path, the level predicted by the linear-plus-quadratic "
           r"interpolation of $C(\rho)$ (the family chosen by leave-one-level-out), and the level predicted by the "
           r"cubic interpolation (sensitivity run), as measured/predicted/predicted; blank: the event does not "
           r"occur for that seed; dash: it occurs on one path only. Max $|\Delta|$ and the count within one grid "
           r"step (0.05) are over the seeds with a measured event. Fit levels: "
           + ", ".join(str(x) for x in fit["sim"]["fit_levels"]) + " (arm) and "
           + ", ".join(str(x) for x in fit["sarcos"]["fit_levels"]) + r" (SARCOS). Within one step: " + sm + ".")
    header = [r" & \multicolumn{5}{c}{Seed: measured/predicted (lin.-quad.)/predicted (cubic)} & "
              r"\multicolumn{2}{c}{Lin.-quad.} & \multicolumn{2}{c}{Cubic} \\",
              r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
              r"Event & 0 & 1 & 2 & 3 & 4 & max $|\Delta|$ & within & max $|\Delta|$ & within \\"]
    write("s7_critical", table("tab:s7crit", cap, "l" + "c" * 9, header, rows, size=r"\tiny", sep="2pt"))


def s7_crossings():
    lab = {"DM_vs_DS_W0": r"D-M and D-S, W0", "DM_vs_DS_W1": r"D-M and D-S, W1",
           "STE_vs_best_window": r"straight-through and best window member", "STE_vs_MLP": r"straight-through and B-N"}
    rows = []
    for plat in PLATS:
        d = rd(plat, "e22_crossings.csv")
        w = rd(plat, "e22_crossings_w16.csv")
        rows.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{PNAME[plat]}}}}} \\")
        for t in ["DM_vs_DS_W0", "DM_vs_DS_W1", "STE_vs_best_window", "STE_vs_MLP"]:
            cells = []
            for s in range(5):
                r = d[(d.type == t) & (d.seed == str(s))]
                cells.append(num(float(r.rho_cross.iloc[0]), 3) if len(r) else "--")
            r = d[(d.type == t) & (d.seed == "mean_curve")]
            mcv = num(float(r.rho_cross.iloc[0]), 3) if len(r) else ""
            rows.append(f"{lab[t]} & " + " & ".join(cells) + f" & {mcv}" + r" \\")
        rows.append(r"\emph{mean curves of the width dictionaries} & triangular & sigmoid & ATan & escape-rate & & \\")
        for sp in ("W0", "W1"):
            vals = []
            for f in ("tri", "sig", "atan", "sre"):
                r = w[(w.a == f"D-W_{f}") & (w.b == "D-S") & (w.space == sp)]
                v = float(r.rho_cross_mean_curve.iloc[0]) if len(r) else np.nan
                vals.append(num(v, 3))
            rows.append(f"D-W and D-S, {sp} & " + " & ".join(vals) + r" & & \\")
        if plat == "sim":
            rows.append(r"\midrule")
    cap = (r"Crossing levels of test-risk curves along the noise path (E22): the level $\rho$ at which the first "
           r"curve falls below the second, per seed and for the seed-mean curves (the mean curve of the "
           r"straight-through/B-N pair was not formed). Best window member: the window member of D-M with the "
           r"lowest risk at that level. Width-dictionary rows: crossing of the seed-mean curves of D-W and D-S for "
           r"each surrogate family (seeds as in Table~\ref{tab:s7width}).")
    header = [r"Curves & seed 0 & 1 & 2 & 3 & 4 & mean curve \\"]
    write("s7_crossings", table("tab:s7cross", cap, "lcccccc", header, rows, sep="3pt"))


KTYPES = [("bn_mlp", "B-N (perceptron)"), ("dm_atan", "D-M, ATan"), ("dm_sigmoid", "D-M, sigmoid"),
          ("dm_triangular", "D-M, triangular"), ("dm_sre", "D-M, escape-rate"), ("dm_ste", "D-M, straight-through"),
          ("ds_", "D-S members"), ("dt_", "D-T members"), ("bl_learnable", "B-L"),
          ("dma_atan", "D-MA, ATan"), ("dma_sigmoid", "D-MA, sigmoid"), ("dma_triangular", "D-MA, triangular"),
          ("dma_sre", "D-MA, escape-rate"), ("dma_ste", "D-MA, straight-through"), ("dsa_", "D-SA members"),
          ("dmas_", r"B-A$^{+}$"), ("dmqs_", r"B-A$^{+}_{q}$"), ("dmq_ste", r"D-MA$_q$, straight-through"), ("dmq_win", r"D-MA$_q$, window members"),
          ("dsq_", r"D-SA$_q$ members"), ("bnqs_mlp", r"B-N$^{+}_{q}$"),
          ("D-M|W0", "D-M, W0 average"), ("D-M|W1", "D-M, W1 average"), ("D-S|W0", "D-S, W0 average"),
          ("D-S|W1", "D-S, W1 average")]


def ktype_rows(d, key):
    if key == "dmq_win":
        return d[d.type.str.startswith("dmq_") & ~d.type.str.startswith("dmq_ste") & ~d.type.str.startswith("dmqs")]
    if key.endswith("_"):
        return d[d.type.str.startswith(key)]
    return d[d.type == key]


def s7_kappa():
    data = {p: rd(p, "e22_kappa.csv") for p in PLATS}
    levels = {"sim": ["0.35", "0.7", "1.0"], "sarcos": ["0.25", "0.5", "0.7"]}
    rows = []
    for key, lab in KTYPES:
        cells = []
        any_ = False
        for p in PLATS:
            r = ktype_rows(data[p], key)
            if r.empty:
                cells += ["--"] * 5
                continue
            any_ = True
            cells += [num(r.kappa_full.mean(), 2), num(r.kappa_frozen.mean(), 2)]
            for lv in levels[p]:
                col = f"kappa_local_{lv}"
                cells.append(num(r[col].mean(), 2) if col in r and r[col].notna().any() else "--")
        if any_:
            rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if key in ("bl_learnable", "bnqs_mlp"):
            rows.append(r"\midrule")
    cap = (r"Noise-scaling exponent $\kappa$ of the test risk, $R(\rho)-R(0)\propto\rho^{\kappa}$ (E22, paired "
           r"noise, mean over seeds and members of a type). Full: log--log least-squares slope over all grid levels "
           r"$0.05\le\rho\le1.2$ (arm) and $0.05\le\rho\le0.9$ (SARCOS); frozen: over the frozen levels "
           r"$\{0.35,0.7,1.0\}$ (arm) and $\{0.5,0.7\}$ (SARCOS); local: central difference over $\rho\pm0.05$ at the "
           r"stated level. $\kappa=2$ is the quadratic growth of a smooth map. On each platform B-A is the "
           r"development-selected member of D-MA. Subscript $q$: augmentation on $(q,\dot q)$ only (arm).")
    header = [r" & \multicolumn{5}{c}{Simulated arm} & \multicolumn{5}{c}{SARCOS} \\",
              r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
              r"Predictor & full & frozen & 0.35 & 0.7 & 1.0 & full & frozen & 0.25 & 0.5 & 0.7 \\"]
    write("s7_kappa", table("tab:s7kappa", cap, "lcccccccccc", header, rows, sep="3pt"))


def s7_wtable():
    rhos = {"sim": [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2],
            "sarcos": [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 0.9]}
    W = {}
    for p in PLATS:
        d = rd(p, "e22_wtable.csv")
        A = np.array([json.loads(x) for x in d.weights])
        d = d.assign(ste=A[:, 2], null=A[:, 5])
        W[p] = d.groupby(["name", "space", "rho"])[["ste", "null"]].mean()
    allr = sorted(set(rhos["sim"]) | set(rhos["sarcos"]))
    rows = []
    for r in allr:
        cells = []
        for p in PLATS:
            if r not in rhos[p]:
                cells += ["--"] * 4
                continue
            g = W[p]
            rr = round(r, 2)
            cells += [num(g.loc[("D-M", "W0", rr), "ste"], 2), num(g.loc[("D-M", "W1", rr), "ste"], 2),
                      num(g.loc[("D-M", "W1", rr), "null"], 2), num(g.loc[("D-S", "W1", rr), "null"], 2)]
        rows.append(f"{r:g} & " + " & ".join(cells) + r" \\")
    cap = (r"Weight table $w^\star(\rho)$ along the noise path (E22): mean over five seeds of the calibrated weight "
           r"on the straight-through member of D-M (STE) and on the null predictor, on W0 and W1. The complete "
           r"weight vectors of every seed, dictionary and level are in the code release; the table is the "
           r"lookup used by the weight-table source of E23.")
    header = [r" & \multicolumn{4}{c}{Simulated arm} & \multicolumn{4}{c}{SARCOS} \\",
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
              r" & \multicolumn{2}{c}{D-M, STE} & D-M & D-S & \multicolumn{2}{c}{D-M, STE} & D-M & D-S \\",
              r"$\rho$ & W0 & W1 & null & null & W0 & W1 & null & null \\"]
    write("s7_wtable", table("tab:s7wtab", cap, "lcccccccc", header, rows, sep="4pt"))


# =========================================================================================
# S7.6: E23 label-free calibration
# =========================================================================================
LF_SOURCES = ["S0", "S2", "S2b", "S3", "S5", "TAB", "TABs"]
LF_CONDS = {"sim": ["O1-L", "O1-M", "O1-H", "O2B", "t3:0.35", "t3:0.7", "t3:1.0", "ar1:0.35", "ar1:0.7", "ar1:1.0",
                    "P1"],
            "sarcos": ["O1-L", "O1-M", "O1-H", "O1qqd", "t3:0.5", "t3:0.7", "ar1:0.5", "ar1:0.7"]}


def s7_labelfree(plat):
    d = rd(plat, "e23_labelfree.csv")
    d = d[d.space == "W1"]
    srcs = [s for s in LF_SOURCES if s in set(d.source)]
    rows = []
    for c in LF_CONDS[plat]:
        for i, m in enumerate(["D-M", "D-S"]):
            x = d[(d.cond == c) & (d.method == m)]
            if x.empty:
                continue
            s1 = x.iloc[0]
            cells = [num(s1.nmse_S1), num(s1.seed_sd_S1)]
            for s in srcs:
                r = x[x.source == s]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                top = num(r.diff_vs_S1)
                if s in ("S2", "S2b"):
                    cells.append(mc(top, iv(r.diff_ci_lo, r.diff_ci_hi, 4)))
                else:
                    cells.append(top)
            rows.append(f"{cl(c) if i == 0 else ''} & {m} & " + " & ".join(cells) + r" \\")
        if c in {"sim": ("O2B", "t3:1.0", "ar1:1.0"), "sarcos": ("O1qqd", "t3:0.7")}[plat]:
            rows.append(r"\midrule")
    srclab = {"S0": "S0", "S2": r"S2 ($\hat v$)", "S2b": "S2b (bias)", "S3": "S3 (true)", "S5": "S5 (rest)",
              "TAB": "TAB", "TABs": "TAB (rest)"}
    cap = (rf"{PNAME[plat]}, calibration without target labels (E23), W1: test nMSE of target-labeled calibration "
           r"(S1, full calibration role) with its seed standard deviation, and the difference of each source to "
           r"S1 (positive: worse than S1). S0: nominal labeled data, no noise. S2: nominal labeled data with "
           r"simulated diagonal Gaussian noise of the variance $\hat v$ estimated from the unlabeled target "
           r"calibration inputs; S2b: a simulated per-trajectory bias of variance $\hat v$ (O2-B). S3: as S2 with "
           r"the true variance. S5: as S2 with the variance of a 10-s recording at rest observed through the "
           r"target noise. TAB: the weight table of Table~\ref{tab:s7wtab} at $\hat\rho$ from $\hat v$; TAB (rest): "
           r"at $\hat\rho$ from the recording at rest. The interval below S2 is the 95\% paired cluster-bootstrap "
           r"interval. $t_3$ and AR(1): Student-$t$ noise with three degrees of freedom and channel-correlated "
           r"Gaussian noise, both simulated as diagonal Gaussian by S2--S5."
           + (" P1 is the payload change (plant shift)." if plat == "sim" else ""))
    header = ["Condition & Dict. & S1 & sd & " + " & ".join(srclab[s] for s in srcs) + r" \\"]
    write(f"s7_labelfree_{plat}", table(f"tab:s7lf{plat}", cap, "ll" + "c" * (2 + len(srcs)), header, rows,
                                        sep="2.5pt"))


def s7_rho():
    rows = []
    for p in PLATS:
        d = rd(p, "e23_labelfree.csv")
        x = d[(d.method == "D-M") & (d.space == "W1") & (d.source == "S1")]
        rows.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{PNAME[p]}}}}} \\")
        for c in LF_CONDS[p]:
            r = x[x.cond == c]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(f"{cl(c)} & {num(r.rho_true, 2)} & {num(r.rho_hat, 3)} & {num(r.rho_static, 3)}" + r" \\")
        if p == "sim":
            rows.append(r"\midrule")
    cap = (r"Noise level estimated without labels (E23): $\hat\rho=\sqrt{\text{mean of }\hat v}$ over the noisy "
           r"channels of the platform from the unlabeled calibration inputs (Eq.~\eqref{eq:s8wiener}) and from the "
           r"10-s recording at rest. Under O1-M $(q,\dot q)$ on SARCOS only 14 of the 21 channels are noisy, so "
           r"$\hat\rho$ averages over channels without noise; under P1 the input distribution changes without "
           r"noise.")
    header = [r"Condition & $\rho$ & $\hat\rho$ ($\hat v$) & $\hat\rho$ (rest) \\"]
    write("s7_rho", table("tab:s7rho", cap, "lccc", header, rows, sep="5pt"))


# =========================================================================================
# S7.7: E24 certificate, E25 budget
# =========================================================================================
def s7_cert():
    rows = []
    msets = {}
    for p in PLATS:
        d = rd(p, "e24_certificate.csv")
        ms = sorted(d.m.unique())
        msets[p] = ms
    allm = sorted(set(msets["sim"]) | set(msets["sarcos"]))
    for p in PLATS:
        d = rd(p, "e24_certificate.csv")
        rows.append(rf"\multicolumn{{{2 + len(allm)}}}{{l}}{{\emph{{{PNAME[p]}}} ($m$ in "
                    + ("trajectories" if p == "sim" else "100-row blocks") + r")} \\")
        for c in CONDS[p]:
            first = True
            for dn in ("D-M", "D-S"):
                for sp in ("W0", "W1"):
                    cells = []
                    for m in allm:
                        r = d[(d.cond == c) & (d.dict == dn) & (d.space == sp) & (d.m == m)]
                        cells.append("" if r.empty else f"{num(r.cert_rate_t.iloc[0], 2)}/{num(r.coverage_t_held.iloc[0], 2)}")
                    rows.append(f"{cl(c) if first else ''} & {dn}, {sp} & " + " & ".join(cells) + r" \\")
                    first = False
        if p == "sim":
            rows.append(r"\midrule")
    cap = (r"Split-sample certificate of the calibration gain (E24), one-sided Student-$t$ bound at 95\%: "
           r"certification rate (fraction of cases with a positive lower bound on the gain of the deployed "
           r"half-sample weights over the best single member of the same half) and empirical coverage (fraction "
           r"of cases in which the bound does not exceed the reference gain, computed on the test role pooled "
           r"with the calibration units the case did not use), as rate/coverage. Cases: five seeds, ten "
           r"calibration groups (arm) or disjoint block sets (SARCOS), and both directions of the cross-fit; "
           r"$m=20$, 40, 100 on the arm merge groups, which leaves 100, 50, 20, and 10 cases for "
           r"$m\le10$, 20, 40, 100. Blank: not run.")
    header = [r"Condition & Dict., space & " + " & ".join(f"$m={m}$" for m in allm) + r" \\"]
    write("s7_cert", table("tab:s7cert", cap, "ll" + "c" * len(allm), header, rows, sep="3pt"))


def s7_certboot():
    rows = []
    for p in PLATS:
        d = rd(p, "e24_certificate.csv")
        ms = sorted(d.m.unique())
        rows.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{PNAME[p]}}}}} \\")
        for c in CONDS[p]:
            for sp in ("W0", "W1"):
                cells = []
                for m in [4, 6, 8, 10, 20, 40, 100]:
                    r = d[(d.cond == c) & (d.dict == "D-M") & (d.space == sp) & (d.m == m)]
                    cells.append("" if r.empty else
                                 f"{num(r.cert_rate_boot.iloc[0], 2)}/{num(r.coverage_boot_held.iloc[0], 2)}")
                rows.append(f"{cl(c) if sp == 'W0' else ''} & {sp} & " + " & ".join(cells) + r" \\")
        if p == "sim":
            rows.append(r"\midrule")
    cap = (r"As Table~\ref{tab:s7cert} for D-M with the percentile-bootstrap lower bound (2{,}000 resamples of the "
           r"per-unit gains of the second half): certification rate/coverage.")
    header = [r"Condition & Space & " + " & ".join(f"$m={m}$" for m in [4, 6, 8, 10, 20, 40, 100]) + r" \\"]
    write("s7_certboot", table("tab:s7certboot", cap, "llccccccc", header, rows, sep="3pt"))


def s7_minm():
    rows = []
    for p in PLATS:
        d = rd(p, "e24_min_m.csv")
        c24 = rd(p, "e24_certificate.csv")
        rows.append(rf"\multicolumn{{9}}{{l}}{{\emph{{{PNAME[p]}}}}} \\")
        for c in CONDS[p]:
            cells = []
            for meth in ("t", "boot"):
                for dn in ("D-M", "D-S"):
                    for sp in ("W0", "W1"):
                        r = d[(d.cond == c) & (d.dict == dn) & (d.space == sp) & (d.method == meth)]
                        v = r.min_m_90.iloc[0] if len(r) else np.nan
                        mx = c24.m.max()
                        cells.append("--" if not np.isfinite(v) else (f"$>{mx}$" if v > mx else f"{int(v)}"))
            rows.append(f"{cl(c)} & " + " & ".join(cells) + r" \\")
        if p == "sim":
            rows.append(r"\midrule")
    h = c24 = rd("sim", "e24_certificate.csv")
    hmax = h.hoeffding_lb_max.max()
    cap = (r"Smallest calibration budget $m$ on the grid of Table~\ref{tab:s7cert} with a certification rate of at "
           r"least 0.9 (E24), for the Student-$t$ and the percentile-bootstrap bound; $m$ counts trajectories on "
           r"the arm and 100-row blocks on SARCOS. For comparison, the Hoeffding bound on the entries of "
           rf"$\widehat C$ is negative in every case on the arm (largest value {num(hmax, 0)}, at $m=100$).")
    header = [r" & \multicolumn{4}{c}{Student $t$} & \multicolumn{4}{c}{Bootstrap} \\",
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
              r" & \multicolumn{2}{c}{D-M} & \multicolumn{2}{c}{D-S} & \multicolumn{2}{c}{D-M} & \multicolumn{2}{c}{D-S} \\",
              r"Condition & W0 & W1 & W0 & W1 & W0 & W1 & W0 & W1 \\"]
    write("s7_minm", table("tab:s7minm", cap, "lcccccccc", header, rows, sep="4pt"))


def s7_budget():
    rows = []
    for p in PLATS:
        d = rd(p, "e25_budget.csv")
        d = d[d.m == -1]
        rows.append(r"\multicolumn{8}{l}{\emph{" + PNAME[p] + r"}} \\")
        for c in CONDS[p]:
            first = True
            for dn in ("D-M", "D-S"):
                for fam, flab in (("Wk", r"W-Q($k$)"), ("Wk+null", r"$k$ + null")):
                    cells = []
                    for k in range(1, 6):
                        r = d[(d.cond == c) & (d.dict == dn) & (d.family == fam) & (d.k == k)]
                        cells.append("--" if r.empty else num(r.nmse.iloc[0]))
                    rows.append(f"{cl(c) if first else ''} & {dn} & {flab} & " + " & ".join(cells) + r" \\")
                    first = False
        if p == "sim":
            rows.append(r"\midrule")
    cap = (r"Execution budget in the relaxed weight space (E25): test nMSE of the best average of at most $k$ "
           r"executed members without the null (W-Q($k$), the budget family of the locked protocol) and with the "
           r"null as a free extra candidate ($k$ + null; $k=5$ is W1), calibrated on the full calibration role "
           r"(mean over five seeds). The arm with $m=10$ (as Table~\ref{tab:s3k}) is in the code release.")
    header = [r"Condition & Dict. & Family & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=5$ \\"]
    write("s7_budget", table("tab:s7budget", cap, "lllccccc", header, rows, sep="4pt"))


def s7_retention():
    rows = []
    for p in PLATS:
        d = rd(p, "e25_retention.csv")
        d = d[d.m == -1]
        rows.append(r"\multicolumn{13}{l}{\emph{" + PNAME[p] + r"}} \\")
        for c in CONDS[p]:
            if c == "N":
                pass
            cells = []
            for dn in ("D-M", "D-S"):
                for k in (2, 3):
                    rs = d[(d.cond == c) & (d.dict == dn) & (d.k == k) & (d.ref == "W-S")]
                    rn = d[(d.cond == c) & (d.dict == dn) & (d.k == k) & (d.ref == "k1+null")]
                    cells += [num(rs.retention_plain.iloc[0], 2), num(rs.retention_null.iloc[0], 2),
                              num(rn.retention_null.iloc[0], 2)]
            rows.append(f"{cl(c)} & " + " & ".join(cells) + r" \\")
        if p == "sim":
            rows.append(r"\midrule")
    cap = (r"Retention of the calibration gain by small budgets (E25, full calibration role, mean over seeds). "
           r"Plain: $[R(\text{W-S})-R(\text{W-Q}(k))]/[R(\text{W-S})-R(\text{W-Q}(5))]$, the ratio of the locked "
           r"protocol. vs W-S: the same with $k$ members plus the null against five members plus the null. "
           r"vs scaled: the reference is the best single member with a free scale ($k=1$ + null), which removes the "
           r"part of the gain that a scalar alone provides.")
    header = [r" & \multicolumn{6}{c}{D-M} & \multicolumn{6}{c}{D-S} \\",
              r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}",
              r" & \multicolumn{3}{c}{$k=2$} & \multicolumn{3}{c}{$k=3$} & \multicolumn{3}{c}{$k=2$} & "
              r"\multicolumn{3}{c}{$k=3$} \\",
              r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
              r"Condition & plain & vs W-S & vs scaled & plain & vs W-S & vs scaled & plain & vs W-S & vs scaled & "
              r"plain & vs W-S & vs scaled \\"]
    write("s7_retention", table("tab:s7ret", cap, "l" + "c" * 12, header, rows, sep="2.5pt"))


def s7_perjoint():
    rows = []
    for p in PLATS:
        d = rd(p, "e25_perjoint.csv")
        rows.append(r"\multicolumn{11}{l}{\emph{" + PNAME[p] + r"}} \\")
        for c in ("O1-M", "O1-H"):
            first = True
            for dn, sp, q, lab in (("D-M", "W2", "ste_weight_by_joint", "D-M, W2: STE"),
                                   ("D-M", "W3", "ste_weight_by_joint", "D-M, W3: STE"),
                                   ("D-M", "W3", "null_weight_by_joint", "D-M, W3: null"),
                                   ("D-S", "W3", "null_weight_by_joint", "D-S, W3: null")):
                g = d[(d.cond == c) & (d.dict == dn) & (d.space == sp)]
                A = np.array([json.loads(v) for v in g[q]]).mean(0)
                cells = [num(v, 2) for v in A]
                cells += [num(g.union_support.mean(), 1), num(g.mean_joint_support.mean(), 1)]
                rows.append(f"{cl(c) if first else ''} & {lab} & " + " & ".join(cells) + r" \\")
                first = False
        if p == "sim":
            rows.append(r"\midrule")
    cap = (r"Per-joint weights (E25, full calibration role, mean over five seeds): weight on the straight-through "
           r"member (STE) of D-M and on the null predictor for each output joint, the number of members executed "
           r"(the union of the per-joint supports), and the mean number of members per joint.")
    header = [r"Condition & Weight & J1 & J2 & J3 & J4 & J5 & J6 & J7 & executed & per joint \\"]
    write("s7_perjoint", table("tab:s7pj", cap, "llccccccccc", header, rows, sep="3pt"))


# =========================================================================================
# S7.8: E26 selection of the weight space
# =========================================================================================
def s7_select():
    rows = []
    for p in PLATS:
        d = rd(p, "e26_select.csv")
        rows.append(r"\multicolumn{11}{l}{\emph{" + PNAME[p] + (r" ($m$ in trajectories)" if p == "sim" else
                                                                r" ($m$ in 100-row blocks; 45 is the full role)")
                    + r"}} \\")
        for c in CONDS[p]:
            first = True
            for dn in ("D-M", "D-S"):
                for m in sorted(d[(d.cond == c) & (d.dict == dn)].m.unique()):
                    r = d[(d.cond == c) & (d.dict == dn) & (d.m == m)].iloc[0]
                    fr = [f"{100 * r[f'freq_W{j}']:.0f}" for j in range(5)]
                    best = min(range(5), key=lambda j: r[f"test_W{j}"])
                    rows.append(f"{cl(c) if first else ''} & {dn} & {int(m)} & {int(r.n)} & " + " & ".join(fr)
                                + f" & {num(r.test_selected)} & {num(r.regret)} & W{best}" + r" \\")
                    first = False
        if p == "sim":
            rows.append(r"\midrule")
    cap = (r"Choosing the weight space from the calibration data (E26): cross-fitting on the calibration subset "
           r"(folds: trajectories, or the five 200-step segments of the single trajectory when $m=1$; SARCOS: "
           r"100-row blocks) selects the space with the smallest held-out risk among W0--W4; the selected space "
           r"is refitted on the whole subset. Columns: number of cases (seeds $\times$ subsets), selection "
           r"frequency of each space in percent, test nMSE of the selected space, regret against the space with "
           r"the lowest test nMSE in each case, and the space with the lowest mean test nMSE.")
    header = [r"Condition & Dict. & $m$ & cases & W0 & W1 & W2 & W3 & W4 & nMSE & regret & best \\"]
    rows = [r.replace(r"\multicolumn{11}", r"\multicolumn{12}") for r in rows]
    write("s7_select", table("tab:s7sel", cap, "lllc" + "c" * 5 + "ccc", header, rows, sep="3pt"))


def main():
    s3_wspace_m()
    s7_ladder("sim", ["N", "O1-L", "O1-M"], "a")
    s7_ladder("sim", ["O1-H", "O2B", "P1"], "b")
    s7_ladder("sarcos", ["N", "O1-L", "O1-M"], "a")
    s7_ladder("sarcos", ["O1-H", "O1qqd"], "b")
    s7_geometry()
    s7_paired()
    s7_attribution()
    s7_width()
    s7_e18()
    s7_bestwidth()
    s7_critical()
    s7_crossings()
    s7_kappa()
    s7_wtable()
    s7_labelfree("sim")
    s7_labelfree("sarcos")
    s7_rho()
    s7_cert()
    s7_certboot()
    s7_minm()
    s7_budget()
    s7_retention()
    s7_perjoint()
    s7_select()


if __name__ == "__main__":
    main()
