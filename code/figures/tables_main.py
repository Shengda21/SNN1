"""Tables II-IV of the main text (LaTeX, booktabs), generated from the local result CSVs.

table2.tex (tab:ladder): core open-loop results, method x (platform, condition, weight space)
    e21_ladder.csv: nmse_mean (W0, W1, W3, W5), n_exec, null_weight of D-M;
    e21_ladder_w16.csv: D-W_tri (W0, W1, W3; seeds listed in its `seeds` column).
table3.tex (tab:attr): attribution of the D-S - D-M simplex margin
    e21_attribution.csv: margin_W0 [_lo, _hi], scale_part, angle_part, scale_share, rem_est_W1,
    rem_perjoint_W3.
table4.tex (tab:cert): (a) gain certificate, e24_certificate.csv (cert_rate_t, coverage_t_held);
    (b) execution budget, e25_budget.csv (nmse, m = -1 = full calibration) and retention computed
    per seed from e25_budget_rows.csv exactly as in e25_budget.py (k = 2 is checked against
    e25_retention.csv); (c) latency, the two execution rows from results/runtime/runtime.json.
Run:  python tables_main.py      (writes build/tables/table{2,3,4}.tex)
"""
import re

import numpy as np
import pandas as pd

import common_style as cs

# ----------------------------------------------------------------------------- Table II
T2_METHODS = [("B-F", "Dev-selected single SNN (B-F)"), ("B-L", "Learned ATan-width SNN (B-L)"),
              ("B-A", "Noise-augmented SNN (B-A)"), ("B-N", "MLP (B-N)"), ("D-S", "Seed dictionary (D-S)"),
              ("D-T", "Snapshot dictionary (D-T)"), ("D-M", "Multi-surrogate dictionary (D-M)")]
T2_DW = ("D-W_tri", r"Width dictionary (D-W)$^{\dagger}$")
T2_CONDS = ["N", "O1-M", "O1-H"]
T2_SPACES = ["W0", "W1", "W3", "W5"]
T2_ND = {"N": 4, "O1-M": 3, "O1-H": 3}

# ----------------------------------------------------------------------------- Table III
T3_ROWS = [("sarcos", "O1-M"), ("sarcos", "O1-H"), ("sim", "O1-M"), ("sim", "O1-H"), ("sim", "O2B")]

# ----------------------------------------------------------------------------- Table IV
T4_CERT_COND = "O1-M"
T4_CERT_M = [10, 20, 40]
T4_BUDGET_ROWS = [("sim", "O1-M"), ("sim", "O1-H"), ("sarcos", "O1-M"), ("sarcos", "O1-H")]
T4_DICTS = ["D-M", "D-S"]


def num(x, nd, sign=False):
    if x is None or not np.isfinite(x):
        return "--"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    if s.startswith("-"):
        s = "$-$" + s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if re.fullmatch(r"(\$-\$)?0\.0+", s):  # avoid "-0.000"
        s = s.replace("$-$", "")
    return s


def write(name, text):
    cs.TAB_OUT.mkdir(parents=True, exist_ok=True)
    path = cs.TAB_OUT / name
    path.write_text(text, encoding="utf-8")
    print("wrote", path)


# ----------------------------------------------------------------------------- Table II
def table2():
    lad = {p: cs.read(p, "e21_ladder.csv", ["cond", "method", "space", "nmse_mean", "n_exec", "null_weight",
                                            "n_seeds"]) for p in cs.PLATFORMS}
    w16 = {p: cs.read(p, "e21_ladder_w16.csv", ["cond", "method", "space", "nmse_mean", "seeds"])
           for p in cs.PLATFORMS}
    cols = [(c, sp) for c in T2_CONDS for sp in T2_SPACES]
    lines = []
    dw_seeds = set()
    n_seeds = set()
    for p in ["sarcos", "sim"]:
        L = lad[p].set_index(["method", "cond", "space"])
        W = w16[p].set_index(["method", "cond", "space"])
        rows = []
        for key, lab in T2_METHODS:
            vals = [L["nmse_mean"].get((key, c, sp), np.nan) for c, sp in cols]
            rows.append((lab, vals))
            n_seeds |= set(lad[p][lad[p].method == key]["n_seeds"].unique())
        dw_vals = [W["nmse_mean"].get((T2_DW[0], c, sp), np.nan) for c, sp in cols]
        if np.isfinite(dw_vals).any():
            rows.append((T2_DW[1], dw_vals))
            dw_seeds |= set(w16[p][w16[p].method == T2_DW[0]]["seeds"].astype(str).unique())
        # bold: lowest displayed value in each column
        txt = [[num(v, T2_ND[c]) for v, (c, _) in zip(vals, cols)] for _, vals in rows]
        for j, (c, _) in enumerate(cols):
            shown = [float(t[j]) if t[j] != "--" else np.inf for t in txt]
            best = min(shown)
            for i in range(len(rows)):
                if np.isfinite(best) and shown[i] == best:
                    txt[i][j] = r"\textbf{" + txt[i][j] + "}"
        lines.append(r"\multicolumn{13}{@{}l}{\textit{" + cs.PLATFORM_NAME[p] + r"}} \\")
        for (lab, _), t in zip(rows, txt):
            lines.append(lab + " & " + " & ".join(t) + r" \\")
        ne = [L["n_exec"].get(("D-M", c, sp), np.nan) for c, sp in cols]
        nw = [L["null_weight"].get(("D-M", c, sp), np.nan) if sp != "W0" else np.nan for c, sp in cols]
        lines.append(r"\quad D-M: executed members & " +
                     " & ".join(num(v, 0 if abs(v - round(v)) < 1e-9 else 1) for v in ne) + r" \\")
        lines.append(r"\quad D-M: null weight & " + " & ".join(num(v, 2) for v in nw) + r" \\")
        lines.append(r"\midrule" if p == "sarcos" else r"\bottomrule")
    ns = "/".join(str(int(n)) for n in sorted(n_seeds))
    dws = "; ".join(sorted(dw_seeds))
    head = r"""\begin{table*}[t]
\centering
\caption{Core open-loop results across weight spaces. Test nMSE (standardized units, Eq.~1), mean over """ + \
        ("five" if ns == "5" else ns) + r""" training seeds; the weights are fitted on the full $\Dcal$ of the same condition and frozen. W0: simplex (W-Q(5) of the earlier version); W1: simplex with null, $\{w\ge 0,\ \one^\top w\le 1\}$, which for a single model is a scalar $c\in[0,1]$; W3: per-joint simplex with null; W5: the label-free input estimator in front of the members, then W1. For a single model W0 is the model itself. Bold: lowest value of the column within a platform. The 95\% two-level cluster-bootstrap intervals (training seed $\times$ test unit) are listed in the Supplementary Material. The last two rows of each block give the executed members and the null weight of the D-M solution. $^{\dagger}$Triangular E16 width members plus the STE member, training seeds """ + dws.replace(",", ", ") + r""" only; W5 was not computed.}
\label{tab:ladder}
\footnotesize
\setlength{\tabcolsep}{2.8pt}
\begin{tabular}{@{}l*{12}{c}@{}}
\toprule
 & \multicolumn{4}{c}{N} & \multicolumn{4}{c}{O1-M} & \multicolumn{4}{c}{O1-H} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}
Method & W0 & W1 & W3 & W5 & W0 & W1 & W3 & W5 & W0 & W1 & W3 & W5 \\
\midrule
"""
    tail = "\n\\end{tabular}\n\\end{table*}\n"
    write("table2.tex", head + "\n".join(lines) + tail)


# ----------------------------------------------------------------------------- Table III
def table3():
    need = ["cond", "margin_W0", "margin_W0_lo", "margin_W0_hi", "scale_part", "angle_part", "scale_share",
            "rem_est_W1", "rem_est_W1_lo", "rem_est_W1_hi", "rem_perjoint_W3", "rem_perjoint_W3_lo",
            "rem_perjoint_W3_hi"]
    att = {p: cs.read(p, "e21_attribution.csv", need).set_index("cond") for p in cs.PLATFORMS}
    lines = []
    prev = None
    for p, c in T3_ROWS:
        r = att[p].loc[c]
        plat = ""
        if p != prev:
            n = sum(1 for q, _ in T3_ROWS if q == p)
            plat = r"\multirow{" + str(n) + "}{*}{" + cs.PLATFORM_SHORT[p] + "}"
            if prev is not None:
                lines.append(r"\midrule")
        prev = p
        margin = num(r.margin_W0, 3) + r"\,{\scriptsize[" + num(r.margin_W0_lo, 3) + ", " + num(r.margin_W0_hi, 3) + "]}"
        share = f"{100 * r.scale_share:.0f}\\%"
        sig = lambda v, lo, hi: num(v, 3) + ("$^{*}$" if (lo > 0 or hi < 0) else "")
        lines.append(" & ".join([plat, c, margin, num(r.scale_part, 3), num(r.angle_part, 3), share,
                                 sig(r.rem_est_W1, r.rem_est_W1_lo, r.rem_est_W1_hi),
                                 sig(r.rem_perjoint_W3, r.rem_perjoint_W3_lo, r.rem_perjoint_W3_hi)]) + r" \\")
    text = r"""\begin{table}[t]
\centering
\caption{Attribution of the multi-surrogate margin. Margin: D-S minus D-M test nMSE in W0, with its 95\% cluster-bootstrap interval; scale: the part removed by admitting null (W0 margin minus W1 margin); angle: the margin that remains in W1; share: scale part over margin. The last two columns give the D-S minus D-M margin that remains after the label-free input estimator (W5) and after per-joint weights with null (W3). Positive values favour D-M. $^{*}$Interval excludes zero.}
\label{tab:attr}
\footnotesize
\setlength{\tabcolsep}{1.8pt}
\begin{tabular}{@{}llcccccc@{}}
\toprule
 & & Margin & & & Scale & \multicolumn{2}{c}{Remaining after} \\
\cmidrule(l){7-8}
 & & (W0) & Scale & Angle & share & estimator & per-joint \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("table3.tex", text)


# ----------------------------------------------------------------------------- Table IV
def retention_from_rows(rows):
    """Retention of the W-S -> (5 + null) gain kept by k members + null, per seed and group, then
    averaged (same definition as e25_budget.py)."""
    out = []
    for (c, dn, m, s, g), sub in rows.groupby(["cond", "dict", "m", "seed", "group"]):
        R = {(f, k): v for f, k, v in zip(sub["family"], sub["k"], sub["nmse"])}
        ws = R[("Wk", 1)]
        den = ws - R[("Wk+null", 5)]
        for k in (1, 2, 3, 5):
            out.append({"cond": c, "dict": dn, "m": m, "k": k,
                        "ret": (ws - R[("Wk+null", k)]) / den if abs(den) > 1e-12 else np.nan})
    return pd.DataFrame(out).groupby(["cond", "dict", "m", "k"])["ret"].mean()


def parse_latency():
    """Median batch-1 latency of k = 1..5 SNN members on the GPU and on one CPU thread, from
    results/runtime/runtime.json (written by code/profile_runtime.py)."""
    import json
    rt = json.load(open(cs.RES / "runtime" / "runtime.json"))
    rows = pd.DataFrame(rt["rows"])
    sel = lambda dev: [float(rows[(rows.device == dev) & (rows.model == "SNN") & (rows.batch == 1) & (rows.k == k)]
                             .iloc[0].latency_ms_median) for k in range(1, 6)]
    return sel("cuda"), sel("cpu"), rt.get("gpu", "GPU"), "CPU"


def table4():
    cert = {p: cs.read(p, "e24_certificate.csv", ["cond", "dict", "space", "m", "cert_rate_t", "coverage_t_held"])
            for p in cs.PLATFORMS}
    bud = {p: cs.read(p, "e25_budget.csv", ["cond", "dict", "m", "family", "k", "nmse"]) for p in cs.PLATFORMS}
    rows = {p: cs.read(p, "e25_budget_rows.csv", ["cond", "seed", "dict", "m", "group", "family", "k", "nmse"])
            for p in cs.PLATFORMS}
    retc = {p: cs.read(p, "e25_retention.csv", ["cond", "dict", "m", "ref", "k", "retention_null"])
            for p in cs.PLATFORMS}

    # (a) certificate
    la = []
    for p in ["sim", "sarcos"]:
        C = cert[p][cert[p].cond == T4_CERT_COND].set_index(["dict", "space", "m"])
        first = True
        for dn in T4_DICTS:
            for sp in ["W0", "W1"]:
                cells = []
                for m in T4_CERT_M:
                    if (dn, sp, m) in C.index:
                        r = C.loc[(dn, sp, m)]
                        cells += [num(r.cert_rate_t, 2), num(r.coverage_t_held, 2)]
                    else:
                        cells += ["--", "--"]
                plat = (r"\multirow{4}{*}{" + cs.PLATFORM_SHORT[p] + "}") if first else ""
                first = False
                la.append(" & ".join([plat, dn, sp] + cells) + r" \\")
        if p == "sim":
            la.append(r"\midrule")
    # (b) budget
    lb = []
    for p in ["sim", "sarcos"]:
        B = bud[p][bud[p].m == -1].set_index(["cond", "dict", "family", "k"])["nmse"]
        ret = retention_from_rows(rows[p][rows[p].m == -1])
        chk = retc[p][(retc[p].m == -1) & (retc[p].ref == "W-S") & (retc[p].k == 2)].set_index(["cond", "dict"])
        for (q, c) in [x for x in T4_BUDGET_ROWS if x[0] == p]:
            for dn in T4_DICTS:
                r2 = ret.get((c, dn, -1, 2), np.nan)
                ref = chk["retention_null"].get((c, dn), np.nan)
                if np.isfinite(ref) and abs(ref - r2) > 1e-9:
                    raise ValueError(f"retention mismatch {p} {c} {dn}: {r2} vs {ref}")
                cells = [num(B.get((c, dn, "Wk", 1), np.nan), 3)]
                for k in (1, 2):
                    cells.append(num(B.get((c, dn, "Wk+null", k), np.nan), 3) + r"\,{\scriptsize(" +
                                 num(ret.get((c, dn, -1, k), np.nan), 2) + ")}")
                cells.append(num(B.get((c, dn, "Wk+null", 5), np.nan), 3))
                lb.append(" & ".join([f"{cs.PLATFORM_SHORT[p]} {c}", dn] + cells) + r" \\")
        if p == "sim":
            lb.append(r"\midrule")
    # (c) latency
    gpu, cpu, gname, cname = parse_latency()
    lc = [r"GPU & " + " & ".join(f"{v:.2f}" for v in gpu) + r" \\",
          r"CPU & " + " & ".join(f"{v:.2f}" for v in cpu) + r" \\"]
    ms = ", ".join(str(m) for m in T4_CERT_M)
    text = r"""\begin{table}[t]
\centering
\caption{Certificate, execution budget, and latency. (a) Gain of the calibrated weights over the best single member at """ + T4_CERT_COND + r""", certified when the one-sided 95\% $t$ lower bound from sample splitting is positive: certification rate (cert.) and coverage (cov., the bound does not exceed the gain on the test role pooled with the unused calibration units) for $m=""" + ms + r"""$ calibration units (arm: trajectories; SARCOS: 100-row blocks). (b) Test nMSE with full calibration of the best unscaled member (W-S) and of $k$ executed members plus null; in parentheses, the fraction of the W-S to (5 + null) gain that is retained. (c) Batch-1 execution time of $k$ SNN members in ms (GPU: """ + gname + r"""; CPU: one thread of an """ + cname + r"""). The null member outputs zero and adds no execution time.}
\label{tab:cert}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lll*{6}{c}@{}}
\multicolumn{9}{@{}l}{\textit{(a) Gain certificate, $t$ lower bound}} \\
\toprule
 & & & \multicolumn{2}{c}{$m=""" + str(T4_CERT_M[0]) + r"""$} & \multicolumn{2}{c}{$m=""" + str(T4_CERT_M[1]) + r"""$} & \multicolumn{2}{c}{$m=""" + str(T4_CERT_M[2]) + r"""$} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
 & & Space & cert. & cov. & cert. & cov. & cert. & cov. \\
\midrule
""" + "\n".join(la) + r"""
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{@{}llcccc@{}}
\multicolumn{6}{@{}l}{\textit{(b) Execution budget, test nMSE (retained gain)}} \\
\toprule
 & & W-S & 1 + null & 2 + null & 5 + null \\
\midrule
""" + "\n".join(lb) + r"""
\bottomrule
\end{tabular}

\vspace{6pt}
\begin{tabular}{@{}l*{5}{c}@{}}
\multicolumn{6}{@{}l}{\textit{(c) Latency (ms), $k$ executed members}} \\
\toprule
 & $k=1$ & 2 & 3 & 4 & 5 \\
\midrule
""" + "\n".join(lc) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("table4.tex", text)


if __name__ == "__main__":
    table2()
    table3()
    table4()
