"""Tables for the attenuation / augmentation analysis (E09-E12).

python build_s8_tables.py --sarcos <results/sarcos> --sim <results/sim> \
       --frozen_sarcos <frozen_sarcos.json> --frozen_sim <frozen_sim.json> --out <tables dir>

Writes table3.tex (main text), s8_ladder_<platform>.tex, s8_pairs_<platform>.tex,
s8_split.tex, s8_weights.tex, s8_geometry.tex and results_manifest_e09.csv.
Every number is read from e09_summary.csv / e09_paired.csv / e09_weights.csv /
e10_members.csv / e10_dicts.csv / e11_frontend_geometry.json of the platform.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

BS = "\\"
EOL = " " + BS + BS

# (method, dictionary, label) rows of the main-text table, in blocks
MAIN_BLOCKS = [
    [("W-Q5", "D-S", "D-S (seeds)"), ("W-Q5", "D-M", "D-M (surrogates)"), ("W-Q5", "D-M-STE", "D-M without straight-through")],
    [("W-Q5", "D-S+0", "D-S + null"), ("W-Q5", "D-M+0", "D-M + null"),
     ("W-Q5", "D-S|wfe", "D-S, input estimator"), ("W-Q5", "D-M|wfe", "D-M, input estimator"),
     ("W-Q5", "D-SW", "D-S, estimator optional")],
    [("B-A", "-", "B-A (one augmented network)"), ("B-A+", "-", "B-A$^{+}$ (augmented validation)"),
     ("B-AQ+", "-", "B-A$^{+}$, noise on $(q,\\dot q)$ only"),
     ("W-Q5", "D-SA", "D-SA (augmented seeds)"), ("W-Q5", "D-MA", "D-MA (augmented surrogates)"),
     ("W-Q5", "D-SQ", "D-SA, noise on $(q,\\dot q)$ only"), ("W-Q5", "D-MQ", "D-MA, noise on $(q,\\dot q)$ only"),
     ("W-Q5", "D-U", "D-M and D-MA together")],
]
FULL_ROWS = [
    ("B-F", "-", "Dev-selected single (B-F)"), ("B-F|wfe", "-", "B-F, input estimator"), ("B-F|wfo", "-", "B-F, oracle estimator"),
    ("B-N", "-", "MLP (B-N)"), ("B-N|wfe", "-", "MLP, input estimator"), ("B-NA", "-", "MLP, augmented (B-NA)"), ("B-NQ", "-", "MLP, augmented on $(q,\\dot q)$"),
    ("B-A", "-", "Augmented SNN (B-A)"), ("B-A|wfe", "-", "B-A, input estimator"), ("B-AQ", "-", "Augmented SNN on $(q,\\dot q)$"),
    ("B-A+", "-", "Augmented SNN, augmented validation (B-A$^{+}$)"), ("B-AQ+", "-", "B-A$^{+}$ on $(q,\\dot q)$"),
    ("B-A+|wfe", "-", "B-A$^{+}$, input estimator"),
    ("B-NA+", "-", "MLP, augmented, augmented validation"), ("B-NQ+", "-", "MLP, augmented on $(q,\\dot q)$, augmented validation"),
    None,
    ("W-Q5", "D-S", "D-S"), ("W-Q5", "D-M", "D-M"), ("W-Q5", "D-M-STE", "D-M without straight-through"),
    ("W-Q5", "D-S+0", "D-S + null"), ("W-Q5", "D-M+0", "D-M + null"),
    ("W-Q5", "D-S|wfe", "D-S, input estimator"), ("W-Q5", "D-M|wfe", "D-M, input estimator"),
    ("W-Q5", "D-S|wfe+0", "D-S, input estimator, + null"), ("W-Q5", "D-M|wfe+0", "D-M, input estimator, + null"),
    ("W-Q5", "D-S|wfo", "D-S, oracle estimator"), ("W-Q5", "D-M|wfo", "D-M, oracle estimator"),
    ("W-Q5", "D-SW", "D-S with and without estimator (10 candidates)"), ("W-Q5", "D-MW", "D-M with and without estimator (10 candidates)"),
    None,
    ("W-S", "D-MA", "D-MA, calibrated single"), ("W-E", "D-MA", "D-MA, equal"), ("W-Q5", "D-MA", "D-MA, W-Q(5)"),
    ("W-Q5", "D-SA", "D-SA, W-Q(5)"), ("W-Q5", "D-MA+0", "D-MA + null"), ("W-Q5", "D-MA|wfe", "D-MA, input estimator"),
    ("W-Q5", "D-M+A", "D-M + B-A (6 candidates)"), ("W-Q5", "D-S+A", "D-S + B-A (6 candidates)"),
    ("W-Q5", "D-M+A+", "D-M + B-A$^{+}$ (6 candidates)"), ("W-Q5", "D-S+A+", "D-S + B-A$^{+}$ (6 candidates)"),
    ("W-Q5", "D-M+AQ+", "D-M + B-A$^{+}$ on $(q,\\dot q)$ (6 candidates)"), ("W-Q5", "D-S+AQ+", "D-S + B-A$^{+}$ on $(q,\\dot q)$ (6 candidates)"),
    ("W-Q2", "D-U", "D-M and D-MA, W-Q(2)"), ("W-Q5", "D-U", "D-M and D-MA, W-Q(5) (10 candidates)"), ("W-Q5", "D-US", "D-S and D-SA, W-Q(5) (10 candidates)"),
    ("W-Q5", "D-MQ", "D-MA on $(q,\\dot q)$, W-Q(5)"), ("W-Q5", "D-SQ", "D-SA on $(q,\\dot q)$, W-Q(5)"), ("W-Q5", "D-UQ", "D-M and D-MA on $(q,\\dot q)$, W-Q(5)"),
]
PAIR_ROWS = [
    ("W-Q5|D-M", "W-Q5|D-S", "D-M $-$ D-S"), ("W-Q5|D-M-STE", "W-Q5|D-S", "D-M without straight-through $-$ D-S"),
    ("W-Q5|D-M+0", "W-Q5|D-S+0", "D-M $-$ D-S, both + null"), ("W-Q5|D-M|wfe", "W-Q5|D-S|wfe", "D-M $-$ D-S, both behind the estimator"),
    ("W-Q5|D-M|wfe+0", "W-Q5|D-S|wfe+0", "D-M $-$ D-S, estimator and null"), ("W-Q5|D-MW", "W-Q5|D-SW", "D-M $-$ D-S, with and without estimator"),
    ("W-Q5|D-MA", "W-Q5|D-SA", "D-MA $-$ D-SA"), ("W-Q5|D-MQ", "W-Q5|D-SQ", "D-MA $-$ D-SA, augmented on $(q,\\dot q)$"), ("W-Q5|D-U", "W-Q5|D-US", "(D-M and D-MA) $-$ (D-S and D-SA)"),
    None,
    ("W-Q5|D-S+0", "W-Q5|D-S", "null added to D-S"), ("W-Q5|D-M+0", "W-Q5|D-M", "null added to D-M"),
    ("W-Q5|D-S|wfe", "W-Q5|D-S", "estimator added to D-S"), ("W-Q5|D-M|wfe", "W-Q5|D-M", "estimator added to D-M"),
    ("W-Q5|D-MW", "W-Q5|D-M", "estimator versions offered to D-M"),
    None,
    ("W-Q5|D-M", "B-A|-", "D-M $-$ B-A"), ("W-Q5|D-S+0", "B-A|-", "(D-S + null) $-$ B-A"), ("W-Q5|D-M|wfe", "B-A|-", "D-M behind estimator $-$ B-A"),
    ("W-Q5|D-MW", "B-A|-", "D-M with and without estimator $-$ B-A"),
    ("W-Q5|D-MA", "B-A|-", "D-MA $-$ B-A"), ("W-Q5|D-SA", "B-A|-", "D-SA $-$ B-A"), ("W-Q5|D-M+A", "B-A|-", "(D-M + B-A) $-$ B-A"),
    ("W-Q5|D-U", "B-A|-", "(D-M and D-MA) $-$ B-A"), ("W-Q5|D-U", "W-Q5|D-M", "(D-M and D-MA) $-$ D-M"), ("W-Q5|D-U", "W-Q5|D-MA", "(D-M and D-MA) $-$ D-MA"),
    ("B-NA|-", "B-A|-", "augmented MLP $-$ B-A"), ("B-AQ|-", "B-A|-", "B-A on $(q,\\dot q)$ $-$ B-A"), ("W-Q5|D-MQ", "B-AQ|-", "D-MA $-$ B-A, both on $(q,\\dot q)$"),
    ("W-Q5|D-MQ", "W-Q5|D-M|wfe", "D-MA on $(q,\\dot q)$ $-$ D-M behind estimator"),
    None,
    ("B-A+|-", "B-A|-", "B-A$^{+}$ $-$ B-A"), ("B-AQ+|-", "B-AQ|-", "B-A$^{+}$ $-$ B-A, both on $(q,\\dot q)$"), ("B-AQ+|-", "B-A+|-", "B-A$^{+}$ on $(q,\\dot q)$ $-$ B-A$^{+}$"),
    ("W-Q5|D-M", "B-A+|-", "D-M $-$ B-A$^{+}$"), ("W-Q5|D-M", "B-AQ+|-", "D-M $-$ B-A$^{+}$ on $(q,\\dot q)$"),
    ("W-Q5|D-S+0", "B-A+|-", "(D-S + null) $-$ B-A$^{+}$"), ("W-Q5|D-S+0", "B-AQ+|-", "(D-S + null) $-$ B-A$^{+}$ on $(q,\\dot q)$"),
    ("W-Q5|D-S|wfe", "B-A+|-", "D-S behind estimator $-$ B-A$^{+}$"), ("W-Q5|D-S|wfe", "B-AQ+|-", "D-S behind estimator $-$ B-A$^{+}$ on $(q,\\dot q)$"),
    ("W-Q5|D-M+A+", "B-A+|-", "(D-M + B-A$^{+}$) $-$ B-A$^{+}$"), ("W-Q5|D-M+A+", "W-Q5|D-S+A+", "(D-M + B-A$^{+}$) $-$ (D-S + B-A$^{+}$)"),
    ("W-Q5|D-M+AQ+", "B-AQ+|-", "(D-M + B-A$^{+}$) $-$ B-A$^{+}$, on $(q,\\dot q)$"), ("W-Q5|D-M+AQ+", "W-Q5|D-S+AQ+", "(D-M + B-A$^{+}$) $-$ (D-S + B-A$^{+}$), on $(q,\\dot q)$"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarcos", required=True); ap.add_argument("--sim", required=True)
    ap.add_argument("--frozen_sarcos", required=True); ap.add_argument("--frozen_sim", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    FZ = {"sarcos": json.load(open(args.frozen_sarcos)), "sim": json.load(open(args.frozen_sim))}
    DIR = {"sarcos": args.sarcos, "sim": args.sim}
    PN = {"sarcos": "SARCOS", "sim": "Simulated arm"}
    S = {p: pd.read_csv(os.path.join(DIR[p], "e09_summary.csv")) for p in DIR}
    P = {p: pd.read_csv(os.path.join(DIR[p], "e09_paired.csv")) for p in DIR}
    W = {p: pd.read_csv(os.path.join(DIR[p], "e09_weights.csv"), keep_default_na=False) for p in DIR}
    EM = {p: pd.read_csv(os.path.join(DIR[p], "e10_members.csv")) for p in DIR}
    ED = {p: pd.read_csv(os.path.join(DIR[p], "e10_dicts.csv")) for p in DIR}
    GEO = {p: json.load(open(os.path.join(DIR[p], "e11_frontend_geometry.json"))) for p in DIR}
    manifest = []

    def conds(p):
        c = list(FZ[p]["conds"])
        return c + (["P1"] if p == "sim" else [])

    def cl(p, c, short=False):
        fz = FZ[p]
        if c in ("N", "P1"):
            return c
        kind, rho = c.split(":"); rho = float(rho)
        lev = {fz["O1_L"]: "L", fz["O1_M"]: "M", fz["O1_H"]: "H"}.get(rho, f"{rho}")
        return {"O1": f"O1-{lev}", "O1qqd": f"O1-{lev} on $(q,{BS}dot q)$", "O2B": "O2-B"}.get(kind, c)

    def get(p, c, method, dn):
        s = S[p]
        r = s[(s.cond == c) & (s.method == method) & (s.dictionary == dn)]
        return None if len(r) == 0 else r.iloc[0]

    def note(table, p, c, method, dn, r, src="e09_summary.csv"):
        manifest.append({"table": table, "platform": p, "cond": c, "method": method, "dictionary": dn, "value": r.nmse_mean,
                         "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "source": f"results/{p}/{src}"})

    # ---------------- Table III (main text)
    cols = [("sarcos", "N"), ("sarcos", f"O1:{FZ['sarcos']['O1_M']}"), ("sarcos", f"O1:{FZ['sarcos']['O1_H']}"),
            ("sim", "N"), ("sim", f"O1:{FZ['sim']['O1_M']}"), ("sim", f"O1:{FZ['sim']['O1_H']}")]
    vals = {}
    for b in MAIN_BLOCKS:
        for method, dn, label in b:
            for p, c in cols:
                r = get(p, c, method, dn)
                if r is not None:
                    vals[(method, dn, p, c)] = r.nmse_mean; note("III", p, c, method, dn, r)
    best = {col: min(v for (m, dn, p, c), v in vals.items() if (p, c) == col) for col in cols}
    lines = [BS + "begin{table}[t]", BS + "centering",
             BS + "caption{Attenuation controls and augmentation (E09--E12). Test nMSE of W-Q(5), mean over five seeds, calibration on the full $" + BS + "Dcal$ of the condition, at most five executed members. ``+ null'' offers the constant predictor (the training-mean torque) as an additional candidate; ``input estimator'' places the label-free linear estimator of Eq.~" + BS + "eqref{eq:wiener} in front of the frozen members; augmented members were trained with the B-A augmentation. Bold: lowest value of the column.}",
             BS + "label{tab:atten}", BS + "footnotesize", BS + "setlength{" + BS + "tabcolsep}{2.6pt}",
             BS + "begin{tabular}{@{}lcccccc@{}}", BS + "toprule",
             " & " + BS + "multicolumn{3}{c}{SARCOS} & " + BS + "multicolumn{3}{c}{Simulated arm}" + EOL,
             "Candidates & N & O1-M & O1-H & N & O1-M & O1-H" + EOL, BS + "midrule"]
    for bi, b in enumerate(MAIN_BLOCKS):
        for method, dn, label in b:
            cells = []
            for col in cols:
                v = vals.get((method, dn) + col)
                if v is None:
                    cells.append("--")
                else:
                    t = f"{v:.3f}" if v >= 0.1 else f"{v:.4f}"
                    cells.append(BS + "textbf{" + t + "}" if abs(v - best[col]) < 5e-5 else t)
            if all(x == "--" for x in cells):
                continue
            lines.append(label + " & " + " & ".join(cells) + EOL)
        if bi < len(MAIN_BLOCKS) - 1:
            lines.append(BS + "midrule")
    lines += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}"]
    open(os.path.join(args.out, "table3.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------------- S8 full ladders and paired differences, per platform
    for p in DIR:
        cs = conds(p)
        lines = [BS + "begin{table}[H]", BS + "centering",
                 BS + "caption{" + PN[p] + ": test nMSE of the attenuation and augmentation candidates under every condition, mean [95" + BS + "% cluster-bootstrap CI] over five seeds and test units. Dictionaries use the stated rule on the full $" + BS + "Dcal$ of the condition with at most five executed members. The oracle estimator uses the true noise variances and is a reference.}",
                 BS + "label{tab:s8ladder" + p + "}", BS + "scriptsize", BS + "resizebox{" + BS + "textwidth}{!}{%",
                 BS + "begin{tabular}{l" + "c" * len(cs) + "}", BS + "toprule",
                 "Candidates & " + " & ".join(cl(p, c) for c in cs) + EOL, BS + "midrule"]
        for row in FULL_ROWS:
            if row is None:
                if lines[-1] != BS + "midrule":
                    lines.append(BS + "midrule")
                continue
            method, dn, label = row
            cells, anyv = [], False
            for c in cs:
                r = get(p, c, method, dn)
                if r is None:
                    cells.append("--")
                else:
                    anyv = True; note("S8-ladder", p, c, method, dn, r)
                    cells.append(f"{r.nmse_mean:.4f} [{r.ci_lo:.4f},{r.ci_hi:.4f}]")
            if anyv:
                lines.append(label + " & " + " & ".join(cells) + EOL)
        lines += [BS + "bottomrule", BS + "end{tabular}}", BS + "end{table}"]
        open(os.path.join(args.out, f"s8_ladder_{p}.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

        pr = P[p]
        lines = [BS + "begin{table}[H]", BS + "centering",
                 BS + "caption{" + PN[p] + ": paired differences in test nMSE under W-Q(5), mean [95" + BS + "% CI] and the number of seeds (of five) in which the first candidate set is better. Negative values favor the first.}",
                 BS + "label{tab:s8pairs" + p + "}", BS + "scriptsize", BS + "resizebox{" + BS + "textwidth}{!}{%",
                 BS + "begin{tabular}{l" + "c" * len(cs) + "}", BS + "toprule",
                 "Difference & " + " & ".join(cl(p, c) for c in cs) + EOL, BS + "midrule"]
        for row in PAIR_ROWS:
            if row is None:
                if lines[-1] != BS + "midrule":
                    lines.append(BS + "midrule")
                continue
            a, b, label = row
            cells, anyv = [], False
            for c in cs:
                r = pr[(pr.cond == c) & (pr.a == a) & (pr.b == b)]
                if len(r) == 0:
                    cells.append("--")
                else:
                    r = r.iloc[0]; anyv = True
                    cells.append(f"${r['diff']:+.4f}$ [${r.ci_lo:+.4f}$, ${r.ci_hi:+.4f}$] {int(r.seeds_a_better)}/{int(r.n_seeds)}")
                    manifest.append({"table": "S8-pairs", "platform": p, "cond": c, "method": a, "dictionary": b, "value": r["diff"],
                                     "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "source": f"results/{p}/e09_paired.csv"})
            if anyv:
                lines.append(label + " & " + " & ".join(cells) + EOL)
        if lines[-1] == BS + "midrule":
            lines.pop()
        lines += [BS + "bottomrule", BS + "end{tabular}}", BS + "end{table}"]
        open(os.path.join(args.out, f"s8_pairs_{p}.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------------- S8 split of the residual (E10), O1-M and O1-H of both platforms
    lines = [BS + "begin{table}[H]", BS + "centering",
             BS + "caption{Split of the test risk at O1-M into the propagated-perturbation term $P_n=" + BS + "E" + BS + "|f(" + BS + "tilde x)-f(x)" + BS + "|^2/d$, the nominal term $P_b=" + BS + "E" + BS + "|f(x)-y" + BS + "|^2/d$ and their cross term (risk $=P_n+P_b+$ cross), with the regression slope of the target on the prediction and the output power of the prediction (the target power is 0.996 on SARCOS and 0.923 on the arm). Member rows are means over the members of the group and over five seeds. Dictionary columns: mean pairwise correlation of the propagated terms and of the nominal terms, and the minimum over the simplex of the propagated term alone.}",
             BS + "label{tab:s8split}", BS + "scriptsize", BS + "setlength{" + BS + "tabcolsep}{4pt}",
             BS + "begin{tabular}{lllccccccccc}", BS + "toprule",
             "Platform & Condition & Members & risk & $P_n$ & $P_b$ & cross & slope & output power & corr($n$) & corr($b$) & floor of $P_n$" + EOL, BS + "midrule"]
    GROUPS = [("D-S", None, "D-S (five seeds)"), ("D-M", "window", "D-M, four window surrogates"), ("D-M", "ste", "D-M, straight-through"),
              ("D-SA", None, "D-SA (augmented seeds)"), ("D-MA", "window", "D-MA, four window surrogates"), ("D-MA", "ste", "D-MA, straight-through"),
              ("D-SQ", None, "D-SA on $(q," + BS + "dot q)$"), ("D-MQ", "window", "D-MA on $(q," + BS + "dot q)$, window"), ("D-MQ", "ste", "D-MA on $(q," + BS + "dot q)$, straight-through"),
              ("D-S|wfe", None, "D-S behind the estimator"), ("D-M|wfe", "window", "D-M behind the estimator, window"), ("D-M|wfe", "ste", "D-M behind the estimator, straight-through"),
              ("B-A", None, "B-A"), ("B-A+", None, "B-A$^{+}$"), ("B-AQ+", None, "B-A$^{+}$ on $(q," + BS + "dot q)$"), ("B-N", None, "MLP"), ("B-NA", None, "MLP, augmented"), ("B-NQ+", None, "MLP, augmented on $(q," + BS + "dot q)$, aug. validation")]
    for p in DIR:
        for c in [f"O1:{FZ[p]['O1_M']}"]:
            em, ed = EM[p][EM[p].cond == c], ED[p][ED[p].cond == c]
            first = True
            for dn, sel, label in GROUPS:
                sub = em[em.dictionary == dn]
                if sel == "window":
                    sub = sub[sub.member != "ste"]
                elif sel == "ste":
                    sub = sub[sub.member == "ste"]
                if len(sub) == 0:
                    continue
                m = sub.mean(numeric_only=True)
                dd = ed[ed.dictionary == dn].mean(numeric_only=True) if (sel != "ste" and len(ed[ed.dictionary == dn])) else None
                if dd is None:
                    extra = ["--", "--", "--"]
                else:
                    extra = [f"{dd.corr_n:.2f}", f"{dd.corr_b:.2f}", f"{dd.simplex_floor_n:.3f}"]
                    if sel == "window":
                        extra = [x + "$^{" + BS + "ast}$" for x in extra]
                lines.append(f"{PN[p] if first else ''} & {cl(p, c) if first else ''} & {label} & {m.risk:.3f} & {m.P_n:.3f} & {m.P_b:.3f} & ${m.cross:+.3f}$ & {m.slope:.2f} & {m.output_power:.2f} & " + " & ".join(extra) + EOL)
                manifest.append({"table": "S8-split", "platform": p, "cond": c, "method": label, "dictionary": dn, "value": m.P_n, "source": f"results/{p}/e10_members.csv"})
                first = False
            lines.append(BS + "midrule")
    lines = lines[:-1] + [BS + "bottomrule", BS + "multicolumn{12}{l}{$^{" + BS + "ast}$ Dictionary-level quantities of the whole five-member dictionary, straight-through member included.}" + EOL,
                          BS + "end{tabular}", BS + "end{table}"]
    open(os.path.join(args.out, "s8_split.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------------- S8 weights
    lines = [BS + "begin{table}[H]", BS + "centering",
             BS + "caption{Mean W-Q(5) weight over five seeds given to selected candidates. For the dictionaries extended by the null predictor the weight on the null is the amount by which the average is shrunk toward the training-mean torque.}",
             BS + "label{tab:s8weights}", BS + "scriptsize", BS + "setlength{" + BS + "tabcolsep}{4pt}"]
    body = []
    WSPEC = [("D-M", ["ste"], "D-M: straight-through"), ("D-M+0", ["ste"], "D-M + null: straight-through"), ("D-M+0", ["null"], "D-M + null: null"),
             ("D-S+0", ["null"], "D-S + null: null"), ("D-M|wfe", ["ste"], "D-M behind estimator: straight-through"),
             ("D-M+A", ["B-A"], "D-M + B-A: B-A"), ("D-M+A", ["ste"], "D-M + B-A: straight-through"),
             ("D-MW", [x + "|wfe" for x in ["atan", "sigmoid", "ste", "triangular", "sre"]], "D-M with and without estimator: estimator versions (sum)"),
             ("D-MA", ["ste+aug"], "D-MA: augmented straight-through"),
             ("D-U", [x + "+aug" for x in ["atan", "sigmoid", "ste", "triangular", "sre"]], "D-M and D-MA: augmented members (sum)"),
             ("D-U", ["ste", "ste+aug"], "D-M and D-MA: both straight-through members (sum)")]
    for p in DIR:
        cs = conds(p)
        w = W[p][W[p].rule == "W-Q5"]
        body += [BS + "begin{tabular}{l" + "c" * len(cs) + "}", BS + "toprule", PN[p] + " & " + " & ".join(cl(p, c) for c in cs) + EOL, BS + "midrule"]
        for dn, mems, label in WSPEC:
            sub = w[(w.dictionary == dn) & (w.member.isin(mems))]
            if len(sub) == 0:
                continue
            g = sub.groupby(["cond", "seed"]).weight.sum().groupby("cond").mean()
            body.append(label + " & " + " & ".join(f"{g.get(c, np.nan):.2f}" for c in cs) + EOL)
        body += [BS + "bottomrule", BS + "end{tabular}", "", BS + "vspace{4pt}", ""]
    lines += body[:-3] + [BS + "end{table}"]
    open(os.path.join(args.out, "s8_weights.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------------- S8 input geometry
    lines = [BS + "begin{table}[H]", BS + "centering",
             BS + "caption{Input geometry and the linear input estimator. Participation ratio $(" + BS + "sum" + BS + "lambda_i)^2/" + BS + "sum" + BS + "lambda_i^2$ of the eigenvalues of the training input covariance (21 channels). Removed: fraction of the input perturbation power removed by the estimator on the test inputs, with estimated and with oracle noise variances, and the fraction an estimator would remove if the channels were uncorrelated.}",
             BS + "label{tab:s8geom}", BS + "scriptsize", BS + "begin{tabular}{llcccc}", BS + "toprule",
             "Platform (participation ratio) & Condition & Removed, estimated & Removed, oracle & If uncorrelated & Mean estimated variance (perturbed channels)" + EOL, BS + "midrule"]
    for p in DIR:
        g = GEO[p]; first = True
        for c in conds(p):
            e = g["conditions"][c]
            vo = np.asarray(e["v_oracle"]); ve = np.asarray(e["v_est"])
            mest = float(ve[vo > 0].mean()) if (vo > 0).any() else float(ve.mean())
            lines.append(f"{PN[p] + ' (' + format(g['participation_ratio'], '.1f') + ')' if first else ''} & {cl(p, c)} & {(format(e['wfe']['removed_fraction_measured'], '.2f') if (vo > 0).any() else '--')} & {(format(e['wfo']['removed_fraction_measured'], '.2f') if (vo > 0).any() else '--')} & {(format(e.get('removed_fraction_if_uncorrelated', 0.0), '.2f') if (vo > 0).any() else '--')} & {mest:.3f}" + (f" (true {vo[vo > 0].mean():.3f})" if (vo > 0).any() else " (true 0)") + EOL)
            first = False
        lines.append(BS + "midrule")
    lines = lines[:-1] + [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}"]
    open(os.path.join(args.out, "s8_geometry.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # ---------------- S8 calibration-budget slice of the attenuation candidates (E11/E12)
    CAND = [("plain", "plain"), ("+null", "+ null"), ("estimator", "estimator"), ("estimator+null", "est. + null"), ("plain+estimator", "with and without est.")]
    lines = [BS + "begin{table}[H]", BS + "centering",
             BS + "caption{Calibration budget of the attenuation candidates: test nMSE of W-Q(5), mean over five seeds and ten calibration groups, when the weights and the noise variances of the input estimator are both obtained from the same calibration subset ($n_{" + BS + "mathrm{cal}}$ rows on SARCOS, $m$ trajectories on the arm).}",
             BS + "label{tab:s8mslice}", BS + "scriptsize", BS + "setlength{" + BS + "tabcolsep}{3.5pt}",
             BS + "begin{tabular}{lll" + "c" * (2 * len(CAND)) + "}", BS + "toprule",
             " & & & " + BS + "multicolumn{" + str(len(CAND)) + "}{c}{D-S} & " + BS + "multicolumn{" + str(len(CAND)) + "}{c}{D-M}" + EOL,
             "Platform & Condition & Budget & " + " & ".join(l for _, l in CAND) + " & " + " & ".join(l for _, l in CAND) + EOL, BS + "midrule"]
    for p in DIR:
        f = os.path.join(DIR[p], "e11_m_slice.csv")
        if not os.path.exists(f):
            continue
        e = pd.read_csv(f)
        g = e.groupby(["cond", "m", "dictionary", "candidates"]).nmse.mean()
        e.groupby(["cond", "m", "dictionary", "candidates"]).agg(nmse=("nmse", "mean"), null_weight=("null_weight", "mean"), estimator_weight=("estimator_weight", "mean")).reset_index().to_csv(os.path.join(args.out, f"s8_m_slice_summary_{p}.csv"), index=False)
        first = True
        for c in list(dict.fromkeys(e.cond)):
            fc = True
            for m in sorted(e.m.unique()):
                cells = [f"{g[(c, m, dn, k)]:.4f}" for dn in ["D-S", "D-M"] for k, _ in CAND]
                lines.append(f"{PN[p] if first else ''} & {cl(p, c) if fc else ''} & {m} & " + " & ".join(cells) + EOL)
                for dn in ["D-S", "D-M"]:
                    for k, _ in CAND:
                        manifest.append({"table": "S8-m", "platform": p, "cond": c, "method": k, "dictionary": dn, "m": m, "value": g[(c, m, dn, k)], "source": f"results/{p}/e11_m_slice.csv"})
                first = fc = False
        lines.append(BS + "midrule")
    lines = lines[:-1] + [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}"]
    open(os.path.join(args.out, "s8_mslice.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    pd.DataFrame(manifest).to_csv(os.path.join(args.out, "results_manifest_e09.csv"), index=False)
    print("S8 tables written;", len(manifest), "manifest rows")


if __name__ == "__main__":
    main()
