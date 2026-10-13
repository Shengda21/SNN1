"""Table and figure of the underfitting control E14.

python build_e14_tables.py --sarcos <results/sarcos> --sim <results/sim> --frozen_sarcos <json> --frozen_sim <json> \
       --out <tables dir> --fig <fig7.pdf>

Writes s8_e14.tex, results_manifest_e14.csv, e14_numbers.json and the figure
(forward response against nominal error along training, with the straight-through member and
the reduced-capacity members). Inputs: e14_curve.csv, e14_summary.csv, e14_paired.csv, e14_weights.csv.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BS = "\\"
EOL = " " + BS + BS
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sarcos", required=True); ap.add_argument("--sim", required=True)
    ap.add_argument("--frozen_sarcos", required=True); ap.add_argument("--frozen_sim", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--fig", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    DIR = {"sarcos": args.sarcos, "sim": args.sim}
    FZ = {"sarcos": json.load(open(args.frozen_sarcos)), "sim": json.load(open(args.frozen_sim))}
    PN = {"sarcos": "SARCOS", "sim": "Simulated arm"}
    manifest, numbers = [], {}
    lines = [BS + "begin{table}[H]", BS + "centering",
             BS + "caption{Underfitting control (E14). Member columns: nominal test nMSE, measured forward perturbation response, and at O1-M the propagated term, the slope of the target on the prediction, and the risk of the member alone; means over five seeds. Dictionary columns: test nMSE of W-Q(5) on the four window-surrogate members of D-M plus the listed member, at O1-M and O1-H, the weight the member receives at O1-M, and the fraction of the straight-through member's contribution that it recovers at O1-M. Early checkpoints are states of the development-selected surrogate along its own training run; reduced-capacity members are trained to convergence under the unchanged protocol.}",
             BS + "label{tab:s8e14}", BS + "scriptsize", BS + "setlength{" + BS + "tabcolsep}{4pt}",
             BS + "begin{tabular}{llccccccccc}", BS + "toprule",
             " & & " + BS + "multicolumn{5}{c}{Member} & " + BS + "multicolumn{4}{c}{Four window members plus this member}" + EOL,
             "Platform & Fifth member & nominal & response & $P_n$ & slope & risk alone & O1-M & O1-H & weight & recovered" + EOL, BS + "midrule"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    for ax, p in zip(axes, ["sarcos", "sim"]):
        fz = FZ[p]; strong = fz["strong_surrogate"]
        O1M, O1H = f"O1:{fz['O1_M']}", f"O1:{fz['O1_H']}"
        c = pd.read_csv(os.path.join(DIR[p], "e14_curve.csv")); s = pd.read_csv(os.path.join(DIR[p], "e14_summary.csv"))
        w = pd.read_csv(os.path.join(DIR[p], "e14_weights.csv"))

        def dv(dn, cond):
            r = s[(s.dictionary == dn) & (s.cond == cond)]
            return None if len(r) == 0 else float(r.iloc[0].nmse_mean)

        def wt(dn, cond, member):
            r = w[(w.dictionary == dn) & (w.cond == cond) & (w.member == member)]
            return float(r.weight.mean()) if len(r) else np.nan

        base, full = dv("D-M-STE", O1M), dv("D-M", O1M)
        fin = c[c.kind == "final member"]
        rows = [("straight-through (D-M)", fin[fin.surrogate == "ste"], "D-M", "ste"),
                ("none (four window members)", None, "D-M-STE", None),
                ("converged window member (mean)", fin[fin.surrogate != "ste"], None, None),
                ("early checkpoint, error just below", c[(c.kind == "early checkpoint b") & (c.surrogate == strong)], f"D-M[STE->early {strong}:b]", "early"),
                ("early checkpoint, error just above", c[(c.kind == "early checkpoint a") & (c.surrogate == strong)], f"D-M[STE->early {strong}:a]", "early")]
        for h in (16, 8, 4):
            rows.append((f"hidden width {h}, converged", c[c.kind == f"hidden {h}"], f"D-M[STE->hidden {h}]", "early"))
        numbers[p] = {}
        first = True
        for label, sub, dn, mem in rows:
            if sub is not None and len(sub) == 0:
                continue
            if sub is None:
                mcells = ["--"] * 5
            else:
                m = sub.mean(numeric_only=True)
                mcells = [f"{m.test_nmse_N:.3f}", f"{m.response:.3f}", f"{m.P_n:.3f}", f"{m.slope:.2f}", f"{m.risk_O1M:.3f}"]
                numbers[p][label] = {"nominal": float(m.test_nmse_N), "response": float(m.response), "P_n": float(m.P_n), "slope": float(m.slope), "risk_alone": float(m.risk_O1M)}
            if dn is None or dv(dn, O1M) is None:
                dcells = ["--"] * 4
            else:
                r1, r2 = dv(dn, O1M), dv(dn, O1H)
                rec = (base - r1) / (base - full)
                dcells = [f"{r1:.3f}", f"{r2:.3f}", "--" if mem is None else f"{wt(dn, O1M, mem):.2f}", "--" if dn in ("D-M-STE",) else f"{100 * rec:.0f}" + BS + "%"]
                numbers[p].setdefault(label, {}).update({"dict_O1M": r1, "dict_O1H": r2, "recovered": rec, "weight": None if mem is None else wt(dn, O1M, mem)})
                manifest.append({"table": "S8-E14", "platform": p, "cond": O1M, "method": "W-Q5", "dictionary": dn, "value": r1, "source": f"results/{p}/e14_summary.csv"})
            lines.append(f"{PN[p] if first else ''} & {label} & " + " & ".join(mcells + dcells) + EOL)
            first = False
        lines.append(BS + "midrule")

        # figure: response against nominal error
        tr = c[c.kind == "trajectory"]
        win = tr[tr.surrogate != "ste"].groupby("epoch").mean(numeric_only=True)
        ax.plot(win.test_nmse_N, win.response, "-", color="#1f6f8b", lw=1.0, label="window surrogates along training")
        st = tr[tr.surrogate == "ste"].groupby("epoch").mean(numeric_only=True)
        ax.plot(st.test_nmse_N, st.response, "-", color="#c0392b", lw=1.0, label="straight-through along training")
        f2 = fin.groupby("surrogate").mean(numeric_only=True)
        ax.scatter(f2.loc[f2.index != "ste", "test_nmse_N"], f2.loc[f2.index != "ste", "response"], s=22, color="#1f6f8b", edgecolor="black", lw=0.3, zorder=3, label="converged window members")
        ax.scatter(f2.loc[["ste"], "test_nmse_N"], f2.loc[["ste"], "response"], s=36, marker="^", color="#c0392b", edgecolor="black", lw=0.3, zorder=4, label="straight-through member")
        cap = c[c.kind.str.startswith("hidden")].groupby("kind").mean(numeric_only=True)
        if len(cap):
            ax.scatter(cap.test_nmse_N, cap.response, s=24, marker="s", color="#e59866", edgecolor="black", lw=0.3, zorder=3, label="reduced hidden width (16, 8, 4)")
            for k, r in cap.iterrows():
                ax.annotate(k.replace("hidden ", ""), (r.test_nmse_N, r.response), textcoords="offset points", xytext=(4, 3), fontsize=5.5)
        ax.set_xscale("log"); ax.set_xlabel("nominal test nMSE"); ax.set_ylabel("forward perturbation response")
        ax.set_title(PN[p]); ax.spines[["top", "right"]].set_visible(False)
    lines = lines[:-1] + [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}"]
    open(os.path.join(args.out, "s8_e14.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=6)
    for a, t in zip(axes, "ab"):
        a.text(-0.14, 1.05, t, transform=a.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(args.fig, bbox_inches="tight"); fig.savefig(os.path.splitext(args.fig)[0] + ".png", dpi=200, bbox_inches="tight")
    json.dump(numbers, open(os.path.join(args.out, "e14_numbers.json"), "w"), indent=1)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out, "results_manifest_e14.csv"), index=False)
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
