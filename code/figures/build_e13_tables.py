"""Tables of the closed-loop follow-up E13 and of the slow/fast residual split E13b.

python build_e13_tables.py --sim <results/sim> --out <tables dir>

Writes table4.tex (main text), s9_cl.tex, s9_pairs.tex, s9_slow.tex,
e13_rank_correlations.json and results_manifest_e13.csv. Inputs: e13_summary.csv,
e13_paired.csv (from summarize_e13.py) and e13b_lowfreq.csv (from analysis_lowfreq.py).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

BS = "\\"
EOL = " " + BS + BS
QQD = "$(q," + BS + "dot q)$"
LABEL = {"PD": "PD only", "B-N": "MLP (B-N)", "B-N|est": "MLP, input estimator", "B-A": "B-A (augmented network)",
         "B-A+(q,qd)": "B-A$^{+}$, noise on " + QQD + " only", "D-M:W-S": "D-M, single (W-S)", "D-S:W-Q5": "D-S",
         "D-M:W-Q5": "D-M", "D-S+null:W-Q5": "D-S + null", "D-M+null:W-Q5": "D-M + null", "D-S|est:W-Q5": "D-S, input estimator",
         "D-M|est:W-Q5": "D-M, input estimator", "D-S,est-optional:W-Q5": "D-S, estimator optional",
         "D-SQ:W-Q5": "D-SA, noise on " + QQD + " only"}
ORDER = ["PD", "B-N", "B-N|est", "B-A", "B-A+(q,qd)", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-M+null:W-Q5",
         "D-S|est:W-Q5", "D-M|est:W-Q5", "D-S,est-optional:W-Q5", "D-SQ:W-Q5"]
MAIN = ["B-N", "D-S:W-Q5", "D-M:W-S", "D-M:W-Q5", "B-A", "D-SQ:W-Q5", "D-S+null:W-Q5", "D-S|est:W-Q5", "B-N|est"]


def spearman(a, b):
    return float(np.corrcoef(pd.Series(list(a)).rank(), pd.Series(list(b)).rank())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S = pd.read_csv(os.path.join(args.sim, "e13_summary.csv")); P = pd.read_csv(os.path.join(args.sim, "e13_paired.csv"))
    lf_full = pd.read_csv(os.path.join(args.sim, "e13b_lowfreq.csv"))      # weights fitted on the full Dcal (reference only)
    O1M = [c for c in lf_full.cond.unique() if c != "N"][0]
    lf = pd.read_csv(os.path.join(args.sim, "e13c_lowfreq_matched.csv"))   # weights of the closed-loop run (E13c)
    lf["cond"] = lf["cond"].map({"N": "N", "O1M": O1M})
    L = lf.groupby(["cond", "method"]).mean(numeric_only=True).reset_index()
    L0 = lf[lf.group == 0].groupby(["cond", "method"]).mean(numeric_only=True).reset_index()   # true-state diagnostic: group 0
    Lfull = lf_full.groupby(["cond", "method"]).mean(numeric_only=True).reset_index()
    manifest = []

    def cl(cond, ctl):
        r = S[(S.cond == cond) & (S.controller == ctl)]
        return None if len(r) == 0 else r.iloc[0]

    def ol(cond, m, tab=None):
        T = L if tab is None else tab
        r = T[(T.cond == cond) & (T.method == m)]
        return None if len(r) == 0 else r.iloc[0]

    # ---------------- closed-loop table (S9)
    lines = [BS + "begin{table}[t]", BS + "centering",
             BS + "caption{Open-loop risk and closed-loop tracking at O1-M on the simulated arm (E13). Open-loop: test nMSE and the nMSE of the 0.25-s moving average of the residual (slow part); W-Q(5) unless stated; mean over five seeds. Tracking: RMSE in $10^{-3}$ rad over 30 episodes, five seeds, and three calibration groups under the shared feedback law, and with the feedback terms computed from the true state (diagnostic, one group). Rows are ordered by tracking error.}",
             BS + "label{tab:cl13}", BS + "footnotesize", BS + "setlength{" + BS + "tabcolsep}{2.4pt}",
             BS + "begin{tabular}{@{}lcccc@{}}", BS + "toprule",
             " & " + BS + "multicolumn{2}{c}{Open loop (nMSE)} & " + BS + "multicolumn{2}{c}{Tracking RMSE}" + EOL,
             "Feedforward model & total & slow part & shared fb. & true-state fb." + EOL, BS + "midrule"]
    main_rows = sorted(MAIN, key=lambda m: cl("O1M", m).rmse)
    for m in main_rows:
        c, t, o = cl("O1M", m), cl("O1M", m + "@true-fb"), ol(O1M, m)
        lines.append(f"{LABEL[m]} & {o.nmse_total:.3f} & {o.nmse_slow_L25:.3f} & {1000 * c.rmse:.1f} & " + ("--" if t is None else f"{1000 * t.rmse:.1f}") + EOL)
        manifest.append({"table": "IV", "platform": "sim", "cond": "O1M", "method": m, "dictionary": "-", "value": c.rmse, "ci_lo": c.ci_lo, "ci_hi": c.ci_hi,
                         "source": "results/sim/e13_summary.csv"})
        manifest.append({"table": "IV", "platform": "sim", "cond": O1M, "method": m + " open-loop total / slow L25", "dictionary": "-", "value": o.nmse_total,
                         "source": "results/sim/e13c_lowfreq_matched.csv"})
    lines += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}"]
    open(os.path.join(args.out, "table4.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------------- S9 full closed-loop table
    cols = [("N", ""), ("O1M", ""), ("O1M", "@true-fb"), ("P1", "")]
    head = ["N", "O1-M", "O1-M, true-state feedback", "P1"]
    lines = [BS + "begin{table}[H]", BS + "centering",
             BS + "caption{Closed loop with the attenuation candidates (E13): tracking RMSE in $10^{-3}$ rad, mean [95" + BS + "% interval of a two-way bootstrap over seeds and episodes], failed episodes included at the threshold. Calibration with $m=10$ trajectories and three groups; feedback gains, references, and failure rule as in E06. The true-state column is a diagnostic in which the feedback terms use the true state and the feedforward the perturbed observation.}",
             BS + "label{tab:s9cl}", BS + "scriptsize", BS + "setlength{" + BS + "tabcolsep}{4pt}", BS + "resizebox{" + BS + "textwidth}{!}{%",
             BS + "begin{tabular}{l" + "c" * len(cols) + "cc}", BS + "toprule",
             "Feedforward model & " + " & ".join(head) + " & Torque RMS at O1-M (N" + BS + ",m) & Saturation at O1-M" + EOL, BS + "midrule"]
    for m in ORDER:
        cells = []
        for cond, sfx in cols:
            r = cl(cond, m + sfx)
            if r is None:
                cells.append("--")
            else:
                cells.append(f"{1000 * r.rmse:.1f} [{1000 * r.ci_lo:.1f}, {1000 * r.ci_hi:.1f}]")
                manifest.append({"table": "S9-cl", "platform": "sim", "cond": cond + sfx, "method": m, "dictionary": "-", "value": r.rmse, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                                 "source": "results/sim/e13_summary.csv"})
        r = cl("O1M", m)
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + f" & {r.tau_rms:.1f} & {100 * r.saturation:.1f}" + BS + "%" + EOL)
    lines += [BS + "bottomrule", BS + "end{tabular}}", BS + "end{table}"]
    open(os.path.join(args.out, "s9_cl.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------------- S9 paired differences against the seed dictionary
    pc = [("N", "D-S:W-Q5", ""), ("O1M", "D-S:W-Q5", ""), ("O1M", "D-S:W-Q5@true-fb", "@true-fb"), ("P1", "D-S:W-Q5", "")]
    lines = [BS + "begin{table}[H]", BS + "centering",
             BS + "caption{Paired differences in tracking RMSE against the seed dictionary under W-Q(5), in $10^{-3}$ rad: mean [95" + BS + "% interval], relative change, and the number of seeds (of five) in which the listed model tracks better. Positive values mean a larger tracking error than the seed dictionary.}",
             BS + "label{tab:s9pairs}", BS + "scriptsize", BS + "setlength{" + BS + "tabcolsep}{4pt}", BS + "resizebox{" + BS + "textwidth}{!}{%",
             BS + "begin{tabular}{l" + "c" * len(pc) + "}", BS + "toprule", "Feedforward model & " + " & ".join(head) + EOL, BS + "midrule"]
    for m in ORDER:
        if m in ("PD", "D-S:W-Q5"):
            continue
        cells = []
        for cond, base, sfx in pc:
            r = P[(P.cond == cond) & (P.a == m + sfx) & (P.b == base)]
            if len(r) == 0:
                cells.append("--")
            else:
                r = r.iloc[0]
                cells.append(f"${1000 * r['diff']:+.2f}$ [${1000 * r.ci_lo:+.2f}$, ${1000 * r.ci_hi:+.2f}$], ${100 * r.rel:+.0f}" + BS + f"%$, {int(r.seeds_a_better)}/{int(r.n_seeds)}")
                manifest.append({"table": "S9-pairs", "platform": "sim", "cond": cond + sfx, "method": m, "dictionary": base, "value": r["diff"], "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                                 "source": "results/sim/e13_paired.csv"})
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + EOL)
    lines += [BS + "bottomrule", BS + "end{tabular}}", BS + "end{table}"]
    open(os.path.join(args.out, "s9_pairs.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---------------- S9 slow / fast split and rank correlations
    lines = [BS + "begin{table}[H]", BS + "centering",
             BS + "caption{Slow part of the open-loop residual (E13c): test nMSE and the nMSE of the moving average of the residual over $L$ control steps along each test trajectory ($L=10$, 25, 50 correspond to 0.1, 0.25, 0.5 s), with the closed-loop tracking RMSE of the same model in $10^{-3}$ rad. Open-loop and tracking columns use the same weights, fitted on the ten trajectories of each calibration group; means over five seeds and the three groups (one group for the uncalibrated networks).}",
             BS + "label{tab:s9slow}", BS + "scriptsize", BS + "setlength{" + BS + "tabcolsep}{4pt}",
             BS + "begin{tabular}{lcccccccccc}", BS + "toprule",
             " & " + BS + "multicolumn{5}{c}{N} & " + BS + "multicolumn{5}{c}{O1-M}" + EOL,
             "Feedforward model & total & $L=10$ & $L=25$ & $L=50$ & tracking & total & $L=10$ & $L=25$ & $L=50$ & tracking" + EOL, BS + "midrule"]
    for m in ORDER:
        if m == "PD":
            continue
        cells = []
        for oc, cc in [("N", "N"), (O1M, "O1M")]:
            o, c = ol(oc, m), cl(cc, m)
            cells += [f"{o.nmse_total:.4f}", f"{o.nmse_slow_L10:.4f}", f"{o.nmse_slow_L25:.4f}", f"{o.nmse_slow_L50:.4f}", f"{1000 * c.rmse:.1f}"]
            manifest.append({"table": "S9-slow", "platform": "sim", "cond": oc, "method": m, "dictionary": "-", "value": o.nmse_slow_L25, "source": "results/sim/e13c_lowfreq_matched.csv"})
        lines.append(f"{LABEL[m]} & " + " & ".join(cells) + EOL)
    corr = {}
    for cc, oc, sfx in [("N", "N", ""), ("O1M", O1M, ""), ("O1M", O1M, "@true-fb")]:
        ms = [m for m in ORDER if m != "PD" and cl(cc, m + sfx) is not None]
        t = [cl(cc, m + sfx).rmse for m in ms]
        tab = L0 if sfx else L          # the true-state diagnostic ran with the weights of group 0
        corr[cc + sfx] = {"n_models": len(ms), "total": spearman([ol(oc, m, tab).nmse_total for m in ms], t)}
        for Lw in (10, 25, 50):
            corr[cc + sfx][f"slow_L{Lw}"] = spearman([ol(oc, m, tab)[f"nmse_slow_L{Lw}"] for m in ms], t)
        corr[cc + sfx + "|full_Dcal_weights"] = {"n_models": len(ms), "total": spearman([ol(oc, m, Lfull).nmse_total for m in ms], t)}
        for Lw in (10, 25, 50):
            corr[cc + sfx + "|full_Dcal_weights"][f"slow_L{Lw}"] = spearman([ol(oc, m, Lfull)[f"nmse_slow_L{Lw}"] for m in ms], t)
    lines.append(BS + "midrule")
    c1, c2, c3 = corr["N"], corr["O1M"], corr["O1M@true-fb"]
    lines.append("Spearman correlation with tracking RMSE & " + " & ".join(f"${c1[k]:+.2f}$" for k in ["total", "slow_L10", "slow_L25", "slow_L50"]) + " & & "
                 + " & ".join(f"${c2[k]:+.2f}$" for k in ["total", "slow_L10", "slow_L25", "slow_L50"]) + " &" + EOL)
    lines.append("Same, true-state feedback (" + str(c3["n_models"]) + " models) & & & & & & " + " & ".join(f"${c3[k]:+.2f}$" for k in ["total", "slow_L10", "slow_L25", "slow_L50"]) + " &" + EOL)
    lines += [BS + "bottomrule", BS + "end{tabular}", BS + "end{table}"]
    open(os.path.join(args.out, "s9_slow.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # per-seed correlations at O1-M (shared feedback): open loop and tracking averaged over the three groups within each seed
    ep = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(args.sim, "e13_episodes_*.csv")))], ignore_index=True)
    ts = ep[ep.cond == "O1M"].groupby(["controller", "seed", "episode"]).rmse.mean().groupby(["controller", "seed"]).mean()
    Lseed = lf[lf.cond == O1M].groupby(["method", "seed"]).mean(numeric_only=True)
    ms13 = [m for m in ORDER if m != "PD" and cl("O1M", m) is not None]
    corr["O1M|per_seed"] = {str(s): {k: spearman([Lseed.loc[(m, s), k] for m in ms13], [ts[(m, s)] for m in ms13])
                                     for k in ["nmse_total", "nmse_slow_L10", "nmse_slow_L25", "nmse_slow_L50"]} for s in sorted(lf.seed.unique())}
    json.dump(corr, open(os.path.join(args.out, "e13_rank_correlations.json"), "w"), indent=1)
    pd.DataFrame(manifest).to_csv(os.path.join(args.out, "results_manifest_e13.csv"), index=False)
    print("E13 tables written;", len(manifest), "manifest rows;", json.dumps({k: {a: round(b, 2) for a, b in v.items()} for k, v in corr.items() if "per_seed" not in k}))


if __name__ == "__main__":
    main()
