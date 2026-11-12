"""Supplementary table for S7.8 (E26 control): choosing the weight space by split-conformal interval
length against cross-fitting.

Reads   results/{sim,sarcos}/e26_conformal.csv   (written by code/e26_conformal.py)
Writes  build/tables/s7_conformal.tex       (label tab:s7conformal)
Prints  the numbers quoted in the text (python code/figures/supp_table_s7_conformal.py)

One row = all calibration sets (both dictionaries, every seed and group) of a platform, condition class
and m. Agreement is the share of random splits in which the conformal rule picks the cross-fitted space;
Delta is the mean over cases of (test nMSE of the conformal choice - test nMSE of the cross-fitted
choice), in brackets the standard deviation over the five seeds of the seed means; regret is against the
post-hoc best space of W0-W4 on the test role.
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SNN29_ROOT", Path(__file__).resolve().parents[2]))
OUT = ROOT / "build" / "tables"

CLASSES = {
    "sim": [("N", ["N"]), ("O1-L/M/H", ["O1-L", "O1-M", "O1-H"]), ("O2-B", ["O2B"])],
    "sarcos": [("N", ["N"]), (r"O1-M/H, O1-$q\dot q$", ["O1-M", "O1-H", "O1qqd"])],
}
PLAT = {"sim": "Arm", "sarcos": "SARCOS"}


def num(x, d=4, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$")


def summarize(sub):
    seed_diff = sub.groupby("seed")["diff"].mean()
    return {"n": len(sub), "agree": sub["agree"].mean(), "conf": sub["test_conf"].mean(), "cf": sub["test_cf"].mean(),
            "diff": sub["diff"].mean(), "sd": float(seed_diff.std(ddof=1)),
            "rconf": sub["regret_conf"].mean(), "rcf": sub["regret_cf"].mean()}


def main():
    body, nums = [], {}
    for plat in ("sim", "sarcos"):
        df = pd.read_csv(ROOT / "results" / plat / "e26_conformal.csv")
        for ci, (clab, conds) in enumerate(CLASSES[plat]):
            for mi, m in enumerate(sorted(df["m"].unique())):
                sub = df[df["cond"].isin(conds) & (df["m"] == m)]
                r = summarize(sub)
                nums[(plat, clab, m)] = r
                first = (ci == 0 and mi == 0)
                body.append(" & ".join([PLAT[plat] if first else "", clab if mi == 0 else "", str(int(m)),
                                        f"{100 * r['agree']:.0f}", num(r["conf"]), num(r["cf"]),
                                        f"{num(r['diff'], 4, True)} ({num(r['sd'])})", num(r["rconf"]), num(r["rcf"])]) + r" \\")
            if ci < len(CLASSES[plat]) - 1:
                body.append(r"\cmidrule{2-9}")
        if plat == "sim":
            body.append(r"\midrule")
    cap = (r"Weight-space choice by split-conformal interval length (ten random half-splits of each calibration set, $\alpha=0.1$) "
           r"against cross-fitting: share of splits with the same chosen space (\%), test nMSE of both choices, their difference "
           r"(conformal minus cross-fit; standard deviation of the five seed means in brackets), and regret against the best of W0--W4 on the test role. "
           r"The arm uses $m$ trajectories ($m=1$: five 200-step segments), SARCOS $m$ blocks of 100 rows; means over both dictionaries, five seeds, and all calibration sets.")
    lines = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{cap}}}", r"\label{tab:s7conformal}", r"\scriptsize",
             r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{0.92}", r"\begin{tabular}{lllcccccc}", r"\toprule",
             r"Platform & Condition & $m$ & Agree & nMSE conf. & nMSE CF & $\Delta$ & Regret conf. & Regret CF \\", r"\midrule"]
    lines += body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s7_conformal.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / "s7_conformal.tex")

    # numbers for the text
    for plat in ("sim", "sarcos"):
        df = pd.read_csv(ROOT / "results" / plat / "e26_conformal.csv")
        nz = df[df["cond"] != "N"]
        print(f"[{plat}] agreement all {df['agree'].mean():.3f}, N {df[df['cond'] == 'N']['agree'].mean():.3f}, noisy {nz['agree'].mean():.3f}")
        for m, sub in df.groupby("m"):
            s = summarize(sub); sn = summarize(sub[sub["cond"] != "N"])
            print(f"  m={m}: agree {s['agree']:.3f} (noisy {sn['agree']:.3f}); diff {s['diff']:+.4f} (noisy {sn['diff']:+.4f}); "
                  f"regret conf {s['rconf']:.4f} cf {s['rcf']:.4f}; max regret conf {sub['regret_conf'].max():.3f} cf {sub['regret_cf'].max():.3f}; "
                  f"sd regret conf {sub['regret_conf'].std():.4f} cf {sub['regret_cf'].std():.4f}")
        sm = pd.read_csv(ROOT / "results" / plat / "e26_conformal_summary.csv")
        fc = sm[[f"freq_conf_W{i}" for i in range(5)]].values.argmax(1)
        ff = sm[[f"freq_cf_W{i}" for i in range(5)]].values.argmax(1)
        print(f"  cells {len(sm)}: modal space equal {np.mean(fc == ff):.2f}; conformal better in {(sm['diff'] < 0).sum()} cells; "
              f"max diff {sm['diff'].max():+.4f}")
    d = pd.read_csv(ROOT / "results" / "sim" / "e26_conformal.csv")
    m1 = d[(d["m"] == 1)]
    for lab, sub in (("m=1 without O2B", m1[m1["cond"] != "O2B"]), ("m=1 O2B", m1[m1["cond"] == "O2B"])):
        s = summarize(sub)
        print(f"  sim {lab}: diff {s['diff']:+.4f} regret conf {s['rconf']:.4f} cf {s['rcf']:.4f}")
    for m, sub in d[d["cond"] != "N"].groupby("m"):
        print(f"  sim noisy m={m}: conf picks W4 {sub['freq_W4'].mean():.3f}, cf picks W4 {(sub['cf_selected'] == 'W4').mean():.3f}")


if __name__ == "__main__":
    main()
