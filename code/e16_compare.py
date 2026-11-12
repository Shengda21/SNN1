"""Markdown tables for the status notes, read from ../results/<platform>/e16_members.csv.

python e16_compare.py detach   detach-reset (w16d) vs original reset (w16 / runs25 dm_ste) at the
                               same (family, HWHM, seed)
python e16_compare.py wide     seed-0 sweep HWHM 0.0625 ... 64 and STE per family, plus the
                               best epochs of every HWHM = 4 member and of the STE members
Columns: best_dev, best_epoch, test nMSE under N, O1-M and O1-H (sim rho 0.7 / 1.0,
SARCOS rho 0.5 / 0.7).
"""
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
COND = {"sim": ("O1:0.7", "O1:1.0"), "sarcos": ("O1:0.5", "O1:0.7")}


def load(p):
    m = pd.read_csv(os.path.join(RES, p, "e16_members.csv"))
    m["hwhm"] = m["hwhm"].astype(float)
    return m


def fmt_h(h):
    return "STE" if math.isinf(h) else f"{h:g}"


def vals(r, p):
    m, h = COND[p]
    return (f"{r.best_dev:.4f}", f"{int(r.best_epoch)}/{int(r.epochs_run)}", f"{r['test_nmse_N']:.4f}",
            f"{r['test_nmse_' + m]:.3f}", f"{r['test_nmse_' + h]:.3f}")


def detach():
    print("| 平台 | 族 | HWHM | 种子 | best_dev 原/detach | best_epoch 原/detach | nMSE(N) 原/detach | O1-M 原/detach | O1-H 原/detach |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p)
        d = m[m.block == "w16d"].sort_values(["family", "hwhm", "seed"])
        for _, r in d.iterrows():
            if r.family == "ste":
                o = m[(m.source == "runs25") & (m.block == "dm") & (m.family == "ste") & (m.seed == r.seed)]
            else:
                o = m[(m.block == "w16") & (m.family == r.family) & (m.hwhm == r.hwhm) & (m.seed == r.seed)]
            vd = vals(r, p)
            vo = vals(o.iloc[0], p) if len(o) else ("—",) * 5
            cells = " | ".join(f"{a} / {b}" for a, b in zip(vo, vd))
            print(f"| {p} | {r.family} | {fmt_h(r.hwhm)} | {int(r.seed)} | {cells} |")


def wide():
    print("| 平台 | 族 | HWHM | best_dev | best_epoch | nMSE(N) | O1-M | O1-H |")
    print("|---|---|---|---|---|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p)
        w = m[(m.block == "w16") & (m.seed == 0)].sort_values(["family", "hwhm"])
        for fam in ("tri", "atan", "sig", "sre"):
            for _, r in w[w.family == fam].iterrows():
                print(f"| {p} | {fam} | {fmt_h(r.hwhm)} | " + " | ".join(vals(r, p)) + " |")
        s = m[(m.source == "runs25") & (m.block == "dm") & (m.family == "ste") & (m.seed == 0)]
        for _, r in s.iterrows():
            print(f"| {p} | ste (dm_ste_s0) | STE | " + " | ".join(vals(r, p)) + " |")
    print()
    print("| 平台 | 成员 | best_epoch / epochs_run | best_dev |")
    print("|---|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p)
        sel = m[((m.block == "w16") & (m.hwhm == 4.0)) | ((m.source == "runs25") & (m.family == "ste"))
                | ((m.block == "w16d") & (m.family == "ste"))]
        for _, r in sel.sort_values(["block", "family", "seed"]).iterrows():
            print(f"| {p} | {r.member_id} | {int(r.best_epoch)}/{int(r.epochs_run)} | {r.best_dev:.4f} |")


if __name__ == "__main__":
    {"detach": detach, "wide": wide}[sys.argv[1]]()
