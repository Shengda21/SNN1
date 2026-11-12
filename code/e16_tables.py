"""Seed-mean markdown tables for the status notes (E16 sweep, control, detach, ultra-wide, E17, E18).

python e16_tables.py > out.md     (reads ../results/<platform>/e16_members.csv)
"""
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
COND = {"sim": ("O1:0.7", "O1:1.0"), "sarcos": ("O1:0.5", "O1:0.7")}
HS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


def load(p):
    m = pd.read_csv(os.path.join(RES, p, "e16_members.csv"))
    m["hwhm"] = m["hwhm"].astype(float)
    return m


def ms(v, d=4):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{d}f}"
    return f"{v.mean():.{d}f}±{v.std(ddof=1):.{d}f}"


def fh(h):
    return "STE" if math.isinf(h) else f"{h:g}"


def sweep():
    print("\n### E16 宽度扫描（峰值归一，3 个种子均值±标准差；test 集）\n")
    for p in ("sim", "sarcos"):
        m = load(p); cm, ch = COND[p]
        w = m[m.block == "w16"]
        ste = m[(m.source == "runs25") & (m.family == "ste") & (m.seed.isin([0, 1, 2]))]
        for col, lab, d in (("test_nmse_N", "nMSE(N)", 4), (f"test_nmse_{cm}", f"nMSE(O1-M, {cm})", 3),
                            (f"test_nmse_{ch}", f"nMSE(O1-H, {ch})", 3), (f"r_{cm}", f"r(O1-M)", 3),
                            (f"beta_{cm}", "β(O1-M)", 3)):
            print(f"\n**{p} — {lab}**\n")
            print("| 族 | " + " | ".join(fh(h) for h in HS) + " | STE（dm_ste s0–2） |")
            print("|---|" + "---|" * (len(HS) + 1))
            for fam in ("tri", "sig", "atan", "sre"):
                cells = [ms(w[(w.family == fam) & (w.hwhm == h)][col], d) for h in HS]
                print(f"| {fam} | " + " | ".join(cells) + f" | {ms(ste[col], d)} |")
        print(f"\n**{p} — 族内 Spearman(HWHM, ·)，3 种子 × 7 宽度**\n")
        print("| 族 | r(O1-M) | β(O1-M) | nMSE(N) | nMSE(O1-M) |")
        print("|---|---|---|---|---|")
        for fam in ("tri", "sig", "atan", "sre"):
            g = w[(w.family == fam) & (w.hwhm <= 4)]
            vals = [spearmanr(g.hwhm, g[c])[0] for c in (f"r_{cm}", f"beta_{cm}", "test_nmse_N", f"test_nmse_{cm}")]
            print(f"| {fam} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")


def control():
    print("\n### 对照组：原稿参数化（峰值 sig k/4、atan α/2）vs 峰值归一，seed 0\n")
    print("| 平台 | 族 | HWHM | 原稿参数化峰值 | nMSE(N) 归一/原稿 | O1-M 归一/原稿 | O1-H 归一/原稿 | r(O1-M) 归一/原稿 | best_epoch 归一/原稿 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p); cm, ch = COND[p]
        for fam in ("sig", "atan"):
            for h in HS:
                a = m[(m.block == "w16") & (m.family == fam) & (m.hwhm == h) & (m.seed == 0)]
                c = m[(m.block == "w16c") & (m.family == fam) & (m.hwhm == h) & (m.seed == 0)]
                if not len(a) or not len(c):
                    continue
                a, c = a.iloc[0], c.iloc[0]
                print(f"| {p} | {fam} | {fh(h)} | {c.peak:.3g} | {a.test_nmse_N:.4f} / {c.test_nmse_N:.4f} | "
                      f"{a['test_nmse_' + cm]:.3f} / {c['test_nmse_' + cm]:.3f} | {a['test_nmse_' + ch]:.3f} / {c['test_nmse_' + ch]:.3f} | "
                      f"{a['r_' + cm]:.3f} / {c['r_' + cm]:.3f} | {int(a.best_epoch)} / {int(c.best_epoch)} |")


def detach():
    print("\n### detach-reset 对照（w16d vs 原复位；3 个种子均值±标准差；HWHM=16 的非 detach 只有 seed 0）\n")
    print("| 平台 | 族 | HWHM | best_dev 原 / detach | best_epoch 原 / detach | nMSE(N) 原 / detach | O1-M 原 / detach | O1-H 原 / detach |")
    print("|---|---|---|---|---|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p); cm, ch = COND[p]
        d = m[m.block == "w16d"]
        for fam in ("ste", "tri", "atan"):
            for h in sorted(d[d.family == fam].hwhm.unique()):
                dd = d[(d.family == fam) & (d.hwhm == h)]
                if fam == "ste":
                    oo = m[(m.source == "runs25") & (m.family == "ste") & (m.seed.isin([0, 1, 2]))]
                else:
                    oo = m[(m.block == "w16") & (m.family == fam) & (m.hwhm == h)]
                cells = []
                for col, dg in (("best_dev", 4), ("best_epoch", 0), ("test_nmse_N", 4), (f"test_nmse_{cm}", 3),
                                (f"test_nmse_{ch}", 3)):
                    if col == "best_epoch":
                        f = lambda s: "—" if not len(s) else ",".join(str(int(x)) for x in s)
                        cells.append(f"{f(oo.sort_values('seed')[col])} / {f(dd.sort_values('seed')[col])}")
                    else:
                        cells.append(f"{ms(oo[col], dg)} / {ms(dd[col], dg)}")
                print(f"| {p} | {fam} | {fh(h)} | " + " | ".join(cells) + " |")


def wide():
    print("\n### 超宽扫描（峰值归一、不 detach，seed 0；HWHM 0.0625–64 与 STE）\n")
    print("| 平台 | 族 | HWHM | best_dev | best_epoch/epochs | nMSE(N) | O1-M | O1-H | r(O1-M) | β(O1-M) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p); cm, ch = COND[p]
        w = m[(m.block == "w16") & (m.seed == 0)].sort_values("hwhm")
        rows = [(fam, r) for fam in ("tri", "atan") for _, r in w[w.family == fam].iterrows()]
        rows += [("ste（dm_ste_s0）", r) for _, r in m[(m.source == "runs25") & (m.family == "ste") & (m.seed == 0)].iterrows()]
        rows += [("ste detach（w16d_ste_s0）", r) for _, r in m[(m.block == "w16d") & (m.family == "ste") & (m.seed == 0)].iterrows()]
        for fam, r in rows:
            print(f"| {p} | {fam} | {fh(r.hwhm)} | {r.best_dev:.4f} | {int(r.best_epoch)}/{int(r.epochs_run)} | {r.test_nmse_N:.4f} | "
                  f"{r['test_nmse_' + cm]:.3f} | {r['test_nmse_' + ch]:.3f} | {r['r_' + cm]:.3f} | {r['beta_' + cm]:.3f} |")
    print("\n**HWHM=4 成员与 STE 成员的 best_epoch / epochs_run**\n")
    print("| 平台 | 组 | 成员 best_epoch/epochs（按种子） |")
    print("|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p)
        groups = [(f"w16 {fam} h4", m[(m.block == "w16") & (m.family == fam) & (m.hwhm == 4.0)]) for fam in ("tri", "sig", "atan", "sre")]
        groups += [(f"w16d {fam} h4", m[(m.block == "w16d") & (m.family == fam) & (m.hwhm == 4.0)]) for fam in ("tri", "atan")]
        groups += [("dm_ste（原复位）", m[(m.source == "runs25") & (m.family == "ste")]),
                   ("w16d_ste（detach）", m[(m.block == "w16d") & (m.family == "ste")])]
        for lab, g in groups:
            g = g.sort_values("seed")
            print(f"| {p} | {lab} | " + ", ".join(f"s{int(r.seed)}:{int(r.best_epoch)}/{int(r.epochs_run)}" for _, r in g.iterrows()) + " |")


def e17():
    print("\n### E17 逐层消融（tri；窄 HWHM=0.25；3 个种子均值±标准差）\n")
    print("| 平台 | 放置 | 宽 HWHM | nMSE(N) | O1-M | O1-H | r(O1-M) | β(O1-M) | 占全宽变化的比例 Δr |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p); cm, ch = COND[p]
        nar = m[(m.block == "w16") & (m.family == "tri") & (m.hwhm == 0.25)]
        rn = nar[f"r_{cm}"].mean()
        cols = ("test_nmse_N", f"test_nmse_{cm}", f"test_nmse_{ch}", f"r_{cm}", f"beta_{cm}")
        print(f"| {p} | 全窄 | 0.25 | " + " | ".join(ms(nar[c], 4 if c == 'test_nmse_N' else 3) for c in cols) + " | 0 |")
        for H in (1.0, 4.0):
            full = m[(m.block == "w16") & (m.family == "tri") & (m.hwhm == H)]
            dr = full[f"r_{cm}"].mean() - rn
            for lab, g in (("仅第 1 层宽", m[(m.block == "w17") & (m.placement == "L1") & (m.hwhm == H)]),
                           ("仅第 2–3 层宽", m[(m.block == "w17") & (m.placement == "L23") & (m.hwhm == H)]),
                           ("全宽（E16）", full)):
                frac = (g[f"r_{cm}"].mean() - rn) / dr if abs(dr) > 1e-9 else float("nan")
                print(f"| {p} | {lab} | {H:g} | " + " | ".join(ms(g[c], 4 if c == 'test_nmse_N' else 3) for c in cols) + f" | {frac:+.2f} |")


def e18():
    print("\n### E18 噪声注入方式与有界激活（3 个种子均值±标准差；强代理原稿宽度）\n")
    print("| 平台 | 方式 | 强度 | best_epoch | nMSE(N) | O1-M | O1-H | r(O1-M) | β(O1-M) | ‖ν‖²(O1-M) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in ("sim", "sarcos"):
        m = load(p); cm, ch = COND[p]
        fr = m[(m.source == "runs25") & (m.block == "dm") & (m.family == ("sig" if p == "sim" else "tri")) & (m.seed.isin([0, 1, 2]))]
        groups = [("基线（dm 强代理）", "—", fr)]
        groups += [("前向膜噪声", f"σ={s:g}", m[(m.block == "w18mem") & (m.membrane_noise == s)]) for s in (0.25, 0.5, 1.0, 2.0)]
        groups += [("输入增广", f"{s:g}×aug", m[(m.block == "w18aug") & (m.aug_scale == s)]) for s in (0.25, 0.5, 1.0, 1.4)]
        groups += [("B-A（runs25 ba，s0–2）", "1×aug", m[(m.source == "runs25") & (m.block == "ba") & (m.seed.isin([0, 1, 2]))])]
        groups += [("逃逸噪声随机前向", f"HWHM={h:g}", m[(m.block == "w18esc") & (m.hwhm == h)]) for h in (0.25, 1.0)]
        groups += [("MLP ReLU（runs25 bn，s0–2）", "—", m[(m.source == "runs25") & (m.block == "bn") & (m.seed.isin([0, 1, 2]))])]
        groups += [(f"MLP {a}", "—", m[(m.block == "w18mlp") & (m.activation == a)]) for a in ("tanh", "crelu", "bin")]
        for lab, s, g in groups:
            ep = ",".join(str(int(x)) for x in g.sort_values("seed").best_epoch)
            print(f"| {p} | {lab} | {s} | {ep} | {ms(g.test_nmse_N, 4)} | {ms(g['test_nmse_' + cm], 3)} | {ms(g['test_nmse_' + ch], 3)} | "
                  f"{ms(g['r_' + cm], 3)} | {ms(g['beta_' + cm], 3)} | {ms(g['nu2_' + cm], 3)} |")


if __name__ == "__main__":
    which = sys.argv[1:] or ["sweep", "control", "detach", "wide", "e17", "e18"]
    for k in which:
        {"sweep": sweep, "control": control, "detach": detach, "wide": wide, "e17": e17, "e18": e18}[k]()
