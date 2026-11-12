"""Supplementary tables for S8 (needed step: task filters, interaction torque, admittance, tracking;
E27-E29) on the simulated arm.

Reads   results/sim/e27_*.csv, e27_29_summary.json, e28_*.csv/.json, e29_*.csv/.json
Writes  build/tables/s8n_*.tex (new names; the older s8_* and s9_* tables are untouched)
Prints  the numbers quoted in the S8 text (python code/figures/supp_tables_s8.py > s8_numbers.txt)

Conventions: per seed the mean over the calibration groups (groups 0-2; group 0 only for the
uncalibrated networks and for E21's optima), then the mean over five seeds; brackets are the range
over seeds (E27) or 95% two-way cluster-bootstrap intervals over seeds and episodes (E28, E29), as
written by the experiment scripts.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SNN29_ROOT", Path(__file__).resolve().parents[2]))
RES = ROOT / "results" / "sim"
OUT = ROOT / "build" / "tables"

TS = 0.01                                   # control period (s), e27_common.TS
FS = 100.0
KP = np.array([100.0, 40.0])                # e27_common.KP, joints 1-4 / 5-7
KD = np.array([8.0, 3.2])                   # e27_common.KD
DAMP = 1.0                                  # e27_common.JOINT_DAMPING
RHO = 0.7                                   # O1-M level of the arm
REF = "D-S:W-Q5"

LAB = {
    "ID-exact": "Exact ID (noise floor)", "oracle": "Ideal sensor",
    "B-N": "MLP (B-N)", "B-N|est": "MLP, input estimator", "B-A": "B-A (augmented network)",
    "B-A+(q,qd)": r"B-A$^{+}$, noise on $(q,\dot q)$ only", "D-M:W-S": "D-M, single (W-S)",
    "D-S:W-Q5": "D-S", "D-M:W-Q5": "D-M", "D-S+null:W-Q5": "D-S + null", "D-M+null:W-Q5": "D-M + null",
    "D-S|est:W-Q5": "D-S, input estimator", "D-M|est:W-Q5": "D-M, input estimator",
    "D-S,est-optional:W-Q5": "D-S, estimator optional", "D-SQ:W-Q5": r"D-SA, noise on $(q,\dot q)$ only",
    "D-S:W1": "D-S, W1 fitted on the group", "D-S:W3": "D-S, W3 fitted on the group",
    "D-M:W1": "D-M, W1 fitted on the group", "D-M:W3": "D-M, W3 fitted on the group",
    "D-S:e21-W1": "D-S, E21 optimum in W1", "D-S:e21-W3": "D-S, E21 optimum in W3",
    "D-M:e21-W1": "D-M, E21 optimum in W1", "D-M:e21-W3": "D-M, E21 optimum in W3",
    "D-S:W-Q5+b": "D-S, affine offset", "D-S:W-Q5@calN": "D-S, weights from N",
    "D-S+null:W-Q5@calN": "D-S + null, weights from N", "D-S|est:W-Q5@calN": "D-S, input estimator, weights from N",
    "B-N@calN": "MLP (B-N), weights from N",
    "D-S:W-QG5": "D-S, task-weighted W-QG(5)", "D-M:W-QG5": "D-M, task-weighted W-QG(5)",
    "D-S:W-Q5@ref": "D-S, reference feedforward", "B-N@ref": "MLP (B-N), reference feedforward",
}
E13 = ["B-N", "B-N|est", "B-A", "B-A+(q,qd)", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-M+null:W-Q5",
       "D-S|est:W-Q5", "D-M|est:W-Q5", "D-S,est-optional:W-Q5", "D-SQ:W-Q5"]
NUMS = {}


def lab(c):
    return LAB.get(c, c)


def num(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def pct(x, d=0):
    return num(100 * x, d, sign=True) + r"\%"


def write(name, caption, label, colspec, header, body, size=r"\scriptsize", tabcolsep="3pt", resize=False,
          placement="!htbp"):
    lines = [rf"\begin{{table}}[{placement}]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", size,
             rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}", r"\renewcommand{\arraystretch}{0.92}"]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"] + header + [r"\midrule"] + body + [r"\bottomrule",
                                                                                                    r"\end{tabular}" + ("}" if resize else "")]
    lines.append(r"\end{table}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", OUT / f"{name}.tex")


def rd(f, **kw):
    return pd.read_csv(RES / f, **kw)


# ----------------------------------------------------------------------------- S8.1 task filters
def leq_exact_lp(fc):
    a = 1.0 - math.exp(-2 * math.pi * fc * TS)
    return (2.0 - a) / a


def leq_pd_discrete(kp, kd, J=1.0):
    """Exact white-noise reduction of the sampled PD loop J e'' + D e' = r - kp e - kd e' (ZOH), computed from
    its impulse response; independent of J to 0.02 for J in [0.1, 5] kg m^2."""
    from scipy.signal import cont2discrete
    A = np.array([[0.0, 1.0], [0.0, -DAMP / J]]); B = np.array([[0.0], [1.0 / J]])
    Ad, Bd, *_ = cont2discrete((A, B, np.array([[1.0, 0.0]]), np.zeros((1, 1))), TS, method="zoh")
    Acl = Ad - Bd @ np.array([[kp, kd]])
    x = Bd.copy(); h = []
    for _ in range(20000):
        h.append(x[0, 0]); x = Acl @ x
    h = np.array(h)
    return h.sum() ** 2 / (h ** 2).sum()


def gain(L):
    return 1.0 / (1.0 + RHO ** 2 / L)


def t_filters():
    F = rd("e27_filtered.csv")
    F = F[(F.joint.astype(str) == "all") & np.isfinite(F.fc)]
    lq = F.groupby("fc")[["leq_cont", "leq_exact"]].first()
    p = json.load(open(RES / "e28_params_c20.json"))
    Md, Bd = np.array(p["Md"]), np.array(p["Bd"])
    L_adm_c = float(np.mean(2 * Md / (Bd * TS)))
    a_adm = float(np.mean(1.0 - TS * Bd / Md))
    L_adm_x = (1 + a_adm) / (1 - a_adm)
    meta = json.load(open(RES / "e29_taskweighted_meta.json"))
    Lp = meta["L_plan"]
    L_pd_c = 2 * (KD + DAMP) / (KP * TS)
    L_pd_x = [leq_pd_discrete(k, d) for k, d in zip(KP, KD)]
    NUMS["filters"] = {"L_adm_cont": L_adm_c, "pole_adm": a_adm, "L_adm_exact": L_adm_x, "L_pd_cont": L_pd_c.tolist(),
                       "L_pd_exact": L_pd_x, "L_plan": [Lp[0], Lp[-1]], "Md_over_Bd": float(np.mean(Md / Bd))}
    rows = []

    def row(task, filt, par, Lc, Lx):
        rows.append(f"{task} & {filt} & {par} & {num(Lc, 1)} & {num(Lx, 1)} & {num(gain(Lc), 2)} & {num(gain(Lx), 2)} \\\\")
    row("Contact or collision detection", "first-order low-pass", r"$f_c=40$\,Hz", lq.loc[40.0, "leq_cont"], lq.loc[40.0, "leq_exact"])
    row("", "", r"$f_c=20$\,Hz", lq.loc[20.0, "leq_cont"], lq.loc[20.0, "leq_exact"])
    row("Intention recognition", "first-order low-pass", r"$f_c=5$\,Hz", lq.loc[5.0, "leq_cont"], lq.loc[5.0, "leq_exact"])
    row("", "", r"$f_c=2$\,Hz", lq.loc[2.0, "leq_cont"], lq.loc[2.0, "leq_exact"])
    row("Admittance control (E28)", r"$1/(M_ds+B_d)$, velocity output", rf"$M_d/B_d={np.mean(Md / Bd):.1f}$\,s", L_adm_c, L_adm_x)
    row("PD tracking (E13, E29)", "decoupled joint loop", "joints 1--4", L_pd_c[0], L_pd_x[0])
    row("", "", "joints 5--7", L_pd_c[1], L_pd_x[1])
    write("s8n_filters",
          r"Task filters at the 100-Hz control rate: equivalent averaging length $\Leq$ (white-noise power reduction "
          r"relative to a constant bias) in the continuous approximation and for the exact discrete filter, and the "
          r"optimal scalar gain $1/(1+\rho^2/\Leq)$ of a linear predictor with uncorrelated input channels at $\rho=0.7$. "
          r"Admittance parameters from Table~\ref{tab:s8nadmpar} ($M_d/B_d=0.1$\,s on every joint); joint-loop gains "
          r"$K_p=100$, $K_d=8$ (joints 1--4) and $40$, $3.2$ (joints 5--7), joint damping $D=1$\,N\,m\,s/rad. The "
          r"continuous $\Leq$ of the low-pass is floored at 1.",
          "tab:s8nfilters", "lllcccc",
          [r" & & & \multicolumn{2}{c}{$\Leq$} & \multicolumn{2}{c}{gain at $\rho=0.7$} \\",
           r"Task & Filter & Parameter & continuous & exact & continuous & exact \\"], rows)


# ----------------------------------------------------------------------------- S8.2 spurious torque
SP_CANDS = ["B-N", "B-N|est", "B-A", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-S|est:W-Q5",
            "D-S:e21-W1", "D-S:e21-W3", "D-M:e21-W3"]


def t_spurious():
    F = rd("e27_filtered.csv")
    F["joint"] = F.joint.astype(str)
    F["fc"] = F.fc.astype(float)
    cell = F.groupby(["cond", "candidate", "joint", "seed", "fc"], as_index=False)[["rmse_Nm", "nmse"]].mean()
    m = cell.groupby(["cond", "candidate", "joint", "fc"])[["rmse_Nm", "nmse"]].mean()
    ref = cell[cell.candidate == REF].set_index(["cond", "joint", "seed", "fc"])
    rows = []
    out = {}
    for c in SP_CANDS:
        v = lambda cond, fc, j="all", col="rmse_Nm": float(m.loc[(cond, c, j, fc), col])
        cc = cell[(cell.candidate == c) & (cell.cond == "O1M") & (cell.joint == "all")].set_index(["seed", "fc"])
        above5 = above40 = "--"
        if c != REF:
            r5 = ref.loc[("O1M", "all")]
            above5 = int(sum(cc.loc[(s, 5.0), "rmse_Nm"] > r5.loc[(s, 5.0), "rmse_Nm"] for s in range(5)))
            below40 = int(sum(cc.loc[(s, 40.0), "rmse_Nm"] < r5.loc[(s, 40.0), "rmse_Nm"] for s in range(5)))
            above5n = int(sum(cc.loc[(s, 5.0), "nmse"] > r5.loc[(s, 5.0), "nmse"] for s in range(5)))
            above40 = f"{above5}/{above5n}/{below40}"
        out[c] = {fc: v("O1M", fc) for fc in [np.inf, 2.0, 5.0, 20.0, 40.0]}
        out[c]["j2_5"] = v("O1M", 5.0, "2")
        out[c]["N5"] = v("N", 5.0); out[c]["Nraw"] = v("N", np.inf); out[c]["P15"] = v("P1", 5.0); out[c]["P1raw"] = v("P1", np.inf)
        out[c]["nmse5"] = v("O1M", 5.0, col="nmse")
        rows.append(f"{lab(c)} & {num(out[c][np.inf])} & {num(out[c][2.0])} & {num(out[c][5.0])} & {num(out[c][20.0])} & "
                    f"{num(out[c][40.0])} & {num(out[c]['nmse5'], 3)} & {num(out[c]['j2_5'])} & {above40} & "
                    f"{num(out[c]['N5'])} & {num(out[c]['P15'])} \\\\")
    NUMS["spurious"] = {lab(c): {str(k): round(x, 3) for k, x in d.items()} for c, d in out.items()}
    write("s8n_spurious",
          r"Spurious interaction torque on the 40 test trajectories without interaction (E27): RMS over the seven "
          r"joints of $\hat\tau_{\mathrm{ext}}=\hat f(\tilde x)-\tau_{\mathrm{motor}}$ in N\,m, unfiltered (raw) and "
          r"after a causal first-order low-pass at $f_c$; weights of the closed-loop calibration groups. nMSE: mean "
          r"over joints of the per-joint normalized mean square after the 5-Hz filter. Seeds: number of seeds (of "
          r"five) in which the candidate exceeds D-S at 5\,Hz in N\,m / in nMSE, and falls below D-S at 40\,Hz in "
          r"N\,m. N and P1: 5-Hz values.",
          "tab:s8nspur", "lcccccccccc",
          [r" & \multicolumn{8}{c}{O1-M} & N & P1 \\ \cmidrule(lr){2-9}\cmidrule(lr){10-10}\cmidrule(lr){11-11}",
           r"Candidate & raw & 2\,Hz & 5\,Hz & 20\,Hz & 40\,Hz & nMSE, 5\,Hz & joint 2, 5\,Hz & seeds & 5\,Hz & 5\,Hz \\"],
          rows, tabcolsep="2.5pt")


CR_PAIRS = [("D-S+null:W-Q5", REF), ("D-M:W-Q5", REF), ("D-S|est:W-Q5", REF), ("D-S|est:W-Q5", "B-N"), ("B-N|est", REF),
            ("D-S:W1", REF), ("D-S:W3", REF), ("D-S:e21-W1", REF), ("D-S:e21-W3", REF), ("D-M:e21-W1", REF),
            ("D-M:e21-W3", REF)]


def t_crossing():
    CR = rd("e27_crossing.csv", dtype={"seed": str})
    CR = CR[CR.cond == "O1M"]
    S = json.loads(open(RES / "e27_29_summary.json").read().replace("NaN", "null"))
    plan = {(c["a"], c["b"]): c["plan_A6_Hz"] for c in S["e27_crossing_O1M"] if c["metric"] == "nmse"}
    rows, nums = [], {}
    for a, b in CR_PAIRS:
        cells = []
        for met in ("nmse", "rms_Nm", "joint2_Nm"):
            s = CR[(CR.a == a) & (CR.b == b) & (CR.metric == met)]
            mn = s[s.seed == "mean"].iloc[0]
            ps = s[s.seed != "mean"].fc_cross_measured.astype(float)
            nch = int(s[s.seed != "mean"].n_sign_changes.max())
            nums[(a, b, met)] = (mn.fc_cross_measured, ps.min(), ps.max(), mn.fc_cross_pred_cont, mn.fc_cross_pred_exact,
                                 mn.L_cross_pred, nch)
            cells.append((mn, ps, nch))
        (n_, pn, chn), (r_, pr, _), (j_, pj, _) = cells
        pl = plan.get((a, b))
        bname = "D-S" if b == REF else "MLP"
        rows.append(f"{lab(a)} vs {bname} & {num(n_.L_cross_pred)} & {num(n_.fc_cross_measured, 1)} "
                    f"[{num(pn.min(), 1)}, {num(pn.max(), 1)}] & {num(n_.fc_cross_pred_cont, 1)} & "
                    f"{num(n_.fc_cross_pred_exact, 1)} & {pl if pl else '--'} & {num(r_.fc_cross_measured, 1)} "
                    f"[{num(pr.min(), 1)}, {num(pr.max(), 1)}] & {num(r_.fc_cross_pred_cont, 1)} & "
                    f"{num(j_.fc_cross_measured, 1)} \\\\")
    NUMS["crossing"] = {f"{a} vs {b} {m}": [round(float(x), 2) for x in v] for (a, b, m), v in nums.items()}
    write("s8n_cross",
          r"Cut-off frequency at which the filtered risk of the first candidate falls below that of the second (O1-M, "
          r"E27). Measured: first sign change on a log grid of 79 cut-offs from 1 to 50\,Hz of the seed-mean curves "
          r"[range of the five per-seed crossings]; every pair and seed changes sign exactly once. Predicted: "
          r"$L^\star=(W_b-W_a)/(S_a-S_b)$ from the slow and white parts of Table~\ref{tab:s8nsw}, converted with the "
          r"continuous $f_c=f_s/(\pi L^\star)$ or the exact discrete relation; plan: the values computed from the "
          r"original Table XI before E27. N\,m: the same for the RMS over joints in physical units and for joint 2.",
          "tab:s8ncross", "lcccccccc",
          [r" & \multicolumn{5}{c}{nMSE} & \multicolumn{2}{c}{RMS over joints (N\,m)} & joint 2 \\"
           r" \cmidrule(lr){2-6}\cmidrule(lr){7-8}\cmidrule(lr){9-9}",
           r"Pair & $L^\star$ & measured (Hz) & cont. & exact & plan & measured (Hz) & cont. & measured \\"],
          rows, tabcolsep="3pt")


SW_CANDS = ["B-N", "B-N|est", "B-A", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-S|est:W-Q5",
            "D-S:e21-W3", "D-M:e21-W3"]


def t_slowwhite():
    SW = rd("e27_slowwhite.csv")
    SW = SW[SW.joint.astype(str) == "all"]
    g = SW.groupby(["cond", "candidate"])[["total", "slow_L10", "slow_L25", "slow_L50", "S", "W", "S_plus_W", "pred_L50"]].mean()
    SWj = rd("e27_slowwhite.csv"); SWj = SWj[SWj.joint.astype(str) == "2"]
    gj = SWj.groupby(["cond", "candidate"])[["S", "W"]].mean()
    rows, nums = [], {}
    for c in SW_CANDS:
        r = g.loc[("O1M", c)]; rj = gj.loc[("O1M", c)]
        nums[lab(c)] = {k: round(float(v), 4) for k, v in r.items()} | {"S_j2": round(float(rj.S), 4), "W_j2": round(float(rj.W), 4)}
        rows.append(f"{lab(c)} & {num(r.total, 3)} & {num(r.slow_L10, 3)} & {num(r.slow_L25, 3)} & {num(r.S, 3)} & "
                    f"{num(r.W, 3)} & {pct(r.S_plus_W / r.total - 1)} & {num(r.slow_L50, 3)} & {num(r.pred_L50, 3)} & "
                    f"{num(rj.S, 3)} & {num(rj.W, 3)} \\\\")
    NUMS["slowwhite_O1M"] = nums
    write("s8n_sw",
          r"Slow and white parts of the open-loop residual at O1-M (E27; nMSE, mean over joints): total risk, risk of "
          r"the $L$-step moving average for $L=10$ and 25, the fit $\mathrm{slow}(L)=S+W/L$ through these two "
          r"windows, the excess of $S+W$ over the total, and the check at $L=50$ (measured and $S+W/50$). Last two "
          r"columns: $S$ and $W$ of joint 2.",
          "tab:s8nsw", "lcccccccccc",
          [r" & & \multicolumn{2}{c}{moving average} & & & & \multicolumn{2}{c}{$L=50$} & \multicolumn{2}{c}{joint 2} \\"
           r" \cmidrule(lr){3-4}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
           r"Candidate & total & $L=10$ & $L=25$ & $S$ & $W$ & $S+W$ vs total & measured & predicted & $S$ & $W$ \\"],
          rows, tabcolsep="3pt")


# ----------------------------------------------------------------------------- S8.3 external torque
EX_CANDS = ["ID-exact", "B-N", "B-N|est", "B-A", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-S|est:W-Q5",
            "D-S:e21-W3"]
EX_CANDS_N = ["ID-exact", "B-N", "B-N|est", "B-A", "D-M:W-S", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5", "D-S|est:W-Q5",
              "D-S:e21-W3"]          # numbers for the text only


def t_ext():
    EX = rd("e27_ext_summary.csv", dtype={"seed": str, "joint": str})
    mean = EX[EX.seed.isin(["mean"]) | ((EX.seed == "-1") & (EX.candidate == "ID-exact"))]
    mean = mean.drop_duplicates(subset=["cond", "setting", "candidate", "filter", "joint"], keep="last")
    k = mean.set_index(["cond", "setting", "candidate", "filter", "joint"])
    rows, nums = [], {}
    blocks = [("O1M", "a", r"O1-M, setting (a): recorded acceleration"),
              ("O1M", "b", r"O1-M, setting (b): acceleration differenced from the observed velocity"),
              ("N", "a", r"N, setting (a)")]
    for cond, st, title in blocks:
        show = cond != "N"                          # the N block is quoted in the text, not tabulated
        if show:
            rows.append(rf"\multicolumn{{12}}{{l}}{{\emph{{{title}}}}} \\")
        for c in (EX_CANDS if show else EX_CANDS_N):
            g = lambda f, col, j="all": float(k.loc[(cond, st, c, f, j), col])
            auc_w = np.mean([g("5Hz", "auc", str(j)) for j in (5, 6, 7)])
            sd = "--" if c in ("ID-exact", REF) else (
                f"{int(g('5Hz', 'seeds_rmse_above_DS'))}/{int(g('5Hz', 'seeds_auc_below_DS'))}")
            nums[(cond, st, lab(c))] = {"raw": g("raw", "rmse_err"), "5Hz": g("5Hz", "rmse_err"), "20Hz": g("20Hz", "rmse_err"),
                                        "auc5": g("5Hz", "auc"), "auc5_j2": g("5Hz", "auc", "2"), "auc5_j567": auc_w,
                                        "det": g("5Hz", "det_rate_cfar"), "lat": g("5Hz", "lat_med_ms_cfar"),
                                        "far": g("5Hz", "far_per_min_cfar"), "det99": g("5Hz", "det_rate_cfar99"),
                                        "lat99": g("5Hz", "lat_med_ms_cfar99"), "far99": g("5Hz", "far_per_min_cfar99"),
                                        "auc_raw": g("raw", "auc"), "auc20": g("20Hz", "auc")}
            if not show:
                continue
            rows.append(f"{lab(c)} & {num(g('raw', 'rmse_err'))} & {num(g('5Hz', 'rmse_err'))} & {num(g('20Hz', 'rmse_err'))} & "
                        f"{num(g('5Hz', 'auc'), 3)} & {num(g('5Hz', 'auc', '2'), 2)} & {num(auc_w, 2)} & "
                        f"{num(g('5Hz', 'det_rate_cfar'), 2)} & {num(g('5Hz', 'lat_med_ms_cfar'), 0)} & "
                        f"{num(g('5Hz', 'far_per_min_cfar'), 0)} & {num(g('20Hz', 'auc'), 3)} & {sd} \\\\")
    NUMS["ext"] = {f"{a}|{b}|{c}": {kk: round(v, 3) for kk, v in d.items()} for (a, b, c), d in nums.items()}
    write("s8n_ext",
          r"Interaction-torque estimation with external torques (E27, 40 trajectories, 265 event--joint pairs): RMSE of "
          r"the filtered estimate against the equally filtered $\tau_{\mathrm{ext}}$ in N\,m, pooled over joints "
          r"(raw, 5\,Hz, 20\,Hz); sample-level AUC of $|\mathrm{LP}(\hat\tau_{\mathrm{ext}})|$ for `event active on the "
          r"joint' at 5\,Hz, pooled, on joint 2, and the mean of joints 5--7; event detection with the per-joint "
          r"CFAR threshold (95th percentile of $|\mathrm{LP}(\hat\tau_{\mathrm{ext}})|$ on the calibration group, "
          r"5\,Hz): detection rate, median latency from onset, false alarms per minute of quiet time; AUC at 20\,Hz; "
          r"seeds (of five) in which the 5-Hz RMSE exceeds and the 5-Hz AUC falls below that of D-S. Exact ID: the "
          r"plant's inverse dynamics at the observed input (no seeds).",
          "tab:s8next", "lccccccccccc",
          [r" & \multicolumn{3}{c}{RMSE (N\,m)} & \multicolumn{3}{c}{AUC, 5\,Hz} & \multicolumn{3}{c}{CFAR, 5\,Hz} & AUC & \\"
           r" \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
           r"Candidate & raw & 5\,Hz & 20\,Hz & pooled & joint 2 & joints 5--7 & detected & latency (ms) & alarms/min & "
           r"20\,Hz & seeds \\"],
          rows, tabcolsep="2.5pt", resize=True)


# ----------------------------------------------------------------------------- S8.4 admittance
def t_adm_params():
    rows = []
    P = {c: json.load(open(RES / f"e28_params_{c}.json")) for c in ("c20", "c50")}
    for c, key, name, d in [("c20", "Bd", r"$B_d$ (N\,m\,s/rad), $c=20$", 1), ("c20", "Md", r"$M_d$ (kg\,m$^2$), $c=20$", 2),
                            ("c50", "Bd", r"$B_d$ (N\,m\,s/rad), $c=50$", 1), ("c50", "Md", r"$M_d$ (kg\,m$^2$), $c=50$", 2),
                            ("c20", "Kh", r"$K_h$ (N\,m/rad), both", 0), ("c20", "Bh", r"$B_h$ (N\,m\,s/rad), both", 1)]:
        rows.append(name + " & " + " & ".join(num(v, d) for v in P[c][key]) + r" \\")
    assert np.allclose(P["c20"]["Kh"], P["c50"]["Kh"]) and np.allclose(P["c20"]["Bh"], P["c50"]["Bh"])
    NUMS["adm_params"] = {"intent_rms_rad": P["c20"]["intent_rms_rad"], "L_eq": P["c20"]["L_eq"]}
    write("s8n_admpar",
          r"Admittance and human-model parameters of E28 per joint: $B_d=c\,s_l$ with $s_l$ the training torque "
          r"standard deviation of joint $l$, $M_d=0.1\,B_d$ ($\Leq=2M_d/(B_dT_s)=20$); human stiffness and damping "
          r"$K_h=100\,s_l$, $B_h=(20/3)\,s_l$ in both settings ($K_h=5B_d$, $B_h=B_d/3$ at $c=20$).",
          "tab:s8nadmpar", "lccccccc",
          [r"Parameter & joint 1 & 2 & 3 & 4 & 5 & 6 & 7 \\"], rows, tabcolsep="4pt")


ADM_ROWS = ["oracle", "ID-exact", "B-N", "B-N|est", "B-A", "D-S:W-Q5", "D-M:W-Q5", "D-S+null:W-Q5",
            "D-S|est:W-Q5", "D-S:e21-W3", "D-M:e21-W3", "D-S:W-Q5+b", "D-S:W-Q5@calN"]
ADM_COLS = [(20.0, "N", "", r"N, $c=20$"), (20.0, "O1M", "", r"O1-M, $c=20$"),
            (20.0, "O1M", "@true-fb", r"O1-M, $c=20$, true state"), (50.0, "O1M", "", r"O1-M, $c=50$"),
            (20.0, "P1", "", r"P1, $c=20$"), (50.0, "P1", "", r"P1, $c=50$")]


def t_adm():
    S = rd("e28_summary.csv").set_index(["c_bd", "cond", "candidate"])
    rows_d, rows_i, nums = [], [], {}
    for c in ADM_ROWS:
        cd, ci = [], []
        for cbd, cond, sfx, _ in ADM_COLS:
            key = (cbd, cond, c + sfx)
            if key not in S.index:
                cd.append("--"); ci.append("--")
                continue
            r = S.loc[key]
            nums[(cbd, cond, c + sfx)] = {k: round(float(r[k]), 4) for k in
                                          ["free_drift_final_max", "free_drift_final_max_lo", "free_drift_final_max_hi",
                                           "free_fail", "free_limit_hit", "free_drift_gt_0p5", "free_tau_hat5_rms",
                                           "intent_tau_h_rms", "intent_intent_err_rms", "intent_fail"]}
            f = r.free_fail
            cd.append(num(r.free_drift_final_max, 2) + (f" ({100 * f:.0f}\\%)" if f > 0 else ""))
            fi = r.intent_fail
            ci.append(f"{num(r.intent_tau_h_rms, 2)} / {num(1e3 * r.intent_intent_err_rms, 0)}" + (f" ({100 * fi:.0f}\\%)" if fi > 0 else ""))
        rows_d.append(f"{lab(c)} & " + " & ".join(cd) + r" \\")
        rows_i.append(f"{lab(c)} & " + " & ".join(ci) + r" \\")
    NUMS["adm"] = {f"{a}|{b}|{c}": d for (a, b, c), d in nums.items()}
    hdr = ["Estimator & " + " & ".join(h for *_, h in ADM_COLS) + r" \\"]
    write("s8n_adm",
          r"Admittance loop (E28), episodes without the human: drift of the admittance reference after 10\,s, "
          r"$\max_l|q_{r,l}(10\,\mathrm{s})-q_{0,l}|$ in rad, mean over seeds and the 30 start poses (failed episodes "
          r"counted at 0.5\,rad), with the failed fraction in parentheses when nonzero. $c$: $B_d=c\,s_l$; noisy "
          r"feedback unless marked; true state: feedback terms from the true state (diagnostic). Ideal sensor: the true "
          r"human torque one step late (zero here); exact ID: the plant's inverse dynamics at the observed input as "
          r"feedforward and estimator. Affine offset and weights from N: P1 recalibration variants (S8.4).",
          "tab:s8nadm", "lcccccc", hdr, rows_d, tabcolsep="3pt")
    # transparency (episodes with the human, rows_i) is quoted in the S8.4 text from NUMS["adm"]; no table


PAIR_ROWS = ["B-N", "B-N|est", "B-A", "D-M:W-S", "D-M:W-Q5", "D-S+null:W-Q5", "D-S|est:W-Q5", "D-S:e21-W3",
             "D-M:e21-W3", "D-S:W-Q5+b", "D-S:W-Q5@calN"]
PAIR_COLS = [(20.0, "N", "", r"N, $c=20$"), (20.0, "O1M", "", r"O1-M, $c=20$"),
             (20.0, "O1M", "@true-fb", r"O1-M, $c=20$, true state"), (50.0, "O1M", "", r"O1-M, $c=50$"),
             (20.0, "P1", "", r"P1, $c=20$")]   # P1 at c=50: |mean difference| <= 0.0053 rad (caption)


def t_adm_pairs():
    P = rd("e28_paired.csv")
    P = P[(P.metric == "drift_final_max") & (P.episode_type == "free")].set_index(["c_bd", "cond", "a"])
    rows, nums = [], {}
    for c in PAIR_ROWS:
        cells = []
        for cbd, cond, sfx, _ in PAIR_COLS:
            key = (cbd, cond, c + sfx)
            if key not in P.index:
                cells.append("--"); continue
            r = P.loc[key]
            nums[f"{cbd}|{cond}|{c + sfx}"] = [round(float(x), 3) for x in (r["diff"], r.ci_lo, r.ci_hi, r.rel)] + [int(r.seeds_a_larger)]
            cells.append(f"{num(r['diff'], 3, True)} [{num(r.ci_lo, 3)}, {num(r.ci_hi, 3)}], {int(r.seeds_a_larger)}/{int(r.n_seeds)}")
        rows.append(f"{lab(c)} & " + " & ".join(cells) + r" \\")
    NUMS["adm_pairs"] = nums
    write("s8n_admpairs",
          r"Admittance drift paired with D-S in the same setting (E28): difference in rad, mean [95\% two-way "
          r"cluster-bootstrap interval over seeds and start poses], and the number of seeds (of five) in which the "
          r"listed estimator drifts more. Columns as in Table~\ref{tab:s8nadm}; positive values mean more drift than D-S. "
          r"Under P1 with $c=50$ every mean difference is within $\pm0.006$\,rad.",
          "tab:s8nadmpairs", "lccccc",
          ["Estimator & " + " & ".join(h for *_, h in PAIR_COLS) + r" \\"], rows, tabcolsep="2.5pt", resize=True)


# ----------------------------------------------------------------------------- S8.5 tracking
TR_ROWS = [("D-S:W-Q5", None), ("D-S:W-Q5@ref", "D-S:W-Q5"), ("B-N", None), ("B-N@ref", "B-N"),
           ("D-S:W-QG5", "D-S:W-Q5"), ("D-M:W-Q5", None), ("D-M:W-QG5", "D-M:W-Q5")]
TR_COLS = [("N", ""), ("O1M", ""), ("O1M", "@true-fb"), ("P1", "")]


def t_track():
    S = rd("e29_tracking_summary.csv").set_index(["cond", "controller"])
    P = rd("e29_tracking_paired.csv").set_index(["cond", "a", "b"])
    rows, nums = [], {}
    for c, base in TR_ROWS:
        cells = []
        for cond, sfx in TR_COLS:
            ctl = c.replace("@ref", "@ref" + sfx) if "@ref" in c else c + sfx
            if (cond, ctl) not in S.index:
                cells.append("--"); continue
            r = S.loc[(cond, ctl)]
            cell = f"{num(1e3 * r.rmse, 1)} [{num(1e3 * r.ci_lo, 1)}, {num(1e3 * r.ci_hi, 1)}]"
            nums[f"{cond}|{ctl}"] = [round(1e3 * float(x), 2) for x in (r.rmse, r.ci_lo, r.ci_hi)]
            if base is not None:
                bctl = base + sfx
                if (cond, bctl) in S.index:
                    rel = r.rmse / S.loc[(cond, bctl), "rmse"] - 1
                    k = (cond, ctl, bctl)
                    sd = f", {int(P.loc[k, 'seeds_a_better'])}/5" if k in P.index else ""
                    cell += f"; {pct(rel, 1)}{sd}"
                    nums[f"{cond}|{ctl}|rel"] = round(float(rel), 4)
            cells.append(cell)
        rows.append(f"{lab(c)} & " + " & ".join(cells) + r" \\")
    NUMS["track"] = nums
    write("s8n_track",
          r"Tracking supplements (E29): RMSE in $10^{-3}$\,rad, mean [95\% two-way bootstrap interval]; after the "
          r"semicolon, the change against the same model with measured-state feedforward and W-Q(5) weights, and the "
          r"seeds (of five) with the smaller error. Reference feedforward: the predictor on $(q_{\mathrm{ref}},\dot "
          r"q_{\mathrm{ref}},\ddot q_{\mathrm{ref}})$, weights fitted on clean calibration inputs. W-QG(5): budgeted "
          r"simplex weights fitted after the joint-loop filter $C_G$ (with null it equals D-S). Three calibration groups "
          r"in every column.",
          "tab:s8ntrack", "lcccc",
          [r"Feedforward & N & O1-M & O1-M, true-state feedback & P1 \\"], rows, tabcolsep="3pt", resize=True)


def t_taskweighted():
    T = rd("e29_taskweighted.csv", dtype={"seed": str})
    m = T[T.seed == "mean"].set_index(["cond", "candidate"])
    meta = json.load(open(RES / "e29_taskweighted_meta.json"))
    rows, nums = [], {}
    order = sorted(E13, key=lambda c: m.loc[("O1M", c), "meas_rmse_mean"])
    for c in order:
        r = m.loc[("O1M", c)]; rn = m.loc[("N", c)]
        nums[lab(c)] = {"meas": 1e3 * r.meas_rmse_mean, "agg": 1e3 * r.pred_agg, "joint": 1e3 * r.pred_joint,
                        "cg": 1e3 * r.pred_cg, "ratio_joint": r.meas_rmse_mean / r.pred_joint, "err_cg": r.err_cg_rel,
                        "share_slow_j2": r.share_slow_joint_j2, "S_j2": r.S_j2, "W_j2": r.W_j2,
                        "N_meas": 1e3 * rn.meas_rmse_mean, "N_joint": 1e3 * rn.pred_joint, "N_cg": 1e3 * rn.pred_cg,
                        "N_err_cg": rn.err_cg_rel}
        rows.append(f"{lab(c)} & {num(1e3 * r.meas_rmse_mean, 1)} & {num(1e3 * r.pred_agg, 1)} & {num(1e3 * r.pred_joint, 1)} & "
                    f"{num(1e3 * r.pred_cg, 1)} & {num(r.meas_rmse_mean / r.pred_joint, 2)} & {pct(r.err_cg_rel)} & "
                    f"{num(r.share_slow_joint_j2, 2)} & {num(r.S_j2, 3)} & {num(r.W_j2, 3)} & "
                    f"{num(1e3 * rn.meas_rmse_mean, 1)} & {num(1e3 * rn.pred_joint, 1)} & {num(1e3 * rn.pred_cg, 1)} \\\\")
    sp = meta["spearman_pred_vs_meas"]
    rows.append(r"\midrule")
    rows.append(rf"Spearman correlation with measured & & {num(sp['O1M']['agg'], 2, True)} & {num(sp['O1M']['joint'], 2, True)} & "
                rf"{num(sp['O1M']['cg'], 2, True)} & & & & & & & {num(sp['N']['joint'], 2, True)} & {num(sp['N']['cg'], 2, True)} \\")
    NUMS["taskweighted"] = {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in nums.items()}
    NUMS["taskweighted_meta"] = {"spearman": sp, "c_l_mrad2": meta["c_l_mrad2"], "joint2_weight": meta["joint2_weight"],
                                 "c_sum": sum(meta["c_l_mrad2"])}
    write("s8n_tw",
          r"Task-weighted prediction of the tracking error (E29) in $10^{-3}$\,rad. Measured: true-state feedback at "
          r"O1-M (three groups) and the shared law under N. Predictions from the open-loop residual of the same weights: "
          r"joint average, $\sum_l c_l(\bar S+\bar W/L_l)$ with joint means $\bar S,\bar W$ (the plan's version); per "
          r"joint, $\sum_l c_l(S_l+W_l/L_l)$, $c_l=(s_l/K_{p,l})^2/7$, $L_l=18$ (joints 1--4) and 21 (joints 5--7); "
          r"$C_G$, residual filtered through the decoupled discrete joint loop. $C_G$ error: prediction minus measured, "
          r"relative. Slow share: fraction of $\sum_l c_lS_l$ on joint 2.",
          "tab:s8ntw", "lcccccccccccc",
          [r" & \multicolumn{9}{c}{O1-M, true-state feedback} & \multicolumn{3}{c}{N} \\"
           r" \cmidrule(lr){2-10}\cmidrule(lr){11-13}",
           r"Feedforward model & measured & joint avg. & per joint & $C_G$ & meas./per joint & $C_G$ error & "
           r"slow share j2 & $S_2$ & $W_2$ & measured & per joint & $C_G$ \\"],
          rows, tabcolsep="2.5pt", resize=True)


def main():
    t_filters()
    t_spurious()
    t_crossing()
    t_slowwhite()
    t_ext()
    t_adm_params()
    t_adm()
    t_adm_pairs()
    t_track()
    t_taskweighted()
    print(json.dumps(NUMS, indent=1, default=float))


if __name__ == "__main__":
    main()
