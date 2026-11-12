"""Fig. 6 (fig:task): the task layer of the simulated arm at O1-M (E27-E29), one row of three panels.

(a) Spurious interaction torque after a causal first-order low-pass at cutoff fc (E27, zero-cost part,
    40 test trajectories without interaction): filtered nMSE (mean over joints of the per-joint normalized
    MSE, e27_filtered_fine.csv) minus that of the seed dictionary D-S, against fc (log axis, 2-40 Hz).
    Per seed the mean over the calibration groups, then the mean over five seeds. Markers on the zero line:
    crossings with D-S, the 'nmse' rows of e27_crossing.csv (seed == 'mean', fc_cross_measured), printed in
    the legend; open circle: D-S + estimator falls below the MLP (same file, pair (D-S|est:W-Q5, B-N)).
(b) Left, interaction-torque estimation with external torques (E27 new simulation, e27_ext_summary.csv,
    cond O1M, filter 5Hz, joint 'all', seed 'mean'): estimation-error RMSE (rmse_err, N m) and sample-level
    AUC, bars = min-max over seeds; filled, acceleration setting (a); open, setting (b).
    Right, admittance drift (E28, e28_paired.csv, drift_final_max, free episodes): candidate minus D-S with
    the 95% two-way bootstrap interval; filled, c_bd 50 with noisy feedback; open, c_bd 20 with true-state
    feedback ('@true-fb'); off-scale values as arrows.
(c) Task-weighted prediction of the tracking RMSE against the measured RMSE (E29, e29_taskweighted.csv,
    cond O1M, seed 'mean', true-state feedback): pred_joint (filled) and pred_cg (open) joined per model;
    Spearman correlations from e29_taskweighted_meta.json; grey, the other E13 models.
One legend below the row identifies the candidates in all panels (colours, line styles and markers from
common_style.TASK; the panel uses shorter names than TASK['label']).
Run:  python fig6_task.py      (writes build/fig/fig6.pdf; SNN29_PNG=<dir> also writes a PNG)
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common_style as cs

cs.FS = cs.FS_SMALL = 6.5  # compact version for the 12-page budget (never below 6.5 pt)
cs.FS_PANEL = 7.5

REF = "D-S:W-Q5"
COND = "O1M"
FC_RANGE = (2.0, 40.0)
FIG_H = 2.8

SHORT = {"ID-exact": "Exact ID", "B-N": "MLP", "B-N|est": "MLP + est.", "B-A": "B-A", "D-S:W-Q5": "D-S",
         "D-M:W-Q5": "D-M", "D-M:W-S": "D-M single", "D-S+null:W-Q5": "D-S + null", "D-S|est:W-Q5": "D-S + est.",
         "D-S:e21-W3": "D-S, E21 W3", "D-M:e21-W3": "D-M, E21 W3"}

# (a) curves; crossing partner whose value is printed in the legend
A_CANDS = ["B-N", "D-M:W-Q5", "D-S:e21-W3", "D-S+null:W-Q5", "D-S|est:W-Q5", "B-N|est", "B-A"]
A_EXTRA = ("D-S|est:W-Q5", "B-N")
A_BANDS = [((2.0, 5.0), "intention"), ((20.0, 40.0), "contact")]
# (b) rows from top to bottom; drift settings (c_bd, candidate suffix, filled marker, legend text)
B_CANDS = ["ID-exact", "B-N", "D-S:W-Q5", "D-M:W-Q5", "B-A", "D-S:e21-W3", "D-M:e21-W3", "D-S+null:W-Q5",
           "B-N|est", "D-S|est:W-Q5"]
B_DRIFT = [(50.0, "", True, r"$B_d=50\,s_l$"), (20.0, "@true-fb", False, r"$B_d=20\,s_l$, true state")]
B_XMAX = 0.85
# (c) coloured models (the other E13 models in grey)
C_KEY = ["B-N", "D-S:W-Q5", "D-M:W-S", "D-M:W-Q5", "B-A", "D-S+null:W-Q5", "B-N|est", "D-S|est:W-Q5"]
# shared legend, column-major in three rows
LEG = ["D-S:W-Q5", "B-N", "D-M:W-Q5", "D-M:W-S", "B-A", "D-S:e21-W3", "D-S+null:W-Q5", "D-S|est:W-Q5", "B-N|est"]


def ax_in(fig, l, b, w, h):
    """Axes placed in inches from the lower-left corner of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes([l / W, b / H, w / W, h / H])


def mstyle(st, filled=None, ms=3.8):
    f = st["filled"] if filled is None else filled
    star = st["marker"] == "*"
    return dict(marker=st["marker"], ms=ms * (1.35 if star else 1.0), mec=st["color"],
                mfc=st["color"] if f else "white", mew=0.8)


# ----------------------------------------------------------------------------- data
def load_a():
    fi = cs.read("sim", "e27_filtered_fine.csv", ["seed", "cond", "group", "candidate", "fc", "nmse"])
    fi = fi[fi.cond == COND]
    per_seed = fi.groupby(["candidate", "seed", "fc"], as_index=False)["nmse"].mean()
    curve = per_seed.groupby(["candidate", "fc"])["nmse"].mean().unstack("fc")
    cr = cs.read("sim", "e27_crossing.csv", ["cond", "a", "b", "metric", "seed", "fc_cross_measured"])
    cr = cr[(cr.cond == COND) & (cr.metric == "nmse") & (cr.seed.astype(str) == "mean")]
    return curve, {(r.a, r.b): float(r.fc_cross_measured) for _, r in cr.iterrows()}


def load_b():
    ex = cs.read("sim", "e27_ext_summary.csv", ["cond", "setting", "candidate", "filter", "joint", "seed", "rmse_err",
                                                 "auc", "rmse_err_min", "rmse_err_max", "auc_min", "auc_max"])
    ex = ex[(ex.cond == COND) & (ex["filter"] == "5Hz") & (ex.joint.astype(str) == "all")
            & (ex.seed.astype(str) == "mean")]
    pr = cs.read("sim", "e28_paired.csv", ["c_bd", "cond", "a", "b", "episode_type", "metric", "diff", "ci_lo", "ci_hi"])
    pr = pr[(pr.cond == COND) & (pr.metric == "drift_final_max") & (pr.episode_type == "free")]
    return ex, pr


def load_c():
    tw = cs.read("sim", "e29_taskweighted.csv", ["cond", "candidate", "seed", "pred_joint", "pred_cg", "meas_rmse_mean"])
    tw = tw[(tw.cond == COND) & (tw.seed.astype(str) == "mean")].copy()
    meta = json.load(open(cs.RES / "sim" / "e29_taskweighted_meta.json"))
    return tw, meta["spearman_pred_vs_meas"][COND]


# ----------------------------------------------------------------------------- (a)
def panel_a(ax, curve, cross):
    fc_all = curve.columns.values.astype(float)
    sel = (fc_all >= FC_RANGE[0] * 0.999) & (fc_all <= FC_RANGE[1] * 1.001)
    fc = fc_all[sel]
    ref = curve.loc[REF].values[sel]
    for (lo, hi), name in A_BANDS:
        ax.axvspan(lo, hi, color="#f1f0eb", lw=0, zorder=0)
        ax.text(np.sqrt(lo * hi), 0.975, name, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=cs.FS_SMALL, color=cs.INK2)
    ax.axhline(0, color=cs.TASK[REF]["color"], lw=1.3, zorder=1)
    diffs = {}
    for cand in A_CANDS:
        st = cs.task_style(cand)
        diffs[cand] = curve.loc[cand].values[sel] - ref
        ax.plot(fc, diffs[cand], color=st["color"], ls=st["ls"], lw=1.0, zorder=3)
        if (cand, REF) in cross:
            ax.plot([cross[(cand, REF)]], [0.0], ls="none", zorder=6, **mstyle(st, ms=4.2))
    a, b = A_EXTRA
    if (a, b) in cross:
        fx = cross[(a, b)]
        yx = np.interp(np.log(fx), np.log(fc), diffs[b])
        ax.plot([fx], [yx], ls="none", marker="o", ms=6.0, mfc="none", mec=cs.INK, mew=0.8, zorder=7)
    ax.set_xscale("log")
    ax.set_xlim(*FC_RANGE)
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(["2", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_ylim(-0.095, 0.088)
    ax.set_yticks([-0.08, -0.04, 0.0, 0.04, 0.08])
    ax.set_yticklabels(["\u22120.08", "\u22120.04", "0", "0.04", "0.08"])
    ax.set_xlabel("Low-pass cutoff $f_c$ (Hz)")
    ax.set_ylabel("Spurious torque, nMSE minus D-S")


# ----------------------------------------------------------------------------- (b)
def dot_row(ax, y, x, lo, hi, st, filled):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo, hi], [y, y], color=st["color"], lw=0.9, solid_capstyle="butt", zorder=3)
    ax.plot([x], [y], ls="none", zorder=5, **mstyle(st, filled=filled))


def panel_b(ax_e, ax_auc, ax_d, ex, pr):
    rows = list(B_CANDS)
    n = len(rows)
    ypos = {c: n - 1 - i for i, c in enumerate(rows)}
    off = 0.2
    for set_, dy, filled in [("a", off, True), ("b", -off, False)]:
        s = ex[ex.setting == set_].set_index("candidate")
        for c in rows:
            if c in s.index:
                r = s.loc[c]
                st = cs.task_style(c)
                dot_row(ax_e, ypos[c] + dy, r.rmse_err, r.rmse_err_min, r.rmse_err_max, st, filled)
                dot_row(ax_auc, ypos[c] + dy, r.auc, r.auc_min, r.auc_max, st, filled)
    for (cbd, sfx, filled, _), dy in zip(B_DRIFT, (off, -off)):
        s = pr[pr.c_bd == cbd].set_index("a")
        for c in rows:
            if c + sfx not in s.index:
                continue
            r = s.loc[c + sfx]
            st = cs.task_style(c)
            if r["diff"] > B_XMAX:   # off scale: arrow at the right edge with the value
                ax_d.annotate("", xy=(B_XMAX, ypos[c] + dy), xytext=(B_XMAX - 0.2, ypos[c] + dy),
                              arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=0.9, mutation_scale=6))
                ax_d.text(B_XMAX - 0.22, ypos[c] + dy, f"+{r['diff']:.2f}", ha="right", va="center",
                          fontsize=cs.FS_SMALL, color=cs.INK)
                continue
            dot_row(ax_d, ypos[c] + dy, r["diff"], r.ci_lo, r.ci_hi, st, filled)
    ax_d.axvline(0, color=cs.MUTED, lw=0.8, zorder=1)
    for ax in (ax_e, ax_auc, ax_d):
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks([ypos[c] for c in rows])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    ax_e.set_yticklabels([SHORT[c] for c in rows])
    for ax in (ax_auc, ax_d):
        ax.set_yticklabels([])
    ax_e.set_xlabel("Error at 5 Hz (N·m)")
    ax_auc.set_xlabel("AUC")
    ax_d.set_xlabel("Drift minus D-S (rad)")
    ax_e.set_xlim(1.0, 2.45)
    ax_e.set_xticks([1.0, 1.5, 2.0])
    ax_auc.set_xlim(0.785, 0.89)
    ax_auc.set_xticks([0.80, 0.86])
    ax_d.set_xlim(-0.3, B_XMAX)
    ax_d.set_xticks([0, 0.4, 0.8])
    g = dict(color=cs.INK2, ls="none", marker="o", ms=3.8, mec=cs.INK2, mew=0.8)
    hs = [Line2D([], [], mfc=cs.INK2, label=r"true $\ddot q$", **g),
          Line2D([], [], mfc="white", label=r"$\ddot q$ from noisy $\dot q$", **g)]
    ax_e.legend(handles=hs, loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=1, frameon=False, borderaxespad=0.1,
                handletextpad=0.2, labelspacing=0.15)
    hd = [Line2D([], [], mfc=cs.INK2 if f else "white", label=lab, **g) for (_, _, f, lab) in B_DRIFT]
    ax_d.legend(handles=hd, loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=1, frameon=False, borderaxespad=0.1,
                handletextpad=0.2, labelspacing=0.15)


# ----------------------------------------------------------------------------- (c)
def panel_c(ax, tw, rho):
    tw = tw.set_index("candidate")
    xl, yl = (6.0, 18.0), (6.0, 23.0)
    ax.plot([6, 23], [6, 23], color=cs.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    for c in [c for c in tw.index if c not in C_KEY]:
        r = tw.loc[c]
        x1, x2, y = 1e3 * r.pred_joint, 1e3 * r.pred_cg, 1e3 * r.meas_rmse_mean
        ax.plot([x1, x2], [y, y], color=cs.MUTED, lw=0.6, zorder=2)
        ax.plot([x1], [y], ls="none", marker="o", ms=2.8, mfc=cs.MUTED, mec=cs.MUTED, zorder=3)
        ax.plot([x2], [y], ls="none", marker="o", ms=2.8, mfc="white", mec=cs.MUTED, mew=0.7, zorder=3)
    for c in C_KEY:
        r = tw.loc[c]
        st = cs.task_style(c)
        x1, x2, y = 1e3 * r.pred_joint, 1e3 * r.pred_cg, 1e3 * r.meas_rmse_mean
        ax.plot([x1, x2], [y, y], color=st["color"], lw=0.8, zorder=4)
        ax.plot([x1], [y], ls="none", zorder=5, **mstyle(st, filled=True))
        ax.plot([x2], [y], ls="none", zorder=5, **mstyle(st, filled=False))
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_xticks([10, 15])
    ax.set_yticks([10, 15, 20])
    ax.set_xlabel("Predicted RMSE (mrad)")
    ax.set_ylabel("Measured RMSE (mrad)")
    g = dict(color=cs.INK2, ls="none", marker="o", ms=3.8, mec=cs.INK2, mew=0.8)
    hs = [Line2D([], [], mfc=cs.INK2, label=f"per joint, {rho['joint']:+.2f}", **g),
          Line2D([], [], mfc="white", label=f"joint loop, {rho['cg']:+.2f}", **g)]
    ax.legend(handles=hs, loc="lower right", frameon=False, borderaxespad=0.1, handletextpad=0.2,
              title=r"Spearman $\rho_s$", title_fontsize=cs.FS, alignment="left", labelspacing=0.15)


def shared_legend(fig, cross):
    hs = []
    for c in LEG:
        st = cs.task_style(c)
        lab = SHORT[c]
        if (c, REF) in cross:
            lab += f" ({cross[(c, REF)]:.1f} Hz)"
        hs.append(Line2D([], [], color=st["color"], ls=st["ls"], lw=1.4 if c == REF else 1.0, label=lab,
                         **mstyle(st, ms=4.0)))
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False, handlelength=2.6,
               columnspacing=1.2, handletextpad=0.4, labelspacing=0.15, borderaxespad=0.0)


def main():
    cs.apply_style()
    curve, cross = load_a()
    ex, pr = load_b()
    tw, rho = load_c()
    fig = plt.figure(figsize=(cs.DOUBLE_COL, FIG_H))
    yb, h = 0.58, 1.88
    ax_a = ax_in(fig, 0.50, yb, 1.66, h)
    panel_a(ax_a, curve, cross)
    ax_e = ax_in(fig, 2.98, yb, 0.70, h)
    ax_auc = ax_in(fig, 3.75, yb, 0.48, h)
    ax_d = ax_in(fig, 4.30, yb, 0.78, h)
    panel_b(ax_e, ax_auc, ax_d, ex, pr)
    ax_c = ax_in(fig, 5.62, yb, 1.48, h)
    panel_c(ax_c, tw, rho)
    shared_legend(fig, cross)
    top = (yb + h + 0.34) / FIG_H
    cs.fig_label(fig, "a", 0.01 / cs.DOUBLE_COL, top)
    cs.fig_label(fig, "b", 2.22 / cs.DOUBLE_COL, top)
    cs.fig_label(fig, "c", 5.20 / cs.DOUBLE_COL, top)
    cs.savefig(fig, "fig6")


if __name__ == "__main__":
    main()
