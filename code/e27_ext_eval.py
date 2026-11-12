"""E27 new simulation, evaluation: interaction-torque estimate tau_ext_hat = f_hat(x~) - tau_motor on the
40 external-torque trajectories (ext_test.npz from e27_gen_ext.py).

Conditions N and O1-M (rho = 0.7 x training std on q, qd; noise seed 2702, shared by every candidate).
Acceleration input:
  (a) the recorded forward-dynamics acceleration (as the original open loop);
  (b) causal backward difference of the observed (noisy) velocity followed by two cascaded first-order
      low-passes at fc_b; fc_b is chosen per condition on the 30 calibration trajectories of groups
      0-2 (minimum normalized acceleration error); all 21 channels are then perturbed.
Weights: the closed-loop calibration groups (m = 10, groups 0-2), fitted with the same input
pipeline as the evaluation (setting b recalibrates on velocity-differenced accelerations).
A reference pseudo-candidate 'ID-exact' uses the plant's exact inverse dynamics at the observed input.

Metrics (steps >= 0.5 s; filters: raw, 5 Hz, 20 Hz causal first-order low-pass):
  * RMSE of the filtered error LP(tau_hat) - LP(tau_ext) and of LP(tau_hat) - tau_ext, per joint (N m);
  * sample-level ROC/AUC: score |LP(tau_hat_j)|, label 'event active on joint j';
  * event onset detection (threshold method) with (i) a fixed 0.5 N m threshold and (ii) a per-joint
    CFAR threshold = 95th percentile of |LP(tau_hat)| on the calibration group (no external torque):
    detection rate within the event, latency from onset, sample false-positive rate and false-alarm
    rate (alarm onsets per minute) in quiet periods (no event on the joint, > 0.2 s after an event).
Outputs: e27_ext_episodes.csv, e27_ext_summary.csv, e27_ext_roc.csv, e27_ext_meta.json.

python e27_ext_eval.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs25/sim \
    --frozen ../runs25/sim/frozen.json --cfg e27_candidates.json --ext ../data_sim/ext_test.npz --out_dir ../results/sim
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import e27_common as C
from sim.panda_env import PandaSim
from sim.data import apply_condition

EXT_NOISE_SEED = 2702
EVAL_START = 50
POST_GUARD = 20
FILTERS = {"raw": None, "5Hz": 5.0, "20Hz": 20.0}
THETA_FIXED = 0.5
CFAR_Q = 95.0
FC_B_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, np.inf]
FPR_GRID = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def accel_from_vel(QD, fc):
    """QD [n, T, 7] physical velocity -> causal acceleration estimate (backward difference, then two
    cascaded first-order low-passes at fc)."""
    d = np.diff(QD, axis=1, prepend=QD[:, :1]) / C.TS
    if not np.isfinite(fc):
        return d
    return C.lowpass(C.lowpass(d, fc), fc)


def auc(score, label):
    pos = label.sum(); neg = label.size - pos
    if pos == 0 or neg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def roc_points(score, label):
    s_pos, s_neg = score[label], score[~label]
    out = []
    for f in FPR_GRID:
        if f <= 0:
            thr = np.inf if len(s_neg) == 0 else s_neg.max()
            out.append(float((s_pos > thr).mean()))
        elif f >= 1:
            out.append(1.0)
        else:
            thr = np.quantile(s_neg, 1.0 - f)
            out.append(float((s_pos > thr).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--frozen", required=True)
    ap.add_argument("--cfg", required=True); ap.add_argument("--ext", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--conds", default="N,O1M"); ap.add_argument("--settings", default="a,b")
    ap.add_argument("--tag", default=""); ap.add_argument("--skip_id", action="store_true")
    ap.add_argument("--names", default="")
    args = ap.parse_args()
    t0 = time.time()
    ctx = C.setup(args.data_dir, args.frozen)
    cfg = C.load_config(args.cfg)
    names = args.names.split(",") if args.names else list(cfg["candidates"])
    xm, xs, ym, ys = ctx["xm"], ctx["xs"], ctx["ym"], ctx["ys"]
    ext = np.load(args.ext)
    nT, T = ext["q"].shape[:2]
    TAU, TEXT = ext["tau"], ext["tau_ext"]
    active = np.abs(TEXT) > 0
    kidx = np.arange(T)
    evalm = np.broadcast_to((kidx >= EVAL_START)[None, :, None], active.shape)
    # quiet mask: no event on the joint and more than POST_GUARD steps after the last event end on it
    recent = np.zeros_like(active)
    ev = {k: ext[k] for k in ["ev_traj", "ev_joint", "ev_on", "ev_off", "ev_amp", "ev_type"]}
    for i in range(len(ev["ev_traj"])):
        t_, j_, off = int(ev["ev_traj"][i]), int(ev["ev_joint"][i]), int(ev["ev_off"][i])
        recent[t_, off:off + POST_GUARD, j_] = True
    quiet = (~active) & (~recent) & evalm
    pos = active & evalm
    labmask = (active | quiet)                  # samples used for ROC/AUC (guard samples excluded)
    sim = PandaSim(args.xml_dir, 0.0)

    # ------------------------------------------------------------ setting (b): accel filter selection
    D = ctx["D"]
    meta = {"fc_b": {}, "rho_qdd_eff_ext": {}, "rho_qdd_eff_cal": {}, "fc_b_grid_err": {}}
    noise = {}
    for cond in args.conds.split(","):
        rng = np.random.default_rng(EXT_NOISE_SEED)
        if cond == "O1M":
            nq = rng.standard_normal((nT, T, 7)) * ctx["rho"] * xs[:7]
            nqd = rng.standard_normal((nT, T, 7)) * ctx["rho"] * xs[7:14]
        else:
            nq = np.zeros((nT, T, 7)); nqd = np.zeros((nT, T, 7))
        noise[cond] = (nq, nqd)
        Xc = apply_condition(D["X"]["cal"], "cal", "N" if cond == "N" else f"O1:{ctx['rho']}").astype(np.float64)
        Xc = Xc.reshape(-1, T, 21)[:30]
        QDc = Xc[:, :, 7:14] * xs[7:14] + xm[7:14]; QDDc = Xc[:, :, 14:21] * xs[14:21] + xm[14:21]
        errs = {}
        for fc in FC_B_GRID:
            a_ = accel_from_vel(QDc, fc)
            errs[fc] = float(np.mean(((a_ - QDDc)[:, EVAL_START:] / xs[14:21]) ** 2))
        fbest = min(errs, key=errs.get)
        meta["fc_b"][cond] = fbest
        meta["fc_b_grid_err"][cond] = {str(k): v for k, v in errs.items()}
        meta["rho_qdd_eff_cal"][cond] = float(np.sqrt(errs[fbest]))
        a_ext = accel_from_vel(ext["qd"] + nqd, fbest)
        meta["rho_qdd_eff_ext"][cond] = (np.sqrt(np.mean(((a_ext - ext["qdd"])[:, EVAL_START:] / xs[14:21]) ** 2, (0, 1)))).tolist()
    print("accel filter:", json.dumps(meta), flush=True)

    def make_xform(fc):
        def xf(X, role):
            Xr = X.reshape(-1, T, 21).copy()
            qd = Xr[:, :, 7:14] * xs[7:14] + xm[7:14]
            Xr[:, :, 14:21] = (accel_from_vel(qd, fc) - xm[14:21]) / xs[14:21]
            return Xr.reshape(-1, 21)
        return xf

    def ext_inputs(cond, setting):
        nq, nqd = noise[cond]
        Q = ext["q"] + nq; QD = ext["qd"] + nqd
        QDD = ext["qdd"] if setting == "a" else accel_from_vel(QD, meta["fc_b"][cond])
        Xp = np.concatenate([Q, QD, QDD], 2)
        return Xp, ((Xp - xm) / xs).reshape(-1, 21)

    def id_exact(Xp):
        flat = Xp.reshape(-1, 21)
        out = np.zeros((len(flat), 7))
        for i in range(len(flat)):
            out[i] = sim.inverse_dynamics(flat[i, :7], flat[i, 7:14], flat[i, 14:21])
        return out.reshape(Xp.shape[:-1] + (7,))

    # ------------------------------------------------------------ metrics of one estimate
    def evaluate(base, That, Rcal):
        """That [nT, T, 7] estimate of tau_ext (N m); Rcal [10, T, 7] estimate on the calibration group
        (true tau_ext = 0 there)."""
        rows_ep, rows_sum, rows_roc = [], [], []
        for fname, fc in FILTERS.items():
            Th = C.lowpass(That, fc) if fc else That
            Tt = C.lowpass(TEXT, fc) if fc else TEXT
            err = (Th - Tt)[:, EVAL_START:]; errv = (Th - TEXT)[:, EVAL_START:]
            stat = np.abs(Th)
            Rc = np.abs(C.lowpass(Rcal, fc) if fc else Rcal)[:, EVAL_START:]
            th_cfar = np.percentile(Rc.reshape(-1, 7), CFAR_Q, axis=0)
            th_cfar99 = np.percentile(Rc.reshape(-1, 7), 99.0, axis=0)
            # per-trajectory per-joint error
            rm_tj = np.sqrt((err ** 2).mean(1))            # [nT, 7]
            res = {}
            for thname, th in [("fixed", np.full(7, THETA_FIXED)), ("cfar", th_cfar), ("cfar99", th_cfar99)]:
                alarm = stat > th[None, None, :]
                rising = alarm & ~np.concatenate([np.zeros_like(alarm[:, :1]), alarm[:, :-1]], 1)
                fa_tj = (rising & quiet).sum(1)            # [nT, 7]
                fp_tj = (alarm & quiet).sum(1)
                lat = np.full(len(ev["ev_traj"]), np.nan)
                pre = np.zeros(len(ev["ev_traj"]), bool)
                for i in range(len(ev["ev_traj"])):
                    t_, j_, on, off = int(ev["ev_traj"][i]), int(ev["ev_joint"][i]), int(ev["ev_on"][i]), int(ev["ev_off"][i])
                    a_ = alarm[t_, on:off, j_]
                    if a_.any():
                        lat[i] = float(np.argmax(a_)) * C.TS * 1000.0
                    pre[i] = alarm[t_, on - 1, j_]
                res[thname] = (fa_tj, fp_tj, lat, pre, th)
            qt = quiet.sum(1)                              # [nT, 7] quiet samples
            for t_ in range(nT):
                r = {**base, "filter": fname, "traj": t_, "rmse_all": float(np.sqrt((rm_tj[t_] ** 2).mean()))}
                for l in range(7):
                    r[f"rmse_j{l + 1}"] = float(rm_tj[t_, l])
                sel = ev["ev_traj"] == t_
                r["n_pairs"] = int(sel.sum())
                for thname in ("fixed", "cfar", "cfar99"):
                    fa_tj, fp_tj, lat, pre, th = res[thname]
                    r[f"n_det_{thname}"] = int(np.isfinite(lat[sel]).sum())
                    r[f"lat_mean_ms_{thname}"] = float(np.nanmean(lat[sel])) if np.isfinite(lat[sel]).any() else np.nan
                    r[f"false_alarms_{thname}"] = int(fa_tj[t_].sum())
                    r[f"fp_samples_{thname}"] = int(fp_tj[t_].sum())
                r["quiet_samples"] = int(qt[t_].sum())
                rows_ep.append(r)
            # summary per joint and pooled
            for l in list(range(7)) + ["all"]:
                jsel = slice(None) if l == "all" else l
                if l == "all":
                    lab = active[labmask]; sc = stat[labmask]
                    rm = float(np.sqrt((err ** 2).mean())); rv = float(np.sqrt((errv ** 2).mean()))
                    evsel = np.ones(len(ev["ev_traj"]), bool)
                else:
                    lm = labmask[:, :, l]
                    lab = active[:, :, l][lm]; sc = stat[:, :, l][lm]
                    rm = float(np.sqrt((err[:, :, l] ** 2).mean())); rv = float(np.sqrt((errv[:, :, l] ** 2).mean()))
                    evsel = ev["ev_joint"] == l
                r = {**base, "filter": fname, "joint": l if l == "all" else l + 1, "rmse_err": rm, "rmse_vs_true": rv,
                     "auc": auc(sc, lab), "n_pairs": int(evsel.sum())}
                for thname in ("fixed", "cfar", "cfar99"):
                    fa_tj, fp_tj, lat, pre, th = res[thname]
                    lt = lat[evsel]
                    qs = qt[:, jsel].sum()
                    r[f"det_rate_{thname}"] = float(np.isfinite(lt).mean()) if len(lt) else np.nan
                    r[f"lat_med_ms_{thname}"] = float(np.nanmedian(lt)) if np.isfinite(lt).any() else np.nan
                    r[f"lat_mean_ms_{thname}"] = float(np.nanmean(lt)) if np.isfinite(lt).any() else np.nan
                    r[f"pre_onset_alarm_{thname}"] = float(pre[evsel].mean()) if len(lt) else np.nan
                    r[f"fpr_{thname}"] = float(fp_tj[:, jsel].sum() / max(qs, 1))
                    r[f"far_per_min_{thname}"] = float(fa_tj[:, jsel].sum() / max(qs * C.TS / 60.0, 1e-9))
                    r[f"theta_{thname}"] = float(np.mean(th[jsel]))
                rows_sum.append(r)
            lab = active[labmask]; sc = stat[labmask]
            tp = roc_points(sc, lab)
            for f, tpr in zip(FPR_GRID, tp):
                rows_roc.append({**base, "filter": fname, "fpr": f, "tpr": tpr})
        return rows_ep, rows_sum, rows_roc

    EP, SU, RO = [], [], []
    seeds = [int(s) for s in args.seeds.split(",")]
    # exact-ID reference (no learned model; one evaluation per condition and setting, no groups)
    for cond in ([] if args.skip_id else args.conds.split(",")):
        for setting in args.settings.split(","):
            Xp, _ = ext_inputs(cond, setting)
            That = id_exact(Xp) - TAU
            xf = None if setting == "a" else make_xform(meta["fc_b"][cond])
            cs = C.cal_split(ctx, cond, 0, xform=xf)
            Xcp = (cs["X"] * xs + xm).reshape(10, T, 21)
            Rcal = id_exact(Xcp) - (cs["Y"] * ys + ym).reshape(10, T, 7)
            base = {"seed": -1, "cond": cond, "setting": setting, "group": 0, "candidate": "ID-exact"}
            a, b, c = evaluate(base, That, Rcal); EP += a; SU += b; RO += c
            print(f"ID-exact {cond} {setting} done {time.time() - t0:.0f}s", flush=True)
    for seed in seeds:
        mem = C.Members(ctx, args.runs, seed)
        for cond in args.conds.split(","):
            for setting in args.settings.split(","):
                Xp, Xs_ = ext_inputs(cond, setting)
                xf = None if setting == "a" else make_xform(meta["fc_b"][cond])
                for g in C.GROUPS:
                    cs = C.cal_split(ctx, cond, g, xform=xf)
                    ctag = f"cal:{cond}:{setting}:g{g}"
                    for name in names:
                        if g not in C.cand_groups(cfg, name):
                            continue
                        items, w = C.candidate_weights(ctx, cfg, mem, name, cs, ctag)
                        P = C.candidate_predict(ctx, mem, items, w, Xs_, cs["W"], f"ext:{cond}:{setting}", f"ext:{cond}:{setting}:g{g}:est")
                        That = (P * ys + ym).reshape(nT, T, 7) - TAU
                        Pc = C.candidate_predict(ctx, mem, items, w, cs["X"], cs["W"], ctag + ":raw", ctag + ":est")
                        Rcal = ((Pc - cs["Y"]) * ys).reshape(10, T, 7)
                        base = {"seed": seed, "cond": cond, "setting": setting, "group": g, "candidate": name}
                        a, b, c = evaluate(base, That, Rcal); EP += a; SU += b; RO += c
                    mem.drop(ctag); mem.drop(f"ext:{cond}:{setting}:g{g}")
                    print(f"seed {seed} {cond} {setting} g{g} done {time.time() - t0:.0f}s", flush=True)
                mem.drop(f"ext:{cond}:{setting}")
        sfx = args.tag
        pd.DataFrame(EP).to_csv(os.path.join(args.out_dir, f"e27_ext_episodes{sfx}.csv"), index=False)
        pd.DataFrame(SU).to_csv(os.path.join(args.out_dir, f"e27_ext_summary_rows{sfx}.csv"), index=False)
        pd.DataFrame(RO).to_csv(os.path.join(args.out_dir, f"e27_ext_roc_rows{sfx}.csv"), index=False)
    json.dump(meta, open(os.path.join(args.out_dir, f"e27_ext_meta{args.tag}.json"), "w"), indent=1)
    print("E27EXT_DONE", time.time() - t0)


if __name__ == "__main__":
    main()
