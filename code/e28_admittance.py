"""E28: admittance control closed loop with the frozen predictor as interaction-torque estimator.

Plant, gains, torque limits and the inner tracking law are those of E13 (BatchedClosedLoop):
    tau = f_hat(q_obs, qd_obs, qdd_r) + Kp (q_r - q_fb) + Kd (qd_r - qd_fb)
The reference (q_r, qd_r, qdd_r) is produced by a joint-space virtual admittance driven by the
interaction-torque estimate of the previous control step,
    tau_hat[k] = f_hat(q_obs[k-1], qd_obs[k-1], qdd[k-1]) - tau_motor[k-1]
(qdd = recorded forward-dynamics acceleration, setting (a) of E27), integrated semi-implicitly:
    qdd_r = (tau_hat - B_d qd_r) / M_d ; qd_r += Ts qdd_r ; q_r += Ts qd_r ,
with L_eq = 2 M_d / (B_d Ts) = 20 (M_d / B_d = 0.1 s); q_r is held inside the joint limits (0.15 rad
margin, velocity zeroed at the limit, counted as a limit hit).
Human: joint-space spring-damper to an intention trajectory, tau_h = K_h (q_int - q) + B_h (qd_int - qd),
applied as MjData.qfrc_applied (zero-order hold); zero in the 'free' episodes.
Episodes: 30 start poses (first states of the 30 closed-loop references, at rest), each run once
without human ('free', drift) and once with the human ('intent'); 10 s each; N, O1-M, P1 with the
noise seeds of E13 (900/901/902; the 'intent' half uses the same noise draws as the 'free' half).
Weights: calibration group 0 (m = 10) of the condition (cal_P1 under P1), as E13; under P1 also the
weights calibrated under N ('@calN', not recalibrated) and an affine variant 'D-S:W-Q5+b' (W-Q5
plus a per-joint constant offset equal to the mean calibration residual of the group).
'oracle': exact inverse dynamics of the simulated plant at the observed state as feedforward and the
true human torque (one step late) as estimate; 'ID-exact': the same exact inverse dynamics used both as
feedforward and as estimator, ID(q_obs, qd_obs, qdd) - tau_motor (noise propagation without model error).
Admittance: B_d = c_bd x (training torque std of the joint), M_d = L_eq B_d Ts / 2 (L_eq = 20, i.e.
M_d / B_d = 0.1 s); human K_h = 5 B_d, B_h = B_d / 3 (closed-loop time constant B_d / K_h = 0.2 s).
Failure (as E13): any joint |q - q_r| > 0.5 rad or a non-finite state stops the episode.

python e28_admittance.py --data_dir ../data_sim --xml_dir ../mujoco_menagerie/franka_emika_panda --runs ../runs25/sim \
    --frozen ../runs25/sim/frozen.json --cfg e27_candidates.json --out ../results/sim/e28_episodes_a.csv --seeds 0,1
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import e27_common as C
from sim.panda_env import PandaSim, TORQUE_LIMIT, FAIL_ERR, CONTROL_HZ

INTENT_SEED = 2801
E28_NAMES = ["D-S:W-Q5", "D-M:W-Q5", "D-M:W-S", "B-N", "B-A", "B-N|est", "D-S+null:W-Q5", "D-S|est:W-Q5",
             "D-S:e21-W1", "D-S:e21-W3", "D-M:e21-W1", "D-M:e21-W3"]
CALN_NAMES = ["D-S:W-Q5", "B-N", "D-S+null:W-Q5", "D-S|est:W-Q5"]


def intent_trajectories(rng, q0s, n_steps, amp=(0.02, 0.08), freq=(0.1, 0.4), H=2):
    t = np.arange(n_steps) / CONTROL_HZ
    Q, QD = [], []
    for q0 in q0s:
        A = rng.uniform(*amp, (7, H)); f = rng.uniform(*freq, (7, H)); ph = rng.uniform(0, 2 * np.pi, (7, H))
        w = 2 * np.pi * f
        arg = w[:, :, None] * t[None, None, :] + ph[:, :, None]
        q = q0[:, None] + (A[:, :, None] * (np.sin(arg) - np.sin(ph)[:, :, None])).sum(1)
        qd = (A[:, :, None] * w[:, :, None] * np.cos(arg)).sum(1)
        Q.append(q.T); QD.append(qd.T)
    return np.stack(Q), np.stack(QD)


class AdmittanceLoop:
    def __init__(self, xml_dir, q0s, q_int, qd_int, human_on, adm, human, payload_kg=0.0, obs_noise=None,
                 noise_seed=0, q_scale=None, qd_scale=None, n_steps=1000):
        self.B = len(q0s)
        self.sims = [PandaSim(xml_dir, payload_kg) for _ in range(self.B)]
        self.idsim = PandaSim(xml_dir, payload_kg)    # inverse dynamics of the simulated plant (oracle feedforward)
        self.q0s, self.q_int, self.qd_int, self.human_on = q0s, q_int, qd_int, human_on
        self.Md, self.Bd = adm
        self.Kh, self.Bh = human
        self.n = n_steps
        self.lo, self.hi = self.sims[0].lo, self.sims[0].hi
        rng = np.random.default_rng(noise_seed)
        half = self.B // 2
        self.noise_q = np.zeros((self.B, self.n, 7)); self.noise_qd = np.zeros((self.B, self.n, 7))
        if obs_noise:
            nq = rng.standard_normal((half, self.n, 7)) * obs_noise * q_scale
            nqd = rng.standard_normal((half, self.n, 7)) * obs_noise * qd_scale
            self.noise_q = np.concatenate([nq, nq], 0); self.noise_qd = np.concatenate([nqd, nqd], 0)

    def run(self, ff=None, oracle=False, prefilter_hz=None, offset=None, fb_true=False):
        B, n = self.B, self.n
        for b in range(B):
            self.sims[b].reset(self.q0s[b], np.zeros(7))
        q_r = self.q0s.copy(); qd_r = np.zeros((B, 7)); qdd_r = np.zeros((B, 7))
        prev_obs = None; prev_qdd = None; prev_tau = None; prev_th = np.zeros((B, 7))
        th_f = np.zeros((B, 7))
        alive = np.ones(B, bool); fail_step = np.full(B, -1)
        L = {k: np.zeros((B, n, 7)) for k in ["q", "q_r", "tau_hat", "tau_h", "tau_app", "e_trk"]}
        limit_hit = np.zeros(B, bool); sat = np.zeros(B)
        a_pf = None if not prefilter_hz else C.lp_alpha(prefilter_hz)
        for k in range(n):
            q = np.stack([s.data.qpos for s in self.sims]).copy(); qd = np.stack([s.data.qvel for s in self.sims]).copy()
            q_obs = q + self.noise_q[:, k]; qd_obs = qd + self.noise_qd[:, k]
            # interaction-torque estimate from the previous step
            if k == 0:
                th = np.zeros((B, 7))
            elif oracle == "true":
                th = prev_th.copy()
            elif oracle == "id":
                th = np.stack([self.idsim.inverse_dynamics(prev_obs[0][b], prev_obs[1][b], prev_qdd[b]) for b in range(B)]) - prev_tau
            else:
                X = np.concatenate([prev_obs[0], prev_obs[1], prev_qdd], 1)
                th = ff(X) - prev_tau
                if offset is not None:
                    th = th + offset
            if a_pf is not None:
                th_f = th_f + a_pf * (th - th_f)
            else:
                th_f = th
            # admittance
            qdd_r = (th_f - self.Bd * qd_r) / self.Md
            qd_r = qd_r + C.TS * qdd_r
            q_r = q_r + C.TS * qd_r
            out_lo = q_r < self.lo; out_hi = q_r > self.hi
            hit = out_lo | out_hi
            if hit.any():
                q_r = np.clip(q_r, self.lo, self.hi); qd_r[hit] = 0.0; qdd_r[hit] = 0.0
                limit_hit |= hit.any(1)
            # inner loop (E13 law)
            q_fb, qd_fb = (q, qd) if fb_true else (q_obs, qd_obs)     # fb_true: diagnostic, as E13's @true-fb
            tau = C.KP * (q_r - q_fb) + C.KD * (qd_r - qd_fb)
            if oracle:
                tff = np.stack([self.idsim.inverse_dynamics(q_obs[b], qd_obs[b], qdd_r[b]) for b in range(B)])
            else:
                Xf = np.concatenate([q_obs, qd_obs, qdd_r], 1)
                tff = ff(Xf)
                if offset is not None:
                    tff = tff + offset        # the affine variant corrects the feedforward too
            tau = tau + tff
            tau_h = self.human_on[:, None] * (self.Kh * (self.q_int[:, k] - q) + self.Bh * (self.qd_int[:, k] - qd))
            qdd_now = np.zeros((B, 7)); tau_app = np.zeros((B, 7))
            for b in range(B):
                if not alive[b]:
                    continue
                s = self.sims[b]
                s.data.qfrc_applied[:] = tau_h[b]
                qdd_b, ta, _ = s.apply_and_step(tau[b])
                qdd_now[b] = qdd_b; tau_app[b] = ta
                e = q[b] - q_r[b]
                if (not np.all(np.isfinite(s.data.qpos))) or np.max(np.abs(e)) > FAIL_ERR:
                    alive[b] = False; fail_step[b] = k
            sat += (np.abs(tau) > TORQUE_LIMIT).mean(1)
            for key, val in [("q", q), ("q_r", q_r), ("tau_hat", th), ("tau_h", tau_h), ("tau_app", tau_app), ("e_trk", q - q_r)]:
                L[key][:, k] = val
            prev_obs = (q_obs, qd_obs); prev_qdd = qdd_now; prev_tau = tau_app; prev_th = tau_h
        L["failed"] = fail_step >= 0; L["fail_step"] = fail_step; L["limit_hit"] = limit_hit; L["sat"] = sat / n
        return L


def episode_metrics(L, q0s, q_int, human_on):
    B, n, _ = L["q"].shape
    rows = []
    for b in range(B):
        fs = L["fail_step"][b]
        end = n if fs < 0 else fs + 1
        qr, q = L["q_r"][b, :end], L["q"][b, :end]
        d_final = qr[-1] - q0s[b]
        r = {"episode_type": "intent" if human_on[b] else "free", "failed": bool(L["failed"][b]), "fail_step": int(fs),
             "limit_hit": bool(L["limit_hit"][b]), "saturation": float(L["sat"][b]),
             "drift_final_max": float(np.abs(d_final).max()), "drift_final_rms": float(np.sqrt((d_final ** 2).mean())),
             "drift_max": float(np.abs(qr - q0s[b]).max()), "q_drift_max": float(np.abs(q - q0s[b]).max()),
             "trk_rmse": float(np.sqrt((L["e_trk"][b, :end] ** 2).mean())),
             "tau_hat_rms": float(np.sqrt((L["tau_hat"][b, 1:end] ** 2).mean())),
             "tau_err_rms": float(np.sqrt(((L["tau_hat"][b, 1:end] - L["tau_h"][b, :end - 1]) ** 2).mean())) if end > 1 else np.nan,
             "tau_h_rms": float(np.sqrt((L["tau_h"][b, :end] ** 2).mean())),
             "intent_err_rms": float(np.sqrt(((q - q_int[b, :end]) ** 2).mean())) if human_on[b] else np.nan,
             "intent_err_max": float(np.abs(q - q_int[b, :end]).max()) if human_on[b] else np.nan,
             "tau_rms": float(np.sqrt((L["tau_app"][b, :end] ** 2).mean()))}
        th5 = C.lowpass(L["tau_hat"][b:b + 1, 1:end], 5.0)[0] if end > 2 else np.zeros((1, 7))
        r["tau_hat5_rms"] = float(np.sqrt((th5 ** 2).mean()))
        for l in range(7):
            r[f"drift_final_j{l + 1}"] = float(d_final[l])
            r[f"tau_hat_mean_j{l + 1}"] = float(L["tau_hat"][b, 1:end, l].mean())
            r[f"tau_h_rms_j{l + 1}"] = float(np.sqrt((L["tau_h"][b, :end, l] ** 2).mean()))
        if fs >= 0:   # failed episodes: drift and errors at the threshold, as E13 fills the error
            r["drift_final_max"] = max(r["drift_final_max"], FAIL_ERR)
            if human_on[b]:
                r["intent_err_rms"] = max(r["intent_err_rms"], FAIL_ERR)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--runs", required=True); ap.add_argument("--frozen", required=True)
    ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--conds", default="N,O1M,P1")
    ap.add_argument("--names", default=",".join(E28_NAMES))
    ap.add_argument("--c_bd", type=float, default=20.0, help="B_d = c_bd x training torque std of the joint")
    ap.add_argument("--leq", type=float, default=20.0)
    ap.add_argument("--kh_ratio", type=float, default=5.0, help="K_h = kh_ratio x B_d (1/s)")
    ap.add_argument("--bh_ratio", type=float, default=1.0 / 3.0, help="B_h = bh_ratio x B_d")
    ap.add_argument("--prefilter_hz", type=float, default=0.0)
    ap.add_argument("--n_episodes", type=int, default=30)
    ap.add_argument("--no_oracle", action="store_true"); ap.add_argument("--no_extras", action="store_true")
    ap.add_argument("--fb_true", action="store_true", help="diagnostic: feedback terms from the true state")
    args = ap.parse_args()
    t00 = time.time()
    ctx = C.setup(args.data_dir, args.frozen)
    cfg = C.load_config(args.cfg)
    xs, ys = ctx["xs"], ctx["ys"]
    Bd = args.c_bd * ys; Md = args.leq * Bd * C.TS / 2.0
    Kh = args.kh_ratio * Bd; Bh = args.bh_ratio * Bd
    rn = np.load(os.path.join(args.data_dir, "closed_refs.npz"))
    q0 = rn["q"][:args.n_episodes, 0].copy()
    qi, qdi = intent_trajectories(np.random.default_rng(INTENT_SEED), q0, rn["q"].shape[1])
    E = len(q0)
    q0s = np.concatenate([q0, q0], 0); q_int = np.concatenate([np.repeat(q0[:, None], qi.shape[1], 1), qi], 0)
    qd_int = np.concatenate([np.zeros_like(qdi), qdi], 0); human_on = np.r_[np.zeros(E), np.ones(E)]
    params = {"c_bd": args.c_bd, "kh_ratio": args.kh_ratio, "bh_ratio": args.bh_ratio, "Md": Md.tolist(), "Bd": Bd.tolist(), "L_eq": args.leq, "Kh": Kh.tolist(), "Bh": Bh.tolist(),
              "prefilter_hz": args.prefilter_hz, "fb_true": bool(args.fb_true), "intent_amp_rad": [0.02, 0.08], "intent_freq_hz": [0.1, 0.4],
              "intent_rms_rad": float(np.sqrt(((qi - q0[:, None]) ** 2).mean())), "intent_seed": INTENT_SEED}
    json.dump(params, open(args.out.replace(".csv", "_params.json"), "w"), indent=1)
    print(json.dumps(params), flush=True)
    rows = []

    def run_one(seed, cond, name, ff, oracle=False, offset=None, extra=None):
        payload = 1.0 if cond == "P1" else 0.0
        noise = ctx["rho"] if cond == "O1M" else None
        t0 = time.time()
        loop = AdmittanceLoop(args.xml_dir, q0s, q_int, qd_int, human_on, (Md, Bd), (Kh, Bh), payload_kg=payload,
                              obs_noise=noise, noise_seed=C.NOISE_SEED[cond], q_scale=xs[:7], qd_scale=xs[7:14],
                              n_steps=q_int.shape[1])
        L = loop.run(ff, oracle=oracle, prefilter_hz=args.prefilter_hz or None, offset=offset, fb_true=args.fb_true)
        wall = time.time() - t0
        ms = episode_metrics(L, q0s, q_int, human_on)
        for e, m in enumerate(ms):
            rows.append({"seed": seed, "cond": cond, "candidate": name + ("@true-fb" if args.fb_true else ""), "episode": e % E, **m,
                         **(extra or {}), "wall_s": wall})
        fr = [m for m in ms if m["episode_type"] == "free"]; it = [m for m in ms if m["episode_type"] == "intent"]
        print(f"seed {seed} {cond} {name}: drift {np.mean([m['drift_final_max'] for m in fr]):.4f} "
              f"fail {np.mean([m['failed'] for m in ms]):.2f} limit {np.mean([m['limit_hit'] for m in ms]):.2f} "
              f"tau_h {np.mean([m['tau_h_rms'] for m in it]):.3f} int_err {np.mean([m['intent_err_rms'] for m in it]):.4f} "
              f"wall {wall:.0f}s total {time.time() - t00:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)

    seeds = [int(s) for s in args.seeds.split(",")]
    if not args.no_oracle:
        for cond in args.conds.split(","):
            run_one(-1, cond, "oracle", None, oracle="true")
            run_one(-1, cond, "ID-exact", None, oracle="id")
    for seed in seeds:
        mem = C.Members(ctx, args.runs, seed)
        for cond in args.conds.split(","):
            cs = C.cal_split(ctx, cond, 0)
            ctag = f"cal:{cond}:g0"
            todo = [(n, cs, ctag, n) for n in args.names.split(",")]
            if cond == "P1" and not args.no_extras:
                csN = C.cal_split(ctx, "N", 0)
                todo += [(n, csN, "cal:N:g0", n + "@calN") for n in CALN_NAMES]
            for name, c_s, tag, label in todo:
                items, w = C.candidate_weights(ctx, cfg, mem, name, c_s, tag)
                ff, n_exec = C.make_ff(ctx, mem, items, w, c_s["W"])
                run_one(seed, cond, label, ff, extra={**C.weight_summary(items, w), "w": json.dumps(np.round(np.asarray(w), 4).tolist())})
            if not args.no_extras:
                items, w = C.candidate_weights(ctx, cfg, mem, "D-S:W-Q5", cs, ctag)
                Pc = C.candidate_predict(ctx, mem, items, w, cs["X"], cs["W"], ctag + ":raw", ctag + ":est")
                off = -((Pc - cs["Y"]) * ys).mean(0)            # correction added to f_hat (feedforward and estimate)
                ff, _ = C.make_ff(ctx, mem, items, w, cs["W"])
                run_one(seed, cond, "D-S:W-Q5+b", ff, offset=off, extra={"offset": json.dumps(np.round(-off, 3).tolist())})
            mem.drop("cal:")
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("E28_DONE", time.time() - t00)


if __name__ == "__main__":
    main()
