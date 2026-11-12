"""E27 new simulation: 40 test trajectories with external joint torques.

The 40 references are those of the open-loop test role (seed 400, regenerated and checked against
test_N.npz). The computed-torque teacher of gen_data.py (exact ID of the nominal plant + PD,
Kp = 100/40, Kd = 8/3.2) tracks them unchanged and does not know the external torque. External
torques tau_ext are applied as generalized joint forces (MjData.qfrc_applied, zero-order hold at the
100-Hz control rate):
  * events at random times, non-overlapping: onset 0.5-1.5 s after the previous event (first one at
    0.5-1.5 s), duration 0.5-2 s, all events end before 9.5 s;
  * each event acts on 1, 2 or 3 joints (p = 0.5, 0.3, 0.2), amplitude per joint 1-3 N m with a
    random sign;
  * type step (rectangular) or ramp (linear rise over 30-70 % of the duration, then hold), p = 0.5.
Recorded at each control instant (same convention as PandaSim.rollout_teacher): true q, qd, the
forward-dynamics acceleration qdd under motor torque + tau_ext, applied motor torque tau, tau_ext.
Identity: ID(q, qd, qdd) = tau + tau_ext, checked with mj_inverse on a separate MjData.

python e27_gen_ext.py --xml_dir ../mujoco_menagerie/franka_emika_panda --data_dir ../data_sim --out ../data_sim/ext_test.npz
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np

import e27_common  # noqa: F401  (sets sys.path)
from sim.panda_env import PandaSim, sample_reference, N_STEPS, CONTROL_HZ, TORQUE_LIMIT
from sim.gen_data import KP_TEACHER, KD_TEACHER, ROLE_SEEDS

EVENT_SEED = 2701


def sample_events(rng, n_steps=N_STEPS):
    ev = []
    t = rng.uniform(0.5, 1.5)
    while True:
        dur = rng.uniform(0.5, 2.0)
        if t + dur > 9.5:
            break
        nj = int(rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
        joints = np.sort(rng.choice(7, nj, replace=False))
        amps = rng.choice([-1.0, 1.0], nj) * rng.uniform(1.0, 3.0, nj)
        typ = int(rng.integers(0, 2))           # 0 step, 1 ramp
        on = int(round(t * CONTROL_HZ)); off = int(round((t + dur) * CONTROL_HZ))
        rise = int(round(rng.uniform(0.3, 0.7) * (off - on))) if typ == 1 else 0
        ev.append({"on": on, "off": off, "joints": joints.tolist(), "amps": amps.tolist(), "type": typ, "rise": rise})
        t += dur + rng.uniform(0.5, 1.5)
    return ev


def profile(events, n_steps=N_STEPS):
    tau = np.zeros((n_steps, 7))
    for e in events:
        for j, A in zip(e["joints"], e["amps"]):
            for k in range(e["on"], e["off"]):
                s = 1.0 if e["type"] == 0 else min(1.0, (k - e["on"] + 1) / max(e["rise"], 1))
                tau[k, j] = A * s
    return tau


def rollout_ext(sim, chk, ref, tau_ext, kp, kd):
    q_ref, qd_ref, qdd_ref = ref["q"], ref["qd"], ref["qdd"]
    n = q_ref.shape[0]
    out = {k: np.zeros((n, 7)) for k in ["q", "qd", "qdd", "tau", "tau_req"]}
    resid = np.zeros(n)
    sim.reset(q_ref[0], qd_ref[0])
    for k in range(n):
        q = sim.data.qpos.copy(); qd = sim.data.qvel.copy()
        tau_ff = sim.inverse_dynamics(q, qd, qdd_ref[k])          # teacher: ID of the plant, blind to tau_ext
        tau = tau_ff + kp * (q_ref[k] - q) + kd * (qd_ref[k] - qd)
        sim.data.qfrc_applied[:] = tau_ext[k]
        qdd, tau_app, _ = sim.apply_and_step(tau)
        out["q"][k], out["qd"][k], out["qdd"][k], out["tau"][k], out["tau_req"][k] = q, qd, qdd, tau_app, tau
        resid[k] = np.max(np.abs(chk.inverse_dynamics(q, qd, qdd) - tau_app - tau_ext[k]))
    return out, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True); ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    t0 = time.time()
    sim = PandaSim(args.xml_dir, 0.0); chk = PandaSim(args.xml_dir, 0.0)
    rng = np.random.default_rng(ROLE_SEEDS["test"])
    refs = [sample_reference(rng, lo=sim.lo, hi=sim.hi) for _ in range(args.n)]
    test = np.load(os.path.join(args.data_dir, "test_N.npz"))
    ref_dev = float(np.abs(np.stack([r["q"] for r in refs]) - test["ref_q"][:args.n]).max())
    erng = np.random.default_rng(EVENT_SEED)
    recs, resid, evrows, taus = [], [], [], []
    for i, ref in enumerate(refs):
        events = sample_events(erng)
        tau_ext = profile(events)
        rec, rs = rollout_ext(sim, chk, ref, tau_ext, KP_TEACHER, KD_TEACHER)
        recs.append(rec); resid.append(rs); taus.append(tau_ext)
        for ei, e in enumerate(events):
            for j, A in zip(e["joints"], e["amps"]):
                evrows.append([i, ei, j, e["on"], e["off"], A, e["type"], e["rise"], len(e["joints"])])
    out = {k: np.stack([r[k] for r in recs]) for k in ["q", "qd", "qdd", "tau", "tau_req"]}
    out["tau_ext"] = np.stack(taus)
    out["id_resid"] = np.stack(resid)
    out["ref_q"] = np.stack([r["q"] for r in refs]); out["ref_qd"] = np.stack([r["qd"] for r in refs])
    out["ref_qdd"] = np.stack([r["qdd"] for r in refs])
    ev = np.array(evrows, dtype=np.float64)
    for c, name in enumerate(["ev_traj", "ev_event", "ev_joint", "ev_on", "ev_off", "ev_amp", "ev_type", "ev_rise", "ev_njoints"]):
        out[name] = ev[:, c]
    np.savez_compressed(args.out, **out)
    active = np.abs(out["tau_ext"]) > 0
    info = {"n_traj": args.n, "ref_max_abs_diff_vs_test_N": ref_dev, "id_identity_max": float(out["id_resid"].max()),
            "n_events": int(len(np.unique(ev[:, 0] * 100 + ev[:, 1]))), "n_event_joint_pairs": int(len(ev)),
            "active_fraction_any_joint": float(active.any(2).mean()), "active_fraction_per_joint": active.mean((0, 1)).tolist(),
            "teacher_saturation_frac": float((np.abs(out["tau_req"]) > TORQUE_LIMIT).mean()),
            "tracking_rmse_rad": float(np.sqrt(((out["q"] - out["ref_q"]) ** 2).mean())),
            "tracking_maxerr_rad": float(np.abs(out["q"] - out["ref_q"]).max()),
            "qdd_absmax_per_joint": np.abs(out["qdd"]).max((0, 1)).tolist(),
            "qdd_std_per_joint": out["qdd"].reshape(-1, 7).std(0).tolist(),
            "qdd_std_test_N": test["qdd"].reshape(-1, 7).std(0).tolist(),
            "event_seed": EVENT_SEED, "wall_s": time.time() - t0}
    json.dump(info, open(args.out.replace(".npz", "_info.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
