"""Generate the simulation records and the E00 checks.

Roles (nominal plant, teacher-collected, actual states and applied torques):
  train 80 traj, dev 20, cal 10 groups x 10 traj, test 40 traj (open loop)
Payload plant P1: cal (same 100 references re-simulated), test (same 40).
Closed-loop reference set: 30 references (shared by every condition/controller).
All references come from the same sampler with role-specific seeds.
E00 checks written to e00_checks.json: inverse-dynamics identity residual,
physics step-halving check, torque saturation fraction of the teacher.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.panda_env import PandaSim, sample_reference, PAYLOAD_KG, N_STEPS, TORQUE_LIMIT

ROLE_SEEDS = {"train": 100, "dev": 200, "cal": 300, "test": 400, "closed": 500}
ROLE_SIZES = {"train": 80, "dev": 20, "cal": 100, "test": 40, "closed": 30}
KP_TEACHER = np.array([100.0, 100, 100, 100, 40, 40, 40])
KD_TEACHER = np.array([8.0, 8, 8, 8, 3.2, 3.2, 3.2])


def gen_role(sim, role, n, seed):
    rng = np.random.default_rng(seed)
    refs, recs = [], []
    for i in range(n):
        ref = sample_reference(rng, lo=sim.lo, hi=sim.hi)
        refs.append(ref)
        recs.append(sim.rollout_teacher(ref, KP_TEACHER, KD_TEACHER))
    out = {k: np.stack([r[k] for r in recs]) for k in ["q", "qd", "qdd", "tau", "tau_req", "id_check", "ncon"]}
    out["ref_q"] = np.stack([r["q"] for r in refs]); out["ref_qd"] = np.stack([r["qd"] for r in refs])
    out["ref_qdd"] = np.stack([r["qdd"] for r in refs])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    checks = {}
    t0 = time.time()
    sim = PandaSim(args.xml_dir, 0.0)
    for role in ["train", "dev", "cal", "test"]:
        rec = gen_role(sim, role, ROLE_SIZES[role], ROLE_SEEDS[role])
        np.savez_compressed(os.path.join(args.out, f"{role}_N.npz"), **rec)
        checks[f"{role}_N"] = {
            "n_traj": int(rec["q"].shape[0]), "id_identity_max": float(rec["id_check"].max()),
            "id_identity_mean": float(rec["id_check"].mean()), "contact_steps": int((rec["ncon"] > 0).sum()),
            "teacher_saturation_frac": float((np.abs(rec["tau_req"]) > TORQUE_LIMIT).mean()),
            "tracking_rmse_rad": float(np.sqrt(((rec["q"] - rec["ref_q"]) ** 2).mean())),
            "q_std": rec["q"].reshape(-1, 7).std(0).tolist(), "qd_std": rec["qd"].reshape(-1, 7).std(0).tolist(),
            "qdd_std": rec["qdd"].reshape(-1, 7).std(0).tolist(), "tau_std": rec["tau"].reshape(-1, 7).std(0).tolist(),
            "tau_absmax": np.abs(rec["tau"]).reshape(-1, 7).max(0).tolist()}
        print(role, checks[f"{role}_N"], time.time() - t0, flush=True)
    # payload plant, same references as cal/test
    simP = PandaSim(args.xml_dir, PAYLOAD_KG)
    for role in ["cal", "test"]:
        rec = gen_role(simP, role, ROLE_SIZES[role], ROLE_SEEDS[role])
        np.savez_compressed(os.path.join(args.out, f"{role}_P1.npz"), **rec)
        nom = np.load(os.path.join(args.out, f"{role}_N.npz"))
        checks[f"{role}_P1"] = {
            "id_identity_max": float(rec["id_check"].max()),
            "tau_shift_rms_per_joint": np.sqrt(((rec["tau"] - nom["tau"]) ** 2).mean((0, 1))).tolist(),
            "tau_std": rec["tau"].reshape(-1, 7).std(0).tolist(),
            "teacher_saturation_frac": float((np.abs(rec["tau_req"]) > TORQUE_LIMIT).mean())}
        print(role, "P1", checks[f"{role}_P1"], flush=True)
    # closed-loop references
    rng = np.random.default_rng(ROLE_SEEDS["closed"])
    refs = [sample_reference(rng, lo=sim.lo, hi=sim.hi) for _ in range(ROLE_SIZES["closed"])]
    np.savez_compressed(os.path.join(args.out, "closed_refs.npz"),
                        q=np.stack([r["q"] for r in refs]), qd=np.stack([r["qd"] for r in refs]),
                        qdd=np.stack([r["qdd"] for r in refs]))
    # step-halving check on one development reference
    rngc = np.random.default_rng(ROLE_SEEDS["dev"])
    ref = sample_reference(rngc, lo=sim.lo, hi=sim.hi)
    r1 = PandaSim(args.xml_dir, 0.0, 0.001).rollout_teacher(ref, KP_TEACHER, KD_TEACHER)
    r2 = PandaSim(args.xml_dir, 0.0, 0.0005).rollout_teacher(ref, KP_TEACHER, KD_TEACHER)
    checks["step_halving"] = {"q_max_abs_diff": float(np.abs(r1["q"] - r2["q"]).max()),
                              "tau_rms_diff": float(np.sqrt(((r1["tau"] - r2["tau"]) ** 2).mean())),
                              "tau_rms": float(np.sqrt((r1["tau"] ** 2).mean()))}
    checks["physics"] = {"physics_dt": 0.001, "control_hz": 100, "integrator": "implicitfast",
                         "torque_limits": TORQUE_LIMIT.tolist(), "payload_kg": PAYLOAD_KG,
                         "teacher_kp": KP_TEACHER.tolist(), "teacher_kd": KD_TEACHER.tolist(),
                         "traj_steps": N_STEPS, "gen_time_s": time.time() - t0}
    with open(os.path.join(args.out, "e00_checks.json"), "w") as f:
        json.dump(checks, f, indent=1)
    print(json.dumps(checks["step_halving"]), json.dumps(checks["physics"]))


if __name__ == "__main__":
    main()
