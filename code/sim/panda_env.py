"""Torque-controlled 7-DOF Franka Panda in MuJoCo.

Plant: mujoco_menagerie panda_nohand.xml with its inertial parameters, joint
damping (1 N m s/rad) and armature (0.1), position actuators replaced by direct
torque motors with the Panda effort limits (87/87/87/87/12/12/12 N m). Physics
1 kHz (implicitfast), control 100 Hz, zero-order-hold torques.

Records at every control instant k: q_k, qd_k (true state), qdd_k = forward
dynamics acceleration under the applied torque at that instant, tau_k = applied
actuator torque after the effort limit. These satisfy the exact inverse-dynamics
relation of the plant, which gen_data.py verifies with mj_inverse.

Payload condition P1: a 1.0 kg sphere (radius 5 cm, consistent inertia) rigidly
attached 5 cm beyond the flange; mass, centre of mass and inertia are updated
together by the compiler.
"""
import os
import re
import numpy as np
import mujoco

TORQUE_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
PHYSICS_DT = 0.001
CONTROL_HZ = 100
SUBSTEPS = int(round(1.0 / CONTROL_HZ / PHYSICS_DT))
TRAJ_SECONDS = 10.0
N_STEPS = int(TRAJ_SECONDS * CONTROL_HZ)
LIMIT_MARGIN = 0.15
FAIL_ERR = 0.5          # rad, any joint -> episode failure (frozen on dev)
PAYLOAD_KG = 1.0
PAYLOAD_R = 0.05


def build_xml(xml_dir, payload_kg=0.0, physics_dt=PHYSICS_DT):
    src = open(os.path.join(xml_dir, "panda_nohand.xml"), encoding="utf-8").read()
    # 1) physics step
    src = src.replace('<option integrator="implicitfast" impratio="10"/>',
                      f'<option timestep="{physics_dt}" integrator="implicitfast" impratio="10"/>')
    # 2) torque motors
    motors = "\n".join(
        f'    <motor name="motor{j+1}" joint="joint{j+1}" ctrllimited="true" ctrlrange="-{TORQUE_LIMIT[j]} {TORQUE_LIMIT[j]}"/>'
        for j in range(7))
    src = re.sub(r"<actuator>.*?</actuator>", "<actuator>\n" + motors + "\n  </actuator>", src, flags=re.S)
    # 3) keyframe ctrl now means torque: zero
    src = re.sub(r'ctrl="[^"]*"', 'ctrl="0 0 0 0 0 0 0"', src)
    # 4) optional payload body attached beyond the flange
    if payload_kg > 0:
        I = 0.4 * payload_kg * PAYLOAD_R ** 2
        payload = (f'<body name="payload" pos="0 0 {PAYLOAD_R}">'
                   f'<inertial mass="{payload_kg}" pos="0 0 0" diaginertia="{I} {I} {I}"/>'
                   f'</body>')
        src = src.replace('<site name="attachment_site"/>', '<site name="attachment_site"/>' + payload)
    return src


class PandaSim:
    def __init__(self, xml_dir, payload_kg=0.0, physics_dt=PHYSICS_DT):
        self.xml = build_xml(xml_dir, payload_kg, physics_dt)
        path = os.path.join(xml_dir, f"panda_torque_{payload_kg}_{physics_dt}.xml")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.xml)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.substeps = int(round(1.0 / CONTROL_HZ / physics_dt))
        self.lo = self.model.jnt_range[:, 0] + LIMIT_MARGIN
        self.hi = self.model.jnt_range[:, 1] - LIMIT_MARGIN
        self.payload_kg = payload_kg

    # -------------------------------------------------------------- state
    def reset(self, q, qd):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q; self.data.qvel[:] = qd
        mujoco.mj_forward(self.model, self.data)

    def inverse_dynamics(self, q, qd, qdd):
        """Exact generalized force that produces qdd at (q, qd) in this plant."""
        d = self.data
        d.qpos[:] = q; d.qvel[:] = qd; d.qacc[:] = qdd
        mujoco.mj_inverse(self.model, d)
        return d.qfrc_inverse.copy()

    def apply_and_step(self, tau, check=False):
        """Apply torque (clipped by the motor limits), record the forward
        acceleration at this instant, then advance one control period.
        check=True also returns |ID(q, qd, qdd) - tau_applied|_max and the number
        of active contacts at the instant (E00 identity check)."""
        d = self.data
        d.ctrl[:] = tau
        mujoco.mj_forward(self.model, d)
        qdd = d.qacc.copy()
        tau_applied = d.actuator_force.copy()
        chk = None
        if check:
            ncon = int(d.ncon)
            mujoco.mj_inverse(self.model, d)          # uses current qpos, qvel, qacc
            chk = (float(np.max(np.abs(d.qfrc_inverse - tau_applied))), ncon)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, d)
        return qdd, tau_applied, chk

    # -------------------------------------------------------------- teacher
    def rollout_teacher(self, ref, kp, kd):
        """Computed-torque teacher (exact ID + PD) tracking a reference; returns
        the recorded inverse-dynamics samples for the actual states."""
        q_ref, qd_ref, qdd_ref = ref["q"], ref["qd"], ref["qdd"]
        n = q_ref.shape[0]
        Q = np.zeros((n, 7)); QD = np.zeros((n, 7)); QDD = np.zeros((n, 7)); TAU = np.zeros((n, 7))
        TAU_REQ = np.zeros((n, 7)); CHK = np.zeros(n); NCON = np.zeros(n, int)
        self.reset(q_ref[0], qd_ref[0])
        for k in range(n):
            q = self.data.qpos.copy(); qd = self.data.qvel.copy()
            tau_ff = self.inverse_dynamics(q, qd, qdd_ref[k])   # qpos/qvel unchanged
            tau = tau_ff + kp * (q_ref[k] - q) + kd * (qd_ref[k] - qd)
            qdd, tau_app, chk = self.apply_and_step(tau, check=True)
            Q[k], QD[k], QDD[k], TAU[k], TAU_REQ[k] = q, qd, qdd, tau_app, tau
            CHK[k], NCON[k] = chk
        return {"q": Q, "qd": QD, "qdd": QDD, "tau": TAU, "tau_req": TAU_REQ, "id_check": CHK,
                "ncon": NCON}


def _load_assets(xml_dir):
    assets = {}
    adir = os.path.join(xml_dir, "assets")
    for f in os.listdir(adir):
        assets[f] = open(os.path.join(adir, f), "rb").read()
    return assets


# ------------------------------------------------------------------ references
def sample_reference(rng, n_steps=N_STEPS, lo=None, hi=None):
    """Joint-space multi-sine reference with analytic derivatives, kept inside the
    joint limits (with margin) by amplitude scaling."""
    t = np.arange(n_steps) / CONTROL_HZ
    H = 3
    A = rng.uniform(0.05, 0.30, (7, H))
    f = rng.uniform(0.10, 0.40, (7, H))
    ph = rng.uniform(0, 2 * np.pi, (7, H))
    c = Q_HOME + rng.uniform(-0.2, 0.2, 7)
    if lo is not None:
        amp = A.sum(1)
        room = np.minimum(c - lo, hi - c)
        scale = np.minimum(1.0, room / np.maximum(amp, 1e-9))
        A = A * scale[:, None]
    w = 2 * np.pi * f
    arg = w[:, :, None] * t[None, None, :] + ph[:, :, None]
    q = c[:, None] + (A[:, :, None] * np.sin(arg)).sum(1)
    qd = (A[:, :, None] * w[:, :, None] * np.cos(arg)).sum(1)
    qdd = -(A[:, :, None] * w[:, :, None] ** 2 * np.sin(arg)).sum(1)
    return {"q": q.T.copy(), "qd": qd.T.copy(), "qdd": qdd.T.copy(),
            "params": {"A": A, "f": f, "ph": ph, "c": c}}


# ------------------------------------------------------------------ closed loop
class BatchedClosedLoop:
    """Run B episodes in lockstep with a shared feedback law and a frozen
    feedforward predictor:
        tau = fhat(q_obs, qd_obs, qdd_ref) + Kp (q_ref - q_obs) + Kd (qd_ref - qd_obs)
    Observations are perturbed by the condition's observer; all controllers of a
    paired comparison receive identical reference and noise draws."""

    def __init__(self, xml_dir, refs, kp, kd, payload_kg=0.0, obs_noise=None, obs_bias=None,
                 noise_seed=0, q_scale=None, qd_scale=None):
        self.B = len(refs)
        self.sims = [PandaSim(xml_dir, payload_kg) for _ in range(self.B)]
        self.refs = refs
        self.kp, self.kd = kp, kd
        self.n = refs[0]["q"].shape[0]
        rng = np.random.default_rng(noise_seed)
        self.noise_q = np.zeros((self.B, self.n, 7)); self.noise_qd = np.zeros((self.B, self.n, 7))
        if obs_noise:   # rho relative to training std of each channel, i.i.d. per step
            self.noise_q = rng.standard_normal((self.B, self.n, 7)) * obs_noise * q_scale
            self.noise_qd = rng.standard_normal((self.B, self.n, 7)) * obs_noise * qd_scale
        if obs_bias:    # constant per episode
            bq = rng.standard_normal((self.B, 1, 7)) * obs_bias * q_scale
            bqd = rng.standard_normal((self.B, 1, 7)) * obs_bias * qd_scale
            self.noise_q = self.noise_q + bq; self.noise_qd = self.noise_qd + bqd

    def run(self, feedforward=None, fb_true=False):
        """feedforward: callable(X [B,21] physical (q_obs, qd_obs, qdd_ref)) -> tau_ff [B,7]
        (already clipped to the torque limits by the caller's contract) or None (PD only).
        fb_true=True is a diagnostic: the feedback terms use the true state while the
        feedforward still receives the perturbed observation (not a deployable controller).
        Returns per-episode metrics and the trajectories."""
        B, n = self.B, self.n
        for b in range(B):
            self.sims[b].reset(self.refs[b]["q"][0], self.refs[b]["qd"][0])
        err = np.zeros((B, n, 7)); tau_req = np.zeros((B, n, 7)); tau_app = np.zeros((B, n, 7))
        q_log = np.zeros((B, n, 7)); alive = np.ones(B, bool); fail_step = np.full(B, -1)
        tau_fb_log = np.zeros((B, n, 7)); tau_ff_log = np.zeros((B, n, 7))
        for k in range(n):
            q = np.stack([s.data.qpos for s in self.sims]); qd = np.stack([s.data.qvel for s in self.sims])
            q_obs = q + self.noise_q[:, k]; qd_obs = qd + self.noise_qd[:, k]
            q_ref = np.stack([r["q"][k] for r in self.refs]); qd_ref = np.stack([r["qd"][k] for r in self.refs])
            qdd_ref = np.stack([r["qdd"][k] for r in self.refs])
            q_fb, qd_fb = (q, qd) if fb_true else (q_obs, qd_obs)
            tau = self.kp * (q_ref - q_fb) + self.kd * (qd_ref - qd_fb)
            tau_fb_log[:, k] = tau
            if feedforward is not None:
                X = np.concatenate([q_obs, qd_obs, qdd_ref], 1)
                tff = feedforward(X)
                tau_ff_log[:, k] = tff
                tau = tau + tff
            for b in range(B):
                if not alive[b]:
                    continue
                _, ta, _ = self.sims[b].apply_and_step(tau[b])
                tau_app[b, k] = ta; tau_req[b, k] = tau[b]
                e = q[b] - q_ref[b]
                err[b, k] = e; q_log[b, k] = q[b]
                if (not np.all(np.isfinite(self.sims[b].data.qpos))) or np.max(np.abs(e)) > FAIL_ERR:
                    alive[b] = False; fail_step[b] = k
            # failed episodes: fill the remaining error with the failure threshold
        for b in range(B):
            if fail_step[b] >= 0:
                err[b, fail_step[b] + 1:] = FAIL_ERR
        sat = (np.abs(tau_req) > TORQUE_LIMIT).mean((1, 2))
        rmse = np.sqrt((err ** 2).mean((1, 2)))
        maxerr = np.abs(err).max((1, 2))
        tau_rms = np.sqrt((tau_app ** 2).mean((1, 2)))
        return {"rmse": rmse, "maxerr": maxerr, "tau_rms": tau_rms, "saturation": sat,
                "failed": fail_step >= 0, "fail_step": fail_step, "err": err, "q": q_log,
                "tau_req": tau_req,
                "tau_fb_rms": np.sqrt((tau_fb_log ** 2).mean((1, 2))), "tau_ff_rms": np.sqrt((tau_ff_log ** 2).mean((1, 2)))}
