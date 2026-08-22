"""
Robust inverse kinematics for the UR5e arm.

Damped least squares with a null-space posture bias and joint-limit clamping.
The posture bias is what fixes the "arm flails to a weird far pose" bug: among
the infinitely many joint solutions that reach a point, it prefers the one
closest to a natural resting posture, and it never chases a target it cannot
actually reach (the solve reports `reachable`).

This is intentionally hand-rolled (not a library) because implementing DLS +
null-space control is one of the learning goals of the project.
"""

from __future__ import annotations

import numpy as np
import mujoco

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# A comfortable "elbow-up, facing the wall" posture the solver biases toward.
Q_NOMINAL = np.array([0.0, -1.1, 1.4, -1.5708, -1.5708, 0.0])


class ArmIK:
    def __init__(self, model, site="tcp"):
        self.m = model
        self.site = model.site(site).id
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.jrange = np.array([model.joint(j).range for j in ARM_JOINTS])
        self._scratch = mujoco.MjData(model)

    def solve(self, data, target, iters=140, tol=1.2e-2, damp=0.07,
              k_posture=0.02, max_step=0.25):
        """Return (q_arm, reachable). Does not modify `data`.

        The posture bias only acts while still far from the target (>5 cm), so
        it shapes the *approach* into a natural pose but never fights final
        convergence — that was the "can't reach / flails" failure mode.
        """
        s = self._scratch
        s.qpos[:] = data.qpos                      # includes current base_x
        q = data.qpos[self.qadr].copy()
        jacp = np.zeros((3, self.m.nv))
        err = np.full(3, 1e3)
        for _ in range(iters):
            s.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            err = target - s.site_xpos[self.site]
            e = np.linalg.norm(err)
            if e < tol:
                break
            mujoco.mj_jacSite(self.m, s, jacp, None, self.site)
            J = jacp[:, self.dadr]                 # 3 x 6
            JtJinv = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3),
                                           np.eye(3))
            dq = JtJinv @ err                      # task step
            if e > 0.05:                           # posture bias only when far
                N = np.eye(6) - JtJinv @ J         # (approx) null-space
                dq += N @ (k_posture * (Q_NOMINAL - q))
            dq = np.clip(dq, -max_step, max_step)
            q = np.clip(q + dq, self.jrange[:, 0], self.jrange[:, 1])
        return q, bool(np.linalg.norm(err) < tol)

    def fk(self, data, q_arm):
        """TCP world position for a given arm configuration (uses scratch)."""
        s = self._scratch
        s.qpos[:] = data.qpos
        s.qpos[self.qadr] = q_arm
        mujoco.mj_kinematics(self.m, s)
        mujoco.mj_comPos(self.m, s)
        return s.site_xpos[self.site].copy()
