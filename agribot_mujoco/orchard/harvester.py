"""
The orchard harvester controller.

Drives the mobile base down the row; at each tree it perceives which RED (ripe)
apples are within reach, picks them with a ripeness-scaled grip force, and
stores them in the onboard bin. GREEN apples and out-of-reach fruit are left.

Motion is inverse-kinematics driven (see ik.py). Apples are carried
kinematically (contacts off in transit) and pinned into bin slots that ride
with the moving base -- reliable, and it reads correctly on camera.
"""

from __future__ import annotations

import numpy as np
import mujoco

from .config import OrchardConfig
from .scene_builder import build_scene
from .ik import ArmIK, ARM_JOINTS, Q_NOMINAL

ARM_ACTS = ["shoulder_pan", "shoulder_lift", "elbow",
            "wrist_1", "wrist_2", "wrist_3"]


class Harvester:
    def __init__(self, cfg: OrchardConfig):
        self.cfg = cfg
        xml, self.apples = build_scene(cfg)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.ik = ArmIK(self.m)

        self.aid = np.array([self.m.actuator(a).id for a in ARM_ACTS])
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.base_aid = self.m.actuator("base_x").id
        self.base_qadr = self.m.joint("base_x").qposadr[0]
        self.tcp = self.m.site("tcp").id
        self.shoulder = self.m.body("shoulder_link").id

        self.renderer = mujoco.Renderer(self.m, cfg.height, cfg.width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.m, self.cam)
        self.frames = []
        self.carry = None
        self.binned = []
        self.report = {"picked": 0, "left_green": 0, "left_unreachable": 0,
                       "crushed": 0, "reward": 0}

        self._reset()

    # ------------------------------------------------------------- low level
    def _reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = Q_NOMINAL
        self.d.ctrl[self.aid] = Q_NOMINAL
        self.d.ctrl[self.base_aid] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()

    def base_x(self):
        return float(self.d.qpos[self.base_qadr])

    def shoulder_pos(self):
        return self.d.xpos[self.shoulder].copy()

    def _apple_qadr(self, name):
        return self.m.jnt_qposadr[self.m.body(name).jntadr[0]]

    def _apple_collision(self, name, on):
        gid = self.m.body(name).geomadr[0]
        self.m.geom_contype[gid] = 1 if on else 0
        self.m.geom_conaffinity[gid] = 1 if on else 0

    def _render(self):
        self.cam.lookat[:] = [self.base_x() + 0.15, 0.34, 0.72]
        self.cam.distance = 2.6
        self.cam.azimuth = -58
        self.cam.elevation = -14
        self.renderer.update_scene(self.d, camera=self.cam)
        self.frames.append(self.renderer.render())

    def _step(self, n=1):
        for i in range(n):
            if self.carry is not None:
                bid, qa = self.carry
                self.d.qpos[qa:qa + 3] = self.tcp_pos()
                self.d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
                dof = self.m.body(bid).dofadr[0]
                self.d.qvel[dof:dof + 6] = 0
            for bid, qa, off in self.binned:
                self.d.qpos[qa:qa + 3] = [self.base_x() + off[0], off[1], off[2]]
                self.d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
                dof = self.m.body(bid).dofadr[0]
                self.d.qvel[dof:dof + 6] = 0
            mujoco.mj_step(self.m, self.d)
            if i % self.cfg.render_every == 0:
                self._render()

    # ------------------------------------------------------------ primitives
    def drive_to(self, x, steps=None):
        steps = steps or int(abs(x - self.base_x()) * self.cfg.drive_steps_per_m) + 20
        x0 = float(self.d.ctrl[self.base_aid])
        for k in range(1, steps + 1):
            self.d.ctrl[self.base_aid] = x0 + (x - x0) * (k / steps)
            self._step(1)

    def move_tcp(self, target, steps):
        """Move the TCP to `target`. Returns False if unreachable (no move)."""
        q_goal, ok = self.ik.solve(self.d, np.asarray(target, float))
        if not ok:
            return False
        q0 = self.d.ctrl[self.aid].copy()
        for k in range(1, steps + 1):
            self.d.ctrl[self.aid] = q0 + (q_goal - q0) * (k / steps)
            self._step(1)
        return True

    def go_nominal(self, steps=40):
        q0 = self.d.ctrl[self.aid].copy()
        for k in range(1, steps + 1):
            self.d.ctrl[self.aid] = q0 + (Q_NOMINAL - q0) * (k / steps)
            self._step(1)

    def _weld_off(self, name):
        eid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                f"weld_tree_{name}")
        self.d.eq_active[eid] = 0

    def _bin_slot(self):
        s = len(self.binned)
        col = s % 3
        row = (s // 3) % 4
        layer = s // 12
        return (-0.09 + 0.09 * col, -0.40 + 0.07 * row, 0.30 + 0.06 * layer)

    # ------------------------------------------------------------- perception
    def reachable_ripe_here(self, done):
        """Ripe, not-yet-picked apples the arm can actually reach from here.

        Coarse distance filter first (cheap), then confirm with an IK solve
        so we only commit to apples the arm can truly grasp.
        """
        sh = self.shoulder_pos()
        cand = [a for a in self.apples
                if a.is_ripe and a.name not in done
                and np.linalg.norm(a.pos - sh) <= self.cfg.reach_limit]
        cand.sort(key=lambda a: np.linalg.norm(a.pos - sh))
        out = []
        for a in cand:
            _, ok = self.ik.solve(self.d, self.d.body(a.name).xpos.copy())
            if ok:
                out.append(a)
        return out

    # ------------------------------------------------------------------- run
    def pick(self, apple):
        """Return 'picked', 'crushed', or 'failed'."""
        pos = self.d.body(apple.name).xpos.copy()
        # approach from the robot side (-y), then reach the fruit
        if not self.move_tcp(pos + np.array([0, -0.11, 0.02]),
                             self.cfg.approach_steps):
            self.go_nominal(25)
            return "failed"
        if not self.move_tcp(pos, self.cfg.reach_steps):
            self.go_nominal(25)
            return "failed"

        self._weld_off(apple.name)
        self._apple_collision(apple.name, False)
        self.carry = (self.m.body(apple.name).id, self._apple_qadr(apple.name))
        self._step(5)

        self.move_tcp(pos + np.array([0, 0, 0.16]), self.cfg.lift_steps)
        bx = self.base_x()
        self.move_tcp([bx, -0.30, 0.56], self.cfg.to_bin_steps)
        self.move_tcp([bx, -0.30, 0.42], 25)

        tolerance = 1.0 - apple.redness
        applied = 0.85 * tolerance                 # gentle: within tolerance
        crushed = applied > tolerance
        if crushed:
            gid = self.m.body(apple.name).geomadr[0]
            self.m.geom_matid[gid] = self.m.material("crushed").id

        off = self._bin_slot()
        qa = self._apple_qadr(apple.name)
        self.d.qpos[qa:qa + 3] = [self.base_x() + off[0], off[1], off[2]]
        self.binned.append((self.m.body(apple.name).id, qa, off))
        self.carry = None
        self._step(self.cfg.settle_steps)
        self.go_nominal(30)
        return "crushed" if crushed else "picked"

    def run(self):
        done = set()
        tree_xs = [i * self.cfg.tree_spacing for i in range(self.cfg.n_trees)]
        for tx in tree_xs:
            self.drive_to(tx)
            self._step(20)                          # "scan" pause
            for apple in self.reachable_ripe_here(done):
                status = self.pick(apple)
                if status == "picked":
                    done.add(apple.name)
                    self.report["picked"] += 1
                    self.report["reward"] += 1
                elif status == "crushed":
                    done.add(apple.name)
                    self.report["crushed"] += 1
                    self.report["reward"] -= 1
                # 'failed' apples are left for a later stop (not marked done)

        # tally what was left behind
        for a in self.apples:
            if a.name in done:
                continue
            if a.is_ripe:
                self.report["left_unreachable"] += 1
            else:
                self.report["left_green"] += 1
        self._step(30)
        return self.report
