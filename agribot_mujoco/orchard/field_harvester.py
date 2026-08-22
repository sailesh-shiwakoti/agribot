"""
Field harvester — the full mobile-manipulation loop.

    SCAN (lidar) -> MAP trees -> pick nearest unvisited -> PLAN a path (A*)
      -> DRIVE there avoiding trunks -> PARK facing the tree
      -> HARVEST reachable red apples (arm IK) -> re-SCAN -> repeat

The robot only ever plans/navigates using trees it has perceived with the
lidar (see lidar.py, navigator.py). Detections accumulate across scans, so
occluded/distant trees are discovered as the robot drives — and one-off
spurious returns never accumulate enough confidence to be visited.
"""

from __future__ import annotations

import numpy as np
import mujoco

from .config import FieldConfig
from .field_scene import build_field
from .lidar import Lidar
from .ik import ArmIK, ARM_JOINTS, Q_NOMINAL
from .navigator import OccupancyGrid, astar, simplify

ARM_ACTS = ["shoulder_pan", "shoulder_lift", "elbow",
            "wrist_1", "wrist_2", "wrist_3"]


class FieldHarvester:
    def __init__(self, cfg: FieldConfig):
        self.cfg = cfg
        xml, self.trees, self.apples = build_field(cfg)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.ik = ArmIK(self.m)
        self.lidar = Lidar(self.m, self.d, cfg)

        self.aid = np.array([self.m.actuator(a).id for a in ARM_ACTS])
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.bx = self.m.actuator("base_x").id
        self.by = self.m.actuator("base_y").id
        self.byaw = self.m.actuator("base_yaw").id
        self.qbx = self.m.joint("base_x").qposadr[0]
        self.qby = self.m.joint("base_y").qposadr[0]
        self.qbyaw = self.m.joint("base_yaw").qposadr[0]
        self.tcp = self.m.site("tcp").id
        self.shoulder = self.m.body("shoulder_link").id

        self.renderer = mujoco.Renderer(self.m, cfg.height, cfg.width)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.m, self.cam)
        self.frames = []
        self.carry = None
        self.binned = []
        self.tree_map = []        # [center(2), weight]
        self.report = {"trees_visited": 0, "picked": 0,
                       "left_green": 0, "left_unreachable": 0}
        self._reset()

    # ------------------------------------------------------------- low level
    def _reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = Q_NOMINAL
        self.d.ctrl[self.aid] = Q_NOMINAL
        mujoco.mj_forward(self.m, self.d)

    def base_pose(self):
        return (float(self.d.qpos[self.qbx]), float(self.d.qpos[self.qby]),
                float(self.d.qpos[self.qbyaw]))

    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()

    def shoulder_pos(self):
        return self.d.xpos[self.shoulder].copy()

    def _bin_world(self, off):
        bx, by, yaw = self.base_pose()
        ox, oy, oz = off
        return [bx + np.cos(yaw) * ox - np.sin(yaw) * oy,
                by + np.sin(yaw) * ox + np.cos(yaw) * oy, oz]

    def _render(self):
        bx, by, _ = self.base_pose()
        self.cam.lookat[:] = [bx, by, 0.5]
        self.cam.distance = 3.6
        self.cam.azimuth = -90
        self.cam.elevation = -30
        self.renderer.update_scene(self.d, camera=self.cam)
        self.frames.append(self.renderer.render())

    def _step(self, n=1):
        for i in range(n):
            if self.carry is not None:
                bid, qa = self.carry
                self.d.qpos[qa:qa + 3] = self.tcp_pos()
                self.d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
                self.d.qvel[self.m.body(bid).dofadr[0]:
                            self.m.body(bid).dofadr[0] + 6] = 0
            for bid, qa, off in self.binned:
                self.d.qpos[qa:qa + 3] = self._bin_world(off)
                self.d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
                self.d.qvel[self.m.body(bid).dofadr[0]:
                            self.m.body(bid).dofadr[0] + 6] = 0
            mujoco.mj_step(self.m, self.d)
            if i % self.cfg.render_every == 0:
                self._render()

    # ------------------------------------------------------------- base motion
    def _drive_to(self, x, y, yaw):
        bx, by, byaw = self.base_pose()
        dist = np.hypot(x - bx, y - by)
        # unwrap yaw to nearest equivalent so we turn the short way
        while yaw - byaw > np.pi:
            yaw -= 2 * np.pi
        while yaw - byaw < -np.pi:
            yaw += 2 * np.pi
        steps = max(12, int(dist * 100) + int(abs(yaw - byaw) * 40))
        for k in range(1, steps + 1):
            f = k / steps
            self.d.ctrl[self.bx] = bx + (x - bx) * f
            self.d.ctrl[self.by] = by + (y - by) * f
            self.d.ctrl[self.byaw] = byaw + (yaw - byaw) * f
            self._step(1)

    def _drive_path(self, waypoints, final_yaw):
        for i, wp in enumerate(waypoints[1:], 1):
            prev = waypoints[i - 1]
            heading = np.arctan2(wp[1] - prev[1], wp[0] - prev[0])
            self._drive_to(wp[0], wp[1], heading)
        # face the tree at the end
        x, y, _ = self.base_pose()
        self._drive_to(x, y, final_yaw)

    # ------------------------------------------------------------- mapping
    def scan_and_map(self):
        for center, npts in self.lidar.detect_trees():
            merged = False
            for e in self.tree_map:
                if np.linalg.norm(center - e[0]) < 0.9:   # same trunk
                    e[0] = (e[0] * e[1] + center * npts) / (e[1] + npts)
                    e[1] += npts
                    merged = True
                    break
            if not merged:
                self.tree_map.append([center.astype(float), npts])

    def committed_trees(self, min_weight=3):
        # min_weight filters one-off spurious returns (real trees accumulate
        # across scans as the robot drives). Kept low so distant trees, which
        # return few points until approached, are still committed.
        return [e[0] for e in self.tree_map if e[1] >= min_weight]

    # ------------------------------------------------------------- arm harvest
    def _move_tcp(self, target, steps):
        q, ok = self.ik.solve(self.d, np.asarray(target, float))
        if not ok:
            return False
        q0 = self.d.ctrl[self.aid].copy()
        for k in range(1, steps + 1):
            self.d.ctrl[self.aid] = q0 + (q - q0) * (k / steps)
            self._step(1)
        return True

    def _go_nominal(self, steps=30):
        q0 = self.d.ctrl[self.aid].copy()
        for k in range(1, steps + 1):
            self.d.ctrl[self.aid] = q0 + (Q_NOMINAL - q0) * (k / steps)
            self._step(1)

    def _apple_qadr(self, name):
        return self.m.jnt_qposadr[self.m.body(name).jntadr[0]]

    def _apple_collision(self, name, on):
        gid = self.m.body(name).geomadr[0]
        self.m.geom_contype[gid] = 1 if on else 0
        self.m.geom_conaffinity[gid] = 1 if on else 0

    def _weld_off(self, name):
        eid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY,
                                f"weld_tree_{name}")
        self.d.eq_active[eid] = 0

    def _bin_slot(self):
        s = len(self.binned)
        return (-0.09 + 0.09 * (s % 3), -0.40 + 0.07 * ((s // 3) % 4),
                0.32 + 0.06 * (s // 12))

    def _pick(self, name):
        pos = self.d.body(name).xpos.copy()
        # approach from the arm side, then reach
        sh = self.shoulder_pos()
        approach = pos + 0.12 * (sh - pos) / (np.linalg.norm(sh - pos) + 1e-6)
        approach[2] = pos[2]
        if not self._move_tcp(approach, self.cfg.approach_steps):
            return False
        if not self._move_tcp(pos, self.cfg.reach_steps):
            self._go_nominal(20)
            return False
        self._weld_off(name)
        self._apple_collision(name, False)
        self.carry = (self.m.body(name).id, self._apple_qadr(name))
        self._step(5)
        self._move_tcp(pos + np.array([0, 0, 0.16]), self.cfg.lift_steps)
        off = self._bin_slot()
        qa = self._apple_qadr(name)
        self.binned.append((self.m.body(name).id, qa, off))
        self.carry = None
        self.d.qpos[qa:qa + 3] = self._bin_world(off)
        self._step(self.cfg.settle_steps)
        self._go_nominal(28)
        return True

    def _harvest_tree(self, tree_center):
        """Pick every reachable ripe apple whose trunk is this tree."""
        sh = self.shoulder_pos()
        near = [a for a in self.apples
                if np.hypot(a.pos[0] - tree_center[0],
                            a.pos[1] - tree_center[1]) < 0.4]
        picked = 0
        for a in sorted(near, key=lambda a: np.linalg.norm(a.pos - sh)):
            if not a.is_ripe:
                self.report["left_green"] += 1
                continue
            if np.linalg.norm(a.pos - self.shoulder_pos()) > self.cfg.reach_limit:
                self.report["left_unreachable"] += 1
                continue
            _, ok = self.ik.solve(self.d, self.d.body(a.name).xpos.copy())
            if not ok:
                self.report["left_unreachable"] += 1
                continue
            if self._pick(a.name):
                self.report["picked"] += 1
                picked += 1
        return picked

    # ------------------------------------------------------------------- run
    def run(self):
        self.scan_and_map()
        self._step(10)
        visited = []
        grid = OccupancyGrid((self.cfg.field_x, self.cfg.field_y),
                             self.cfg.grid_res, self.cfg.robot_radius)

        while True:
            bx, by, _ = self.base_pose()
            base_xy = np.array([bx, by])
            todo = [c for c in self.committed_trees()
                    if all(np.linalg.norm(c - v) > 1.2 for v in visited)]
            if not todo:
                break
            target = min(todo, key=lambda c: np.linalg.norm(c - base_xy))

            # park pose: park_dist from the trunk on the approach side
            u = (target - base_xy) / (np.linalg.norm(target - base_xy) + 1e-6)
            park = target - self.cfg.park_dist * u
            yaw = np.arctan2(u[1], u[0]) - np.pi / 2   # arm front (+y) -> tree

            grid.block_trees(self.committed_trees(), skip=target)
            path = astar(grid, base_xy, park)
            if path is None:
                path = [base_xy, park]                 # fallback: straight line
            wps = simplify(path)
            wps.append(park)                           # end at the EXACT park pose
            self._drive_path(wps, yaw)

            self._harvest_tree(target)
            visited.append(target)
            self.report["trees_visited"] += 1
            self.scan_and_map()                        # discover more trees

        self._step(20)
        return self.report
