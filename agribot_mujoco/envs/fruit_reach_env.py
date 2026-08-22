"""
FruitReachEnv — a Gymnasium environment where a UR5e arm must move its
tool-center-point (TCP) to a randomly placed "fruit" in an orchard scene.

This is the state-based reaching task from Phase 4 of the AgriBot plan: the
policy sees joint state + the vector to the fruit, and learns to reach it.
It is deliberately the *well-conditioned* version of the problem (dense reward,
full state) so SAC/PPO converge quickly on a laptop. Image-based and
obstacle-aware variants are stretch goals that subclass this.

Physics/model notes:
  - The UR5e actuators are position servos (see ur5e.xml): ctrl = target joint
    angle. Our action is a *delta* applied to those targets, which keeps the
    policy output bounded and the motion smooth.
  - The fruit is a MoCap body, so we teleport it each episode by writing
    data.mocap_pos — no free joint, no settling, no contact tuning.
  - The TCP is ur5e.xml's <site name="attachment_site"> at the tool flange.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# Path to the orchard MJCF (lives beside ur5e.xml so meshes resolve).
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    _HERE, "menagerie", "universal_robots_ur5e", "agribot_orchard.xml"
)

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
ARM_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


class FruitReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        render_mode: Optional[str] = None,
        action_scale: float = 0.12,   # rad added to joint targets per step
        success_radius: float = 0.06, # m — TCP within 6 cm counts as a reach
        max_steps: int = 180,
        n_substeps: int = 5,          # physics steps per env step
        control_cost: float = 0.01,
        randomize: float = 1.0,       # 0..1 scales the target sampling volume
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.action_scale = action_scale
        self.success_radius = success_radius
        self.max_steps = max_steps
        self.n_substeps = n_substeps
        self.control_cost = control_cost
        self.randomize = float(np.clip(randomize, 0.0, 1.0))

        # Resolve model indices once.
        self._qpos_adr = np.array(
            [self.model.joint(n).qposadr[0] for n in ARM_JOINTS]
        )
        self._dof_adr = np.array(
            [self.model.joint(n).dofadr[0] for n in ARM_JOINTS]
        )
        self._act_id = np.array(
            [self.model.actuator(n).id for n in ARM_ACTUATORS]
        )
        self._ctrl_range = self.model.actuator_ctrlrange[self._act_id].copy()
        self._tcp_site = self.model.site("attachment_site").id
        self._fruit_mocap = self.model.body("fruit").mocapid[0]
        self._shoulder_body = self.model.body("shoulder_link").id

        # home keyframe target (for reset + as the ctrl baseline)
        self._home_key = self.model.key("home").id
        self._home_qpos = self.model.key("home").qpos.copy()
        self._home_ctrl = self.model.key("home").ctrl.copy()

        # Spaces. Action: per-joint delta in [-1, 1]. Obs: joint pos/vel +
        # vector from TCP to fruit.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        obs_dim = 6 + 6 + 3  # qpos, qvel, tcp->fruit
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._step_count = 0
        self._target = np.zeros(3)
        self._renderer: Optional[mujoco.Renderer] = None

        if seed is not None:
            self.reset(seed=seed)

    # ------------------------------------------------------------------ utils
    def _tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site].copy()

    # Per-joint sampling half-range (rad) around the home pose. Targets are the
    # TCP positions these configs produce, so every target is reachable by
    # construction — the key trick that makes a reach task learnable. Scaled by
    # `randomize` for a curriculum (small volume -> whole frontal workspace).
    _JOINT_SPREAD = np.array([1.2, 0.8, 1.0, 1.2, 1.2, 0.0])

    def _sample_target(self) -> np.ndarray:
        """Sample a reachable TCP via forward kinematics from a random pose.

        We perturb the home configuration, run FK on a scratch copy of the
        state, and read the resulting TCP. Rejection-sample until the point is
        off the floor (z > 0.25). Guaranteed reachable, always in front/above.
        """
        rng = self.np_random
        saved_qpos = self.data.qpos.copy()
        target = None
        for _ in range(20):
            q = self._home_qpos.copy()
            q[self._qpos_adr] += rng.uniform(
                -self._JOINT_SPREAD, self._JOINT_SPREAD
            ) * self.randomize
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            p = self._tcp_pos()
            if p[2] > 0.25:
                target = p
                break
        # restore state so the caller's reset logic is unaffected
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        if target is None:
            target = self._tcp_pos() + np.array([0.0, 0.0, 0.1])
        return target

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self._qpos_adr]
        qvel = self.data.qvel[self._dof_adr]
        tcp_to_fruit = self._target - self._tcp_pos()
        return np.concatenate([qpos, qvel, tcp_to_fruit]).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        # small joint noise so the policy doesn't overfit one start pose
        self.data.qpos[self._qpos_adr] += self.np_random.uniform(
            -0.05, 0.05, size=6
        )
        self.data.ctrl[self._act_id] = self._home_ctrl[self._act_id]
        mujoco.mj_forward(self.model, self.data)

        self._target = self._sample_target()
        self.data.mocap_pos[self._fruit_mocap] = self._target

        self._step_count = 0
        return self._get_obs(), {"target": self._target.copy()}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        target_ctrl = self.data.ctrl[self._act_id] + action * self.action_scale
        target_ctrl = np.clip(
            target_ctrl, self._ctrl_range[:, 0], self._ctrl_range[:, 1]
        )
        self.data.ctrl[self._act_id] = target_ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        dist = float(np.linalg.norm(self._target - self._tcp_pos()))
        success = dist < self.success_radius

        reward = -dist
        reward -= self.control_cost * float(np.sum(np.square(action)))
        if success:
            reward += 10.0

        terminated = success
        truncated = self._step_count >= self.max_steps
        info = {"distance": dist, "is_success": success}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=-1)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Register so SB3 / gym.make can find it.
gym.register(
    id="FruitReach-v0",
    entry_point="envs.fruit_reach_env:FruitReachEnv",
    max_episode_steps=180,
)
