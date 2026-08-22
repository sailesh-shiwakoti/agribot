"""
Procedurally generate the orchard MuJoCo scene (MJCF as a string).

Produces a row of N trees (trunk + branches + foliage) with many apples --
RED (ripe, with per-apple redness) and GREEN (unripe) -- a mobile base on a
slide joint carrying the UR5e, an onboard collection bin, and per-apple welds
so fruit hangs on the branch until picked.

`build_scene(cfg)` returns (xml_string, apples) where `apples` is the list of
Apple(name, pos, redness) with redness=None for green ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OrchardConfig, ASSETS_DIR

# UR5e kinematic tree (reproduced from Menagerie ur5e.xml, explicit params,
# visual-only geoms). Mounted on the mobile base; ends at a "tcp" site with a
# simple two-finger gripper visual.
_ARM_BODY = """
      <body name="base" pos="0 0 0.11" quat="0 0 0 -1">
        <inertial mass="4.0" pos="0 0 0" diaginertia="0.00443333156 0.00443333156 0.0072"/>
        <geom mesh="base_0" material="black" class="vis"/>
        <geom mesh="base_1" material="jointgray" class="vis"/>
        <body name="shoulder_link" pos="0 0 0.163">
          <inertial mass="3.7" pos="0 0 0" diaginertia="0.0102675 0.0102675 0.00666"/>
          <joint name="shoulder_pan_joint" axis="0 0 1" range="-6.28319 6.28319" armature="0.1" damping="5"/>
          <geom mesh="shoulder_0" material="urblue" class="vis"/>
          <geom mesh="shoulder_1" material="black" class="vis"/>
          <geom mesh="shoulder_2" material="jointgray" class="vis"/>
          <body name="upper_arm_link" pos="0 0.138 0" quat="1 0 1 0">
            <inertial mass="8.393" pos="0 0 0.2125" diaginertia="0.133886 0.133886 0.0151074"/>
            <joint name="shoulder_lift_joint" axis="0 1 0" range="-6.28319 6.28319" armature="0.1" damping="5"/>
            <geom mesh="upperarm_0" material="linkgray" class="vis"/>
            <geom mesh="upperarm_1" material="black" class="vis"/>
            <geom mesh="upperarm_2" material="jointgray" class="vis"/>
            <geom mesh="upperarm_3" material="urblue" class="vis"/>
            <body name="forearm_link" pos="0 -0.131 0.425">
              <inertial mass="2.275" pos="0 0 0.196" diaginertia="0.0311796 0.0311796 0.004095"/>
              <joint name="elbow_joint" axis="0 1 0" range="-3.1415 3.1415" armature="0.1" damping="5"/>
              <geom mesh="forearm_0" material="urblue" class="vis"/>
              <geom mesh="forearm_1" material="linkgray" class="vis"/>
              <geom mesh="forearm_2" material="black" class="vis"/>
              <geom mesh="forearm_3" material="jointgray" class="vis"/>
              <body name="wrist_1_link" pos="0 0 0.392" quat="1 0 1 0">
                <inertial mass="1.219" pos="0 0.127 0" diaginertia="0.0025599 0.0025599 0.0021942"/>
                <joint name="wrist_1_joint" axis="0 1 0" range="-6.28319 6.28319" armature="0.1" damping="2"/>
                <geom mesh="wrist1_0" material="black" class="vis"/>
                <geom mesh="wrist1_1" material="urblue" class="vis"/>
                <geom mesh="wrist1_2" material="jointgray" class="vis"/>
                <body name="wrist_2_link" pos="0 0.127 0">
                  <inertial mass="1.219" pos="0 0 0.1" diaginertia="0.0025599 0.0025599 0.0021942"/>
                  <joint name="wrist_2_joint" axis="0 0 1" range="-6.28319 6.28319" armature="0.1" damping="2"/>
                  <geom mesh="wrist2_0" material="black" class="vis"/>
                  <geom mesh="wrist2_1" material="urblue" class="vis"/>
                  <geom mesh="wrist2_2" material="jointgray" class="vis"/>
                  <body name="wrist_3_link" pos="0 0 0.1">
                    <inertial mass="0.1889" pos="0 0.0771683 0" quat="1 0 0 1"
                      diaginertia="0.000132134 9.90863e-05 9.90863e-05"/>
                    <joint name="wrist_3_joint" axis="0 1 0" range="-6.28319 6.28319" armature="0.1" damping="2"/>
                    <geom mesh="wrist3" material="linkgray" class="vis"/>
                    <geom name="finger_l" type="box" size="0.008 0.012 0.03" pos="0.025 0.11 0" material="black"/>
                    <geom name="finger_r" type="box" size="0.008 0.012 0.03" pos="-0.025 0.11 0" material="black"/>
                    <geom name="palm" type="box" size="0.03 0.012 0.01" pos="0 0.085 0" material="jointgray"/>
                    <site name="tcp" pos="0 0.14 0" quat="-1 1 0 0" size="0.005"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
"""

_MESHES = [
    "base_0", "base_1", "shoulder_0", "shoulder_1", "shoulder_2",
    "upperarm_0", "upperarm_1", "upperarm_2", "upperarm_3",
    "forearm_0", "forearm_1", "forearm_2", "forearm_3",
    "wrist1_0", "wrist1_1", "wrist1_2", "wrist2_0", "wrist2_1", "wrist2_2",
    "wrist3",
]


@dataclass
class Apple:
    name: str
    pos: np.ndarray
    redness: float | None       # None -> green/unripe

    @property
    def is_ripe(self) -> bool:
        return self.redness is not None


def _apple_rgba(redness: float | None) -> str:
    if redness is None:
        return "0.30 0.60 0.15 1"                       # green
    r = 0.55 + 0.35 * redness                           # redder as it ripens
    g = 0.32 * (1.0 - redness) + 0.04
    return f"{r:.3f} {g:.3f} 0.05 1"


def build_scene(cfg: OrchardConfig):
    rng = np.random.default_rng(cfg.seed)
    apples: list[Apple] = []
    tree_xml, apple_xml, weld_xml = [], [], []

    x_end = (cfg.n_trees - 1) * cfg.tree_spacing
    idx = 0
    for i in range(cfg.n_trees):
        tx = i * cfg.tree_spacing
        # trunk + a couple of angled branches + a foliage blob
        branches = ""
        for b in range(rng.integers(2, 4)):
            bz = rng.uniform(0.7, 1.2)
            ang = rng.uniform(-0.9, 0.9)
            branches += (
                f'<geom type="cylinder" size="0.022 0.28" '
                f'pos="{rng.uniform(-0.12,0.12):.2f} 0.02 {bz:.2f}" '
                f'euler="1.2 {ang:.2f} 0" material="bark"/>')
        tree_xml.append(
            f'<body name="tree_{i}" pos="{tx:.3f} {cfg.row_y:.3f} 0">'
            f'<geom type="cylinder" size="0.05 0.7" pos="0 0 0.7" material="bark"/>'
            f'{branches}'
            f'<geom type="ellipsoid" size="0.34 0.28 0.42" pos="0 0.06 1.35" material="foliage"/>'
            f'</body>')

        # apples on this tree
        n_ap = int(rng.integers(cfg.apples_per_tree[0], cfg.apples_per_tree[1] + 1))
        for _ in range(n_ap):
            ax = tx + rng.uniform(-cfg.fruit_dx, cfg.fruit_dx)
            ay = rng.uniform(*cfg.fruit_y)
            az = rng.uniform(*cfg.fruit_z)
            if rng.random() < cfg.green_fraction:
                redness = None
            else:
                redness = float(rng.uniform(0.55, 0.92))
            tag = "g" if redness is None else f"r{int(round(redness*100)):02d}"
            name = f"apple_{tag}_{idx:03d}"
            idx += 1
            apples.append(Apple(name, np.array([ax, ay, az]), redness))
            apple_xml.append(
                f'<body name="{name}" pos="{ax:.3f} {ay:.3f} {az:.3f}"><freejoint/>'
                f'<geom type="sphere" size="{cfg.fruit_radius}" mass="0.15" '
                f'rgba="{_apple_rgba(redness)}"/></body>')
            weld_xml.append(
                f'<weld name="weld_tree_{name}" body1="tree_{i}" '
                f'body2="{name}" active="true"/>')

    base_lo, base_hi = -0.3, x_end + 0.3
    mesh_decl = "".join(f'<mesh file="{m}.obj"/>' for m in _MESHES)

    xml = f"""<mujoco model="orchard harvester">
  <compiler angle="radian" meshdir="{ASSETS_DIR}" autolimits="true"/>
  <!-- Contacts disabled: the base rides an actuated slide joint and every
       apple is handled kinematically (welded on the tree, carried, or pinned
       in the bin), so no collision physics is needed. This also stops the
       wheels/chassis from jamming the slide joint against the floor. -->
  <option integrator="implicitfast" timestep="0.002">
    <flag contact="disable"/>
  </option>

  <visual>
    <global offwidth="{cfg.width}" offheight="{cfg.height}" azimuth="-58" elevation="-16"/>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <map haze="0.1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.2 0.35 0.55" width="512" height="3072"/>
    <texture type="2d" name="grass" builtin="checker" mark="edge" rgb1="0.24 0.38 0.20" rgb2="0.20 0.32 0.16"
      markrgb="0.35 0.45 0.30" width="300" height="300"/>
    <material name="grass" texture="grass" texuniform="true" texrepeat="10 10"/>
    <material name="black" rgba="0.033 0.033 0.033 1" specular="0.5" shininess="0.25"/>
    <material name="jointgray" rgba="0.278 0.278 0.278 1" specular="0.5" shininess="0.25"/>
    <material name="linkgray" rgba="0.82 0.82 0.82 1" specular="0.5" shininess="0.25"/>
    <material name="urblue" rgba="0.49 0.678 0.8 1" specular="0.5" shininess="0.25"/>
    <material name="chassis" rgba="0.15 0.15 0.18 1" specular="0.4" shininess="0.3"/>
    <material name="wheel" rgba="0.05 0.05 0.05 1"/>
    <material name="bin" rgba="0.45 0.32 0.15 1"/>
    <material name="bark" rgba="0.36 0.22 0.09 1"/>
    <material name="foliage" rgba="0.14 0.40 0.15 0.5"/>
    <material name="crushed" rgba="0.35 0.12 0.10 1"/>
    {mesh_decl}
  </asset>

  <default>
    <default class="vis"><geom type="mesh" contype="0" conaffinity="0" group="2"/></default>
  </default>

  <worldbody>
    <light name="sun" pos="{x_end/2:.2f} -1 3" dir="-0.1 0.3 -1" directional="true" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grass"/>

    <body name="mobile_base" pos="0 0 0.12">
      <joint name="base_x" type="slide" axis="1 0 0" range="{base_lo:.2f} {base_hi:.2f}" damping="60"/>
      <inertial pos="0 0 0" mass="45" diaginertia="4 4 4"/>
      <geom name="chassis" type="box" size="0.28 0.34 0.10" material="chassis"/>
      <geom name="wheel_fl" type="cylinder" size="0.09 0.03" pos="0.2 0.30 -0.06" euler="1.5708 0 0" material="wheel"/>
      <geom name="wheel_fr" type="cylinder" size="0.09 0.03" pos="0.2 -0.30 -0.06" euler="1.5708 0 0" material="wheel"/>
      <geom name="wheel_bl" type="cylinder" size="0.09 0.03" pos="-0.2 0.30 -0.06" euler="1.5708 0 0" material="wheel"/>
      <geom name="wheel_br" type="cylinder" size="0.09 0.03" pos="-0.2 -0.30 -0.06" euler="1.5708 0 0" material="wheel"/>
      <geom name="bin_floor" type="box" size="0.16 0.15 0.01" pos="0 -0.30 0.12" material="bin"/>
      <geom name="bin_w1" type="box" size="0.16 0.01 0.05" pos="0 -0.45 0.17" material="bin"/>
      <geom name="bin_w2" type="box" size="0.16 0.01 0.05" pos="0 -0.15 0.17" material="bin"/>
      <geom name="bin_w3" type="box" size="0.01 0.15 0.05" pos="0.16 -0.30 0.17" material="bin"/>
      <geom name="bin_w4" type="box" size="0.01 0.15 0.05" pos="-0.16 -0.30 0.17" material="bin"/>
      {_ARM_BODY}
    </body>

    {"".join(tree_xml)}
    {"".join(apple_xml)}
  </worldbody>

  <equality>
    {"".join(weld_xml)}
  </equality>

  <actuator>
    <position name="base_x" joint="base_x" kp="6000" kv="600" forcerange="-3000 3000" ctrlrange="{base_lo:.2f} {base_hi:.2f}"/>
    <position name="shoulder_pan" joint="shoulder_pan_joint" kp="2000" kv="200" forcerange="-150 150"/>
    <position name="shoulder_lift" joint="shoulder_lift_joint" kp="2000" kv="200" forcerange="-150 150"/>
    <position name="elbow" joint="elbow_joint" kp="2000" kv="200" forcerange="-150 150"/>
    <position name="wrist_1" joint="wrist_1_joint" kp="500" kv="50" forcerange="-28 28"/>
    <position name="wrist_2" joint="wrist_2_joint" kp="500" kv="50" forcerange="-28 28"/>
    <position name="wrist_3" joint="wrist_3_joint" kp="500" kv="50" forcerange="-28 28"/>
  </actuator>
</mujoco>"""
    return xml, apples
