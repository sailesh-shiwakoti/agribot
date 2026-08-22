"""
Procedurally generate a 2D orchard *field* (not a tidy row): trees of random
heights at random, drivably-spaced positions, apples around each trunk, and a
mobile base with a PLANAR joint set (x, y, yaw) carrying the UR5e + bin.

Unlike scene_builder (single row + slide joint), here the robot must find and
drive between scattered trees, so the base has full planar mobility and the
demo adds lidar tree-detection + path planning on top.

`build_field(cfg)` -> (xml, trees, apples)
    trees  = list of Tree(name, x, y, radius, height)
    apples = list of Apple(name, pos, redness, tree)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FieldConfig, ASSETS_DIR
from .scene_builder import _ARM_BODY, _MESHES, Apple


@dataclass
class Tree:
    name: str
    x: float
    y: float
    radius: float
    height: float


def _sample_trees(cfg, rng):
    trees = []
    tries = 0
    while len(trees) < cfg.n_trees and tries < 2000:
        tries += 1
        x = rng.uniform(*cfg.field_x)
        y = rng.uniform(*cfg.field_y)
        if np.hypot(x, y) < cfg.start_clear:
            continue
        if any(np.hypot(x - t.x, y - t.y) < cfg.min_tree_sep for t in trees):
            continue
        trees.append(Tree(
            name=f"tree_{len(trees)}",
            x=x, y=y,
            radius=rng.uniform(*cfg.trunk_radius),
            height=rng.uniform(*cfg.trunk_height)))
    return trees


def _apple_rgba(redness):
    if redness is None:
        return "0.30 0.60 0.15 1"
    r = 0.55 + 0.35 * redness
    g = 0.32 * (1.0 - redness) + 0.04
    return f"{r:.3f} {g:.3f} 0.05 1"


def build_field(cfg: FieldConfig):
    rng = np.random.default_rng(cfg.seed)
    trees = _sample_trees(cfg, rng)
    apples: list[Apple] = []
    tree_xml, apple_xml, weld_xml = [], [], []
    idx = 0

    for ti, t in enumerate(trees):
        h = t.height
        canopy_r = 0.28 + 0.18 * (h - 0.9)
        branches = ""
        for _ in range(rng.integers(2, 4)):
            bz = rng.uniform(0.6 * h, 0.95 * h)
            branches += (
                f'<geom type="cylinder" size="0.02 0.24" '
                f'pos="0 0 {bz:.2f}" euler="1.2 {rng.uniform(0,6.28):.2f} 0" '
                f'material="bark"/>')
        tree_xml.append(
            f'<body name="{t.name}" pos="{t.x:.3f} {t.y:.3f} 0">'
            f'<geom type="cylinder" size="{t.radius:.3f} {h/2:.3f}" '
            f'pos="0 0 {h/2:.3f}" material="bark"/>{branches}'
            f'<geom type="ellipsoid" size="{canopy_r:.2f} {canopy_r:.2f} '
            f'{canopy_r*1.1:.2f}" pos="0 0 {h+canopy_r*0.4:.2f}" '
            f'material="foliage"/></body>')

        n_ap = int(rng.integers(cfg.apples_per_tree[0], cfg.apples_per_tree[1] + 1))
        z_hi = min(0.95, h - 0.08)              # keep fruit in the reachable band
        for _ in range(n_ap):
            ang = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(*cfg.fruit_ring)
            ax = t.x + rr * np.cos(ang)
            ay = t.y + rr * np.sin(ang)
            az = rng.uniform(0.5, max(0.55, z_hi))
            redness = None if rng.random() < cfg.green_fraction \
                else float(rng.uniform(0.55, 0.92))
            tag = "g" if redness is None else f"r{int(round(redness*100)):02d}"
            name = f"apple_{tag}_{idx:03d}"
            idx += 1
            apples.append(Apple(name, np.array([ax, ay, az]), redness))
            apple_xml.append(
                f'<body name="{name}" pos="{ax:.3f} {ay:.3f} {az:.3f}"><freejoint/>'
                f'<geom type="sphere" size="{cfg.fruit_radius}" mass="0.15" '
                f'rgba="{_apple_rgba(redness)}"/></body>')
            weld_xml.append(
                f'<weld name="weld_tree_{name}" body1="{t.name}" '
                f'body2="{name}" active="true"/>')

    mesh_decl = "".join(f'<mesh file="{m}.obj"/>' for m in _MESHES)
    lim = 8.0

    xml = f"""<mujoco model="orchard field harvester">
  <compiler angle="radian" meshdir="{ASSETS_DIR}" autolimits="true"/>
  <option integrator="implicitfast" timestep="0.002">
    <flag contact="disable"/>
  </option>

  <visual>
    <global offwidth="{cfg.width}" offheight="{cfg.height}" azimuth="-90" elevation="-35"/>
    <quality shadowsize="4096"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <map haze="0.12"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.2 0.35 0.55" width="512" height="3072"/>
    <texture type="2d" name="grass" builtin="checker" mark="edge" rgb1="0.24 0.38 0.20" rgb2="0.20 0.32 0.16"
      markrgb="0.33 0.43 0.28" width="300" height="300"/>
    <material name="grass" texture="grass" texuniform="true" texrepeat="16 16"/>
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
    <light name="sun" pos="3 0 5" dir="-0.1 0.1 -1" directional="true" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grass"/>

    <body name="mobile_base" pos="0 0 0.12">
      <joint name="base_x"   type="slide" axis="1 0 0" range="-{lim} {lim}" damping="40"/>
      <joint name="base_y"   type="slide" axis="0 1 0" range="-{lim} {lim}" damping="40"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" range="-100 100" damping="20"/>
      <inertial pos="0 0 0" mass="45" diaginertia="4 4 4"/>
      <!-- base geoms are group 2 (rendered, but excluded from the lidar
           raycast, which scans only group-0 geoms = the trees). -->
      <geom name="chassis" type="box" size="0.28 0.34 0.10" material="chassis" group="2"/>
      <geom name="wheel_fl" type="cylinder" size="0.09 0.03" pos="0.2 0.30 -0.06" euler="1.5708 0 0" material="wheel" group="2"/>
      <geom name="wheel_fr" type="cylinder" size="0.09 0.03" pos="0.2 -0.30 -0.06" euler="1.5708 0 0" material="wheel" group="2"/>
      <geom name="wheel_bl" type="cylinder" size="0.09 0.03" pos="-0.2 0.30 -0.06" euler="1.5708 0 0" material="wheel" group="2"/>
      <geom name="wheel_br" type="cylinder" size="0.09 0.03" pos="-0.2 -0.30 -0.06" euler="1.5708 0 0" material="wheel" group="2"/>
      <!-- direction marker so yaw is visible -->
      <geom name="nose" type="box" size="0.06 0.03 0.02" pos="0.32 0 0.02" material="urblue" group="2"/>
      <geom name="bin_floor" type="box" size="0.16 0.15 0.01" pos="0 -0.30 0.12" material="bin" group="2"/>
      <geom name="bin_w1" type="box" size="0.16 0.01 0.05" pos="0 -0.45 0.17" material="bin" group="2"/>
      <geom name="bin_w2" type="box" size="0.16 0.01 0.05" pos="0 -0.15 0.17" material="bin" group="2"/>
      <geom name="bin_w3" type="box" size="0.01 0.15 0.05" pos="0.16 -0.30 0.17" material="bin" group="2"/>
      <geom name="bin_w4" type="box" size="0.01 0.15 0.05" pos="-0.16 -0.30 0.17" material="bin" group="2"/>
      <!-- lidar scan origin -->
      <site name="lidar" pos="0 0 {cfg.lidar_z - 0.12:.3f}"/>
      {_ARM_BODY}
    </body>

    {"".join(tree_xml)}
    {"".join(apple_xml)}
  </worldbody>

  <equality>
    {"".join(weld_xml)}
  </equality>

  <actuator>
    <position name="base_x"   joint="base_x"   kp="8000" kv="1200" forcerange="-6000 6000"/>
    <position name="base_y"   joint="base_y"   kp="8000" kv="1200" forcerange="-6000 6000"/>
    <position name="base_yaw" joint="base_yaw" kp="2000" kv="300"  forcerange="-1500 1500"/>
    <position name="shoulder_pan" joint="shoulder_pan_joint" kp="2000" kv="200" forcerange="-150 150"/>
    <position name="shoulder_lift" joint="shoulder_lift_joint" kp="2000" kv="200" forcerange="-150 150"/>
    <position name="elbow" joint="elbow_joint" kp="2000" kv="200" forcerange="-150 150"/>
    <position name="wrist_1" joint="wrist_1_joint" kp="500" kv="50" forcerange="-28 28"/>
    <position name="wrist_2" joint="wrist_2_joint" kp="500" kv="50" forcerange="-28 28"/>
    <position name="wrist_3" joint="wrist_3_joint" kp="500" kv="50" forcerange="-28 28"/>
  </actuator>
</mujoco>"""
    return xml, trees, apples
