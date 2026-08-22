"""Tunable parameters for the orchard-harvesting simulation.

One place to change the orchard size, fruit density, robot reach, and motion
speed. Everything downstream (scene_builder, harvester) reads from here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Absolute path to the vendored UR5e mesh assets (so the generated MJCF can set
# meshdir regardless of where it's compiled from a string).
ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "envs", "menagerie", "universal_robots_ur5e", "assets",
)


@dataclass
class OrchardConfig:
    # ---- orchard layout ----
    n_trees: int = 10
    tree_spacing: float = 0.85      # metres between trunks along the row (+x)
    row_y: float = 0.60             # trunk y (the fruiting wall is at +y)
    apples_per_tree: tuple = (3, 5)  # inclusive random range
    green_fraction: float = 0.35    # share of apples that are unripe (green)

    # fruit is placed in this box, relative to each trunk, on the base's side
    fruit_dx: float = 0.20          # +/- x spread around the trunk
    fruit_y: tuple = (0.18, 0.30)   # y in front of the trunk (toward the robot)
    fruit_z: tuple = (0.45, 1.00)   # height range
    fruit_radius: float = 0.035

    # ---- robot ----
    reach_limit: float = 0.85       # max shoulder->fruit distance the arm attempts
    bin_capacity_slots: int = 24

    # ---- motion / control (sim steps; timestep = 0.002 s) ----
    reach_steps: int = 36
    approach_steps: int = 26
    lift_steps: int = 22
    to_bin_steps: int = 46
    drive_steps_per_m: int = 130    # base speed (bigger = slower/smoother)
    settle_steps: int = 10

    # ---- rendering ----
    width: int = 1280
    height: int = 720
    render_every: int = 4           # capture 1 of N sim steps
    fps: int = 40

    seed: int = 7


DEFAULT = OrchardConfig()


@dataclass
class FieldConfig:
    """Parameters for the 2D-field navigation + harvest demo."""
    # ---- field ----
    n_trees: int = 9
    field_x: tuple = (1.2, 6.0)       # tree-placement bounds (m)
    field_y: tuple = (-2.4, 2.4)
    min_tree_sep: float = 1.5         # keep trees drivable-apart
    start_clear: float = 1.1          # keep start (0,0) clear of trees
    trunk_height: tuple = (0.9, 1.7)  # random per tree (various heights)
    trunk_radius: tuple = (0.06, 0.10)
    apples_per_tree: tuple = (3, 6)
    green_fraction: float = 0.35
    fruit_ring: tuple = (0.13, 0.20)  # radial offset of fruit from trunk axis
    fruit_z_frac: tuple = (0.45, 0.85)  # (unused; fruit height is absolute now)
    fruit_radius: float = 0.035

    # ---- robot / navigation ----
    reach_limit: float = 0.88
    park_dist: float = 0.55           # base parks this far from the trunk
    robot_radius: float = 0.40        # for obstacle inflation
    n_lidar: int = 240                # rays in the 360-deg scan
    lidar_range: float = 6.0
    lidar_z: float = 0.30             # scan height (hits trunks, not canopy)

    # ---- planner / grid ----
    grid_res: float = 0.20            # occupancy-grid cell size (m)

    # ---- motion (sim steps) ----
    nav_speed: float = 0.9            # m/s of base travel
    yaw_speed: float = 1.2            # rad/s
    reach_steps: int = 34
    approach_steps: int = 24
    lift_steps: int = 20
    to_bin_steps: int = 42
    settle_steps: int = 8

    # ---- rendering ----
    width: int = 1280
    height: int = 720
    render_every: int = 7
    fps: int = 40
    seed: int = 3


FIELD = FieldConfig()
