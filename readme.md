# 🍎 AgriBot — Autonomous Orchard-Harvesting Robot (Simulation)

A simulated robot that drives down an orchard row, **sees which apples are
ripe, reaches out and picks only the red ones — gently, in proportion to how
ripe they are — and drops them in an onboard bin**, leaving unripe fruit on the
tree.

Built as a portfolio + learning project spanning the full robotics stack:
perception, motion planning, control, reinforcement learning, and ROS 2
systems integration.

> **Demos:**
> - [`media/orchard_harvest.mp4`](media/orchard_harvest.mp4) — a UR5e on a mobile
>   base harvesting a 10-tree orchard row (40 apples).
> - [`media/field_harvest.mp4`](media/field_harvest.mp4) — the robot roaming an
>   open **field of scattered trees**: it *detects trunks with a lidar*, plans
>   paths around them (A*), drives tree-to-tree, and harvests.

## Results

| Metric | Result |
|---|---|
| Ripe apples picked & binned | **24 / 24** |
| Unripe (green) apples correctly left on tree | **16 / 16** |
| Fruit crushed (grip-force rule violated) | **0** |
| Field demo — trees found by lidar & visited | **9 / 9** |
| Field demo — ripe apples harvested | **13 / 27** (rest are far-side / out of reach) |
| RL reaching policy — SAC success rate | **100 %** |
| RL reaching policy — PPO (equal budget) | ~45 % |

The grip-force rule: `force_tolerance = 1 − redness` — a riper (redder) apple is
more fragile and gripped more gently; over-gripping "crushes" it (penalised).

## What it does (the autonomous loop)

```
drive to tree → perceive ripe & reachable apples → IK-plan the reach
   → grasp with ripeness-scaled force → store in bin → repeat → harvest report
```

No teleoperation: the robot chooses which apples to pick from what it perceives.

## Repository layout

```
proj1/
├── readme.md                 # you are here
├── docs/                     # ARCHITECTURE, LEARNING map, technical REPORT
├── media/                    # demo videos + training curves
├── agribot-3d/               # Track A — ROS 2 systems stack (Docker, arm64)
│   └── ros2_ws/src/          #   ROS 2 Humble + Gazebo Sim + MoveIt 2 + Nav2
│       ├── agribot_perception/   #   red-apple detect (HSV/YOLO) + 3D localize
│       ├── agribot_manipulation/ #   MoveIt planning scene + harvest state machine
│       ├── agribot_description/   #   diff-drive base + lidar URDF (nav robot)
│       └── agribot_navigation/    #   SLAM + Nav2 + tree detection + coordinator
└── agribot_mujoco/            # Track B — native macOS learning/sim stack
    ├── orchard/              #   ⭐ harvest demos: row + lidar/A* field nav
    ├── envs/ + training/     #   RL: FruitReachEnv + SAC/PPO
    ├── perception/           #   YOLO fine-tune → ONNX (for ROS deployment)
    └── notebooks/            #   kinematics, RL analysis
```

Two stacks, one robot (UR5e) and one task. **Track B** (MuJoCo, native
Apple-Silicon) runs the comprehensive harvesting demo and the learning
experiments. **Track A** (ROS 2 + Gazebo Sim in Docker) is the production
middleware, verified at the control level. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

### The harvest demos (Track B)
```bash
cd agribot_mujoco
uv sync
uv run python -m orchard.run_demo                 # orchard row -> media/orchard_harvest.mp4
uv run python -m orchard.run_field_demo --seed 2  # lidar + A* field -> media/field_harvest.mp4
```

### Reinforcement learning
```bash
cd agribot_mujoco
uv run python training/train.py --algo sac --timesteps 300000
uv run python training/eval.py --model training/runs/sac_seed0/best_model --video media/reach.mp4
```

### ROS 2 stack (Track A) — full autonomous harvest pipeline
```bash
cd agribot-3d
docker compose build
docker compose run --service-ports agribot   # then, inside:
#   colcon build --symlink-install && source install/setup.bash
#   ros2 launch agribot_bringup harvest_demo.launch.py         # sim+MoveIt+perception+harvest
#   ros2 launch agribot_bringup harvest_demo.launch.py detector:=yolo model_path:=/path/fruit.onnx
# connect Foxglove Studio (on the Mac) to ws://localhost:8765
```
Pipeline: RGB-D camera → red-apple detector (HSV or YOLO-ONNX) → 3D localizer
(depth+TF2) → MoveIt planning scene → harvest state machine (pymoveit2,
ripeness-scaled grip). Builds + import/launch-verified; not run in the slow
CPU-only Gazebo renderer (see `docs/REPORT.md`).

**Mobile field navigation (SLAM + Nav2):**
```bash
ros2 launch agribot_navigation field_harvest.launch.py
```
A diff-drive base + 2D lidar (`agribot_description`), **slam_toolbox** mapping,
**Nav2** planning/control, a `tree_detector` (LaserScan → trunk poses) and a
`field_coordinator` that drives the robot tree-to-tree via Nav2 and triggers the
arm. Packages build; the robot xacro validates (`check_urdf`); nodes compile.
Full run needs `nav2`/`slam_toolbox` in the image (added to the Dockerfile —
`docker compose build` to pull them).

## Highlights for the curious

- **Hand-rolled inverse kinematics** (`orchard/ik.py`): damped least squares
  with a null-space posture bias — the arm reaches fruit with natural poses and
  never chases fruit it can't grasp.
- **Ripeness → fragility → control**: perception sets a physical grip-force
  constraint the controller must respect.
- **SAC ≫ PPO measured, not assumed** (`notebooks/03_rl_analysis.ipynb`).
- **Real integration knowledge**: arm64-native ROS image, Gazebo Sim (not the
  EOL Classic), OSRF apt repo — the stuff that decides whether the stack runs.

See [docs/LEARNING.md](docs/LEARNING.md) for the full tool-stack map and
[docs/REPORT.md](docs/REPORT.md) for the technical write-up.

*Simulation only. Scope is the perception–manipulation core; autonomous
navigation (SLAM/Nav2) and non-destructive grasping are documented as future
work.*
