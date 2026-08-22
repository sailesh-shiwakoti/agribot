# AgriBot — Tool-Stack Learning Map

This project doubles as a hands-on curriculum across the robotics stack. Each
area below is anchored to a concrete file/subsystem you can point at and
explain — the goal being breadth *with* depth, defensible in an interview.

| Area | Tools / concepts | Where it lives |
|---|---|---|
| **Robotics middleware** | ROS 2 (rclpy, topics, QoS, TF2, launch, params), colcon, Docker (arm64) | `agribot-3d/` — Dockerfile, `agribot_bringup`, headless launch |
| **Simulation** | Gazebo Sim / SDF (systems, sensors, physics); MuJoCo / MJCF | `worlds/orchard.sdf`; `orchard/scene_builder.py` |
| **Robot description** | URDF/xacro, `ros2_control`, kinematic trees | `ur_description` (UR5e); the MJCF arm tree |
| **Kinematics** | forward kinematics, Jacobians, **damped least-squares IK**, null-space posture control, reachability | `orchard/ik.py` |
| **Motion planning** | MoveIt 2, OMPL (RRT-family), planning scene | Track A `pymoveit2` (`harvest_sequencer`) |
| **Mobile navigation** | 2D lidar, trunk detection (cluster+circle-fit), occupancy grids, **A\***, SLAM (slam_toolbox), **Nav2** | `orchard/lidar.py`, `orchard/navigator.py`, `agribot_navigation` |
| **Mobile robot description** | diff-drive URDF, wheel odometry, `cmd_vel` | `agribot_description` |
| **Classical CV** | OpenCV, HSV color segmentation, pinhole camera model, intrinsics/extrinsics, depth back-projection | `agribot_perception`, `perception/` |
| **Deep CV** | PyTorch, YOLOv8 (Ultralytics), synthetic data + auto-labeling, mAP, **ONNX export/deploy** | `perception/train_yolo.py`, `export_onnx.py`, `eval_map.py` |
| **Reinforcement learning** | Gymnasium API, Stable-Baselines3 (**SAC vs PPO**), reward shaping, domain randomization, evaluation methodology | `envs/fruit_reach_env.py`, `training/` |
| **Manipulation / grasping** | end-effector poses, grasp abstraction, ripeness-scaled force, pick-and-place | `orchard/harvester.py` |
| **3D perception** | RGB-D cameras, depth → 3D, TF frames, (point clouds — stretch) | `worlds/orchard.sdf` camera, localizer |
| **Sensors** | camera models, update rates, noise, occlusion | orchard camera mast, fruiting-wall occlusion |
| **Tooling** | Foxglove Studio, TensorBoard, `uv`, ONNX Runtime, headless rendering | throughout |

## What was learned by *doing* (not just reading)

- **Damped least-squares IK with a null-space posture bias** — implemented from
  scratch in `orchard/ik.py`. The posture term is what makes the arm reach fruit
  with a natural, human-like pose instead of contorting; the reachability check
  stops it chasing fruit it physically cannot grasp.
- **SAC ≫ PPO for continuous control at equal budget** — measured, not assumed:
  100% vs ~45% reach success. (`training/`, `media/rl_curves.png`.)
- **Ripeness → fragility → control** — a small but real closed loop: perception
  (redness) sets a physical constraint (grip-force tolerance) that the
  controller must respect, or the fruit is "crushed" (negative reward).
- **Real deployment friction** — arm64 base images, Gazebo's EOL split, the
  OSRF apt repo, QEMU-vs-native: the unglamorous integration knowledge that
  separates "ran a tutorial" from "stood up the stack."

## Suggested reading order for the code

1. `orchard/scene_builder.py` → how a world is described (MJCF).
2. `orchard/ik.py` → the math that turns "go here" into joint angles.
3. `orchard/harvester.py` → the perceive → plan → act → store loop.
4. `envs/fruit_reach_env.py` + `training/train.py` → the learned alternative.
5. `agribot-3d/` → the same robot on the production ROS 2 stack.
