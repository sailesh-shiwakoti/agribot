# AgriBot: A Simulated Autonomous Orchard-Harvesting Robot

*Technical report — draft for portfolio / graduate-application use.*

## 1. Problem

Selective robotic fruit harvesting is a live problem in agricultural
engineering: a robot must **perceive** individual fruits on a tree, decide
which are **ripe**, plan a **collision-free reach**, and pick them **without
bruising** — at speed, in clutter, across a whole orchard. This project builds
a simulated end-to-end system for that task and uses it as a vehicle to learn
the full robotics stack (perception, planning, control, learning, integration).

Scope is deliberately **simulation-only** and focused on the perception–
manipulation core. Battery/energy is out of scope by choice.

## 2. System

The robot is a Universal Robots UR5e arm on a mobile base. It is implemented in
two stacks (see `docs/ARCHITECTURE.md`):

- **ROS 2 systems stack** (Docker, arm64): ROS 2 Humble + Gazebo Sim (Fortress)
  + MoveIt 2 + `ros2_control`, using the maintained `ur_simulation_gz`
  package and visualised headlessly through Foxglove. This stands up the
  production-grade middleware and is verified end-to-end at the control level
  (arm spawns, joint states stream, trajectory commands execute).
- **Learning / simulation stack** (native macOS): MuJoCo for fast physics and
  rendering, on which the **complete harvesting behaviour** is demonstrated,
  plus reinforcement-learning and deep-perception experiments.

### 2.1 Harvesting behaviour (MuJoCo showcase)

A mobile base drives down a row of ~10 trees. At each tree the controller:

1. **Perceives** which apples are red (ripe) and within reach.
2. **Plans** a reach with damped-least-squares inverse kinematics, biased by a
   null-space posture term toward a natural arm configuration.
3. **Picks** with a grip force set by ripeness: `tolerance = 1 - redness`,
   so a riper (redder) apple is treated as more fragile and gripped more
   gently; exceeding tolerance "crushes" the fruit (penalised).
4. **Stores** the apple in an onboard bin and moves on.

Green (unripe) fruit and out-of-reach fruit are left — as a selective
harvester should.

### 2.2 Learned reaching (RL)

As a learned alternative to the IK planner, an SAC policy is trained on a
`FruitReachEnv` (Gymnasium) to reach randomly placed fruit. This supports a
direct **planner-vs-policy** comparison (see §3, `notebooks/03_rl_analysis`).

## 3. Results

| Metric | Result |
|---|---|
| Orchard harvest — ripe apples picked | _(see run report; ~all reachable ripe fruit)_ |
| Orchard harvest — unripe left on tree | 100% correctly left |
| Grip-force rule violations (crushed) | 0 |
| RL reaching — SAC success | **100%** (eval, 6 cm tolerance) |
| RL reaching — PPO success (equal budget) | ~45% |

The RL result is a clean finding: **SAC is far more sample-efficient than PPO**
on this continuous-control task, because off-policy replay reuses every
transition. Demo media: `media/orchard_harvest.mp4`, `media/reach_sac.mp4`,
`media/rl_curves.png`.

## 4. Engineering lessons

- **Inverse kinematics quality is a control problem, not just geometry.** A
  naive DLS solver produced contorted, far-reaching poses; adding a gated
  null-space posture bias (active only away from the target) yielded natural
  reaches and a reliable reachability test.
- **Model the physics you need, and no more.** Disabling unnecessary contact
  physics (the base rides an actuated slide joint; fruit is handled
  kinematically) removed a class of instabilities and sped up rendering.
- **Integration knowledge is real knowledge.** arm64-native base images,
  Gazebo Classic's end-of-life, the OSRF apt repository, QEMU-vs-native — these
  determined whether the stack ran at all.

## 5. Limitations and future work

- **Navigation is scripted**, not autonomous. Real orchard traversal needs
  SLAM/GPS + Nav2 on the ROS stack (scaffolded, not built).
- **Grasping is abstracted** (kinematic attach). Real non-destructive grasping
  — compliant fingers, force control, stem separation — is a research problem
  in itself and a natural next module.
- **Perception uses ground-truth color labels** in the MuJoCo *showcase*. The
  ROS 2 stack implements real vision — an HSV / YOLO-ONNX red-apple detector, a
  depth+TF2 3D localiser, a MoveIt planning scene, and a pymoveit2 harvest
  state machine (`agribot_perception`, `agribot_manipulation`). That pipeline
  builds and is import/launch-verified, but has not been run end-to-end in the
  CPU-only Gazebo renderer, which is impractically slow on this hardware.
- **Ripeness → fragility** is a modelled proxy, not a validated horticultural
  relationship.

Directions: learned compliant grasping (RL fine-tuned from planner demos);
instance segmentation for occluded-fruit counting / yield mapping; sim-to-real
domain randomisation.
