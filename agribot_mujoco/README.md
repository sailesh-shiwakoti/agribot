# AgriBot — Learning Track (Track B)

Native macOS / Apple-Silicon stack for the learning-heavy parts of AgriBot:
MuJoCo reinforcement learning and YOLO perception training. Runs outside Docker
so it can use Metal (MPS) GPU acceleration and MuJoCo's fast native renderer.

## Setup

```bash
cd agribot_mujoco
uv sync                       # creates .venv with Python 3.11 + all deps
uv run python scripts/smoke_test.py
```

## Layout

| Path | Purpose |
|---|---|
| `envs/` | MuJoCo MJCF scenes + Gymnasium `FruitReachEnv` |
| `training/` | SAC / PPO training scripts + configs + TensorBoard logs |
| `perception/` | YOLO fine-tuning, ONNX export, mAP evaluation |
| `datasets/` | Auto-labeled synthetic images (gitignored; generators kept) |
| `notebooks/` | Kinematics, camera models, RL analysis |
| `scripts/` | Setup / smoke-test utilities |

See the top-level project plan and `docs/LEARNING.md` for how this fits the
whole AgriBot system.
