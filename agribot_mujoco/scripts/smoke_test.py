"""
Phase 0 smoke test for the learning track. Verifies the native macOS stack:
  1. MuJoCo loads the orchard scene (mesh paths resolve).
  2. PyTorch sees the Metal (MPS) GPU.
  3. Offscreen rendering works headless -> saves a PNG.
  4. FruitReachEnv steps and returns sane observations/rewards.

Run:  uv run python scripts/smoke_test.py
"""

import os
import sys

import numpy as np

# Make the project root importable (envs/ package).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ok = True

    # 1. MuJoCo + model load ------------------------------------------------
    import mujoco
    from envs.fruit_reach_env import DEFAULT_MODEL_PATH

    model = mujoco.MjModel.from_xml_path(DEFAULT_MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"[ok] MuJoCo {mujoco.__version__}: loaded orchard scene "
          f"({model.nq} dof, {model.nbody} bodies)")

    # 2. PyTorch / MPS ------------------------------------------------------
    import torch
    mps = torch.backends.mps.is_available()
    print(f"[{'ok' if mps else '!!'}] PyTorch {torch.__version__}: "
          f"MPS available = {mps}")
    if not mps:
        print("     (RL will fall back to CPU — fine but slower)")

    # 3. Offscreen render ---------------------------------------------------
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera=-1)
        frame = renderer.render()
        out = os.path.join(os.path.dirname(__file__), "smoke_render.png")
        import imageio.v2 as imageio
        imageio.imwrite(out, frame)
        renderer.close()
        print(f"[ok] Offscreen render {frame.shape} -> {out}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[!!] Offscreen render FAILED: {e}")

    # 4. Env rollout --------------------------------------------------------
    from envs.fruit_reach_env import FruitReachEnv
    env = FruitReachEnv(seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, "obs shape mismatch"
    total_r = 0.0
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total_r += r
        if term or trunc:
            break
    env.close()
    print(f"[ok] FruitReachEnv: obs={obs.shape}, 20-step return={total_r:.2f}, "
          f"last dist={info['distance']:.3f} m")

    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
