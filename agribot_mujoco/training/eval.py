"""
Evaluate a trained reaching policy: report success rate over N randomized
episodes and save a rollout video (for media/ and the write-up).

Examples:
    uv run python training/eval.py --model training/runs/sac_seed0/best_model
    uv run python training/eval.py --model training/runs/sac_seed0/best_model \
        --episodes 100 --video media/reach_sac.mp4
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC, PPO

from envs.fruit_reach_env import FruitReachEnv


def load_model(path: str):
    # Try both; SB3 .zip stores the class but this keeps the CLI simple.
    for cls in (SAC, PPO):
        try:
            return cls.load(path, device="cpu")
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"could not load model from {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--randomize", type=float, default=1.0)
    p.add_argument("--video", default=None, help="path to save an mp4/gif")
    p.add_argument("--video-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    model = load_model(args.model)

    # --- quantitative eval (no rendering) ---
    env = FruitReachEnv(randomize=args.randomize)
    successes, dists, steps_to_success = 0, [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        n = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            n += 1
            done = term or trunc
        dists.append(info["distance"])
        if info["is_success"]:
            successes += 1
            steps_to_success.append(n)
    env.close()

    rate = successes / args.episodes
    print(f"Success rate: {rate:.1%}  ({successes}/{args.episodes})")
    print(f"Final TCP-fruit distance: mean={np.mean(dists):.3f} m, "
          f"median={np.median(dists):.3f} m")
    if steps_to_success:
        print(f"Steps to success: mean={np.mean(steps_to_success):.1f}")

    # --- qualitative rollout video ---
    if args.video:
        import imageio.v2 as imageio
        renv = FruitReachEnv(render_mode="rgb_array", randomize=args.randomize)
        frames = []
        for ep in range(args.video_episodes):
            obs, _ = renv.reset(seed=args.seed + 1000 + ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = renv.step(action)
                frames.append(renv.render())
                done = term or trunc
        renv.close()
        os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
        fps = 30
        if args.video.endswith(".gif"):
            imageio.mimsave(args.video, frames, fps=fps)
        else:
            imageio.mimsave(args.video, frames, fps=fps, macro_block_size=None)
        print(f"Saved {len(frames)} frames -> {args.video}")


if __name__ == "__main__":
    main()
