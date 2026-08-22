"""
Train a reaching policy on FruitReachEnv with Stable-Baselines3.

Supports SAC (default; strong for continuous control) and PPO (for a sample-
efficiency comparison — a good talking point in the write-up). Logs to
TensorBoard, checkpoints periodically, and evaluates on held-out seeds.

Examples:
    uv run python training/train.py --algo sac --timesteps 200000
    uv run python training/train.py --algo ppo --timesteps 400000 --n-envs 8

Note on device: the policies here are small MLPs. On Apple Silicon, tiny MLPs
run *faster* on CPU than MPS (kernel-launch overhead dominates), so we default
to CPU for RL. MPS is where it counts for YOLO (Phase 3), not here.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

from envs.fruit_reach_env import FruitReachEnv

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def make_env_fn(**kwargs):
    def _init():
        return FruitReachEnv(**kwargs)
    return _init


def build_model(algo: str, env, tb_log: str, seed: int):
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        device="cpu",
        tensorboard_log=tb_log,
    )
    if algo == "sac":
        return SAC(
            **common,
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            gamma=0.98,
            tau=0.02,
            train_freq=1,
            gradient_steps=1,
            learning_starts=1_000,
        )
    if algo == "ppo":
        return PPO(
            **common,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.98,
            gae_lambda=0.95,
            ent_coef=0.0,
        )
    raise ValueError(f"unknown algo: {algo}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--randomize", type=float, default=1.0,
                   help="0..1 target-sampling volume (curriculum knob)")
    p.add_argument("--tag", default=None, help="run name suffix")
    args = p.parse_args()

    run_name = f"{args.algo}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    env_kwargs = dict(randomize=args.randomize)
    train_env = VecMonitor(
        make_vec_env(
            make_env_fn(**env_kwargs),
            n_envs=args.n_envs,
            seed=args.seed,
        )
    )
    eval_env = VecMonitor(
        make_vec_env(make_env_fn(**env_kwargs), n_envs=1, seed=args.seed + 999)
    )

    model = build_model(args.algo, train_env, tb_log=RUNS_DIR, seed=args.seed)

    callbacks = [
        CheckpointCallback(
            save_freq=max(20_000 // args.n_envs, 1),
            save_path=run_dir,
            name_prefix="ckpt",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=max(10_000 // args.n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
        ),
    ]

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=True,
    )
    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"\nSaved final model to {final_path}.zip")
    print(f"TensorBoard: uv run tensorboard --logdir {RUNS_DIR}")


if __name__ == "__main__":
    main()
