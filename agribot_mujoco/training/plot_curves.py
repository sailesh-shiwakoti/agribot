"""
Plot evaluation curves (success rate + mean reward vs. timesteps) from one or
more SB3 runs, for the write-up / media/. Reads each run's evaluations.npz
(written by EvalCallback).

Example:
    uv run python training/plot_curves.py \
        training/runs/sac_seed0_reach training/runs/ppo_seed0_reach \
        --out media/rl_curves.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_run(run_dir: str):
    npz = os.path.join(run_dir, "evaluations.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(npz)
    data = np.load(npz)
    ts = data["timesteps"]
    rewards = data["results"].mean(axis=1)
    # is_success is logged when the env reports it in info (ours does).
    succ = (data["successes"].mean(axis=1)
            if "successes" in data else np.full_like(rewards, np.nan))
    return ts, rewards, succ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("--out", default="media/rl_curves.png")
    args = p.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for run_dir in args.runs:
        label = os.path.basename(run_dir.rstrip("/"))
        ts, rewards, succ = load_run(run_dir)
        ax1.plot(ts, rewards, marker="o", ms=3, label=label)
        ax2.plot(ts, succ * 100, marker="o", ms=3, label=label)

    ax1.set(xlabel="timesteps", ylabel="mean eval return", title="Return")
    ax2.set(xlabel="timesteps", ylabel="success rate (%)",
            title="Reach success", ylim=(0, 100))
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
