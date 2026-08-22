"""
Run the full orchard-harvesting demo and render a video.

    uv run python -m orchard.run_demo
    uv run python -m orchard.run_demo --trees 12 --out media/orchard_harvest.mp4

Produces media/orchard_harvest.mp4 and prints a harvest report.
"""

from __future__ import annotations

import argparse
import os
import time

import imageio.v2 as imageio

from .config import OrchardConfig
from .harvester import Harvester


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trees", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = OrchardConfig()
    if args.trees is not None:
        cfg.n_trees = args.trees
    if args.seed is not None:
        cfg.seed = args.seed

    # project root is two levels above agribot_mujoco/orchard/
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = args.out or os.path.join(root, "media", "orchard_harvest.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    h = Harvester(cfg)
    n_red = sum(a.is_ripe for a in h.apples)
    n_green = len(h.apples) - n_red
    print(f"Orchard: {cfg.n_trees} trees, {len(h.apples)} apples "
          f"({n_red} ripe, {n_green} unripe). Harvesting...")

    rep = h.run()
    imageio.mimsave(out, h.frames, fps=cfg.fps, macro_block_size=None)
    h.renderer.close()

    print("\n===== HARVEST REPORT =====")
    print(f"  ripe apples picked + binned : {rep['picked']} / {n_red}")
    print(f"  crushed (over-force)        : {rep['crushed']}")
    print(f"  ripe left (out of reach)    : {rep['left_unreachable']}")
    print(f"  unripe left on tree (green) : {rep['left_green']} / {n_green}")
    print(f"  score (reward)              : {rep['reward']}")
    print(f"  video ({len(h.frames)} frames)  : {out}")
    print(f"  wall time                   : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
