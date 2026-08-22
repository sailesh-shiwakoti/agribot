"""
Run the field navigation + harvest demo and render a video.

    uv run python -m orchard.run_field_demo
    uv run python -m orchard.run_field_demo --trees 12 --seed 5

The robot detects trees with its lidar, plans paths around them, drives to each,
and harvests the ripe apples. Output: media/field_harvest.mp4 + a report.
"""

from __future__ import annotations

import argparse
import os
import time

import imageio.v2 as imageio

from .config import FieldConfig
from .field_harvester import FieldHarvester


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trees", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = FieldConfig()
    if args.trees is not None:
        cfg.n_trees = args.trees
    if args.seed is not None:
        cfg.seed = args.seed

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = args.out or os.path.join(root, "media", "field_harvest.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    t0 = time.time()
    h = FieldHarvester(cfg)
    n_red = sum(a.is_ripe for a in h.apples)
    print(f"Field: {len(h.trees)} trees, {len(h.apples)} apples ({n_red} ripe). "
          f"Detecting + harvesting...")
    rep = h.run()
    imageio.mimsave(out, h.frames, fps=cfg.fps, macro_block_size=None)
    h.renderer.close()

    print("\n===== FIELD HARVEST REPORT =====")
    print(f"  trees detected (lidar)   : {len(h.committed_trees())} / {len(h.trees)}")
    print(f"  trees visited            : {rep['trees_visited']}")
    print(f"  ripe apples picked       : {rep['picked']} / {n_red}")
    print(f"  ripe left (unreachable)  : {rep['left_unreachable']}")
    print(f"  unripe left on tree      : {rep['left_green']}")
    print(f"  video ({len(h.frames)} frames)  : {out}")
    print(f"  wall time                : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
