"""
Evaluate a trained YOLO detector on a held-out split and report mAP + a
precision-recall curve image for the write-up.

Example:
    uv run python perception/eval_map.py \
        --weights perception/runs/fruit_yolov8n/weights/best.pt \
        --data datasets/orchard/data.yaml
"""

from __future__ import annotations

import argparse
import shutil

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--pr-out", default="media/yolo_pr_curve.png")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, device=device,
                        plots=True)

    print(f"\n=== YOLO detection metrics ({args.split}) ===")
    print(f"mAP50    : {metrics.box.map50:.3f}")
    print(f"mAP50-95 : {metrics.box.map:.3f}")
    print(f"precision: {metrics.box.mp:.3f}")
    print(f"recall   : {metrics.box.mr:.3f}")

    # Ultralytics writes PR_curve.png into its save_dir; copy it out.
    pr_src = f"{metrics.save_dir}/PR_curve.png"
    try:
        shutil.copy(pr_src, args.pr_out)
        print(f"PR curve -> {args.pr_out}")
    except FileNotFoundError:
        print(f"(PR curve not found at {pr_src})")


if __name__ == "__main__":
    main()
