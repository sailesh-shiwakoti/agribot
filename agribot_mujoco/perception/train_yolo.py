"""
Fine-tune YOLOv8 to detect orchard fruit, on Apple-Silicon GPU (MPS).

Consumes a YOLO-format dataset produced by generate_dataset.py (a data.yaml
plus images/ + labels/). Logs to runs/ and prints mAP. Export to ONNX with
export_onnx.py afterwards for deployment in the ROS perception node.

Example:
    uv run python perception/train_yolo.py \
        --data datasets/orchard/data.yaml --epochs 60 --model yolov8n.pt
"""

from __future__ import annotations

import argparse

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to YOLO data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="base weights")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--name", default="fruit_yolov8n")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="perception/runs",
        name=args.name,
        # A single class ("fruit"); light augmentation since our synthetic set
        # already domain-randomizes lighting/positions.
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.4,
        fliplr=0.5, mosaic=1.0,
        patience=20,
    )
    metrics = model.val(data=args.data, device=device)
    print(f"\nmAP50 = {metrics.box.map50:.3f}   mAP50-95 = {metrics.box.map:.3f}")
    print(f"Best weights: perception/runs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
