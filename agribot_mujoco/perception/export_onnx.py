"""
Export a trained YOLOv8 detector to ONNX for deployment in the ROS 2
perception node (agribot_perception/yolo_onnx_detector), which runs it with
onnxruntime on CPU inside the container.

We sanity-check the exported graph with onnxruntime here so deployment
surprises surface on the Mac, not in the container.

Example:
    uv run python perception/export_onnx.py \
        --weights perception/runs/fruit_yolov8n/weights/best.pt \
        --out ../agribot-3d/ros2_ws/src/agribot_perception/models/fruit_yolov8n.onnx
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True, help="destination .onnx path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    args = p.parse_args()

    model = YOLO(args.weights)
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset)
    print(f"Exported: {onnx_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    shutil.copy(onnx_path, args.out)
    print(f"Copied -> {args.out}")

    # --- onnxruntime sanity check ---
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dummy = np.zeros(
        [d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32
    )
    out = sess.run(None, {inp.name: dummy})
    print(f"[ok] onnxruntime forward pass: input {inp.shape} "
          f"-> output {[o.shape for o in out]}")


if __name__ == "__main__":
    main()
