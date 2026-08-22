"""
yolo_onnx_detector — learned red-apple detector, deploying the YOLOv8 model
trained natively on the Mac (agribot_mujoco/perception) and exported to ONNX.

Runs with onnxruntime on CPU inside the container. Publishes the EXACT same
vision_msgs/Detection2DArray interface as red_apple_detector, so the harvest
pipeline is identical regardless of which detector is launched
(`detector:=hsv|yolo`).

Ripeness (hypothesis score) is estimated from the detected crop's mean
saturation, as in the HSV baseline, so downstream force scaling is consistent.

Params:
    model_path   absolute path to the .onnx file
    image_topic  RGB image topic
    conf_thresh  detection confidence threshold
    iou_thresh   NMS IoU threshold
"""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import (
    Detection2D, Detection2DArray, ObjectHypothesisWithPose,
)
import onnxruntime as ort


class YoloOnnxDetector(Node):
    def __init__(self):
        super().__init__("yolo_onnx_detector")
        self.declare_parameter("model_path", "")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_thresh", 0.35)
        self.declare_parameter("iou_thresh", 0.5)
        self.declare_parameter("publish_debug", True)

        model_path = self.get_parameter("model_path").value
        if not model_path:
            raise RuntimeError("yolo_onnx_detector requires 'model_path'")
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf_thresh").value)
        self.iou = float(self.get_parameter("iou_thresh").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        self.sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.get_parameter("image_topic").value, self.on_image, 10)
        self.pub = self.create_publisher(Detection2DArray, "~/detections", 10)
        self.dbg = self.create_publisher(Image, "~/debug_image", 1)
        self.get_logger().info(f"yolo_onnx_detector loaded {model_path}")

    # ---------------- pre / post processing ----------------
    def _letterbox(self, img):
        h, w = img.shape[:2]
        s = self.imgsz / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, s, left, top

    def on_image(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        canvas, s, padx, pady = self._letterbox(bgr)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]        # 1x3xHxW

        out = self.sess.run(None, {self.in_name: blob})[0]  # 1 x (4+nc) x N
        pred = np.squeeze(out, 0).T                          # N x (4+nc)
        boxes_xywh = pred[:, :4]
        scores = pred[:, 4:].max(axis=1)
        keep = scores > self.conf
        boxes_xywh, scores = boxes_xywh[keep], scores[keep]

        # xywh (letterbox px) -> xyxy in original image px
        xyxy = np.empty_like(boxes_xywh)
        xyxy[:, 0] = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - padx) / s
        xyxy[:, 1] = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pady) / s
        xyxy[:, 2] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - padx) / s
        xyxy[:, 3] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pady) / s

        idxs = cv2.dnn.NMSBoxes(
            [[float(a), float(b), float(c - a), float(d - b)]
             for a, b, c, d in xyxy],
            scores.tolist(), self.conf, self.iou)
        idxs = np.array(idxs).flatten() if len(idxs) else np.array([], int)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        det_arr = Detection2DArray()
        det_arr.header = msg.header
        dbg = bgr.copy() if self.publish_debug else None
        H, W = bgr.shape[:2]

        for i in idxs:
            x1, y1, x2, y2 = xyxy[i]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            ripeness = float(np.clip(
                hsv[y1:y2, x1:x2, 1].mean() / 255.0, 0, 1))

            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = (x1 + x2) / 2.0
            det.bbox.center.position.y = (y1 + y2) / 2.0
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "apple"
            hyp.hypothesis.score = ripeness
            det.results.append(hyp)
            det_arr.detections.append(det)
            if dbg is not None:
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.pub.publish(det_arr)
        if dbg is not None:
            self.dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = YoloOnnxDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
