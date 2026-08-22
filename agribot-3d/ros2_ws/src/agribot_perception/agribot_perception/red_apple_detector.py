"""
red_apple_detector — detect RED (ripe) apples in the RGB image and publish
standard vision_msgs/Detection2DArray. GREEN (unripe) fruit is ignored by
construction, because we threshold only the red hue bands.

Each detection carries an estimated *ripeness* in [0, 1] (from color
saturation) as the hypothesis score, so the manipulation stage can scale grip
force (riper = more fragile = gentler).

Subscribes:  <image_topic>            sensor_msgs/Image  (rgb8/bgr8)
Publishes:   ~/detections             vision_msgs/Detection2DArray
             ~/debug_image            sensor_msgs/Image  (boxes drawn, for viz)

This is the classical-CV baseline. yolo_onnx_detector publishes the identical
interface using a trained network, so downstream nodes are agnostic to which
detector runs (see the `detector` launch argument).
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


class RedAppleDetector(Node):
    def __init__(self):
        super().__init__("red_apple_detector")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("min_area_px", 60)
        self.declare_parameter("publish_debug", True)
        image_topic = self.get_parameter("image_topic").value
        self.min_area = int(self.get_parameter("min_area_px").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        self.bridge = CvBridge()
        # Red wraps the hue circle, so we need two bands.
        self.lower1, self.upper1 = np.array([0, 110, 70]), np.array([10, 255, 255])
        self.lower2, self.upper2 = np.array([170, 110, 70]), np.array([180, 255, 255])

        self.sub = self.create_subscription(Image, image_topic,
                                            self.on_image, 10)
        self.pub = self.create_publisher(Detection2DArray, "~/detections", 10)
        self.dbg = self.create_publisher(Image, "~/debug_image", 1)
        self.get_logger().info(f"red_apple_detector watching {image_topic}")

    def on_image(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.lower1, self.upper1),
            cv2.inRange(hsv, self.lower2, self.upper2),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        out = Detection2DArray()
        out.header = msg.header
        dbg = bgr.copy() if self.publish_debug else None

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w / 2.0, y + h / 2.0
            # ripeness ~ mean saturation of the blob (riper apples are deeper red)
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            ripeness = float(np.clip(hsv[:, :, 1][blob > 0].mean() / 255.0, 0, 1))

            det = Detection2D()
            det.header = msg.header
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "apple"
            hyp.hypothesis.score = ripeness
            det.results.append(hyp)
            out.detections.append(det)

            if dbg is not None:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(dbg, f"{ripeness:.2f}", (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self.pub.publish(out)
        if dbg is not None:
            self.dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = RedAppleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
