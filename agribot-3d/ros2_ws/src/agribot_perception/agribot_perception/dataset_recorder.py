"""
dataset_recorder — auto-labelled synthetic dataset generator for training the
YOLO red-apple detector (Phase 3), with zero manual labelling.

For each camera frame it projects every RED fruit model's 3D position through
the camera intrinsics + TF into image pixels, and writes a YOLO-format label
(class 0 = ripe apple) plus the image. GREEN fruit is skipped, so the network
learns "ripe apple" directly. Run it while the sim is up and let a domain
randomizer move fruit/lighting between frames.

Ground truth comes from `/model_poses` (bridged gz model poses) — the fruit
names encode ripeness (fruit_red_* vs fruit_green_*).

Output: <out_dir>/images/*.png  +  <out_dir>/labels/*.txt

    ros2 run agribot_perception dataset_recorder --ros-args -p out_dir:=/ros2_ws/datasets/orchard
"""

from __future__ import annotations

import os

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_msgs.msg import TFMessage
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point


class DatasetRecorder(Node):
    def __init__(self):
        super().__init__("dataset_recorder")
        self.declare_parameter("out_dir", "/ros2_ws/datasets/orchard")
        self.declare_parameter("fruit_radius", 0.035)
        self.declare_parameter("optical_frame", "camera_optical_frame")
        self.out = self.get_parameter("out_dir").value
        self.radius = float(self.get_parameter("fruit_radius").value)
        self.optical = self.get_parameter("optical_frame").value
        os.makedirs(os.path.join(self.out, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.out, "labels"), exist_ok=True)

        self.bridge = CvBridge()
        self.K = None
        self.red_fruit = {}                     # name -> world position
        self.frame = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._info, 10)
        self.create_subscription(TFMessage, "/model_poses", self._poses, 10)

        img_sub = message_filters.Subscriber(self, Image, "/camera/image")
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_image)
        self.get_logger().info(f"recording YOLO dataset to {self.out}")

    def _info(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def _poses(self, msg: TFMessage):
        for tf in msg.transforms:
            n = tf.child_frame_id
            if n.startswith("fruit_red"):
                t = tf.transform.translation
                self.red_fruit[n] = np.array([t.x, t.y, t.z])

    def _project(self, world_pos, stamp):
        pt = PointStamped()
        pt.header.frame_id = "base_link"        # base_link == world origin
        pt.point.x, pt.point.y, pt.point.z = map(float, world_pos)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.optical, "base_link", rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        p = do_transform_point(pt, tf).point
        if p.z <= 0.05:
            return None                          # behind the camera
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        u = fx * p.x / p.z + cx
        v = fy * p.y / p.z + cy
        r_px = fx * self.radius / p.z            # projected radius
        return u, v, r_px

    def _on_image(self, msg: Image):
        if self.K is None or not self.red_fruit:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = img.shape[:2]
        lines = []
        for pos in self.red_fruit.values():
            proj = self._project(pos, msg.header.stamp)
            if proj is None:
                continue
            u, v, r = proj
            if not (0 <= u < w and 0 <= v < h):
                continue
            bw, bh = 2.4 * r, 2.4 * r            # a little padding
            lines.append(f"0 {u/w:.6f} {v/h:.6f} {bw/w:.6f} {bh/h:.6f}")
        if not lines:
            return
        stem = f"frame_{self.frame:05d}"
        cv2.imwrite(os.path.join(self.out, "images", stem + ".png"), img)
        with open(os.path.join(self.out, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        self.frame += 1
        if self.frame % 25 == 0:
            self.get_logger().info(f"saved {self.frame} labelled frames")


def main(args=None):
    rclpy.init(args=args)
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
