"""
fruit_localizer — turn 2D red-apple detections into 3D fruit poses in the
robot's base frame, ready for motion planning.

For each detection it:
  1. samples the depth image at the bbox center (median over a small patch to
     reject noise / edge pixels),
  2. back-projects (u, v, Z) through the pinhole model using camera_info
     intrinsics -> a 3D point in the camera optical frame,
  3. transforms that point into `base_link` with tf2,
  4. republishes as vision_msgs/Detection3DArray (ripeness kept as the score),
     plus a MarkerArray for Foxglove/RViz.

Subscribes:  <detections_topic>   vision_msgs/Detection2DArray
             <depth_topic>        sensor_msgs/Image (32FC1, metres)
             <camera_info_topic>  sensor_msgs/CameraInfo
Publishes:   ~/fruit_3d           vision_msgs/Detection3DArray  (base_link)
             ~/markers            visualization_msgs/MarkerArray
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_point


class FruitLocalizer(Node):
    def __init__(self):
        super().__init__("fruit_localizer")
        self.declare_parameter("detections_topic", "/red_apple_detector/detections")
        self.declare_parameter("depth_topic", "/camera/depth_image")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("patch", 5)          # depth sampling half-window
        self.target_frame = self.get_parameter("target_frame").value
        self.patch = int(self.get_parameter("patch").value)

        self.bridge = CvBridge()
        self.K = None                                # camera intrinsics
        self.optical_frame = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self.on_info, 10)

        # Sync detections with the depth frame they were computed from.
        det_sub = message_filters.Subscriber(
            self, Detection2DArray, self.get_parameter("detections_topic").value)
        depth_sub = message_filters.Subscriber(
            self, Image, self.get_parameter("depth_topic").value)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [det_sub, depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.on_pair)

        self.pub = self.create_publisher(Detection3DArray, "~/fruit_3d", 10)
        self.markers = self.create_publisher(MarkerArray, "~/markers", 10)
        self.get_logger().info("fruit_localizer ready")

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)
        self.optical_frame = msg.header.frame_id

    def _depth_at(self, depth, u, v):
        h, w = depth.shape
        u, v = int(round(u)), int(round(v))
        p = self.patch
        win = depth[max(0, v - p):min(h, v + p + 1),
                    max(0, u - p):min(w, u + p + 1)]
        vals = win[np.isfinite(win) & (win > 0.05)]
        return float(np.median(vals)) if vals.size else None

    def on_pair(self, dets: Detection2DArray, depth_msg: Image):
        if self.K is None:
            return
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = np.asarray(depth, dtype=np.float32)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # transform optical_frame -> base_link (once for this batch)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.optical_frame,
                rclpy.time.Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"tf {self.optical_frame}->{self.target_frame}: {e}",
                                   throttle_duration_sec=2.0)
            return

        out = Detection3DArray()
        out.header.stamp = dets.header.stamp
        out.header.frame_id = self.target_frame
        marr = MarkerArray()

        for i, d in enumerate(dets.detections):
            u = d.bbox.center.position.x
            v = d.bbox.center.position.y
            Z = self._depth_at(depth, u, v)
            if Z is None:
                continue
            # pinhole back-projection in the optical frame
            pt = PointStamped()
            pt.header.frame_id = self.optical_frame
            pt.point.x = (u - cx) * Z / fx
            pt.point.y = (v - cy) * Z / fy
            pt.point.z = float(Z)
            pb = do_transform_point(pt, tf)         # -> base_link

            det3 = Detection3D()
            det3.header = out.header
            det3.bbox.center.position = pb.point
            det3.bbox.size.x = det3.bbox.size.y = det3.bbox.size.z = 0.07
            det3.results = d.results                # keep class + ripeness score
            out.detections.append(det3)

            m = Marker()
            m.header = out.header
            m.ns = "fruit"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = pb.point
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.07
            score = d.results[0].hypothesis.score if d.results else 0.5
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0 - score, 0.0, 0.9
            m.lifetime.sec = 1
            marr.markers.append(m)

        self.pub.publish(out)
        self.markers.publish(marr)


def main(args=None):
    rclpy.init(args=args)
    node = FruitLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
