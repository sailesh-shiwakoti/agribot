"""
tree_detector — detect tree trunks from the 2D lidar and publish their map-frame
positions. The ROS 2 counterpart of the MuJoCo demo's lidar detection.

Pipeline: LaserScan -> Cartesian points (sensor frame) -> segment into clusters
(range discontinuities) -> keep trunk-sized clusters -> circle-fit each to
recover the centre -> transform to the `map` frame with tf2 -> publish a
PoseArray (+ markers) of detected trees. A downstream coordinator drives the
robot to each.

Subscribes:  /scan                     sensor_msgs/LaserScan
Publishes:   ~/detected_trees          geometry_msgs/PoseArray   (map frame)
             ~/markers                 visualization_msgs/MarkerArray
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf2_geometry_msgs import do_transform_point


class TreeDetector(Node):
    def __init__(self):
        super().__init__("tree_detector")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("cluster_gap", 0.15)     # range jump -> new cluster
        self.declare_parameter("max_trunk_radius", 0.25)
        self.declare_parameter("min_points", 2)
        self.map_frame = self.get_parameter("map_frame").value
        self.cluster_gap = float(self.get_parameter("cluster_gap").value)
        self.max_r = float(self.get_parameter("max_trunk_radius").value)
        self.min_pts = int(self.get_parameter("min_points").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.pub = self.create_publisher(PoseArray, "~/detected_trees", 10)
        self.markers = self.create_publisher(MarkerArray, "~/markers", 10)
        self.get_logger().info("tree_detector ready")

    def on_scan(self, scan: LaserScan):
        ranges = np.asarray(scan.ranges)
        n = len(ranges)
        angles = scan.angle_min + np.arange(n) * scan.angle_increment
        valid = np.isfinite(ranges) & (ranges > scan.range_min) & \
            (ranges < scan.range_max)

        # segment into clusters by range discontinuity
        clusters, cur = [], []
        prev_r = None
        for i in range(n):
            if not valid[i]:
                if cur:
                    clusters.append(cur); cur = []
                prev_r = None
                continue
            r = ranges[i]
            if prev_r is not None and abs(r - prev_r) > self.cluster_gap:
                if cur:
                    clusters.append(cur); cur = []
            cur.append((ranges[i] * np.cos(angles[i]),
                        ranges[i] * np.sin(angles[i])))
            prev_r = r
        if cur:
            clusters.append(cur)

        # transform sensor -> map once
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, scan.header.frame_id, rclpy.time.Time())
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"tf to {self.map_frame}: {e}",
                                   throttle_duration_sec=2.0)
            return

        poses = PoseArray()
        poses.header.frame_id = self.map_frame
        poses.header.stamp = scan.header.stamp
        marr = MarkerArray()
        mid = 0
        for c in clusters:
            if len(c) < self.min_pts:
                continue
            center, radius = self._fit_circle(np.array(c))
            if radius > self.max_r:
                continue
            pt = PointStamped()
            pt.header.frame_id = scan.header.frame_id
            pt.point.x, pt.point.y = float(center[0]), float(center[1])
            pm = do_transform_point(pt, tf)

            pose = Pose()
            pose.position = pm.point
            pose.orientation.w = 1.0
            poses.poses.append(pose)

            m = Marker()
            m.header = poses.header
            m.ns, m.id, m.type, m.action = "trees", mid, Marker.CYLINDER, Marker.ADD
            mid += 1
            m.pose = pose
            m.scale.x = m.scale.y = max(0.1, 2 * radius)
            m.scale.z = 1.2
            m.pose.position.z = 0.6
            m.color.r, m.color.g, m.color.b, m.color.a = 0.4, 0.25, 0.1, 0.7
            m.lifetime.sec = 1
            marr.markers.append(m)

        self.pub.publish(poses)
        self.markers.publish(marr)

    @staticmethod
    def _fit_circle(pts):
        """Kasa algebraic circle fit; centroid + nominal radius for few points.
        The lidar sees only the near arc, so a fit recovers the trunk centre."""
        if len(pts) < 3:
            c = pts.mean(axis=0)
            # push outward from the sensor origin by a nominal trunk radius
            n = np.linalg.norm(c)
            return (c + 0.08 * c / n if n > 1e-6 else c), 0.08
        x, y = pts[:, 0], pts[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones(len(pts))]
        b = x ** 2 + y ** 2
        try:
            cx, cy, cc = np.linalg.lstsq(A, b, rcond=None)[0]
            r = float(np.sqrt(max(cc + cx ** 2 + cy ** 2, 1e-4)))
            return np.array([cx, cy]), r
        except np.linalg.LinAlgError:
            return pts.mean(axis=0), 0.08


def main(args=None):
    rclpy.init(args=args)
    node = TreeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
