"""
field_coordinator — the mobile-harvest mission controller.

Consumes tree detections, keeps a running map (dedup + confidence), and for each
unvisited tree: computes a parking pose beside the trunk, sends it to **Nav2**
(NavigateToPose) to drive there while avoiding the other trunks, then triggers
the arm to harvest (std_srvs/Trigger on /harvest_here). Repeat until all
detected trees are done.

Nav2 handles SLAM-based localization, global + local planning, and recovery;
this node is the high-level orchestrator on top of it — the same role
field_harvester.run() plays in the MuJoCo demo.

Subscribes:  /tree_detector/detected_trees   geometry_msgs/PoseArray (map)
Action:      /navigate_to_pose               nav2_msgs/action/NavigateToPose
Service:     /harvest_here                   std_srvs/srv/Trigger  (to the arm)
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from std_srvs.srv import Trigger
import tf2_ros
from tf_transformations import quaternion_from_euler


class FieldCoordinator(Node):
    def __init__(self):
        super().__init__("field_coordinator")
        self.declare_parameter("park_dist", 0.7)
        self.declare_parameter("visit_radius", 0.9)
        self.declare_parameter("merge_radius", 0.7)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.park_dist = float(self.get_parameter("park_dist").value)
        self.visit_r = float(self.get_parameter("visit_radius").value)
        self.merge_r = float(self.get_parameter("merge_radius").value)
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._trees = {}                 # id -> [xy, weight]
        self._visited = []
        self._lock = threading.Lock()
        self._next_id = 0

        self.create_subscription(PoseArray, "/tree_detector/detected_trees",
                                 self._on_trees, 10)
        self.nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.harvest_cli = self.create_client(Trigger, "/harvest_here")

    # ------------------------------------------------------------- mapping
    def _on_trees(self, msg: PoseArray):
        with self._lock:
            for p in msg.poses:
                xy = np.array([p.position.x, p.position.y])
                for e in self._trees.values():
                    if np.linalg.norm(xy - e[0]) < self.merge_r:
                        e[0] = (e[0] * e[1] + xy) / (e[1] + 1)
                        e[1] += 1
                        break
                else:
                    self._trees[self._next_id] = [xy, 1]
                    self._next_id += 1

    def _robot_xy(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            return np.array([tf.transform.translation.x,
                             tf.transform.translation.y])
        except tf2_ros.TransformException:
            return np.zeros(2)

    def _next_tree(self, min_weight=3):
        rob = self._robot_xy()
        with self._lock:
            cand = [e[0] for e in self._trees.values() if e[1] >= min_weight]
        cand = [c for c in cand
                if all(np.linalg.norm(c - v) > self.visit_r for v in self._visited)]
        if not cand:
            return None
        return min(cand, key=lambda c: np.linalg.norm(c - rob))

    # ------------------------------------------------------------- actions
    def _goto(self, xy, yaw) -> bool:
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(xy[0]), float(xy[1])
        q = quaternion_from_euler(0, 0, yaw)
        goal.pose.pose.orientation.x = q[0]
        goal.pose.pose.orientation.y = q[1]
        goal.pose.pose.orientation.z = q[2]
        goal.pose.pose.orientation.w = q[3]

        self.nav.wait_for_server()
        send = self.nav.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send)
        handle = send.result()
        if not handle.accepted:
            self.get_logger().warn("Nav2 rejected the goal")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return True

    def _harvest(self):
        if not self.harvest_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("/harvest_here unavailable; skipping arm")
            return
        fut = self.harvest_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut)

    # ------------------------------------------------------------- mission
    def run(self):
        self.get_logger().info("waiting for first tree detections...")
        idle = 0
        while rclpy.ok():
            target = self._next_tree()
            if target is None:
                idle += 1
                if idle > 30:
                    self.get_logger().info("no more trees — mission complete")
                    break
                self._sleep(0.5)
                continue
            idle = 0
            rob = self._robot_xy()
            u = (target - rob) / (np.linalg.norm(target - rob) + 1e-6)
            park = target - self.park_dist * u
            yaw = float(np.arctan2(u[1], u[0]))    # face the tree
            self.get_logger().info(
                f"driving to tree at {np.round(target,2)} (park {np.round(park,2)})")
            if self._goto(park, yaw):
                self.get_logger().info("arrived; harvesting")
                self._harvest()
            self._visited.append(target)

    def _sleep(self, s):
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=s))


def main(args=None):
    rclpy.init(args=args)
    node = FieldCoordinator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
