"""
harvest_sequencer — the autonomous pick loop.

State machine driving the UR5e via MoveIt (pymoveit2):

    SCAN  -> read 3D fruit detections (base_link frame)
    SELECT-> choose the ripest not-yet-picked reachable apple
    PRE_GRASP -> plan to a pose offset back from the fruit
    APPROACH  -> cartesian move onto the fruit
    HARVEST   -> "grasp" (grip force scaled by ripeness) + detach the fruit
    RETREAT   -> cartesian move back, then repeat

Perception (red-only detection) already guarantees we never target green
fruit. Grip force follows the project rule: force_tolerance = 1 - ripeness, so
riper apples are gripped more gently.

The physical grasp is abstracted for simulation: on HARVEST we call the Gazebo
DeleteEntity service to remove the fruit model (the sim stand-in for a gripper
close + pick). Swapping in a real gripper controller + grasp plugin is the only
change needed for hardware.

Subscribes:  <fruit_topic>   vision_msgs/Detection3DArray  (base_link)
             /model_poses    tf2_msgs/TFMessage  (bridged gz model poses)
Service:     /world/orchard/remove  ros_gz_interfaces/srv/DeleteEntity
"""

from __future__ import annotations

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from vision_msgs.msg import Detection3DArray
from tf2_msgs.msg import TFMessage

from pymoveit2 import MoveIt2

UR_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
# "elbow up, facing the fruiting wall" scan pose
SCAN_JOINTS = [1.5708, -1.4, 1.4, -1.5708, -1.5708, 0.0]
# tool0 z-axis -> +y (into the wall): rotation of -90 deg about x
GRASP_QUAT = [-0.7071, 0.0, 0.0, 0.7071]
APPROACH_OFFSET = np.array([0.0, -0.12, 0.0])   # back off toward the robot


class HarvestSequencer(Node):
    def __init__(self):
        super().__init__("harvest_sequencer")
        self.declare_parameter("fruit_topic", "/fruit_localizer/fruit_3d")
        self.declare_parameter("dedup_radius", 0.06)
        self.declare_parameter("reach_radius", 0.9)
        self.dedup = float(self.get_parameter("dedup_radius").value)
        self.reach = float(self.get_parameter("reach_radius").value)

        cbg = ReentrantCallbackGroup()
        self.moveit2 = MoveIt2(
            node=self, joint_names=UR_JOINTS,
            base_link_name="base_link", end_effector_name="tool0",
            group_name="ur_manipulator", callback_group=cbg)
        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.3

        self._fruit = []            # list of (pos(3), ripeness)
        self._model_poses = {}      # name -> np.array(3)
        self._picked = []           # positions already harvested
        self._lock = threading.Lock()

        self.create_subscription(
            Detection3DArray, self.get_parameter("fruit_topic").value,
            self._on_fruit, 10, callback_group=cbg)
        self.create_subscription(
            TFMessage, "/model_poses", self._on_poses, 10, callback_group=cbg)

        from ros_gz_interfaces.srv import DeleteEntity
        self._delete_cli = self.create_client(
            DeleteEntity, "/world/orchard/remove", callback_group=cbg)
        self._DeleteEntity = DeleteEntity

    # ------------------------------------------------------------- callbacks
    def _on_fruit(self, msg: Detection3DArray):
        fruit = []
        for d in msg.detections:
            p = d.bbox.center.position
            score = d.results[0].hypothesis.score if d.results else 0.5
            fruit.append((np.array([p.x, p.y, p.z]), float(score)))
        with self._lock:
            self._fruit = fruit

    def _on_poses(self, msg: TFMessage):
        with self._lock:
            for tf in msg.transforms:
                if tf.child_frame_id.startswith("fruit"):
                    t = tf.transform.translation
                    self._model_poses[tf.child_frame_id] = np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------- helpers
    def _already_picked(self, pos):
        return any(np.linalg.norm(pos - q) < self.dedup for q in self._picked)

    def _next_target(self):
        with self._lock:
            fruit = list(self._fruit)
        cand = [(p, r) for p, r in fruit
                if not self._already_picked(p) and np.linalg.norm(p) < self.reach]
        if not cand:
            return None
        cand.sort(key=lambda pr: pr[1], reverse=True)   # ripest first
        return cand[0]

    def _nearest_model(self, pos):
        with self._lock:
            items = list(self._model_poses.items())
        best, bestd = None, 1e9
        for name, mp in items:
            d = np.linalg.norm(pos - mp)
            if d < bestd:
                best, bestd = name, d
        return best if bestd < 0.15 else None

    def _detach_fruit(self, pos):
        name = self._nearest_model(pos)
        if name is None:
            self.get_logger().warn("no matching fruit entity to remove")
            return
        if not self._delete_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("DeleteEntity service unavailable; "
                                   "skipping physical detach")
            return
        req = self._DeleteEntity.Request()
        req.entity.name = name
        req.entity.type = 2                              # MODEL
        self._delete_cli.call_async(req)
        self.get_logger().info(f"removed entity '{name}'")

    def _move(self, pos, cartesian=False) -> bool:
        self.moveit2.move_to_pose(position=list(pos), quat_xyzw=GRASP_QUAT,
                                  cartesian=cartesian)
        return bool(self.moveit2.wait_until_executed())

    # ------------------------------------------------------------- main loop
    def run(self):
        self.get_logger().info("harvest sequencer: moving to scan pose")
        self.moveit2.move_to_configuration(SCAN_JOINTS)
        self.moveit2.wait_until_executed()

        idle = 0
        while rclpy.ok():
            target = self._next_target()
            if target is None:
                idle += 1
                if idle > 20:
                    self.get_logger().info("no more reachable fruit — done")
                    break
                self._sleep(0.5)
                continue
            idle = 0
            pos, ripeness = target
            force_tol = 1.0 - ripeness
            self.get_logger().info(
                f"SELECT fruit at {np.round(pos,3)} ripeness={ripeness:.2f} "
                f"-> grip force <= {force_tol:.2f}")

            pre = pos + APPROACH_OFFSET
            if not self._move(pre):
                self.get_logger().warn("PRE_GRASP plan failed; skipping")
                self._picked.append(pos)
                continue
            self._move(pos, cartesian=True)              # APPROACH
            self._detach_fruit(pos)                      # HARVEST
            self._move(pre, cartesian=True)              # RETREAT
            self._picked.append(pos)
            self.get_logger().info(f"HARVESTED ({len(self._picked)} total)")

        self.moveit2.move_to_configuration(SCAN_JOINTS)
        self.moveit2.wait_until_executed()

    def _sleep(self, sec):
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=sec))


def main(args=None):
    rclpy.init(args=args)
    node = HarvestSequencer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
