"""
planning_scene_publisher — register the orchard's static geometry (tree
trunks + ground) with MoveIt as CollisionObjects, so the motion planner avoids
them when reaching for fruit.

This is the piece the original scaffold flagged as "the next real work": the
trunks exist in Gazebo, but MoveIt only avoids what is in its *planning scene*.
We publish a PlanningScene diff on /planning_scene (the standard MoveIt topic).

Trunk poses/dimensions mirror agribot_bringup/worlds/orchard.sdf. base_link is
at the world origin (the UR5e base), so world and base_link coordinates match.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

# (id, x, y, z, radius, length) — from orchard.sdf trellis posts
TRUNKS = [
    ("tree_trunk_1", 0.25, 0.42, 0.7, 0.06, 1.4),
    ("tree_trunk_2", 0.75, 0.42, 0.7, 0.06, 1.4),
]


class PlanningScenePublisher(Node):
    def __init__(self):
        super().__init__("planning_scene_publisher")
        self.declare_parameter("frame_id", "base_link")
        self.frame = self.get_parameter("frame_id").value

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL   # latched
        self.pub = self.create_publisher(PlanningScene, "/planning_scene", qos)
        # publish a few times so late-joining move_group receives it
        self.timer = self.create_timer(1.0, self._publish)
        self._count = 0

    def _cylinder(self, name, x, y, z, r, length) -> CollisionObject:
        co = CollisionObject()
        co.header.frame_id = self.frame
        co.id = name
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.CYLINDER
        prim.dimensions = [float(length), float(r)]        # [height, radius]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.w = 1.0
        co.primitives.append(prim)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        return co

    def _ground(self) -> CollisionObject:
        co = CollisionObject()
        co.header.frame_id = self.frame
        co.id = "ground"
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [10.0, 10.0, 0.02]
        pose = Pose()
        pose.position.z = -0.01
        pose.orientation.w = 1.0
        co.primitives.append(prim)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        return co

    def _publish(self):
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(self._ground())
        for t in TRUNKS:
            scene.world.collision_objects.append(self._cylinder(*t))
        self.pub.publish(scene)
        self._count += 1
        if self._count == 1:
            self.get_logger().info(
                f"published planning scene: ground + {len(TRUNKS)} trunks")
        if self._count >= 5:
            self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = PlanningScenePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
