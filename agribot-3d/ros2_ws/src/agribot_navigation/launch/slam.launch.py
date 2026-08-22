"""Online-async SLAM (slam_toolbox) — builds the field map from /scan."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nav_share = get_package_share_directory("agribot_navigation")
    slam_params = os.path.join(nav_share, "config", "slam_toolbox.yaml")

    slam = Node(
        package="slam_toolbox", executable="async_slam_toolbox_node",
        name="slam_toolbox", output="screen",
        parameters=[slam_params, {"use_sim_time": True}],
    )
    return LaunchDescription([slam])
