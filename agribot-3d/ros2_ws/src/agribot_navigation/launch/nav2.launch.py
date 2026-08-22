"""
Bring up the Nav2 stack (controller, planner, behaviors, bt_navigator,
costmaps, smoother, lifecycle manager) with our params. Map/localization is
provided by slam_toolbox (map -> odom), so no AMCL / map_server here.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nav_share = get_package_share_directory("agribot_navigation")
    params = os.path.join(nav_share, "config", "nav2_params.yaml")
    use_sim = {"use_sim_time": True}

    nodes = [
        Node(package="nav2_controller", executable="controller_server",
             output="screen", parameters=[params, use_sim]),
        Node(package="nav2_planner", executable="planner_server",
             output="screen", parameters=[params, use_sim]),
        Node(package="nav2_behaviors", executable="behavior_server",
             output="screen", parameters=[params, use_sim]),
        Node(package="nav2_bt_navigator", executable="bt_navigator",
             output="screen", parameters=[params, use_sim]),
        Node(package="nav2_velocity_smoother", executable="velocity_smoother",
             output="screen", parameters=[params, use_sim]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_navigation", output="screen",
             parameters=[use_sim, {
                 "autostart": True,
                 "node_names": ["controller_server", "planner_server",
                                "behavior_server", "bt_navigator",
                                "velocity_smoother"]}]),
    ]
    return LaunchDescription(nodes)
