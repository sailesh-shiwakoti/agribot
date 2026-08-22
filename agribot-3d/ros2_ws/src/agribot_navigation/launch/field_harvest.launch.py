"""
Full mobile field-harvest demo (ROS 2 / Gazebo).

    Gazebo field world + AgriBot mobile base + lidar
  + SLAM (slam_toolbox)   -> map
  + Nav2                  -> autonomous navigation between trees
  + tree_detector         -> trunk positions from /scan
  + field_coordinator     -> visits each tree, drives via Nav2, triggers harvest

    ros2 launch agribot_navigation field_harvest.launch.py

Perception/planning/navigation start after the sim + robot are up. Visualize in
Foxglove (map, /scan, costmaps, detected trees, the robot driving the field).
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, TimerAction, ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    nav_share = get_package_share_directory("agribot_navigation")
    desc_share = get_package_share_directory("agribot_description")
    world = os.path.join(nav_share, "config", "field.sdf")

    # headless Gazebo Sim with the field world
    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])),
        launch_arguments={"gz_args": f"-s -r -v 2 {world}"}.items(),
    )

    spawn = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_share, "launch", "spawn_robot.launch.py")),
        launch_arguments={"x": "0.0", "y": "0.0"}.items(),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_share, "launch", "slam.launch.py")))
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_share, "launch", "nav2.launch.py")))

    tree_detector = Node(
        package="agribot_navigation", executable="tree_detector",
        name="tree_detector", output="screen",
        parameters=[{"use_sim_time": True}])
    coordinator = Node(
        package="agribot_navigation", executable="field_coordinator",
        name="field_coordinator", output="screen",
        parameters=[{"use_sim_time": True}])

    foxglove = Node(
        package="foxglove_bridge", executable="foxglove_bridge",
        parameters=[{"port": 8765}], output="screen")

    return LaunchDescription([
        gz,
        TimerAction(period=4.0, actions=[spawn, foxglove]),
        TimerAction(period=10.0, actions=[slam, nav2, tree_detector]),
        TimerAction(period=18.0, actions=[coordinator]),
    ])
