"""
Phase 0 — headless bring-up of the AgriBot cell.

Starts the official UR5e Gazebo-Sim simulation (arm + ros2_control) in our
orchard world with NO Gazebo GUI and NO RViz, then launches foxglove_bridge so
the whole scene (TF, robot model, later camera + markers) can be viewed in
Foxglove Studio on the Mac at ws://localhost:8765 — no XQuartz required.

Usage:
    ros2 launch agribot_bringup headless_sim.launch.py
    # optional: ros2 launch agribot_bringup headless_sim.launch.py foxglove_port:=8765
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = get_package_share_directory("agribot_bringup")
    orchard_world = os.path.join(bringup_share, "worlds", "orchard.sdf")

    foxglove_port = LaunchConfiguration("foxglove_port")

    ur_sim_launch = PathJoinSubstitution(
        [FindPackageShare("ur_simulation_gz"), "launch", "ur_sim_control.launch.py"]
    )

    ur_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_sim_launch),
        launch_arguments={
            "ur_type": "ur5e",
            "world_file": orchard_world,
            "gazebo_gui": "false",   # headless: gzserver only (-s -r)
            "launch_rviz": "false",  # we visualise in Foxglove instead
        }.items(),
    )

    # foxglove_bridge serves a WebSocket on 0.0.0.0:<port>; docker-compose maps
    # 8765 to the Mac so Foxglove Studio can connect.
    foxglove = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        parameters=[{
            "port": foxglove_port,
            "address": "0.0.0.0",
            "topic_whitelist": [".*"],
            "send_buffer_limit": 20_000_000,
        }],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("foxglove_port", default_value="8765"),
        ur_sim,
        foxglove,
    ])
