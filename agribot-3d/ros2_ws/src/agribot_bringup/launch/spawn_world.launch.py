"""
GUI fallback bring-up (optional) — same as headless_sim.launch.py but WITH the
Gazebo Sim GUI and RViz. Only useful if you have X11 forwarding (XQuartz on
macOS) working; the headless + Foxglove path is the recommended default.

Usage (requires an X server / XQuartz):
    ros2 launch agribot_bringup spawn_world.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = get_package_share_directory("agribot_bringup")
    orchard_world = os.path.join(bringup_share, "worlds", "orchard.sdf")

    ur_sim_launch = PathJoinSubstitution(
        [FindPackageShare("ur_simulation_gz"), "launch", "ur_sim_control.launch.py"]
    )

    ur_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_sim_launch),
        launch_arguments={
            "ur_type": "ur5e",
            "world_file": orchard_world,
            "gazebo_gui": "true",
            "launch_rviz": "true",
        }.items(),
    )

    return LaunchDescription([ur_sim])
