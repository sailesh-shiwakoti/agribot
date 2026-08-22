"""
Spawn the AgriBot mobile base into a running Gazebo world and bridge its
topics: robot_state_publisher (URDF TF), ros_gz spawn, and the cmd_vel / odom /
scan / joint_states / tf bridges.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    desc_share = get_package_share_directory("agribot_description")
    xacro_file = os.path.join(desc_share, "urdf", "agribot_base.urdf.xacro")
    robot_desc = Command(["xacro ", xacro_file])

    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")

    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        parameters=[{"robot_description": robot_desc, "use_sim_time": True}],
        output="screen",
    )

    spawn = Node(
        package="ros_gz_sim", executable="create",
        arguments=["-name", "agribot", "-topic", "robot_description",
                   "-x", x, "-y", y, "-z", "0.15"],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry",
            "/scan@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan",
            "/joint_states@sensor_msgs/msg/JointState@ignition.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V",
            "/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("x", default_value="0.0"),
        DeclareLaunchArgument("y", default_value="0.0"),
        rsp, spawn, bridge,
    ])
