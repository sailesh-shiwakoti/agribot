"""
Full autonomous harvest demo (Track A).

Composes the whole pipeline:
    headless Gazebo Sim + UR5e + Foxglove   (headless_sim.launch.py)
  + MoveIt 2 move_group                      (ur_moveit_config)
  + camera bridge + red-apple perception     (perception.launch.py)
  + MoveIt planning scene (trunks/ground)
  + harvest state machine                    (agribot_manipulation)

    ros2 launch agribot_bringup harvest_demo.launch.py
    ros2 launch agribot_bringup harvest_demo.launch.py detector:=yolo model_path:=/path/fruit.onnx

Perception + planning + the sequencer start after a delay so move_group and the
controllers are up first. Watch it in Foxglove (ws://localhost:8765): camera
image + detection boxes, the 3D fruit markers, and the arm planning to each.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bringup_share = get_package_share_directory("agribot_bringup")
    detector = LaunchConfiguration("detector")
    model_path = LaunchConfiguration("model_path")
    use_sim_time = {"use_sim_time": True}

    # 1) headless sim: gz + UR5e + ros2_control + foxglove
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "headless_sim.launch.py")))

    # 2) MoveIt move_group (no RViz; we visualise in Foxglove)
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare("ur_moveit_config"), "launch", "ur_moveit.launch.py"])),
        launch_arguments={
            "ur_type": "ur5e",
            "launch_rviz": "false",
            "use_sim_time": "true",
        }.items(),
    )

    # 3) perception (bridge + detector + localizer)
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, "launch", "perception.launch.py")),
        launch_arguments={"detector": detector, "model_path": model_path}.items(),
    )

    # 4) MoveIt planning scene (trunks + ground)
    planning_scene = Node(
        package="agribot_manipulation", executable="planning_scene_publisher",
        name="planning_scene_publisher", parameters=[use_sim_time],
        output="screen",
    )

    # 5) the harvest state machine
    sequencer = Node(
        package="agribot_manipulation", executable="harvest_sequencer",
        name="harvest_sequencer",
        parameters=[use_sim_time, {"fruit_topic": "/fruit_3d"}],
        output="screen",
    )

    # start move_group + perception once the sim/controllers are up,
    # then the planning scene and sequencer once move_group is ready.
    return LaunchDescription([
        DeclareLaunchArgument("detector", default_value="hsv"),
        DeclareLaunchArgument("model_path", default_value=""),
        sim,
        TimerAction(period=12.0, actions=[moveit, perception]),
        TimerAction(period=22.0, actions=[planning_scene]),
        TimerAction(period=26.0, actions=[sequencer]),
    ])
