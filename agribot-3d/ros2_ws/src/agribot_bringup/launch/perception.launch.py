"""
Perception bring-up: bridge the Gazebo camera into ROS, publish the camera's
static transform, and run the red-apple detector + 3D localizer.

    detector:=hsv   -> classical HSV red-apple detector (default)
    detector:=yolo  -> trained YOLOv8-ONNX detector (needs model_path:=...)

Both detectors publish to /detections, so the localizer (and the harvest
sequencer downstream) are agnostic to which one runs. Output: /fruit_3d.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory("agribot_bringup")
    bridge_config = os.path.join(bringup_share, "config", "ros_gz_bridge.yaml")

    detector = LaunchConfiguration("detector")
    model_path = LaunchConfiguration("model_path")
    is_hsv = IfCondition(PythonExpression(["'", detector, "' == 'hsv'"]))
    is_yolo = IfCondition(PythonExpression(["'", detector, "' == 'yolo'"]))

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        parameters=[{"config_file": bridge_config}],
        output="screen",
    )

    # base_link -> camera optical frame (from the mast pose in orchard.sdf;
    # quaternion verified: optical z/view axis -> +y, tilted ~12 deg down)
    cam_tf = Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="camera_static_tf",
        arguments=[
            "--x", "0.45", "--y", "-0.9", "--z", "0.95",
            "--qx", "-0.77732", "--qy", "0.0", "--qz", "0.0", "--qw", "0.6291",
            "--frame-id", "base_link", "--child-frame-id", "camera_optical_frame",
        ],
    )

    hsv = Node(
        package="agribot_perception", executable="red_apple_detector",
        name="red_apple_detector",
        parameters=[{"image_topic": "/camera/image"}],
        remappings=[("~/detections", "/detections"),
                    ("~/debug_image", "/detections_debug")],
        condition=is_hsv,
    )
    yolo = Node(
        package="agribot_perception", executable="yolo_onnx_detector",
        name="yolo_onnx_detector",
        parameters=[{"image_topic": "/camera/image", "model_path": model_path}],
        remappings=[("~/detections", "/detections"),
                    ("~/debug_image", "/detections_debug")],
        condition=is_yolo,
    )

    localizer = Node(
        package="agribot_perception", executable="fruit_localizer",
        name="fruit_localizer",
        parameters=[{
            "detections_topic": "/detections",
            "depth_topic": "/camera/depth_image",
            "camera_info_topic": "/camera/camera_info",
            "target_frame": "base_link",
        }],
        remappings=[("~/fruit_3d", "/fruit_3d"),
                    ("~/markers", "/fruit_markers")],
    )

    return LaunchDescription([
        DeclareLaunchArgument("detector", default_value="hsv",
                              description="hsv | yolo"),
        DeclareLaunchArgument("model_path", default_value="",
                              description="ONNX model path for detector:=yolo"),
        bridge, cam_tf, hsv, yolo, localizer,
    ])
