# AgriBot Track A — ROS 2 Humble + Gazebo Sim (Fortress) + MoveIt 2, arm64-native.
#
# Why this base:
#   osrf/ros:humble-desktop-full is amd64-ONLY, so on Apple Silicon it runs
#   under QEMU emulation — fatal for Gazebo performance. The official
#   `ros:humble` image is multi-arch, so we build on it natively (arm64) and
#   apt-install exactly the stack we need.
#
# Why Gazebo *Sim* (gz / Fortress) instead of Gazebo Classic:
#   Classic is end-of-life (Jan 2025) and its UR sim package isn't released for
#   Humble. `ur_simulation_gz` is the maintained package and pulls the modern
#   gz-sim + ros_gz bridge stack — a better portfolio story too.
FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Gazebo Fortress: the ROS repo's gz metapackages need libignition-sensors6
# >= 6.8.1, which only lives in the OSRF Gazebo apt repo (the base image ships
# 6.8.0). Enable that repo first or the install hits "held broken packages".
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget gnupg lsb-release ca-certificates && \
    wget -qO /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
      https://packages.osrfoundation.org/gazebo.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
#   ur-simulation-gz    -> UR5e in Gazebo Sim + gz_ros2_control wiring
#   ur-moveit-config    -> official MoveIt 2 config for UR arms (OMPL/RRT*)
#   moveit              -> the planning framework (move_group) pymoveit2 drives
#   ros-gz              -> gz-sim, ros_gz_bridge, ros_gz_image (ROS<->gz topics)
#   gz-ros2-control     -> ros2_control hardware interface inside gz-sim
#   foxglove-bridge     -> WebSocket viz on port 8765 (no XQuartz needed)
#   vision-msgs         -> Detection2DArray for the perception nodes
#   cv-bridge/vision-opencv, tf-transformations -> perception + TF helpers
#   controllers         -> joint trajectory + state broadcaster (usually pulled
#                          in by ur-simulation-gz, listed for clarity)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ur-simulation-gz \
    ros-humble-ur-moveit-config \
    ros-humble-moveit \
    ros-humble-ros-gz \
    ros-humble-gz-ros2-control \
    ros-humble-foxglove-bridge \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-tf-transformations \
    ros-humble-joint-trajectory-controller \
    ros-humble-joint-state-broadcaster \
    ros-humble-controller-manager \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-robot-state-publisher \
    ros-humble-xacro \
    python3-colcon-common-extensions \
    python3-pip \
    python3-transforms3d \
    && rm -rf /var/lib/apt/lists/*

# Python deps for perception:
#  - onnxruntime: run the YOLO model exported from the Mac (Phase 3), CPU/arm64
#  - opencv-python-headless: OpenCV without GUI libs (we render via Foxglove)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    onnxruntime \
    opencv-python-headless

WORKDIR /ros2_ws

# Source ROS + the workspace overlay on every interactive shell.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /ros2_ws/install/setup.bash ]; then source /ros2_ws/install/setup.bash; fi" >> /root/.bashrc

# gz-sim renders camera sensors with OGRE2. Docker Desktop on Mac has no GPU
# passthrough, so force software GL (llvmpipe). Slow but functional for our
# low-res camera. Headless rendering is selected in the launch files.
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV QT_QPA_PLATFORM=offscreen

CMD ["bash"]
