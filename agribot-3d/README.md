# AgriBot-3D — Phase 1 & 2 Starter

ROS 2 Humble + Gazebo + MoveIt 2, running in Docker on macOS.

## What this scaffold gives you

- A `Dockerfile` / `docker-compose.yml` that boots ROS 2 Humble with Gazebo and
  MoveIt 2, using GUI forwarding to your Mac's display.
- `agribot_bringup`: launches the **official UR5e Gazebo simulation** (from
  `ur_simulation_gazebo`) inside a custom `orchard.world` containing a
  placeholder tree trunk + red "fruit" target.
- `agribot_perception`: a working (if simple) red-fruit tracker node using
  OpenCV color segmentation — this is your Phase 3 starting point.

We're using the official UR packages instead of hand-rolling URDF/MoveIt
config. That's the same shortcut real teams take — the arm model and its
MoveIt config are a solved problem; your actual project work is the orchard
world, the perception node, and the visual servoing loop.

---

## Step 0 — One-time Mac setup

1. Install [XQuartz](https://www.xquartz.org/) (X11 server for macOS).
2. Open XQuartz → Settings → Security → check **"Allow connections from
   network clients"**. Restart XQuartz.
3. In a Mac terminal (not inside Docker):
   ```bash
   xhost + 127.0.0.1
   ```
   You'll need to re-run this after every Mac reboot.
4. Make sure Docker Desktop is running.

## Step 1 — Build the image

```bash
git init agribot-3d && cd agribot-3d
# (copy these files into this folder)
docker compose build
```

This will take a while the first time — it's pulling a full desktop ROS 2
image plus Gazebo and MoveIt packages (~a few GB).

## Step 2 — Boot the container and build the workspace

```bash
docker compose run agribot
```

Inside the container:

```bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Step 3 — Launch Gazebo + the UR5e in your orchard world

```bash
ros2 launch agribot_bringup spawn_world.launch.py
```

You should see Gazebo open on your Mac screen with a UR5e arm standing next
to a brown cylinder (trunk) and small red sphere (fruit), plus RViz showing
the planning scene.

**If nothing appears / it hangs:** almost always an X11 problem, not a ROS
problem. Sanity-check the display pipeline first:
```bash
# inside the container
xeyes
```
If `xeyes` doesn't pop up on your Mac, fix that before touching ROS — Gazebo
will never render until basic X11 forwarding works.

**If Gazebo opens but is extremely slow / stutters:** this is expected.
Docker Desktop on Mac doesn't pass through your GPU, so Gazebo is doing
physics + rendering entirely on CPU (`LIBGL_ALWAYS_SOFTWARE=1` in the
Dockerfile is already forcing software rendering so it doesn't just crash).
Two ways around this as the project grows:
- Run `gzserver` headless (no rendering) for actual physics/planning work,
  and only launch `gzclient` (the GUI) when you need to look at something.
- If it's unusably slow even headless, that's a sign to move Docker to a
  Linux box or a cloud VM with GPU passthrough — very common for ROS/Gazebo
  work, nothing wrong with your setup.

## Step 4 — MoveIt 2 motion planning

In a second terminal, attach to the running container:
```bash
docker exec -it agribot-3d bash
source /ros2_ws/install/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true
```
In RViz's MoveIt panel you can now drag the interactive marker to a pose and
hit "Plan & Execute" — this is RRT* (OMPL) planning around the scene, which
is Phase 2 in one command. Your actual Phase 2 work is publishing the
tree/branch geometry as real `CollisionObject` messages so MoveIt treats
them as obstacles (right now the orchard world's collision geometry exists
in Gazebo but isn't yet registered with MoveIt's planning scene — that
wiring is the next thing to build).

## Step 5 — Run the fruit tracker (Phase 3 starting point)

```bash
ros2 run agribot_perception fruit_tracker
ros2 topic echo /fruit_target/pixel
```
You should see pixel coordinates streaming once the fruit is in the camera's
view. From here, the two things worth building next are noted at the top of
`fruit_tracker_node.py`.

---

## Repo layout

```
agribot-3d/
├── Dockerfile
├── docker-compose.yml
└── ros2_ws/src/
    ├── agribot_bringup/       # launch files + orchard.world
    └── agribot_perception/    # camera -> pixel coordinate tracker node
```

## A note on scope

This scaffold gets you a running Gazebo + MoveIt 2 loop with a placeholder
scene fast, so you're not stuck debugging Docker/X11 for days before writing
any robotics code. The two most valuable things to do next, in order:
1. Get `Step 3` actually rendering on your machine (this is the part most
   likely to eat real time — budget for it).
2. Register the orchard collision geometry with MoveIt's planning scene
   (Phase 2, item 2 in the original blueprint) — that's real robotics work,
   not boilerplate.
