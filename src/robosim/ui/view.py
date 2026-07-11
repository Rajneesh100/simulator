"""Isaac-Sim-style interactive viewer, built on MuJoCo's native "Simulate" GUI.

Orbit/pan/zoom camera, wireframe/shadow/contact toggles, and the native stats overlay
(sim time, step count) come for free from mujoco.viewer — we extend it rather than
building a custom GUI (per idea.md UI guidance). This module drives the step loop so it
can add: camera follow-robot mode, debug overlays (lidar rays, trajectory trail, goal
marker) drawn into the viewer's user scene, and a console HUD (pose/velocity/reward)
printed at a fixed rate — a plain-text stand-in for Isaac's dockable HUD panel.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from robosim.core.world import World, build_world

TRAIL_MAX_POINTS = 300
RGBA_TRAIL = (0.2, 0.6, 1.0, 0.6)
RGBA_LIDAR = (1.0, 0.2, 0.2, 0.8)
RGBA_GOAL = (0.1, 0.9, 0.2, 0.9)


class ViewerState:
    def __init__(self):
        self.follow_camera = False
        self.show_lidar = False
        self.show_trail = True
        self.paused = False
        self.single_step = False
        self.trail: list[np.ndarray] = []


def _make_key_callback(state: ViewerState, world: World):
    def key_callback(keycode):
        char = chr(keycode) if 0 < keycode < 256 else ""
        if char.lower() == "c":
            state.follow_camera = not state.follow_camera
        elif char.lower() == "l":
            state.show_lidar = not state.show_lidar
        elif char.lower() == "t":
            state.show_trail = not state.show_trail
            state.trail.clear()
        elif char == " ":
            state.paused = not state.paused
        elif char == ".":
            state.single_step = True
        elif char.lower() == "r":
            world.reset()

    return key_callback


def _add_marker(scene, pos, rgba, size=0.02, geom_type=mujoco.mjtGeom.mjGEOM_SPHERE):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=geom_type,
        size=[size, size, size],
        pos=np.array(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _update_overlays(viewer, world: World, state: ViewerState, lidar_sensor=None):
    scene = viewer.user_scn
    scene.ngeom = 0

    pos, _ = world.robot_pose()
    if state.show_trail:
        state.trail.append(pos.copy())
        if len(state.trail) > TRAIL_MAX_POINTS:
            state.trail.pop(0)
        for p in state.trail[::3]:
            _add_marker(scene, [p[0], p[1], p[2]], RGBA_TRAIL, size=0.012)

    goal = world.scene.task_info().get("goal")
    if goal is not None:
        _add_marker(scene, [goal[0], goal[1], 0.05], RGBA_GOAL, size=0.06)

    if state.show_lidar and lidar_sensor is not None:
        chassis_id = world.model.body(world.root_body_name).id
        origin, ray_dirs, ranges = lidar_sensor.debug_rays(
            world.model, world.data, exclude_body_id=chassis_id
        )
        for d, r in zip(ray_dirs, ranges):
            hit = origin + d * r
            _add_marker(scene, hit, RGBA_LIDAR, size=0.015)


def run_view(
    scene_arg: str = "flat",
    robot_arg: str = "diffdrive",
    speed: float = 1.0,
    debug_lidar: bool = False,
    hud_hz: float = 2.0,
) -> None:
    world = build_world(scene_arg, robot_arg)
    state = ViewerState()
    state.show_lidar = debug_lidar

    lidar_sensor = None
    try:
        from robosim.sensors.lidar import Lidar

        lidar_sensor = Lidar()
    except Exception:
        lidar_sensor = None

    print("Controls: [space]=pause  [.]=single-step  [r]=reset  [c]=camera-follow  "
          "[l]=lidar-debug  [t]=trajectory-trail")

    with mujoco.viewer.launch_passive(
        world.model, world.data, key_callback=_make_key_callback(state, world)
    ) as viewer:
        last_hud = 0.0
        while viewer.is_running():
            step_start = time.time()

            if state.follow_camera:
                viewer.cam.trackbodyid = world.model.body(world.root_body_name).id
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

            if not state.paused or state.single_step:
                mujoco.mj_step(world.model, world.data)
                state.single_step = False

            with viewer.lock():
                _update_overlays(viewer, world, state, lidar_sensor)
            viewer.sync()

            if time.time() - last_hud > 1.0 / hud_hz:
                pos, _ = world.robot_pose()
                lin_vel = world.data.qvel[world.root_joint_qvel_adr : world.root_joint_qvel_adr + 3]
                print(
                    f"t={world.data.time:6.2f}s  pos=({pos[0]:+.2f},{pos[1]:+.2f})  "
                    f"vel={np.linalg.norm(lin_vel):.2f} m/s",
                    end="\r",
                )
                last_hud = time.time()

            dt_left = world.model.opt.timestep / max(speed, 1e-3) - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)
