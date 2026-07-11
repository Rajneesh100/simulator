"""Isaac-Sim-style interactive viewer, built on MuJoCo's native "Simulate" GUI.

Orbit/pan/zoom mouse camera, the collapsible side panels, and rendering toggles come
free from mujoco.viewer (Tab opens the panel, F1 shows native help). This module drives
the step loop and layers the Isaac-feel chrome on top via ui/overlay.py: dark-themed
world (core/theme.py), on-screen HUD (sim time / speed / RTF / physics rate / robot
pose+vel), key-toggleable debug overlays (lidar rays, contacts, trajectory trail, goal
beacon), camera-follow mode, runtime sim-speed control, and a screenshot key.

Pacing: physics advances by a wall-clock budget (speed-scaled) each UI frame, so the UI
stays ~60 fps regardless of sim speed, and the HUD's real-time factor is honest.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from robosim.core.world import World, build_world
from robosim.ui import overlay

UI_FRAME_DT = 1.0 / 60.0
MAX_SIM_CHUNK = 0.1  # max seconds of sim time to run per UI frame (guards long stalls)
HUD_PERIOD = 0.1  # seconds between HUD text refreshes


def _make_key_callback(state: overlay.OverlayState, world: World):
    def key_callback(keycode):
        overlay.handle_common_key(state, keycode, world=world)

    return key_callback


def _update_user_scn(viewer, world: World, state: overlay.OverlayState, lidar) -> None:
    scn = viewer.user_scn
    scn.ngeom = 0
    pos, _ = world.robot_pose()

    if state.show_trail:
        overlay.push_trail(state, pos)
        overlay.draw_trail(scn, state.trail)

    if state.show_goal:
        goal = world.scene.task_info().get("goal")
        if goal is not None:
            overlay.draw_goal(scn, goal)

    if state.show_lidar and lidar is not None:
        chassis_id = world.model.body(world.root_body_name).id
        origin, dirs, ranges_m = lidar.debug_rays(
            world.model, world.data, exclude_body_id=chassis_id
        )
        overlay.draw_lidar(scn, origin, dirs, ranges_m, lidar.max_range)


def run_view(
    scene_arg: str = "flat",
    robot_arg: str = "diffdrive",
    speed: float = 1.0,
    debug_lidar: bool = False,
) -> None:
    world = build_world(scene_arg, robot_arg)
    state = overlay.OverlayState(speed=max(0.1, min(5.0, speed)), show_lidar=debug_lidar)
    timer = overlay.HudTimer()

    from robosim.sensors.lidar import Lidar

    lidar = Lidar()
    dt = world.model.opt.timestep

    print("robosim view — mouse: orbit/pan/zoom · Tab: native panel · F1: native help")
    print("keys: space pause · . step · r reset · -/= speed · c follow · "
          "l lidar · x contacts · t trail · g goal · p screenshot · h hud", flush=True)

    with mujoco.viewer.launch_passive(
        world.model, world.data, key_callback=_make_key_callback(state, world)
    ) as viewer:
        overlay.init_camera(viewer.cam, world.spawn_pos)
        last_wall = time.perf_counter()
        last_hud = 0.0
        sim_budget = 0.0

        while viewer.is_running():
            frame_start = time.perf_counter()
            wall_dt = frame_start - last_wall
            last_wall = frame_start

            steps = 0
            if state.single_step:
                steps = 1
                state.single_step = False
                sim_budget = 0.0
            elif not state.paused:
                sim_budget = min(sim_budget + wall_dt * state.speed, MAX_SIM_CHUNK)
                steps = int(sim_budget / dt)
                sim_budget -= steps * dt
            for _ in range(steps):
                mujoco.mj_step(world.model, world.data)
            timer.update(world.data.time, steps)

            with viewer.lock():
                overlay.apply_camera(viewer.cam, world.model, world.root_body_name, state)
                overlay.apply_vis_flags(viewer.opt, state)
                _update_user_scn(viewer, world, state, lidar)

            if frame_start - last_hud > HUD_PERIOD:
                last_hud = frame_start
                if state.show_hud:
                    pos, quat = world.robot_pose()
                    qvel = world.data.qvel
                    adr = world.root_joint_qvel_adr
                    speed_mps = float((qvel[adr] ** 2 + qvel[adr + 1] ** 2) ** 0.5)
                    texts = overlay.hud_status_texts(
                        "view",
                        world.data.time,
                        state,
                        timer,
                        pos,
                        overlay.quat_to_yaw_deg(quat),
                        speed_mps,
                    )
                    texts.append(overlay.overlay_toggle_line(state))
                    texts.append(
                        (
                            overlay.FONT_HINTS,
                            overlay.GRID.mjGRID_TOPRIGHT,
                            "scene\nrobot",
                            f"{scene_arg}\n{robot_arg}",
                        )
                    )
                    overlay.update_hud(viewer, texts)
                else:
                    overlay.update_hud(viewer, [])

            viewer.sync()

            if state.screenshot_requested:
                state.screenshot_requested = False
                try:
                    path = overlay.save_screenshot(world.model, world.data, viewer.cam)
                    print(f"\n[view] screenshot saved: {path}")
                except Exception as e:  # GL context quirks shouldn't kill the session
                    print(f"\n[view] screenshot failed: {e}")

            elapsed = time.perf_counter() - frame_start
            if elapsed < UI_FRAME_DT:
                time.sleep(UI_FRAME_DT - elapsed)
