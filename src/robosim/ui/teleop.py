"""WASD keyboard driving in the viewer, with optional trajectory recording.

MuJoCo's key_callback only reports key-press/repeat events (no key-up), so teleop is
tap-to-set: W/S bump forward/back speed, A/D bump turn rate, space stops. The full
Isaac-feel chrome from ui/overlay.py applies here too (HUD, follow-cam, debug overlays,
screenshot); the top-right HUD box shows the live drive command and a REC counter when
`--record` is on. Space is "stop", not "pause" — pausing mid-drive isn't useful and the
key is muscle-memory for braking.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from robosim.core.world import build_world
from robosim.envs.obs import build_obs
from robosim.sensors.lidar import Lidar
from robosim.ui import overlay

SPEED_STEP = 0.25  # fraction of max, added/removed per keypress
UI_FRAME_DT = 1.0 / 60.0
MAX_SIM_CHUNK = 0.05
HUD_PERIOD = 0.1


class DriveState:
    def __init__(self):
        self.v = 0.0
        self.omega = 0.0


def _make_key_callback(drive: DriveState, state: overlay.OverlayState, world):
    def key_callback(keycode):
        char = chr(keycode).lower() if 0 < keycode < 256 else ""
        if char == "w":
            drive.v = min(1.0, drive.v + SPEED_STEP)
        elif char == "s":
            drive.v = max(-1.0, drive.v - SPEED_STEP)
        elif char == "a":
            drive.omega = min(1.0, drive.omega + SPEED_STEP)
        elif char == "d":
            drive.omega = max(-1.0, drive.omega - SPEED_STEP)
        elif char == " ":
            drive.v = 0.0
            drive.omega = 0.0
        elif char == "r":
            world.reset()
            drive.v = 0.0
            drive.omega = 0.0
            state.trail.clear()
        elif char != ".":  # no pause/step in teleop — space is the brake
            overlay.handle_common_key(state, keycode, world=None)

    return key_callback


def run_teleop(
    scene_arg: str = "flat",
    robot_arg: str = "diffdrive",
    record_path: str | None = None,
) -> None:
    world = build_world(scene_arg, robot_arg)
    drive = DriveState()
    state = overlay.OverlayState()
    timer = overlay.HudTimer()
    lidar = Lidar()
    chassis_id = world.model.body(world.root_body_name).id
    dt = world.model.opt.timestep

    recording: list[dict] = []
    record_enabled = record_path is not None
    prev_action = np.zeros(2, dtype=np.float32)

    print("robosim teleop — w/s: fwd/back · a/d: turn · space: stop · r: reset")
    print("keys: -/= speed · c follow · l lidar · x contacts · t trail · g goal · "
          "p screenshot · h hud", flush=True)
    if record_enabled:
        print(f"Recording trajectory to {Path(record_path).with_suffix('.npz')}", flush=True)

    with mujoco.viewer.launch_passive(
        world.model, world.data, key_callback=_make_key_callback(drive, state, world)
    ) as viewer:
        overlay.init_camera(viewer.cam, world.spawn_pos)
        last_wall = time.perf_counter()
        last_hud = 0.0
        sim_budget = 0.0

        while viewer.is_running():
            frame_start = time.perf_counter()
            wall_dt = frame_start - last_wall
            last_wall = frame_start

            action = np.array([drive.v, drive.omega], dtype=np.float32)
            ctrl_map = world.robot.action_to_ctrl(action)
            for name, value in ctrl_map.items():
                world.data.actuator(name).ctrl[0] = value

            sim_budget = min(sim_budget + wall_dt * state.speed, MAX_SIM_CHUNK)
            steps = int(sim_budget / dt)
            sim_budget -= steps * dt
            for _ in range(steps):
                mujoco.mj_step(world.model, world.data)
                if record_enabled:
                    pos, quat = world.robot_pose()
                    proprio = world.robot.read_proprio(world.model, world.data)
                    ranges = lidar.sense(world.model, world.data, exclude_body_id=chassis_id)
                    goal = world.scene.task_info().get("goal", (0.0, 0.0))
                    obs = build_obs(ranges, pos[:2], quat, goal, proprio, prev_action)
                    recording.append({"obs": obs, "action": action.copy()})
                    prev_action = action
            timer.update(world.data.time, steps)

            with viewer.lock():
                overlay.apply_camera(viewer.cam, world.model, world.root_body_name, state)
                overlay.apply_vis_flags(viewer.opt, state)
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
                if state.show_lidar:
                    origin, dirs, ranges_m = lidar.debug_rays(
                        world.model, world.data, exclude_body_id=chassis_id
                    )
                    overlay.draw_lidar(scn, origin, dirs, ranges_m, lidar.max_range)

            if frame_start - last_hud > HUD_PERIOD:
                last_hud = frame_start
                if state.show_hud:
                    pos, quat = world.robot_pose()
                    qvel = world.data.qvel
                    adr = world.root_joint_qvel_adr
                    speed_mps = float((qvel[adr] ** 2 + qvel[adr + 1] ** 2) ** 0.5)
                    texts = overlay.hud_status_texts(
                        "teleop",
                        world.data.time,
                        state,
                        timer,
                        pos,
                        overlay.quat_to_yaw_deg(quat),
                        speed_mps,
                    )
                    rec_line = (
                        f"REC {len(recording)} steps" if record_enabled else "off"
                    )
                    texts.append(
                        (
                            overlay.FONT_HUD,
                            overlay.GRID.mjGRID_TOPRIGHT,
                            "drive v\ndrive ω\nrecord",
                            f"{drive.v:+.2f}\n{drive.omega:+.2f}\n{rec_line}",
                        )
                    )
                    texts.append(overlay.overlay_toggle_line(state))
                    overlay.update_hud(viewer, texts)
                else:
                    overlay.update_hud(viewer, [])

            viewer.sync()

            if state.screenshot_requested:
                state.screenshot_requested = False
                try:
                    path = overlay.save_screenshot(world.model, world.data, viewer.cam)
                    print(f"\n[teleop] screenshot saved: {path}")
                except Exception as e:
                    print(f"\n[teleop] screenshot failed: {e}")

            elapsed = time.perf_counter() - frame_start
            if elapsed < UI_FRAME_DT:
                time.sleep(UI_FRAME_DT - elapsed)

    if record_enabled and recording:
        out = Path(record_path).with_suffix(".npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        obs_arr = np.stack([r["obs"] for r in recording])
        act_arr = np.stack([r["action"] for r in recording])
        np.savez(out, obs=obs_arr, actions=act_arr)
        print(f"\nSaved {len(recording)} steps to {out}")
