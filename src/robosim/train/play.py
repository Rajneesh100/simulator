"""Load a trained checkpoint and run it in the interactive viewer.

Same Isaac-feel chrome as `robosim view` (ui/overlay.py) plus a live reward readout:
per-step reward, episode return, distance-to-goal, and success count. Space pauses the
policy, '.' advances one control step, -/= changes playback speed, c toggles follow-cam.
"""

from __future__ import annotations

import time

import mujoco.viewer
import numpy as np

from robosim.envs.base_env import RoboSimEnv
from robosim.train.run_config import env_kwargs_from_run
from robosim.ui import overlay

HUD_PERIOD = 0.1


def _load_model(checkpoint: str):
    try:
        from stable_baselines3 import PPO

        return PPO.load(checkpoint, device="cpu"), False
    except Exception:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(checkpoint, device="cpu"), True


def run_play(
    checkpoint: str,
    scene: str = "flat",
    robot: str = "diffdrive",
    task: str = "goal_reach",
    episodes: int = 5,
) -> None:
    model, is_recurrent = _load_model(checkpoint)
    env = RoboSimEnv(scene=scene, robot=robot, task=task, **env_kwargs_from_run(checkpoint))
    world = env.world
    state = overlay.OverlayState()
    timer = overlay.HudTimer()
    control_dt = world.model.opt.timestep * env._steps_per_action

    print("robosim play — space pause · . step · -/= speed · c follow · "
          "1 lidar · x contacts · t trail · 2 goal · p screenshot · h hud", flush=True)

    def key_callback(keycode):
        overlay.handle_common_key(state, keycode, world=None)  # no mid-episode reset

    successes = 0

    with mujoco.viewer.launch_passive(
        world.model,
        world.data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        overlay.init_camera(viewer.cam, world.spawn_pos)
        last_hud = 0.0

        for ep in range(episodes):
            if not viewer.is_running():
                break
            obs, info = env.reset(seed=ep)
            state.trail.clear()
            lstm_state = None
            episode_start = np.array([True])
            done = False
            total_reward = 0.0
            reward = 0.0

            while not done and viewer.is_running():
                frame_start = time.perf_counter()

                stepped = False
                if not state.paused or state.single_step:
                    state.single_step = False
                    if is_recurrent:
                        action, lstm_state = model.predict(
                            obs, state=lstm_state, episode_start=episode_start,
                            deterministic=True,
                        )
                        episode_start = np.array([False])
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                    stepped = True
                timer.update(world.data.time, env._steps_per_action if stepped else 0)

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
                        goal = env._task_state.get("goal")
                        if goal is not None:
                            overlay.draw_goal(scn, goal)
                    if state.show_lidar:
                        origin, dirs, ranges_m = env.lidar.debug_rays(
                            world.model, world.data, exclude_body_id=env._chassis_id
                        )
                        overlay.draw_lidar(scn, origin, dirs, ranges_m, env.lidar.max_range)

                if frame_start - last_hud > HUD_PERIOD:
                    last_hud = frame_start
                    if state.show_hud:
                        pos, quat = world.robot_pose()
                        qvel = world.data.qvel
                        adr = world.root_joint_qvel_adr
                        speed_mps = float((qvel[adr] ** 2 + qvel[adr + 1] ** 2) ** 0.5)
                        texts = overlay.hud_status_texts(
                            "play",
                            world.data.time,
                            state,
                            timer,
                            pos,
                            overlay.quat_to_yaw_deg(quat),
                            speed_mps,
                        )
                        dist = info.get("distance_to_goal", float("nan"))
                        texts.append(
                            (
                                overlay.FONT_HUD,
                                overlay.GRID.mjGRID_TOPRIGHT,
                                "episode\nreward\nreturn\ndist\nsuccess",
                                f"{ep + 1}/{episodes}\n{reward:+.3f}\n"
                                f"{total_reward:+.2f}\n{dist:.2f} m\n"
                                f"{successes}/{ep}",
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
                        print(f"\n[play] screenshot saved: {path}")
                    except Exception as e:
                        print(f"\n[play] screenshot failed: {e}")

                target_dt = control_dt / state.speed if stepped else 1.0 / 60.0
                elapsed = time.perf_counter() - frame_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

            if info.get("success", False):
                successes += 1
            print(
                f"Episode {ep}: return={total_reward:.2f} "
                f"success={info.get('success', False)}",
                flush=True,
            )
        print(f"Done: {successes}/{episodes} successful episodes", flush=True)
