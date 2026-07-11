"""WASD keyboard driving in the viewer, with optional trajectory recording.

MuJoCo's key_callback only reports key-press/repeat events (no key-up), so this models
teleop as tap-to-set commands rather than continuous hold-to-drive: W/S set forward/back
speed, A/D set turning rate, space stops. Simple and adequate for recording demonstration
trajectories for later behavior-cloning (idea.md M5) — noted in CLAUDE.md decision log.
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

SPEED_STEP = 0.25  # fraction of max, added/removed per keypress


class TeleopState:
    def __init__(self):
        self.v = 0.0
        self.omega = 0.0
        self.recording: list[dict] = []
        self.record_enabled = False


def _make_key_callback(state: TeleopState):
    def key_callback(keycode):
        char = chr(keycode) if 0 < keycode < 256 else ""
        c = char.lower()
        if c == "w":
            state.v = min(1.0, state.v + SPEED_STEP)
        elif c == "s":
            state.v = max(-1.0, state.v - SPEED_STEP)
        elif c == "a":
            state.omega = min(1.0, state.omega + SPEED_STEP)
        elif c == "d":
            state.omega = max(-1.0, state.omega - SPEED_STEP)
        elif char == " ":
            state.v = 0.0
            state.omega = 0.0

    return key_callback


def run_teleop(
    scene_arg: str = "flat",
    robot_arg: str = "diffdrive",
    record_path: str | None = None,
) -> None:
    world = build_world(scene_arg, robot_arg)
    lidar = Lidar()
    chassis_id = world.model.body(world.root_body_name).id
    state = TeleopState()
    state.record_enabled = record_path is not None
    prev_action = np.zeros(2, dtype=np.float32)

    print("Teleop controls: [w/s]=forward/back  [a/d]=turn left/right  [space]=stop")
    if state.record_enabled:
        print(f"Recording trajectory to {record_path}.npz")

    with mujoco.viewer.launch_passive(
        world.model, world.data, key_callback=_make_key_callback(state)
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()
            action = np.array([state.v, state.omega], dtype=np.float32)
            ctrl_map = world.robot.action_to_ctrl(action)
            for name, value in ctrl_map.items():
                world.data.actuator(name).ctrl[0] = value
            mujoco.mj_step(world.model, world.data)

            if state.record_enabled:
                pos, quat = world.robot_pose()
                proprio = world.robot.read_proprio(world.model, world.data)
                ranges = lidar.sense(world.model, world.data, exclude_body_id=chassis_id)
                goal = world.scene.task_info().get("goal", (0.0, 0.0))
                obs = build_obs(ranges, pos[:2], quat, goal, proprio, prev_action)
                state.recording.append({"obs": obs, "action": action.copy(), "time": world.data.time})
            prev_action = action

            viewer.sync()
            dt_left = world.model.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)

    if state.record_enabled and state.recording:
        out = Path(record_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        obs_arr = np.stack([r["obs"] for r in state.recording])
        act_arr = np.stack([r["action"] for r in state.recording])
        np.savez(out.with_suffix(".npz"), obs=obs_arr, actions=act_arr)
        print(f"\nSaved {len(state.recording)} steps to {out.with_suffix('.npz')}")
