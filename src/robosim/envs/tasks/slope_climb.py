"""T4: climb a slope. reward = height progress + upright bonus - collision/time penalty.

Use with `--scene slope:angle=15`. Traction/friction limits and slope-angle domain
randomization determine how far the diffdrive robot can climb before wheels slip.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosim.core.registry import register_task
from robosim.envs.tasks.base import Task

SUCCESS_XY_RADIUS = 0.4
TIME_PENALTY = 0.01
COLLISION_PENALTY = 1.0
SUCCESS_BONUS = 10.0
UPRIGHT_WEIGHT = 0.02  # penalty per radian of |pitch|+|roll| beyond a small margin
UPRIGHT_MARGIN = 0.15  # radians (~8.6deg) of tilt tolerated before penalizing


@register_task("slope_climb")
class SlopeClimbTask(Task):
    def reset(self, world, rng: np.random.Generator) -> dict[str, Any]:
        info = dict(world.scene.task_info())
        pos, _ = world.robot_pose()
        info["best_height"] = float(pos[2])
        return info

    def reward(self, world, task_state, action, collided) -> tuple[float, bool, dict[str, Any]]:
        pos, _ = world.robot_pose()
        goal = task_state["goal"]
        proprio = world.robot.read_proprio(world.model, world.data)
        pitch, roll = float(proprio[2]), float(proprio[3])

        height_gain = max(0.0, float(pos[2]) - task_state["best_height"])
        task_state["best_height"] = max(task_state["best_height"], float(pos[2]))

        tilt = abs(pitch) + abs(roll)
        upright_penalty = UPRIGHT_WEIGHT * max(0.0, tilt - UPRIGHT_MARGIN)

        reward = height_gain - upright_penalty - TIME_PENALTY
        terminated = False
        distance_xy = float(np.hypot(goal[0] - pos[0], goal[1] - pos[1]))
        info: dict[str, Any] = {"height": float(pos[2]), "distance_to_goal": distance_xy}

        if collided:
            reward -= COLLISION_PENALTY
        if distance_xy < SUCCESS_XY_RADIUS:
            reward += SUCCESS_BONUS
            terminated = True
            info["success"] = True
        return reward, terminated, info
