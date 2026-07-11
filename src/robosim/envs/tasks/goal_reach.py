"""T1: navigate to a goal point. reward = progress - time - collision penalty."""

from __future__ import annotations

from typing import Any

import numpy as np

from robosim.core.registry import register_task
from robosim.envs.tasks.base import Task

SUCCESS_RADIUS = 0.25
TIME_PENALTY = 0.01
COLLISION_PENALTY = 1.0
SUCCESS_BONUS = 10.0


@register_task("goal_reach")
class GoalReachTask(Task):
    def reset(self, world, rng: np.random.Generator) -> dict[str, Any]:
        info = dict(world.scene.task_info())
        pos, _ = world.robot_pose()
        info["prev_distance"] = float(np.hypot(info["goal"][0] - pos[0], info["goal"][1] - pos[1]))
        return info

    def reward(self, world, task_state, action, collided) -> tuple[float, bool, dict[str, Any]]:
        pos, _ = world.robot_pose()
        goal = task_state["goal"]
        distance = float(np.hypot(goal[0] - pos[0], goal[1] - pos[1]))
        progress = task_state["prev_distance"] - distance
        task_state["prev_distance"] = distance

        reward = progress - TIME_PENALTY
        terminated = False
        info: dict[str, Any] = {"distance_to_goal": distance}

        if collided:
            reward -= COLLISION_PENALTY
        if distance < SUCCESS_RADIUS:
            reward += SUCCESS_BONUS
            terminated = True
            info["success"] = True
        return reward, terminated, info
