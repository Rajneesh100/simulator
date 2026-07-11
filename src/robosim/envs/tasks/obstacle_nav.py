"""T2: navigate to a goal through an obstacle field. goal_reach reward + clearance shaping."""

from __future__ import annotations

from typing import Any

import numpy as np

from robosim.core.registry import register_task
from robosim.envs.tasks.goal_reach import COLLISION_PENALTY, SUCCESS_BONUS, SUCCESS_RADIUS, TIME_PENALTY
from robosim.envs.tasks.base import Task

CLEARANCE_THRESHOLD = 0.3  # normalized lidar range below which we penalize closeness
CLEARANCE_WEIGHT = 0.05


@register_task("obstacle_nav")
class ObstacleNavTask(Task):
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
        info: dict[str, Any] = {"distance_to_goal": distance}

        min_range = task_state.get("last_min_lidar")
        if min_range is not None and min_range < CLEARANCE_THRESHOLD:
            reward -= CLEARANCE_WEIGHT * (CLEARANCE_THRESHOLD - min_range) / CLEARANCE_THRESHOLD

        terminated = False
        if collided:
            reward -= COLLISION_PENALTY
        if distance < SUCCESS_RADIUS:
            reward += SUCCESS_BONUS
            terminated = True
            info["success"] = True
        return reward, terminated, info
