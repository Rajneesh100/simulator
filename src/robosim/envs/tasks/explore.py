"""T6: wander without collisions. reward = new-area coverage - collision penalty.

Coverage is tracked via a coarse grid of visited cells (idea.md: "wander behavior for
the real robot" — no fixed goal, so goal-based obs terms just point at the origin).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosim.core.registry import register_task
from robosim.envs.tasks.base import Task

CELL_SIZE = 0.5
COLLISION_PENALTY = 1.0
NEW_CELL_REWARD = 1.0
TIME_PENALTY = 0.01


@register_task("explore")
class ExploreTask(Task):
    def reset(self, world, rng: np.random.Generator) -> dict[str, Any]:
        info = dict(world.scene.task_info())
        info.setdefault("goal", (0.0, 0.0))
        info["visited_cells"] = set()
        return info

    def _cell(self, pos) -> tuple[int, int]:
        return (int(np.floor(pos[0] / CELL_SIZE)), int(np.floor(pos[1] / CELL_SIZE)))

    def reward(self, world, task_state, action, collided) -> tuple[float, bool, dict[str, Any]]:
        pos, _ = world.robot_pose()
        cell = self._cell(pos)
        visited = task_state["visited_cells"]
        reward = -TIME_PENALTY
        is_new = cell not in visited
        if is_new:
            visited.add(cell)
            reward += NEW_CELL_REWARD
        if collided:
            reward -= COLLISION_PENALTY
        return reward, False, {"new_cell": is_new, "cells_visited": len(visited)}
