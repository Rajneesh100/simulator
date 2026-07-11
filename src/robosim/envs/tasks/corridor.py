"""T3: corridor/maze — goal not always visible, memory helps (P3/P4 vs P1/P2).

Use with `--scene obstacles:maze=1,seed=N`; reward is identical to goal_reach (the
memory requirement comes entirely from the scene's occluding walls, not the reward).
"""

from __future__ import annotations

from robosim.core.registry import register_task
from robosim.envs.tasks.goal_reach import GoalReachTask


@register_task("corridor")
class CorridorTask(GoalReachTask):
    pass
