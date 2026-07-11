"""T7: obstacle_nav + scripted movers. Use with `--scene obstacles:moving=3,seed=N`.

Same reward shape as obstacle_nav (T2 + dynamic avoidance comes from the scene's
scripted mocap movers, not the reward function).
"""

from __future__ import annotations

from robosim.core.registry import register_task
from robosim.envs.tasks.obstacle_nav import ObstacleNavTask


@register_task("moving_obstacle")
class MovingObstacleTask(ObstacleNavTask):
    pass
