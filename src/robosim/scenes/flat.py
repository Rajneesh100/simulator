"""Empty flat ground — the default scene."""

from __future__ import annotations

import mujoco

from robosim.core.registry import register_scene
from robosim.scenes.base import Scene


@register_scene("flat")
class FlatScene(Scene):
    def __init__(self, **params: str):
        super().__init__(**params)
        self.size = float(params.get("size", 10.0))
        goal = params.get("goal", "3.0,0.0")
        gx, gy = (float(v) for v in goal.split(","))
        self.goal = (gx, gy)

    def build(self, spec) -> None:
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[self.size, self.size, 0.1],
            rgba=[0.5, 0.5, 0.55, 1.0],
        )
        spec.worldbody.add_geom(
            name="goal_marker",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[self.goal[0], self.goal[1], 0.01],
            size=[0.15, 0.01, 0.0],
            rgba=[0.1, 0.9, 0.2, 0.5],
            contype=0,
            conaffinity=0,
        )

    def spawn_pose(self) -> tuple[tuple[float, float, float], float]:
        """(x, y, terrain_z) — terrain_z is ground height, not robot origin height."""
        return (0.0, 0.0, 0.0), 0.0

    def task_info(self) -> dict:
        return {"goal": self.goal, "bounds": (-self.size, self.size, -self.size, self.size)}
