"""Parametric stairs: `--scene stairs:height=0.08,depth=0.25,count=8,width=1.0`."""

from __future__ import annotations

import mujoco

from robosim.core.registry import register_scene
from robosim.scenes.base import Scene

START_X = 1.0


@register_scene("stairs")
class StairsScene(Scene):
    def __init__(self, **params: str):
        super().__init__(**params)
        self.height = float(params.get("height", 0.08))
        self.depth = float(params.get("depth", 0.25))
        self.count = int(params.get("count", 8))
        self.width = float(params.get("width", 1.0))

    def build(self, spec) -> None:
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, 10, 0.1],
            rgba=[0.5, 0.5, 0.55, 1.0],
        )
        for i in range(self.count):
            top_height = self.height * (i + 1)
            center_x = START_X + self.depth * i + self.depth / 2
            spec.worldbody.add_geom(
                name=f"step_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[center_x, 0, top_height / 2],
                size=[self.depth / 2, self.width / 2, top_height / 2],
                rgba=[0.6, 0.6, 0.65, 1.0],
            )
        goal_x = START_X + self.depth * self.count + 0.75
        goal_z = self.height * self.count
        spec.worldbody.add_geom(
            name="goal_marker",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[goal_x, 0, goal_z + 0.01],
            size=[0.15, 0.01, 0.0],
            rgba=[0.1, 0.9, 0.2, 0.5],
            contype=0,
            conaffinity=0,
        )
        self._goal = (goal_x, 0.0)

    def spawn_pose(self) -> tuple[tuple[float, float, float], float]:
        return (0.0, 0.0, 0.0), 0.0

    def task_info(self) -> dict:
        return {"goal": self._goal, "top_height": self.height * self.count}
