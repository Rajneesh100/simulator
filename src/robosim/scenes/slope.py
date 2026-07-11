"""Parametric ramp: `--scene slope:angle=15,length=4.0,width=1.5`."""

from __future__ import annotations

import numpy as np
import mujoco

from robosim.core.registry import register_scene
from robosim.scenes.base import Scene

START_X = 0.5
THICKNESS = 0.1


@register_scene("slope")
class SlopeScene(Scene):
    def __init__(self, **params: str):
        super().__init__(**params)
        self.angle_deg = float(params.get("angle", 15.0))
        self.length = float(params.get("length", 4.0))
        self.width = float(params.get("width", 1.5))

    def build(self, spec) -> None:
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, 10, 0.1],
            rgba=[0.5, 0.5, 0.55, 1.0],
        )
        angle_rad = np.deg2rad(self.angle_deg)
        # Ramp centered so its low edge sits at START_X, ground level.
        cx = START_X + (self.length / 2) * np.cos(angle_rad)
        cz = (self.length / 2) * np.sin(angle_rad) - THICKNESS / 2 * np.cos(angle_rad)
        spec.worldbody.add_geom(
            name="ramp",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[cx, 0, cz],
            euler=[0, -self.angle_deg, 0],
            size=[self.length / 2, self.width / 2, THICKNESS / 2],
            rgba=[0.55, 0.55, 0.6, 1.0],
        )
        top_x = START_X + self.length * np.cos(angle_rad)
        top_z = self.length * np.sin(angle_rad)
        self._goal = (top_x + 0.5, 0.0)
        spec.worldbody.add_geom(
            name="goal_marker",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[self._goal[0], self._goal[1], top_z + 0.01],
            size=[0.15, 0.01, 0.0],
            rgba=[0.1, 0.9, 0.2, 0.5],
            contype=0,
            conaffinity=0,
        )

    def spawn_pose(self) -> tuple[tuple[float, float, float], float]:
        return (0.0, 0.0, 0.0), 0.0

    def task_info(self) -> dict:
        return {"goal": self._goal, "angle_deg": self.angle_deg}
