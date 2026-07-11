"""Noise-based heightfield terrain (seedable), with a flat safe zone at the spawn point.

`--scene rough_terrain:roughness=0.15,area=4.0,seed=0,resolution=40`
"""

from __future__ import annotations

import numpy as np
import mujoco

from robosim.core.registry import register_scene
from robosim.scenes.base import Scene

SAFE_RADIUS_FRAC = 0.12  # fraction of grid radius kept flat around the spawn point


@register_scene("rough_terrain")
class RoughTerrainScene(Scene):
    def __init__(self, **params: str):
        super().__init__(**params)
        self.roughness = float(params.get("roughness", 0.15))
        self.area = float(params.get("area", 4.0))
        self.seed = int(params.get("seed", 0))
        self.resolution = int(params.get("resolution", 40))
        goal = params.get("goal", f"{self.area * 0.7},0.0")
        gx, gy = (float(v) for v in goal.split(","))
        self._goal = (gx, gy)

    def _generate_heightfield(self) -> np.ndarray:
        n = self.resolution
        rng = np.random.default_rng(self.seed)
        raw = rng.uniform(0.0, 1.0, size=(n, n))
        # simple box-blur smoothing pass so bumps aren't single-cell spikes
        smoothed = raw.copy()
        for _ in range(3):
            smoothed = (
                smoothed
                + np.roll(smoothed, 1, axis=0)
                + np.roll(smoothed, -1, axis=0)
                + np.roll(smoothed, 1, axis=1)
                + np.roll(smoothed, -1, axis=1)
            ) / 5.0
        smoothed -= smoothed.min()
        if smoothed.max() > 0:
            smoothed /= smoothed.max()

        yy, xx = np.mgrid[0:n, 0:n]
        cx, cy = n / 2, n / 2
        dist = np.hypot(xx - cx, yy - cy) / (n / 2)
        safe_mask = dist < SAFE_RADIUS_FRAC
        smoothed[safe_mask] = 0.0
        return smoothed.astype(np.float64)

    def build(self, spec) -> None:
        data = self._generate_heightfield().flatten()
        hf = spec.add_hfield(
            name="terrain_hf",
            nrow=self.resolution,
            ncol=self.resolution,
            size=[self.area, self.area, self.roughness, 0.05],
            userdata=data,
        )
        spec.worldbody.add_geom(
            name="terrain",
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname="terrain_hf",
            rgba=[0.5, 0.45, 0.35, 1.0],
        )
        spec.worldbody.add_geom(
            name="goal_marker",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[self._goal[0], self._goal[1], 0.05],
            size=[0.15, 0.05, 0.0],
            rgba=[0.1, 0.9, 0.2, 0.5],
            contype=0,
            conaffinity=0,
        )

    def spawn_pose(self) -> tuple[tuple[float, float, float], float]:
        return (0.0, 0.0, 0.0), 0.0

    def task_info(self) -> dict:
        return {"goal": self._goal, "roughness": self.roughness}
