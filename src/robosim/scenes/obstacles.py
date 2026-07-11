"""Randomized obstacle field (seedable), with an optional 'maze' corridor layout.

`--scene obstacles:count=12,area=6.0,seed=0,goal_distance=5.0`
`--scene obstacles:maze=1,seed=3` for a zigzag corridor (T3: goal not always visible).

Supports curriculum difficulty via `count`/`area` — envs can rebuild the scene with
harder params between training stages (idea.md M5 curriculum hook).
"""

from __future__ import annotations

import numpy as np
import mujoco

from robosim.core.registry import register_scene
from robosim.scenes.base import Scene

SPAWN_CLEARANCE = 0.6
GOAL_CLEARANCE = 0.4


@register_scene("obstacles")
class ObstaclesScene(Scene):
    def __init__(self, **params: str):
        super().__init__(**params)
        self.count = int(params.get("count", 12))
        self.area = float(params.get("area", 6.0))
        self.seed = int(params.get("seed", 0))
        self.goal_distance = float(params.get("goal_distance", 5.0))
        self.maze = bool(int(params.get("maze", 0)))
        self.moving = int(params.get("moving", 0))
        self._movers: list[dict] = []

    def build(self, spec) -> None:
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[self.area, self.area, 0.1],
            rgba=[0.5, 0.5, 0.55, 1.0],
        )
        rng = np.random.default_rng(self.seed)
        self._goal = (self.goal_distance, 0.0)

        if self.maze:
            self._build_maze(spec, rng)
        else:
            self._build_scatter(spec, rng)

        if self.moving:
            self._build_movers(spec, rng)

        spec.worldbody.add_geom(
            name="goal_marker",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[self._goal[0], self._goal[1], 0.01],
            size=[0.15, 0.01, 0.0],
            rgba=[0.1, 0.9, 0.2, 0.5],
            contype=0,
            conaffinity=0,
        )

    def _build_scatter(self, spec, rng) -> None:
        placed = 0
        attempts = 0
        while placed < self.count and attempts < self.count * 20:
            attempts += 1
            x = rng.uniform(-self.area * 0.8, self.area * 0.8)
            y = rng.uniform(-self.area * 0.8, self.area * 0.8)
            if np.hypot(x, y) < SPAWN_CLEARANCE:
                continue
            if np.hypot(x - self._goal[0], y - self._goal[1]) < GOAL_CLEARANCE:
                continue
            is_box = rng.random() < 0.5
            size = rng.uniform(0.15, 0.35)
            height = rng.uniform(0.15, 0.4)
            if is_box:
                spec.worldbody.add_geom(
                    name=f"obs_{placed}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[x, y, height / 2],
                    size=[size, size, height / 2],
                    rgba=[0.7, 0.4, 0.2, 1.0],
                )
            else:
                spec.worldbody.add_geom(
                    name=f"obs_{placed}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[x, y, height / 2],
                    size=[size, height / 2, 0.0],
                    rgba=[0.3, 0.4, 0.7, 1.0],
                )
            placed += 1

    def _build_maze(self, spec, rng) -> None:
        """Zigzag corridor of wall segments; goal is occluded around at least one bend."""
        n_walls = max(4, self.count)
        corridor_width = 1.2
        segment_len = self.goal_distance / n_walls
        wall_thick = 0.15
        wall_height = 0.5
        for i in range(n_walls):
            x = (i + 1) * segment_len
            side = 1 if i % 2 == 0 else -1
            y = side * corridor_width / 2
            spec.worldbody.add_geom(
                name=f"wall_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, wall_height / 2],
                size=[wall_thick, corridor_width, wall_height / 2],
                rgba=[0.6, 0.3, 0.3, 1.0],
            )

    def _build_movers(self, spec, rng) -> None:
        """T7: scripted moving obstacles along the path to the goal, oscillating along y."""
        for i in range(self.moving):
            frac = (i + 1) / (self.moving + 1)
            x = frac * self._goal[0]
            name = f"mover_{i}"
            spec.worldbody.add_body(name=name, pos=[x, 0.0, 0.2], mocap=True).add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.15, 0.15, 0.2],
                rgba=[0.8, 0.2, 0.7, 1.0],
            )
            self._movers.append(
                {
                    "name": name,
                    "axis": "y",
                    "amplitude": rng.uniform(0.6, 1.2),
                    "period": rng.uniform(3.0, 6.0),
                    "base_pos": (x, 0.0, 0.2),
                }
            )

    def spawn_pose(self) -> tuple[tuple[float, float, float], float]:
        return (0.0, 0.0, 0.0), 0.0

    def task_info(self) -> dict:
        return {"goal": self._goal, "seed": self.seed, "maze": self.maze}

    def moving_bodies(self) -> list[dict]:
        return self._movers
