"""Closed stadium racing circuit: two straights + two 180-degree turns, walled on both
sides, with optional static obstacles on the road and scripted opponent cars lapping it.

`--scene racetrack:straight=5.0,radius=2.5,road_width=2.6,obstacles=6,opponents=2,seed=0`

The centerline is a waypoint polyline (task_info()["waypoints"]) that the race task uses
for progress/lap accounting and that opponent movers follow. Walls are physical (crashing
into them is a collision); the road surface is a visual-only strip so "off the racing
line" is judged by the task, not by contacts.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robosim.core.registry import register_scene
from robosim.scenes.base import Scene

WALL_HALF_THICK = 0.04
ROAD_RGBA = [0.13, 0.13, 0.16, 1.0]
OPPONENT_RGBA = [
    [0.15, 0.35, 0.9, 1.0],
    [0.95, 0.75, 0.1, 1.0],
    [0.2, 0.8, 0.4, 1.0],
    [0.7, 0.2, 0.8, 1.0],
]


def _stadium_centerline(straight: float, radius: float, n: int) -> np.ndarray:
    """N points, uniform in arclength, CCW around the stadium. s=0 is the start of the
    bottom straight at (-straight, -radius), heading +x."""
    total = 4 * straight + 2 * np.pi * radius
    pts = []
    for s in np.linspace(0.0, total, n, endpoint=False):
        if s < 2 * straight:  # bottom straight, heading +x
            pts.append((-straight + s, -radius))
        elif s < 2 * straight + np.pi * radius:  # right turn
            a = (s - 2 * straight) / radius  # 0..pi
            pts.append((straight + radius * np.sin(a), -radius * np.cos(a)))
        elif s < 4 * straight + np.pi * radius:  # top straight, heading -x
            d = s - (2 * straight + np.pi * radius)
            pts.append((straight - d, radius))
        else:  # left turn
            a = (s - (4 * straight + np.pi * radius)) / radius
            pts.append((-straight - radius * np.sin(a), radius * np.cos(a)))
    return np.array(pts)


def _add_strip(spec, name: str, a, b, half_width: float, half_height: float,
               z: float, rgba, collide: bool, overlap: float = 1.25) -> None:
    """One box segment covering the polyline edge a->b (with overlap so joints seal)."""
    mid = (a + b) / 2.0
    d = b - a
    seg_half = float(np.hypot(d[0], d[1])) / 2.0
    yaw_deg = float(np.degrees(np.arctan2(d[1], d[0])))
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(mid[0]), float(mid[1]), z],
        size=[seg_half * overlap, half_width, half_height],
        euler=[0, 0, yaw_deg],
        rgba=rgba,
        contype=1 if collide else 0,
        conaffinity=1 if collide else 0,
    )


@register_scene("racetrack")
class RacetrackScene(Scene):
    def __init__(self, **params: str):
        super().__init__(**params)
        self.straight = float(params.get("straight", 5.0))
        self.radius = float(params.get("radius", 2.5))
        self.road_width = float(params.get("road_width", 2.6))
        self.wall_height = float(params.get("wall_height", 0.35))
        self.obstacles = int(params.get("obstacles", 6))
        self.opponents = int(params.get("opponents", 0))
        self.opponent_speed = float(params.get("opponent_speed", 1.2))
        self.seed = int(params.get("seed", 0))
        self.n_wp = int(params.get("n_wp", 96))
        self._movers: list[dict] = []

        self._center = _stadium_centerline(self.straight, self.radius, self.n_wp)
        self._track_length = 4 * self.straight + 2 * np.pi * self.radius

    def _offset_line(self, offset: float) -> np.ndarray:
        """Centerline shifted laterally by `offset` (positive = left of travel)."""
        pts = self._center
        nxt = np.roll(pts, -1, axis=0)
        d = nxt - pts
        d /= np.hypot(d[:, 0], d[:, 1])[:, None]
        normals = np.stack([-d[:, 1], d[:, 0]], axis=1)  # left normal
        return pts + offset * normals

    def build(self, spec) -> None:
        extent = self.straight + self.radius + self.road_width + 2.0
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[extent, extent, 0.1],
            rgba=[0.35, 0.42, 0.35, 1.0],
        )

        half_road = self.road_width / 2.0
        road = self._center
        inner = self._offset_line(+half_road + WALL_HALF_THICK)
        outer = self._offset_line(-(half_road + WALL_HALF_THICK))
        n = len(road)
        for i in range(n):
            j = (i + 1) % n
            # alternate road-strip heights by a hair so overlapping segments don't z-fight
            _add_strip(spec, f"road_{i}", road[i], road[j], half_road, 0.002,
                       0.004 + (i % 2) * 0.0008, ROAD_RGBA, collide=False)
            _add_strip(spec, f"wall_inner_{i}", inner[i], inner[j], WALL_HALF_THICK,
                       self.wall_height / 2, self.wall_height / 2,
                       [0.75, 0.16, 0.16, 1.0], collide=True)
            _add_strip(spec, f"wall_outer_{i}", outer[i], outer[j], WALL_HALF_THICK,
                       self.wall_height / 2, self.wall_height / 2,
                       [0.9, 0.9, 0.9, 1.0], collide=True)

        # start/finish line at s=0, perpendicular white stripe
        spawn = self._center[0]
        d = self._center[1] - self._center[0]
        yaw_deg = float(np.degrees(np.arctan2(d[1], d[0])))
        spec.worldbody.add_geom(
            name="start_line",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[float(spawn[0]), float(spawn[1]), 0.006],
            size=[0.08, half_road, 0.005],
            euler=[0, 0, yaw_deg],
            rgba=[0.95, 0.95, 0.95, 1.0],
            contype=0,
            conaffinity=0,
        )

        self._build_obstacles(spec)
        if self.opponents:
            self._build_opponents(spec)

    def _build_obstacles(self, spec) -> None:
        rng = np.random.default_rng(self.seed)
        half_road = self.road_width / 2.0
        for i in range(self.obstacles):
            # keep the first stretch after the start clear
            s_frac = rng.uniform(0.12, 0.95)
            idx = int(s_frac * self.n_wp)
            lateral = rng.uniform(-(half_road - 0.45), half_road - 0.45)
            base = self._offset_line(lateral)[idx]
            is_box = rng.random() < 0.5
            size = rng.uniform(0.10, 0.18)
            height = rng.uniform(0.25, 0.4)
            common = dict(
                name=f"obs_{i}",
                pos=[float(base[0]), float(base[1]), height / 2],
                rgba=[0.85, 0.55, 0.1, 1.0],
            )
            if is_box:
                spec.worldbody.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[size, size, height / 2],
                    **common,
                )
            else:
                spec.worldbody.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[size, height / 2, 0.0],
                    **common,
                )

    def _build_opponents(self, spec) -> None:
        rng = np.random.default_rng(self.seed + 1)
        lap_time = self._track_length / max(self.opponent_speed, 0.1)
        for i in range(self.opponents):
            # each opponent holds its own lane so they don't form one solid wall
            lane = rng.uniform(-0.45, 0.45)
            lane_pts = self._offset_line(lane)
            name = f"opponent_{i}"
            body = spec.worldbody.add_body(name=name, pos=[0.0, 0.0, 0.08], mocap=True)
            body.add_geom(
                name=f"{name}_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.2, 0.1, 0.06],
                rgba=OPPONENT_RGBA[i % len(OPPONENT_RGBA)],
            )
            body.add_geom(
                name=f"{name}_cabin",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[-0.03, 0.0, 0.08],
                size=[0.09, 0.07, 0.025],
                rgba=[0.12, 0.12, 0.15, 1.0],
                contype=0,
                conaffinity=0,
            )
            self._movers.append(
                {
                    "name": name,
                    "path": "loop",
                    "waypoints": [tuple(p) for p in lane_pts],
                    "period": lap_time * rng.uniform(0.9, 1.15),
                    "phase": (i + 1) / (self.opponents + 1),
                    "z": 0.08,
                }
            )

    def spawn_pose(self) -> tuple[tuple[float, float, float], float]:
        x, y = self._center[0]
        d = self._center[1] - self._center[0]
        return (float(x), float(y), 0.0), float(np.arctan2(d[1], d[0]))

    def task_info(self) -> dict:
        return {
            "waypoints": [tuple(p) for p in self._center],
            "track_length": float(self._track_length),
            "road_half_width": self.road_width / 2.0,
            "goal": tuple(self._center[0]),  # start/finish, for the viewer beacon
        }

    def moving_bodies(self) -> list[dict]:
        return self._movers
