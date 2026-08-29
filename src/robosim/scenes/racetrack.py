"""Closed racing circuit, walled on both sides, with optional elevation changes, static
obstacles on the road, and scripted opponent cars lapping it.

Layouts:
- `stadium` (default): two straights + two 180-degree turns.
  `--scene racetrack:straight=5.0,radius=2.5,road_width=2.6,obstacles=6,opponents=2`
- `gp`: procedural grand-prix circuit — a seeded smooth closed loop with sweeping
  corners of varying radius, F1-style. `--scene racetrack:layout=gp,scale=12,seed=3`
  (`scale` is the mean centerline radius in meters; bigger = longer lap).

Hills: `hills=H` (meters of elevation amplitude, default 0 = flat) adds a seeded,
periodic elevation profile along the lap. With hills the road becomes a REAL physical
surface — pitched box segments the cars drive on — so climbs cost motor torque and
descents add speed; `max_grade` (default 0.12) caps the steepness. With hills=0 the
geometry is exactly the original flat track (road is a visual strip on the ground
plane), keeping older checkpoints valid.

The centerline is a waypoint polyline — (x, y, z) tuples in task_info()["waypoints"] —
that the race task uses for progress/lap accounting (projecting in xy) and that opponent
movers follow. Everything — walls, road strips, obstacles, opponent lanes, spawn — derives
from that one polyline, so a new layout is just a new centerline generator. Walls are
physical (crashing into them is a collision); staying "on the racing line" is judged by
the task, not by contacts.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robosim.core.registry import register_scene
from robosim.scenes.base import Scene

WALL_HALF_THICK = 0.04
ROAD_RGBA = [0.13, 0.13, 0.16, 1.0]
ROAD_SLAB_HALF_H = 0.06  # physical road thickness (hills mode)
OPPONENT_RGBA = [
    [0.15, 0.35, 0.9, 1.0],
    [0.95, 0.75, 0.1, 1.0],
    [0.2, 0.8, 0.4, 1.0],
    [0.7, 0.2, 0.8, 1.0],
]


def _gp_centerline(scale: float, seed: int, n: int) -> np.ndarray:
    """Seeded smooth closed circuit: radius-modulated circle (low-frequency harmonics
    only, amplitudes damped by 1/k so curvature stays drivable), resampled to n points
    uniform in arclength. Different seeds give genuinely different F1-ish circuits."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2 * np.pi, 2048, endpoint=False)
    r = np.full_like(theta, scale)
    for k in range(2, 6):
        amp = scale * rng.uniform(0.05, 0.11) * (2.0 / k)
        phase = rng.uniform(0, 2 * np.pi)
        r += amp * np.cos(k * theta + phase)
    dense = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    seg = np.diff(np.vstack([dense, dense[:1]]), axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    targets = np.linspace(0.0, total, n, endpoint=False)
    idx = np.searchsorted(cum, targets, side="right") - 1
    frac = (targets - cum[idx]) / np.maximum(seg_len[idx], 1e-12)
    nxt = (idx + 1) % len(dense)
    return dense[idx] + frac[:, None] * (dense[nxt] - dense[idx])


def _polyline_length(pts: np.ndarray) -> float:
    seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


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


def _elevation_profile(track_length: float, n: int, hills: float, max_grade: float,
                       hill_freq: int, seed: int) -> np.ndarray:
    """Periodic z(s) sampled at the n uniform-arclength waypoints: seeded sine harmonics
    k=1..hill_freq (mild 1/k^0.7 damping, so high hill_freq = rally-style bumpy), scaled
    so the peak grade stays below max_grade. Physics, not magic: elevation is bounded by
    grade x wavelength, so if the requested amplitude can't fit it is scaled down — and
    SAYS SO, with what you actually got and which knobs buy more."""
    if hills <= 0:
        return np.zeros(n)
    rng = np.random.default_rng(seed + 7)
    s = np.arange(n) * (track_length / n)
    z = np.zeros(n)
    slope = np.zeros(n)
    for k in range(1, max(1, hill_freq) + 1):
        a = hills * rng.uniform(0.5, 1.0) / k**0.7
        phase = rng.uniform(0, 2 * np.pi)
        w = 2 * np.pi * k / track_length
        z += a * np.sin(w * s + phase)
        slope += a * w * np.cos(w * s + phase)
    peak_grade = float(np.abs(slope).max())
    if peak_grade > max_grade:
        z *= max_grade / peak_grade
    got = float(z.max() - z.min())
    if got < hills * 0.9:
        print(f"[racetrack] hills={hills:g} m doesn't fit this lap (would need "
              f"{peak_grade:.0%} peak grade) -> built {got:.1f} m at max_grade={max_grade:g}. "
              f"For more: raise max_grade (rally ~0.3) or scale (longer lap), or lower hill_freq.",
              flush=True)
    return z - z.min()  # lowest point at ground level so no section sinks underground


def _seg_quat(d: np.ndarray) -> list[float]:
    """Orientation quat (wxyz) aligning a box's local +x with 3D direction d:
    yaw about world z, then pitch about the yawed y axis."""
    yaw = np.arctan2(d[1], d[0])
    pitch = -np.arcsin(np.clip(d[2] / max(np.linalg.norm(d), 1e-12), -1.0, 1.0))
    c1, s1 = np.cos(yaw / 2), np.sin(yaw / 2)
    c2, s2 = np.cos(pitch / 2), np.sin(pitch / 2)
    return [float(c1 * c2), float(-s1 * s2), float(c1 * s2), float(c2 * s1)]


def _add_strip(spec, name: str, a3, b3, half_width: float, half_height: float,
               z_center_offset: float, rgba, collide: bool, overlap: float = 1.25) -> None:
    """One box segment covering the 3D polyline edge a3->b3 (overlapped so joints seal),
    pitched to follow the elevation. z_center_offset positions the box center relative
    to the surface line (e.g. +h/2 for a wall standing on it, -h for a slab under it)."""
    a3, b3 = np.asarray(a3, dtype=np.float64), np.asarray(b3, dtype=np.float64)
    mid = (a3 + b3) / 2.0
    d = b3 - a3
    seg_half = float(np.linalg.norm(d)) / 2.0
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(mid[0]), float(mid[1]), float(mid[2] + z_center_offset)],
        size=[seg_half * overlap, half_width, half_height],
        quat=_seg_quat(d),
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
        self.layout = params.get("layout", "stadium")
        self.scale = float(params.get("scale", 9.0))
        self.hills = float(params.get("hills", 0.0))
        self.max_grade = float(params.get("max_grade", 0.12))
        self.hill_freq = int(params.get("hill_freq", 3))
        self._movers: list[dict] = []

        if self.layout == "gp":
            # waypoint density scales with lap length so wall segments stay short
            default_wp = max(96, int(2 * np.pi * self.scale * 1.3 / 0.6))
            self.n_wp = int(params.get("n_wp", default_wp))
            self._center = _gp_centerline(self.scale, self.seed, self.n_wp)
            self._track_length = _polyline_length(self._center)
        elif self.layout == "stadium":
            self.n_wp = int(params.get("n_wp", 96))
            self._center = _stadium_centerline(self.straight, self.radius, self.n_wp)
            self._track_length = 4 * self.straight + 2 * np.pi * self.radius
        else:
            raise ValueError(f"unknown racetrack layout '{self.layout}' (stadium | gp)")

        self._center_z = _elevation_profile(
            self._track_length, self.n_wp, self.hills, self.max_grade, self.hill_freq,
            self.seed,
        )

    def _line3(self, offset: float) -> np.ndarray:
        """3D centerline shifted laterally by `offset` (positive = left of travel);
        z rides the elevation profile unchanged."""
        pts = self._center
        nxt = np.roll(pts, -1, axis=0)
        d = nxt - pts
        d /= np.hypot(d[:, 0], d[:, 1])[:, None]
        normals = np.stack([-d[:, 1], d[:, 0]], axis=1)  # left normal
        xy = pts + offset * normals
        return np.column_stack([xy, self._center_z])

    def build(self, spec) -> None:
        extent = float(np.abs(self._center).max()) + self.road_width + 2.0
        spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[extent, extent, 0.1],
            rgba=[0.22, 0.36, 0.20, 1.0],  # grass
        )

        half_road = self.road_width / 2.0
        hilly = self.hills > 0
        road = self._line3(0.0)
        inner = self._line3(+half_road + WALL_HALF_THICK)
        outer = self._line3(-(half_road + WALL_HALF_THICK))
        kerb_red = [0.85, 0.12, 0.12, 1.0]
        kerb_white = [0.93, 0.93, 0.93, 1.0]
        n = len(road)
        dash_off = 0.004 if hilly else 0.012  # centerline dashes sit on the road top
        for i in range(n):
            j = (i + 1) % n
            if hilly:
                # physical slab the cars drive on; top face at the elevation profile.
                # 0.5 mm alternation avoids z-fighting without a feelable step.
                _add_strip(spec, f"road_{i}", road[i], road[j], half_road,
                           ROAD_SLAB_HALF_H,
                           -ROAD_SLAB_HALF_H - (i % 2) * 0.0005,
                           ROAD_RGBA, collide=True)
            else:
                # flat: visual-only strip on the ground plane (original geometry)
                _add_strip(spec, f"road_{i}", road[i], road[j], half_road, 0.002,
                           0.004 + (i % 2) * 0.0008, ROAD_RGBA, collide=False)
            # F1-style kerbs: walls alternate red/white per segment (offset by one
            # between sides so the stripes interleave visually)
            _add_strip(spec, f"wall_inner_{i}", inner[i], inner[j], WALL_HALF_THICK,
                       self.wall_height / 2, self.wall_height / 2,
                       kerb_red if i % 2 == 0 else kerb_white, collide=True)
            _add_strip(spec, f"wall_outer_{i}", outer[i], outer[j], WALL_HALF_THICK,
                       self.wall_height / 2, self.wall_height / 2,
                       kerb_white if i % 2 == 0 else kerb_red, collide=True)
            # dashed white centerline (visual only)
            if i % 4 == 0:
                mid = (road[i] + road[j]) / 2.0
                _add_strip(spec, f"dash_{i}", road[i], mid, 0.035, 0.002, dash_off,
                           [0.85, 0.85, 0.85, 1.0], collide=False, overlap=0.6)

        self._build_start_checkers(spec, road, half_road, dash_off)

        self._build_obstacles(spec)
        if self.opponents:
            self._build_opponents(spec)

    def _build_start_checkers(self, spec, road: np.ndarray, half_road: float,
                              z_off: float) -> None:
        """Checkered start/finish line: 2 rows x 8 cells of alternating black/white."""
        d = road[1] - road[0]
        dxy = d[:2] / max(float(np.hypot(d[0], d[1])), 1e-9)
        normal = np.array([-dxy[1], dxy[0], 0.0])
        along = np.array([dxy[0], dxy[1], 0.0])
        quat = _seg_quat(d)
        n_cells = 8
        cell = 2 * half_road / n_cells
        for row in range(2):
            for col in range(n_cells):
                lat = -half_road + (col + 0.5) * cell
                pos = road[0] + normal * lat + along * (row - 0.5) * cell
                white = (row + col) % 2 == 0
                spec.worldbody.add_geom(
                    name=f"start_check_{row}_{col}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[float(pos[0]), float(pos[1]), float(pos[2]) + z_off],
                    size=[cell / 2, cell / 2, 0.002],
                    quat=quat,
                    rgba=[0.95, 0.95, 0.95, 1.0] if white else [0.05, 0.05, 0.05, 1.0],
                    contype=0,
                    conaffinity=0,
                )

    def _build_obstacles(self, spec) -> None:
        rng = np.random.default_rng(self.seed)
        half_road = self.road_width / 2.0
        for i in range(self.obstacles):
            # keep the first stretch after the start clear
            s_frac = rng.uniform(0.12, 0.95)
            idx = int(s_frac * self.n_wp)
            lateral = rng.uniform(-(half_road - 0.45), half_road - 0.45)
            base = self._line3(lateral)[idx]
            is_box = rng.random() < 0.5
            size = rng.uniform(0.10, 0.18)
            height = rng.uniform(0.25, 0.4)
            common = dict(
                name=f"obs_{i}",
                pos=[float(base[0]), float(base[1]), float(base[2]) + height / 2],
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
            lane_pts = self._line3(lane)
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
                    "waypoints": [tuple(p) for p in lane_pts],  # 3D: z rides the hills
                    "period": lap_time * rng.uniform(0.9, 1.15),
                    "phase": (i + 1) / (self.opponents + 1),
                    "z": 0.08,  # body-center height above the road surface
                }
            )

    def spawn_pose(self) -> tuple[tuple[float, float, float], float]:
        x, y = self._center[0]
        d = self._center[1] - self._center[0]
        return (float(x), float(y), float(self._center_z[0])), float(np.arctan2(d[1], d[0]))

    def task_info(self) -> dict:
        waypoints3 = np.column_stack([self._center, self._center_z])
        return {
            "waypoints": [tuple(p) for p in waypoints3],  # (x, y, z) per waypoint
            "track_length": float(self._track_length),
            "road_half_width": self.road_width / 2.0,
            "goal": tuple(self._center[0]),  # start/finish, for the viewer beacon
        }

    def collision_exempt_geom_names(self) -> list[str]:
        # with hills the road is a physical surface — scraping it is driving, not crashing
        return ["ground"] + [f"road_{i}" for i in range(self.n_wp)]

    def moving_bodies(self) -> list[dict]:
        return self._movers
