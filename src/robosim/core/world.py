"""Builds a compiled MuJoCo model from a base world + selected scene + robot.

This is the one place scene and robot specs get merged. It never needs editing to add a
new scene or robot — those register themselves via core/registry.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from robosim.core import physics, theme
from robosim.core.registry import get_robot, get_scene


def _parse_scene_arg(scene_arg: str) -> tuple[str, dict[str, str]]:
    """Parse '--scene stairs:height=0.1,count=8' -> ('stairs', {'height': '0.1', 'count': '8'})."""
    if ":" not in scene_arg:
        return scene_arg, {}
    name, _, param_str = scene_arg.partition(":")
    params: dict[str, str] = {}
    for pair in param_str.split(","):
        if not pair:
            continue
        key, _, val = pair.partition("=")
        params[key] = val
    return name, params


@dataclass
class World:
    model: mujoco.MjModel
    data: mujoco.MjData
    scene: Any
    robot: Any
    root_body_name: str
    root_joint_qpos_adr: int
    root_joint_qvel_adr: int
    spawn_pos: tuple[float, float, float]
    spawn_yaw: float
    extra: dict[str, Any] = field(default_factory=dict)

    def reset(self, seed: int | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        x, y, z = self.spawn_pos
        yaw = self.spawn_yaw
        self.data.qpos[self.root_joint_qpos_adr : self.root_joint_qpos_adr + 3] = [x, y, z]
        half = yaw / 2.0
        self.data.qpos[self.root_joint_qpos_adr + 3 : self.root_joint_qpos_adr + 7] = [
            np.cos(half),
            0.0,
            0.0,
            np.sin(half),
        ]
        mujoco.mj_forward(self.model, self.data)

    def robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (xy position, quaternion wxyz) of the robot root body."""
        pos = self.data.qpos[self.root_joint_qpos_adr : self.root_joint_qpos_adr + 3].copy()
        quat = self.data.qpos[self.root_joint_qpos_adr + 3 : self.root_joint_qpos_adr + 7].copy()
        return pos, quat


@dataclass
class MultiWorld:
    """N copies of the same robot compiled into one world — used by `robosim race`.

    Car i's names carry the attach prefix `prefixes[i]` (e.g. "c0_car_chassis"); use
    `car_pose(i)` / `set_ctrl(i, ...)` instead of touching names directly.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    scene: Any
    robots: list[Any]
    prefixes: list[str]
    root_body_names: list[str]  # prefixed
    qpos_adrs: list[int]
    qvel_adrs: list[int]
    spawn_poses: list[tuple[tuple[float, float, float], float]]

    @property
    def n_cars(self) -> int:
        return len(self.robots)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for i, ((x, y, z), yaw) in enumerate(self.spawn_poses):
            adr = self.qpos_adrs[i]
            half = yaw / 2.0
            self.data.qpos[adr : adr + 3] = [x, y, z]
            self.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        mujoco.mj_forward(self.model, self.data)

    def car_pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        adr = self.qpos_adrs[i]
        return (
            self.data.qpos[adr : adr + 3].copy(),
            self.data.qpos[adr + 3 : adr + 7].copy(),
        )

    def car_speed(self, i: int) -> float:
        adr = self.qvel_adrs[i]
        return float(np.hypot(self.data.qvel[adr], self.data.qvel[adr + 1]))

    def set_ctrl(self, i: int, ctrl_map: dict[str, float]) -> None:
        for name, value in ctrl_map.items():
            self.data.actuator(self.prefixes[i] + name).ctrl[0] = value


def _grid_spawn_poses(scene, robot, n_cars: int) -> list[tuple[tuple[float, float, float], float]]:
    """F1-style staggered starting grid: single file, one car per row, alternating left/
    right of the racing line. Side-by-side pairs proved fatal for solo-trained policies —
    both partners converge onto the centerline within a second and wedge each other, so
    each car gets its own row. On a track scene the grid follows the centerline
    backwards; otherwise it steps straight back along the spawn heading."""
    (sx, sy, terrain_z), yaw = scene.spawn_pose()
    z = terrain_z + robot.spawn_z_offset()
    info = scene.task_info()
    waypoints = info.get("waypoints")
    poses: list[tuple[tuple[float, float, float], float]] = []

    row_gap, lane_offset = 1.4, 0.4
    if waypoints is not None:
        pts = np.asarray(waypoints, dtype=np.float64)
        nxt = np.roll(pts, -1, axis=0)
        seg_len = np.hypot(*(nxt - pts).T)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
        total = float(info["track_length"])
        for i in range(n_cars):
            s = (-row_gap * (i + 1)) % total
            idx = int(np.searchsorted(cum, s, side="right") - 1)
            j = (idx + 1) % len(pts)
            frac = (s - cum[idx]) / max(seg_len[idx], 1e-9)
            p = pts[idx] + frac * (pts[j] - pts[idx])
            d = pts[j] - pts[idx]
            tangent_yaw = float(np.arctan2(d[1], d[0]))
            normal = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9)
            side = lane_offset if i % 2 == 0 else -lane_offset
            poses.append(((float(p[0] + side * normal[0]), float(p[1] + side * normal[1]), z), tangent_yaw))
    else:
        back = np.array([-np.cos(yaw), -np.sin(yaw)])
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        for i in range(n_cars):
            offset = back * row_gap * (i + 1) + left * (lane_offset if i % 2 == 0 else -lane_offset)
            poses.append(((sx + float(offset[0]), sy + float(offset[1]), z), yaw))
    return poses


def build_multi_world(
    scene_arg: str = "racetrack",
    robot_arg: str = "racecar",
    n_cars: int = 4,
    add_ground_light: bool = True,
) -> MultiWorld:
    """Compile one world holding `n_cars` copies of the same robot on a starting grid."""
    scene_name, scene_params = _parse_scene_arg(scene_arg)
    robot_name, robot_params = _parse_scene_arg(robot_arg)

    spec = mujoco.MjSpec()
    physics.apply_to_spec(spec)
    if add_ground_light:
        theme.apply_visual_theme(spec)
        spec.worldbody.add_light(
            name="sun",
            pos=[0, 0, 3],
            dir=[-0.3, 0.2, -0.9],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

    scene_cls = get_scene(scene_name)
    scene = scene_cls(**scene_params)
    scene.build(spec)
    if add_ground_light:
        theme.assign_ground_material(spec, scene.ground_geom_names())

    robot_cls = get_robot(robot_name)
    robots, prefixes, spawn_poses = [], [], []
    probe_robot = robot_cls(**robot_params)
    grid = _grid_spawn_poses(scene, probe_robot, n_cars)
    for i in range(n_cars):
        robot = robot_cls(**robot_params)
        (pos, yaw) = grid[i]
        half = yaw / 2.0
        frame = spec.worldbody.add_frame(
            pos=list(pos), quat=[np.cos(half), 0.0, 0.0, np.sin(half)]
        )
        frame.attach_body(robot.build_spec().worldbody.first_body(), f"c{i}_", "")
        robots.append(robot)
        prefixes.append(f"c{i}_")
        spawn_poses.append(grid[i])

    model = spec.compile()
    data = mujoco.MjData(model)

    root_names, qpos_adrs, qvel_adrs = [], [], []
    for i in range(n_cars):
        root = prefixes[i] + robots[i].root_body_name()
        body = model.body(root)
        joint_id = body.jntadr[0]
        root_names.append(root)
        qpos_adrs.append(model.jnt_qposadr[joint_id])
        qvel_adrs.append(model.jnt_dofadr[joint_id])

    world = MultiWorld(
        model=model,
        data=data,
        scene=scene,
        robots=robots,
        prefixes=prefixes,
        root_body_names=root_names,
        qpos_adrs=qpos_adrs,
        qvel_adrs=qvel_adrs,
        spawn_poses=spawn_poses,
    )
    world.reset()
    return world


def build_world(
    scene_arg: str = "flat",
    robot_arg: str = "diffdrive",
    add_ground_light: bool = True,
) -> World:
    scene_name, scene_params = _parse_scene_arg(scene_arg)
    robot_name, robot_params = _parse_scene_arg(robot_arg)

    spec = mujoco.MjSpec()
    physics.apply_to_spec(spec)

    if add_ground_light:
        theme.apply_visual_theme(spec)
        spec.worldbody.add_light(
            name="sun",
            pos=[0, 0, 3],
            dir=[-0.3, 0.2, -0.9],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

    scene_cls = get_scene(scene_name)
    scene = scene_cls(**scene_params)
    scene.build(spec)
    if add_ground_light:
        theme.assign_ground_material(spec, scene.ground_geom_names())

    robot_cls = get_robot(robot_name)
    robot = robot_cls(**robot_params)
    robot_spec = robot.build_spec()

    (sx, sy, terrain_z), spawn_yaw = scene.spawn_pose()
    spawn_pos = (sx, sy, terrain_z + robot.spawn_z_offset())
    half = spawn_yaw / 2.0
    frame = spec.worldbody.add_frame(
        pos=list(spawn_pos),
        quat=[np.cos(half), 0.0, 0.0, np.sin(half)],
    )
    robot_root = robot_spec.worldbody.first_body()
    frame.attach_body(robot_root, "", "")

    model = spec.compile()
    data = mujoco.MjData(model)

    root_body_name = robot.root_body_name()
    body = model.body(root_body_name)
    joint_id = body.jntadr[0]
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]

    world = World(
        model=model,
        data=data,
        scene=scene,
        robot=robot,
        root_body_name=root_body_name,
        root_joint_qpos_adr=qpos_adr,
        root_joint_qvel_adr=qvel_adr,
        spawn_pos=spawn_pos,
        spawn_yaw=spawn_yaw,
    )
    world.reset()
    return world
