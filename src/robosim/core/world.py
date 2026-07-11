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
