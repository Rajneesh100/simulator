"""Differential-drive wheeled robot: 2 driven wheels + a passive caster.

Matches the real WALL-E-style robot's drive train. This is the primary robot for phase 1.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robosim.core import physics
from robosim.core.registry import register_robot
from robosim.robots.base import Robot

CHASSIS_HALF = (0.12, 0.09, 0.05)
WHEEL_RADIUS = 0.045
WHEEL_HALF_WIDTH = 0.02
WHEEL_MOUNT_Z = -0.03  # wheel body offset below chassis origin, in chassis-local frame
TRACK_HALF_WIDTH = 0.11  # distance from center to each wheel
CASTER_RADIUS = 0.02
MAX_LINEAR_VEL = 0.5  # m/s
MAX_ANGULAR_VEL = 2.5  # rad/s


def _velocity_actuator(
    spec, name: str, joint_name: str, kv: float, ctrl_limit: float, max_torque: float
):
    act = spec.add_actuator(
        name=name,
        target=joint_name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        ctrlrange=[-ctrl_limit, ctrl_limit],
        forcerange=[-max_torque, max_torque],
    )
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    gainprm = np.zeros(10)
    gainprm[0] = kv
    act.gainprm = gainprm
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    biasprm = np.zeros(10)
    biasprm[2] = -kv
    act.biasprm = biasprm
    return act


@register_robot("diffdrive")
class DiffDriveRobot(Robot):
    def __init__(self, **params: str):
        super().__init__(**params)
        self.wheel_radius = float(params.get("wheel_radius", WHEEL_RADIUS))
        self.track_half_width = float(params.get("track_half_width", TRACK_HALF_WIDTH))
        self.max_linear_vel = float(params.get("max_linear_vel", MAX_LINEAR_VEL))
        self.max_angular_vel = float(params.get("max_angular_vel", MAX_ANGULAR_VEL))

    def build_spec(self):
        spec = mujoco.MjSpec()
        chassis = spec.worldbody.add_body(name="chassis")
        chassis.add_freejoint(name="chassis_free")
        chassis.add_geom(
            name="chassis_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(CHASSIS_HALF),
            rgba=[0.8, 0.3, 0.1, 1.0],
            friction=list(physics.DEFAULT_FRICTION),
        )
        chassis.add_site(name="imu_site", pos=[0, 0, 0])
        chassis.add_site(
            name="lidar_site", pos=[CHASSIS_HALF[0] - 0.01, 0, CHASSIS_HALF[2] - 0.005]
        )

        for side, y in (("left", self.track_half_width), ("right", -self.track_half_width)):
            wheel = chassis.add_body(name=f"wheel_{side}", pos=[0, y, WHEEL_MOUNT_Z])
            wheel.add_joint(
                name=f"wheel_{side}_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 1, 0],
            )
            wheel.add_geom(
                name=f"wheel_{side}_geom",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.wheel_radius, WHEEL_HALF_WIDTH, 0],
                euler=[90, 0, 0],
                rgba=[0.1, 0.1, 0.1, 1.0],
                friction=list(physics.WHEEL_FRICTION),
            )
            _velocity_actuator(
                spec,
                f"wheel_{side}_motor",
                f"wheel_{side}_joint",
                kv=4.0,
                ctrl_limit=30.0,
                max_torque=0.6,
            )

        wheel_bottom_local_z = WHEEL_MOUNT_Z - self.wheel_radius
        caster_z = wheel_bottom_local_z + CASTER_RADIUS
        chassis.add_geom(
            name="caster_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[-CHASSIS_HALF[0] + CASTER_RADIUS, 0, caster_z],
            size=[CASTER_RADIUS, 0, 0],
            rgba=[0.2, 0.2, 0.2, 1.0],
            friction=[0.05, 0.001, 0.0001],
        )

        spec.add_sensor(
            name="imu_gyro",
            type=mujoco.mjtSensor.mjSENS_GYRO,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="imu_site",
        )
        spec.add_sensor(
            name="imu_quat",
            type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="imu_site",
        )
        spec.add_sensor(
            name="wheel_left_vel",
            type=mujoco.mjtSensor.mjSENS_JOINTVEL,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname="wheel_left_joint",
        )
        spec.add_sensor(
            name="wheel_right_vel",
            type=mujoco.mjtSensor.mjSENS_JOINTVEL,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname="wheel_right_joint",
        )
        return spec

    def root_body_name(self) -> str:
        return "chassis"

    def spawn_z_offset(self) -> float:
        """Height of the chassis origin above ground level when wheels just touch down."""
        return self.wheel_radius - WHEEL_MOUNT_Z + 0.002  # small clearance margin

    def action_dim(self) -> int:
        return 2

    def action_to_ctrl(self, action: np.ndarray) -> dict[str, float]:
        v = float(np.clip(action[0], -1.0, 1.0)) * self.max_linear_vel
        omega = float(np.clip(action[1], -1.0, 1.0)) * self.max_angular_vel
        v_l = v - omega * self.track_half_width
        v_r = v + omega * self.track_half_width
        return {
            "wheel_left_motor": v_l / self.wheel_radius,
            "wheel_right_motor": v_r / self.wheel_radius,
        }

    def sensor_names(self) -> dict[str, str]:
        return {
            "gyro": "imu_gyro",
            "orientation": "imu_quat",
            "wheel_left_vel": "wheel_left_vel",
            "wheel_right_vel": "wheel_right_vel",
        }

    def proprio_dim(self) -> int:
        return 5  # wheel_left_vel, wheel_right_vel, pitch, roll, yaw_rate

    def read_proprio(self, model, data, prefix: str = "") -> np.ndarray:
        from robosim.sensors.proprio import read_proprio

        return read_proprio(model, data, self, prefix=prefix)

    def collision_geom_names(self) -> list[str]:
        return ["chassis_geom"]
