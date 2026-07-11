"""4-wheel Ackermann-steered racecar: driven rear axle, steered front wheels.

Action stays 2-dim like every robot — (throttle, steer) instead of (v, omega) — so the
whole env/task/training stack works unchanged. `action_to_ctrl` does the Ackermann
geometry (per-wheel steering angles) and acts as an open differential on the rear axle
(inner/outer wheel speeds from the commanded turn curvature), so corners don't scrub.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robosim.core import physics
from robosim.core.registry import register_robot
from robosim.robots.base import Robot

CHASSIS_HALF = (0.22, 0.11, 0.04)
WHEEL_RADIUS = 0.06
WHEEL_HALF_WIDTH = 0.025
WHEEL_MOUNT_Z = -0.02  # axle height below chassis origin, chassis-local
WHEELBASE_HALF = 0.15  # front/rear axle x offset from chassis center
TRACK_HALF_WIDTH = 0.13
MAX_SPEED = 3.0  # m/s — a racecar, not a household robot
MAX_STEER = 0.55  # rad (~31 deg)


@register_robot("racecar")
class RaceCarRobot(Robot):
    def __init__(self, **params: str):
        super().__init__(**params)
        self.wheel_radius = float(params.get("wheel_radius", WHEEL_RADIUS))
        self.max_speed = float(params.get("max_speed", MAX_SPEED))
        self.max_steer = float(params.get("max_steer", MAX_STEER))
        self.wheelbase = 2 * WHEELBASE_HALF
        self.track_half_width = TRACK_HALF_WIDTH

    def build_spec(self):
        spec = mujoco.MjSpec()
        chassis = spec.worldbody.add_body(name="car_chassis")
        chassis.add_freejoint(name="car_free")
        chassis.add_geom(
            name="car_chassis_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(CHASSIS_HALF),
            rgba=[0.85, 0.1, 0.1, 1.0],
            friction=list(physics.DEFAULT_FRICTION),
        )
        # cabin: visual-only so the car reads as a car in the viewer
        chassis.add_geom(
            name="car_cabin_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[-0.04, 0.0, CHASSIS_HALF[2] + 0.03],
            size=[0.10, 0.08, 0.03],
            rgba=[0.15, 0.15, 0.18, 1.0],
            contype=0,
            conaffinity=0,
        )
        chassis.add_site(name="imu_site", pos=[0, 0, 0])
        chassis.add_site(
            name="lidar_site", pos=[CHASSIS_HALF[0] - 0.01, 0, CHASSIS_HALF[2] - 0.005]
        )

        def add_wheel(parent, name: str, pos, steered: bool):
            if steered:
                knuckle = parent.add_body(name=f"{name}_knuckle", pos=pos)
                knuckle.add_joint(
                    name=f"{name}_steer_joint",
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    axis=[0, 0, 1],
                    range=[-self.max_steer * 1.3, self.max_steer * 1.3],
                )
                knuckle.add_geom(  # tiny massful geom so the knuckle body is valid
                    name=f"{name}_knuckle_geom",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.01, 0, 0],
                    rgba=[0.3, 0.3, 0.3, 1.0],
                    contype=0,
                    conaffinity=0,
                )
                wheel_parent = knuckle
                wheel_pos = [0.0, 0.0, 0.0]
            else:
                wheel_parent = parent
                wheel_pos = pos
            wheel = wheel_parent.add_body(name=f"{name}_wheel", pos=wheel_pos)
            wheel.add_joint(
                name=f"{name}_roll_joint",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=[0, 1, 0],
            )
            wheel.add_geom(
                name=f"{name}_wheel_geom",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.wheel_radius, WHEEL_HALF_WIDTH, 0],
                euler=[90, 0, 0],
                rgba=[0.08, 0.08, 0.08, 1.0],
                friction=list(physics.WHEEL_FRICTION),
            )

        for side, y in (("left", self.track_half_width), ("right", -self.track_half_width)):
            add_wheel(chassis, f"front_{side}", [WHEELBASE_HALF, y, WHEEL_MOUNT_Z], steered=True)
            add_wheel(chassis, f"rear_{side}", [-WHEELBASE_HALF, y, WHEEL_MOUNT_Z], steered=False)

            # front wheels are grandchildren of the chassis (via the steering knuckle),
            # so MuJoCo's automatic parent-child contact exclusion doesn't cover them
            spec.add_exclude(
                name=f"chassis_front_{side}",
                bodyname1="car_chassis",
                bodyname2=f"front_{side}_wheel",
            )

            # steering: position servo on the knuckle hinge
            act = spec.add_actuator(
                name=f"front_{side}_steer",
                target=f"front_{side}_steer_joint",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                ctrlrange=[-self.max_steer * 1.2, self.max_steer * 1.2],
                forcerange=[-6.0, 6.0],
            )
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            gainprm = np.zeros(10)
            gainprm[0] = 25.0
            act.gainprm = gainprm
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            biasprm = np.zeros(10)
            biasprm[1] = -25.0  # position feedback
            biasprm[2] = -1.0  # velocity damping
            act.biasprm = biasprm

            # drive: velocity servo on the rear wheel
            max_wheel_speed = self.max_speed / self.wheel_radius
            act = spec.add_actuator(
                name=f"rear_{side}_motor",
                target=f"rear_{side}_roll_joint",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                ctrlrange=[-max_wheel_speed * 1.3, max_wheel_speed * 1.3],
                forcerange=[-2.5, 2.5],
            )
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            gainprm = np.zeros(10)
            gainprm[0] = 6.0
            act.gainprm = gainprm
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            biasprm = np.zeros(10)
            biasprm[2] = -6.0
            act.biasprm = biasprm

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
            name="rear_left_vel",
            type=mujoco.mjtSensor.mjSENS_JOINTVEL,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname="rear_left_roll_joint",
        )
        spec.add_sensor(
            name="rear_right_vel",
            type=mujoco.mjtSensor.mjSENS_JOINTVEL,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname="rear_right_roll_joint",
        )
        return spec

    def root_body_name(self) -> str:
        return "car_chassis"

    def spawn_z_offset(self) -> float:
        return self.wheel_radius - WHEEL_MOUNT_Z + 0.002

    def action_dim(self) -> int:
        return 2

    def action_to_ctrl(self, action: np.ndarray) -> dict[str, float]:
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steer_cmd = float(np.clip(action[1], -1.0, 1.0))
        v = throttle * self.max_speed
        delta = steer_cmd * self.max_steer

        L, w = self.wheelbase, self.track_half_width
        if abs(delta) < 1e-4:
            delta_l = delta_r = delta
            curvature = 0.0
        else:
            turn_radius = L / np.tan(delta)  # signed; positive = left turn
            delta_l = float(np.arctan(L / (turn_radius - w)))
            delta_r = float(np.arctan(L / (turn_radius + w)))
            curvature = 1.0 / turn_radius

        # open differential: inner rear wheel slows, outer speeds up
        v_rl = v * (1.0 - curvature * w)
        v_rr = v * (1.0 + curvature * w)
        return {
            "front_left_steer": delta_l,
            "front_right_steer": delta_r,
            "rear_left_motor": v_rl / self.wheel_radius,
            "rear_right_motor": v_rr / self.wheel_radius,
        }

    def sensor_names(self) -> dict[str, str]:
        return {
            "gyro": "imu_gyro",
            "orientation": "imu_quat",
            "wheel_left_vel": "rear_left_vel",
            "wheel_right_vel": "rear_right_vel",
        }

    def proprio_dim(self) -> int:
        return 5

    def read_proprio(self, model, data, prefix: str = "") -> np.ndarray:
        from robosim.sensors.proprio import read_proprio

        proprio = read_proprio(model, data, self, prefix=prefix)
        # wheel speeds reach ~50 rad/s at full throttle — normalize so the obs
        # stays in the same range every other robot produces
        max_wheel_speed = self.max_speed / self.wheel_radius
        proprio[0] /= max_wheel_speed
        proprio[1] /= max_wheel_speed
        return proprio

    def collision_geom_names(self) -> list[str]:
        return ["car_chassis_geom"]
