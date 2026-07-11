"""Proprioceptive sensing: wheel velocities + IMU (pitch, roll, yaw rate).

Reads mujoco sensors by name (via robot.sensor_names()) so this stays robot-agnostic —
any robot that provides the same sensor roles gets the same proprio vector layout.
"""

from __future__ import annotations

import numpy as np


def _sensordata(model, data, sensor_name: str) -> np.ndarray:
    sensor = model.sensor(sensor_name)
    adr = sensor.adr[0]
    dim = sensor.dim[0]
    return np.array(data.sensordata[adr : adr + dim], dtype=np.float32)


def quat_to_pitch_roll(quat_wxyz: np.ndarray) -> tuple[float, float]:
    """Standard aerospace pitch/roll extraction from a (w, x, y, z) quaternion."""
    w, x, y, z = quat_wxyz
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    return float(pitch), float(roll)


def read_proprio(model, data, robot) -> np.ndarray:
    names = robot.sensor_names()
    wheel_left = _sensordata(model, data, names["wheel_left_vel"])[0]
    wheel_right = _sensordata(model, data, names["wheel_right_vel"])[0]
    quat = _sensordata(model, data, names["orientation"])
    pitch, roll = quat_to_pitch_roll(quat)
    yaw_rate = _sensordata(model, data, names["gyro"])[2]
    return np.array([wheel_left, wheel_right, pitch, roll, yaw_rate], dtype=np.float32)
