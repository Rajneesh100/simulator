"""Canonical observation layout — the ONE place this is defined (CLAUDE.md rule).

`[lidar ranges (n_beams, normalized) | goal in robot frame (distance, sin, cos bearing) |
wheel velocities | IMU (pitch, roll, yaw rate) | previous action]`

Every task builds its observation through `build_obs` so the real robot can be made to
emit an identical vector later. Distance is normalized by `goal_norm_dist`; everything
else is already unit-scale (lidar in [0,1], sin/cos in [-1,1], action in [-1,1]).
"""

from __future__ import annotations

import numpy as np

from robosim.sensors.proprio import quat_to_pitch_roll

LIDAR_MAX_RANGE_NORM = 1.0
GOAL_NORM_DIST = 10.0  # meters; distance term is clipped+scaled by this


def obs_dim(n_lidar_beams: int, action_dim: int) -> int:
    return n_lidar_beams + 3 + 2 + 3 + action_dim


def goal_in_robot_frame(robot_xy: np.ndarray, robot_quat_wxyz: np.ndarray, goal_xy) -> tuple[float, float, float]:
    w, x, y, z = robot_quat_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    dx, dy = goal_xy[0] - robot_xy[0], goal_xy[1] - robot_xy[1]
    distance = float(np.hypot(dx, dy))
    bearing = float(np.arctan2(dy, dx) - yaw)
    return distance, float(np.sin(bearing)), float(np.cos(bearing))


def build_obs(
    lidar_ranges: np.ndarray,
    robot_xy: np.ndarray,
    robot_quat_wxyz: np.ndarray,
    goal_xy,
    proprio: np.ndarray,
    prev_action: np.ndarray,
) -> np.ndarray:
    distance, sin_b, cos_b = goal_in_robot_frame(robot_xy, robot_quat_wxyz, goal_xy)
    goal_term = np.array([min(distance / GOAL_NORM_DIST, 1.0), sin_b, cos_b], dtype=np.float32)
    return np.concatenate(
        [
            lidar_ranges.astype(np.float32),
            goal_term,
            proprio.astype(np.float32),
            prev_action.astype(np.float32),
        ]
    )
