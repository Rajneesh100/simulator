"""Canonical observation layout — the ONE place this is defined (CLAUDE.md rule).

`[lidar ranges (n_beams, normalized) | goal in robot frame (distance, sin, cos bearing) |
wheel velocities | IMU (pitch, roll, yaw rate) | previous action | combat block (opt)]`

Every task builds its observation through `build_obs` so the real robot can be made to
emit an identical vector later. Distance is normalized by `goal_norm_dist`; everything
else is already unit-scale (lidar in [0,1], sin/cos in [-1,1], action in [-1,1]).

Combat block (only with --firearms, 16 floats, see `combat_block`):
  for the COMBAT_K=3 nearest cars, nearest first (zeros when fewer):
      [distance/COMBAT_NORM_DIST, sin bearing, cos bearing, closing speed/COMBAT_SPEED_NORM]
  [weapon ready 0/1]
  [threat distance/COMBAT_NORM_DIST, sin, cos]  nearest rocket flying at me (zeros if none)
"""

from __future__ import annotations

import numpy as np

from robosim.sensors.proprio import quat_to_pitch_roll

LIDAR_MAX_RANGE_NORM = 1.0
GOAL_NORM_DIST = 10.0  # meters; distance term is clipped+scaled by this

COMBAT_K = 3
COMBAT_DIM = COMBAT_K * 4 + 1 + 3  # 16
COMBAT_NORM_DIST = 20.0
COMBAT_SPEED_NORM = 12.0
THREAT_CONE_DEG = 30.0


def obs_dim(n_lidar_beams: int, action_dim: int, combat: bool = False) -> int:
    return n_lidar_beams + 3 + 2 + 3 + action_dim + (COMBAT_DIM if combat else 0)


def _yaw(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def goal_in_robot_frame(robot_xy: np.ndarray, robot_quat_wxyz: np.ndarray, goal_xy) -> tuple[float, float, float]:
    yaw = _yaw(robot_quat_wxyz)
    dx, dy = goal_xy[0] - robot_xy[0], goal_xy[1] - robot_xy[1]
    distance = float(np.hypot(dx, dy))
    bearing = float(np.arctan2(dy, dx) - yaw)
    return distance, float(np.sin(bearing)), float(np.cos(bearing))


def combat_block(
    me_xy, me_quat_wxyz, me_vel_xy, others, weapon_ready: bool, rockets,
) -> np.ndarray:
    """See module docstring. `others` = [(xy, vel_xy), ...] of live rival cars;
    `rockets` = [(pos3, vel3), ...] of rockets in flight (any owner). The threat
    bearing points from me TO the rocket, so the policy knows which side it comes from."""
    me_xy = np.asarray(me_xy, dtype=np.float64)
    me_vel = np.asarray(me_vel_xy, dtype=np.float64)
    yaw = _yaw(me_quat_wxyz)
    out = np.zeros(COMBAT_DIM, dtype=np.float32)
    rows = []
    for xy, vel in others:
        d = np.asarray(xy, dtype=np.float64) - me_xy
        dist = float(np.hypot(*d))
        bearing = float(np.arctan2(d[1], d[0]) - yaw)
        rel_v = np.asarray(vel, dtype=np.float64) - me_vel
        closing = float(-np.dot(rel_v, d) / dist) if dist > 1e-6 else 0.0
        rows.append((dist, bearing, closing))
    rows.sort(key=lambda r: r[0])
    for k, (dist, bearing, closing) in enumerate(rows[:COMBAT_K]):
        out[4 * k : 4 * k + 4] = [
            min(dist / COMBAT_NORM_DIST, 1.0), np.sin(bearing), np.cos(bearing),
            np.clip(closing / COMBAT_SPEED_NORM, -1.0, 1.0),
        ]
    out[4 * COMBAT_K] = 1.0 if weapon_ready else 0.0
    cone = np.cos(np.deg2rad(THREAT_CONE_DEG))
    best = None
    for pos, vel in rockets:
        to_me = me_xy - np.asarray(pos[:2], dtype=np.float64)
        dist = float(np.hypot(*to_me))
        v_xy = np.asarray(vel[:2], dtype=np.float64)
        v_norm = float(np.hypot(*v_xy))
        if dist < 1e-6 or v_norm < 1e-6:
            continue
        if np.dot(v_xy, to_me) / (v_norm * dist) < cone:
            continue  # not flying at me
        if best is None or dist < best[0]:
            best = (dist, float(np.arctan2(-to_me[1], -to_me[0]) - yaw))
    if best is not None:
        dist, bearing = best
        out[4 * COMBAT_K + 1 :] = [min(dist / COMBAT_NORM_DIST, 1.0), np.sin(bearing), np.cos(bearing)]
    return out


def build_obs(
    lidar_ranges: np.ndarray,
    robot_xy: np.ndarray,
    robot_quat_wxyz: np.ndarray,
    goal_xy,
    proprio: np.ndarray,
    prev_action: np.ndarray,
    combat: np.ndarray | None = None,
) -> np.ndarray:
    distance, sin_b, cos_b = goal_in_robot_frame(robot_xy, robot_quat_wxyz, goal_xy)
    goal_term = np.array([min(distance / GOAL_NORM_DIST, 1.0), sin_b, cos_b], dtype=np.float32)
    parts = [
        lidar_ranges.astype(np.float32),
        goal_term,
        proprio.astype(np.float32),
        prev_action.astype(np.float32),
    ]
    if combat is not None:
        parts.append(np.asarray(combat, dtype=np.float32))
    return np.concatenate(parts)
