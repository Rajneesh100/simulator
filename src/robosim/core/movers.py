"""Scripted mocap-body movers, shared by the env step loop and the viewers.

Scenes describe movers declaratively via Scene.moving_bodies(); this module derives the
per-model state (mocap ids, path tables) and advances them each physics step. Two path
types:

- axis-sine (the original T7 movers): {"name", "axis": "x"|"y", "amplitude", "period",
  "base_pos"} — oscillates along one axis around base_pos.
- waypoint loop (racetrack opponents): {"name", "path": "loop", "waypoints": [(x, y) or
  (x, y, z), ...], "period", "phase", "z"} — drives around the closed polyline at constant
  speed, yaw following the path tangent. 3D waypoints ride the track's elevation; "z" is
  the body-center height above the waypoint. "phase" in [0, 1) staggers multiple movers.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def derive_movers(model, scene) -> list[dict[str, Any]]:
    """Resolve scene mover specs against a compiled model (mocap ids, path tables)."""
    movers: list[dict[str, Any]] = []
    for mover in scene.moving_bodies():
        m = dict(mover)
        m["mocap_id"] = model.body(m["name"]).mocapid[0]
        if m.get("path") == "loop":
            pts = np.asarray(m["waypoints"], dtype=np.float64)
            closed = np.vstack([pts, pts[:1]])  # close the loop
            seg = np.diff(closed, axis=0)
            seg_len = np.hypot(seg[:, 0], seg[:, 1])
            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            m["_pts"] = closed
            m["_cum"] = cum
            m["_total"] = float(cum[-1])
        else:
            m["axis_vec"] = np.array(
                [1.0, 0.0, 0.0] if m["axis"] == "x" else [0.0, 1.0, 0.0]
            )
        movers.append(m)
    return movers


def _loop_pose(m: dict[str, Any], t: float) -> tuple[np.ndarray, float]:
    """Position and tangent yaw at time t along the closed waypoint loop."""
    s = ((t / m["period"]) + m.get("phase", 0.0)) % 1.0 * m["_total"]
    cum, pts = m["_cum"], m["_pts"]
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(i, len(pts) - 2)
    seg_len = cum[i + 1] - cum[i]
    frac = (s - cum[i]) / seg_len if seg_len > 1e-9 else 0.0
    pos = pts[i] + frac * (pts[i + 1] - pts[i])
    d = pts[i + 1] - pts[i]
    return pos, float(np.arctan2(d[1], d[0]))


def update_movers(data, movers: list[dict[str, Any]], t: float) -> None:
    for m in movers:
        if m.get("path") == "loop":
            p, yaw = _loop_pose(m, t)
            base_z = p[2] if len(p) == 3 else 0.0
            data.mocap_pos[m["mocap_id"]] = [p[0], p[1], base_z + m.get("z", 0.0)]
            half = yaw / 2.0
            data.mocap_quat[m["mocap_id"]] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        else:
            offset = m["amplitude"] * np.sin(2 * np.pi * t / m["period"]) * m["axis_vec"]
            data.mocap_pos[m["mocap_id"]] = np.array(m["base_pos"]) + offset
