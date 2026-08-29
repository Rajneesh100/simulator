"""Race task: lap the circuit as fast as possible, on the road, without hitting anything.

Reward per control step = meters of progress along the track centerline (so speed IS the
reward), minus a small time penalty, minus a centering penalty that grows toward the
walls. Collisions (walls, obstacles, opponent cars) end the episode with a crash penalty;
completing a lap ends it with a bonus and success=True.

The obs "goal" slot is repointed every step to a lookahead point on the centerline ahead
of the car, so the standard observation vector doubles as "where does the track go next".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosim.core.registry import register_task
from robosim.envs.tasks.base import Task

LOOKAHEAD_M = 2.5
TIME_PENALTY = 0.02
CENTER_PENALTY = 0.05  # scaled by |lateral| / road_half_width
OFFROAD_PENALTY = 2.0  # per step spent beyond the road edge (walls make this rare)
COLLISION_PENALTY = 8.0
LAP_BONUS = 30.0
MAX_DS_PER_STEP = 1.0  # meters; guards projection glitches from counting as progress


def _project(pts: np.ndarray, cum: np.ndarray, xy: np.ndarray) -> tuple[float, float]:
    """Project xy onto the closed polyline. Returns (arclength s, lateral distance)."""
    a = pts
    b = np.roll(pts, -1, axis=0)
    ab = b - a
    seg_len_sq = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t = np.clip(((xy - a) * ab).sum(axis=1) / seg_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * ab
    d2 = ((xy - closest) ** 2).sum(axis=1)
    i = int(np.argmin(d2))
    seg_len = float(np.sqrt(seg_len_sq[i]))
    s = float(cum[i] + t[i] * seg_len)
    return s, float(np.sqrt(d2[i]))


def _point_at(pts: np.ndarray, cum: np.ndarray, total: float, s: float) -> tuple[float, float]:
    s = s % total
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(i, len(pts) - 1)
    nxt = pts[(i + 1) % len(pts)]
    seg = float(cum[i + 1] - cum[i]) if i + 1 < len(cum) else total - float(cum[i])
    frac = (s - float(cum[i])) / seg if seg > 1e-9 else 0.0
    p = pts[i] + frac * (nxt - pts[i])
    return float(p[0]), float(p[1])


@register_task("race")
class RaceTask(Task):
    """Params (via `--task race:laps=N,lookahead=4,lookahead_time=0.6`):
    - laps: laps to finish for success (default 1); 0 = endless (lap bonuses keep
      coming, episode ends only on crash or the step cap) — the continuous-demo mode.
    - lookahead: meters ahead on the centerline for the obs goal point (default 2.5).
    - lookahead_time: seconds; if > 0 the goal moves to max(lookahead, speed x this),
      so a fast car sees further ahead — the standard fix for braking too late into
      hairpins. Changes the obs distribution: train and play with the same values.
    - random_heading: probability (0..1) that an episode starts facing a RANDOM
      direction. Solo training otherwise never visits "pointing the wrong way" states,
      so policies can't course-correct after a spin; a third of episodes starting
      disoriented teaches the U-turn."""

    def __init__(self, **params: str):
        super().__init__(**params)
        self.laps = int(params.get("laps", 1))
        self.lookahead = float(params.get("lookahead", LOOKAHEAD_M))
        self.lookahead_time = float(params.get("lookahead_time", 0.0))
        self.random_heading = float(params.get("random_heading", 0.0))

    def _lookahead_m(self, world) -> float:
        if self.lookahead_time <= 0:
            return self.lookahead
        adr = world.root_joint_qvel_adr
        speed = float(np.hypot(world.data.qvel[adr], world.data.qvel[adr + 1]))
        return max(self.lookahead, speed * self.lookahead_time)

    def reset(self, world, rng: np.random.Generator) -> dict[str, Any]:
        if self.random_heading > 0 and rng.random() < self.random_heading:
            import mujoco

            yaw = world.spawn_yaw + rng.uniform(-np.pi, np.pi)
            half = yaw / 2.0
            adr = world.root_joint_qpos_adr
            world.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
            mujoco.mj_forward(world.model, world.data)

        info = dict(world.scene.task_info())
        # waypoints may be (x, y) or (x, y, z); progress/lap math is in the xy plane
        pts = np.asarray(info["waypoints"], dtype=np.float64)[:, :2]
        nxt = np.roll(pts, -1, axis=0)
        seg_len = np.hypot(*(nxt - pts).T)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
        total = float(info["track_length"])

        pos, _ = world.robot_pose()
        s0, _ = _project(pts, cum, pos[:2])
        info.update(
            {
                "_pts": pts,
                "_cum": cum,
                "s": s0,
                "progress": 0.0,
                "laps_done": 0,
                "goal": _point_at(pts, cum, total, s0 + self._lookahead_m(world)),
            }
        )
        return info

    def reward(self, world, task_state, action, collided) -> tuple[float, bool, dict[str, Any]]:
        pts, cum = task_state["_pts"], task_state["_cum"]
        total = float(task_state["track_length"])
        half_width = float(task_state["road_half_width"])

        pos, _ = world.robot_pose()
        s, lateral = _project(pts, cum, pos[:2])
        ds = s - task_state["s"]
        if ds < -total / 2:
            ds += total
        elif ds > total / 2:
            ds -= total
        ds = float(np.clip(ds, -MAX_DS_PER_STEP, MAX_DS_PER_STEP))
        task_state["s"] = s
        task_state["progress"] += ds
        task_state["goal"] = _point_at(pts, cum, total, s + self._lookahead_m(world))

        reward = ds - TIME_PENALTY - CENTER_PENALTY * min(lateral / half_width, 1.5)
        terminated = False
        info: dict[str, Any] = {
            "progress_m": task_state["progress"],
            "lap_fraction": task_state["progress"] / total,
            "lateral_m": lateral,
        }

        if lateral > half_width:
            reward -= OFFROAD_PENALTY
        if collided:
            reward -= COLLISION_PENALTY
            terminated = True

        laps_now = int(task_state["progress"] // total)
        if laps_now > task_state["laps_done"]:
            task_state["laps_done"] = laps_now
            reward += LAP_BONUS
        info["laps_done"] = task_state["laps_done"]
        if self.laps > 0 and task_state["laps_done"] >= self.laps:
            terminated = True
            info["success"] = True
        return reward, terminated, info
