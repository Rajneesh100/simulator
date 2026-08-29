"""What a driver can actually see: line-of-sight, range-limited perception of rivals
and rockets, with velocities ESTIMATED from successive sightings.

The physics engine knows every car's exact position and velocity; a driver does not.
`Perception` models eyes + mirrors instead of an oracle:

- A rival is visible only if it is within `view_range` AND one ray from the observer's
  eye point to the rival's chassis center reaches that rival first — a wall, a hill
  crest, or another car in between hides it. 360 degrees (front, sides, mirrors).
- Velocity is not read from physics. The observer remembers where it last SAW each
  rival and differences the two sightings over the elapsed time (two glances). First
  sighting -> velocity 0; reappearing after occlusion restarts the estimate. Leading a
  target therefore takes real skill, and long-range shots inherit estimation error.
- Rockets in flight are visible under the same rule (you can see the exhaust coming,
  not through a wall); their velocity is taken as observed (the streak direction).

`mode="oracle"` returns the physics truth (for debugging / ablations). Both the learner
and the scripted gunners look through the same `Perception`, so nobody shoots through
walls.
"""

from __future__ import annotations

import mujoco
import numpy as np

from robosim.combat.rockets import RocketSystem

EYE_HEIGHT = 0.25  # m above chassis center: clears own cabin and wheels
_OWN_MARCH = 4  # ray may start inside/over own geometry; step past it this many times


class Perception:
    def __init__(self, model, system: RocketSystem, view_range: float = 12.0,
                 mode: str = "los"):
        if mode not in ("los", "oracle"):
            raise ValueError(f"unknown combat sensor '{mode}' (los | oracle)")
        self.system = system
        self.view_range = float(view_range)
        self.mode = mode
        self._last_seen: dict[tuple[int, int], tuple[np.ndarray, float, np.ndarray]] = {}
        self._geomid = np.zeros(1, dtype=np.int32)

    def reset(self) -> None:
        self._last_seen.clear()

    # ------------------------------------------------------------------ sight

    def eye(self, data, observer: int) -> np.ndarray:
        return self.system.car_pos(data, observer) + np.array([0.0, 0.0, EYE_HEIGHT])

    def line_of_sight(self, model, data, observer: int, point, target_owner: int = -1) -> bool:
        """True if the ray eye -> point reaches `point` (or the geom of `target_owner`)
        before anything else. Own geometry along the first bit of the ray is skipped."""
        eye = self.eye(data, observer)
        vec = np.asarray(point, dtype=np.float64) - eye
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return True
        direction = vec / dist
        traveled = 0.0
        for _ in range(_OWN_MARCH):
            d = mujoco.mj_ray(
                model, data, eye + traveled * direction, direction, None, 1, -1, self._geomid
            )
            if d < 0:
                return True  # nothing in the way at all
            owner = int(self.system.geom_owner[self._geomid[0]])
            if owner == observer:
                traveled += d + 0.01  # own cabin / wheel: look past it
                continue
            if traveled + d >= dist - 1e-6:
                return True  # first obstacle is at or beyond the point itself
            return owner == target_owner and target_owner >= 0
        return False

    # ------------------------------------------------------------------ rivals

    def rivals(self, model, data, observer: int, now: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """[(index, pos3, vel3_estimate), ...] for every live rival the observer can see."""
        out = []
        me = self.eye(data, observer)
        for i in range(self.system.n_cars):
            if i == observer or self.system.is_dead(i):
                continue
            pos = self.system.car_pos(data, i)
            if self.mode == "oracle":
                out.append((i, pos, self.system.car_vel(data, i)))
                continue
            key = (observer, i)
            if np.linalg.norm(pos - me) > self.view_range or not self.line_of_sight(
                model, data, observer, pos, target_owner=i
            ):
                self._last_seen.pop(key, None)  # lost sight: the estimate restarts
                continue
            prev = self._last_seen.get(key)
            if prev is None:
                vel = np.zeros(3)
            elif now - prev[1] > 1e-9:
                vel = (pos - prev[0]) / (now - prev[1])
            else:
                vel = prev[2]  # asked again within the same glance (obs + gunner)
                pos = prev[0]
            self._last_seen[key] = (pos.copy(), now, vel)
            out.append((i, pos, vel))
        return out

    # ------------------------------------------------------------------ rockets

    def rockets(self, model, data, observer: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(pos3, vel3), ...] for rockets in flight the observer can see (not its own)."""
        out = []
        me = self.eye(data, observer)
        for r in self.system.rockets:
            if r.owner == observer:
                continue
            if self.mode == "oracle":
                out.append((r.pos, r.vel))
                continue
            if np.linalg.norm(r.pos - me) > self.view_range:
                continue
            if self.line_of_sight(model, data, observer, r.pos):
                out.append((r.pos, r.vel))
        return out
