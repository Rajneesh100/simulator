"""Scripted rocket aiming: the exact moving-platform / moving-target intercept.

For a flight time t the rocket must leave the shooter with world velocity
    v_req(t) = (target_xy + target_vel_xy*t - shooter_xy) / t   (horizontal)
               (dz + 0.5*g*t^2) / t                              (vertical)
and the muzzle vector is m = v_req - shooter_vel. The physical constraint is
|m| = rocket_speed; f(t) = |m(t)| - speed is decreasing for the low arc, so bisection
on t finds the solution (or proves there is none in range).

Used for opponents in versus training (the frozen driver never held a launcher)
and for weaponless checkpoints in `robosim race`. The learner gets none of this:
it must discover the lead and the elevation from its own observation.
"""

from __future__ import annotations

import numpy as np

from robosim.combat.rockets import (
    LAUNCH_HEIGHT, MAX_ELEVATION, MIN_ARM_M, MIN_ELEVATION, RocketSystem,
)

SELF_SAFETY_MARGIN = 1.0  # m beyond the blast radius the shooter keeps at impact time


def solve_intercept(
    shooter_pos, shooter_vel, target_pos, target_vel, speed: float, g: float,
    t_max: float = 3.0,
) -> tuple[float, float, float] | None:
    """Return (heading_world, elevation, t_flight) or None if no low-arc solution."""
    sp = np.asarray(shooter_pos, dtype=np.float64)
    sv = np.asarray(shooter_vel, dtype=np.float64)
    tp = np.asarray(target_pos, dtype=np.float64)
    tv = np.asarray(target_vel, dtype=np.float64)

    def muzzle(t: float) -> np.ndarray:
        d = tp + tv * t - sp
        v_req = np.array([d[0] / t, d[1] / t, (d[2] + 0.5 * g * t * t) / t])
        return v_req - sv

    def f(t: float) -> float:
        return float(np.linalg.norm(muzzle(t))) - speed

    lo, hi = 0.01, t_max
    if f(lo) < 0:  # target practically on top of us: shortest flight
        t = lo
    elif f(hi) > 0:
        return None  # even the longest allowed flight can't reach
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        t = hi
    m = muzzle(t)
    horiz = float(np.hypot(m[0], m[1]))
    el = float(np.arctan2(m[2], horiz))
    if el < MIN_ELEVATION or el > MAX_ELEVATION:
        return None
    heading = float(np.arctan2(m[1], m[0]))
    return heading, el, t


class ScriptedGunner:
    """Fires at the nearest live rival within `gun_range` whenever an intercept exists
    and the predicted impact is outside the shooter's own blast radius."""

    def __init__(self, gun_range: float = 40.0, sigma_deg: float = 2.0,
                 rng: np.random.Generator | None = None):
        self.gun_range = float(gun_range)
        self.sigma = np.deg2rad(sigma_deg)
        self.rng = rng if rng is not None else np.random.default_rng()

    def decide(self, system: RocketSystem, data, shooter: int, perception=None,
               model=None, now: float = 0.0,
               ignore_cooldown: bool = False) -> tuple[float, float] | None:
        """Pick a target and return (az_rel, el), or None. With `perception` (a
        `combat.perception.Perception`, needs `model` and the sim time `now`) the gunner
        only knows rivals it can SEE and their estimated velocities; without it, it reads
        the physics state (oracle)."""
        if system.is_dead(shooter):
            return None
        if not ignore_cooldown and not system.ready(shooter):
            return None  # ignore_cooldown: "is a shot viable now?" — used for labels
        sp = system.car_pos(data, shooter) + np.array([0.0, 0.0, LAUNCH_HEIGHT])
        sv = system.car_vel(data, shooter)
        yaw = system.car_yaw(data, shooter)
        my_pos = system.car_pos(data, shooter)
        if perception is not None:
            rivals = perception.rivals(model, data, shooter, now)
        else:
            rivals = [
                (i, system.car_pos(data, i), system.car_vel(data, i))
                for i in range(system.n_cars) if i != shooter and not system.is_dead(i)
            ]
        candidates = []
        for i, tp, tv in rivals:
            dist = float(np.linalg.norm(tp - sp))
            if dist <= self.gun_range:
                candidates.append((dist, i, tp, tv))
        candidates.sort(key=lambda c: c[0])
        # nearest viable target: skip rivals with no ballistic solution, rivals inside
        # the fuze's arming distance (the rocket would be a dud), and rivals so close
        # that the blast would catch the shooter — judged at the shooter's PREDICTED
        # position at impact time, since a chasing car drives into its own blast
        for _, i, tp, tv in candidates:
            sol = solve_intercept(sp, sv, tp, tv, system.rocket_speed, system.gravity)
            if sol is None:
                continue
            heading, el, t = sol
            impact = tp + tv * t
            if np.linalg.norm(impact - sp) < MIN_ARM_M + 0.5:
                continue
            me_at_impact = my_pos + sv * t
            if np.linalg.norm(impact - me_at_impact) <= system.blast_radius + SELF_SAFETY_MARGIN:
                continue
            break
        else:
            return None
        az = heading - yaw
        az = float(np.arctan2(np.sin(az), np.cos(az)))  # wrap to [-pi, pi]
        if self.sigma > 0:
            az += float(self.rng.normal(0.0, self.sigma))
            el = float(np.clip(el + self.rng.normal(0.0, self.sigma), MIN_ELEVATION, MAX_ELEVATION))
        return az, el
