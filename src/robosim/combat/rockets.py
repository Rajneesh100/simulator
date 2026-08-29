"""Rocket projectiles with real ballistics, simulated in Python beside mj_step.

A rocket is a point mass: launched with the firing car's world velocity plus
`rocket_speed` along the aim direction, it falls under the model's gravity (a true
parabola) and detonates on the first geom its swept path crosses (car, wall, road,
terrain) — found with `mj_ray` every physics substep. Detonation destroys every car
whose chassis center lies within `blast_radius`, the shooter included.

No MuJoCo bodies are added: existing models, checkpoints, and scenes are untouched.
The same `RocketSystem` drives training (`envs/versus_env.py`) and the viewer
(`ui/race.py`), so what the policy learned is exactly what you watch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

ROCKET_SPEED = 40.0  # m/s muzzle speed; range = v^2 sin(2 el) / g
BLAST_RADIUS = 2.0  # m; everything inside dies, shooter included
COOLDOWN_S = 2.0  # reload
ARMING_M = 0.5  # rocket ignores its own launcher for this much flight
MAX_FLIGHT_S = 4.0  # self-destruct (no blast) after this
LAUNCH_HEIGHT = 0.2  # m above chassis center, clears cabin and wheels
MAX_ELEVATION = np.pi / 4  # low-arc family only
TRAIL_LEN = 40  # substep positions kept for the exhaust animation
KILL_IMPULSE_UP = 4.0  # m/s added to a destroyed car
KILL_IMPULSE_RADIAL = 3.0
KILL_SPIN = 6.0  # rad/s
BLAST_KEEP_S = 1.0  # blasts stay listed this long for the animation


def yaw_from_quat(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def aim_direction(yaw: float, az: float, el: float) -> np.ndarray:
    """Unit launch direction for azimuth `az` relative to heading `yaw` and elevation
    `el` above horizontal."""
    h = yaw + az
    c = np.cos(el)
    return np.array([c * np.cos(h), c * np.sin(h), np.sin(el)], dtype=np.float64)


def action_to_aim(az_cmd: float, el_cmd: float) -> tuple[float, float]:
    """Policy action -> angles: az [-1,1] -> [-pi, pi], el [-1,1] -> [0, MAX_ELEVATION]."""
    az = float(np.clip(az_cmd, -1.0, 1.0)) * np.pi
    el = (float(np.clip(el_cmd, -1.0, 1.0)) + 1.0) / 2.0 * MAX_ELEVATION
    return az, el


@dataclass
class Rocket:
    owner: int
    pos: np.ndarray
    vel: np.ndarray
    age: float = 0.0
    traveled: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    alive: bool = True


@dataclass
class Blast:
    pos: np.ndarray
    owner: int
    time: float
    victims: list[int]


@dataclass
class Launcher:
    car: int
    cooldown_left: float = 0.0
    last_dir: np.ndarray | None = None
    shots: int = 0
    kills: int = 0
    deaths: int = 0
    dead_for: float = -1.0  # < 0 alive; >= 0 seconds since destroyed


class RocketSystem:
    def __init__(
        self,
        model,
        chassis_body_ids: list[int],
        body_subtrees: list[frozenset[int]],
        qvel_adrs: list[int],
        rocket_speed: float = ROCKET_SPEED,
        blast_radius: float = BLAST_RADIUS,
        cooldown_s: float = COOLDOWN_S,
        rng: np.random.Generator | None = None,
    ):
        self.n_cars = len(chassis_body_ids)
        self.chassis_body_ids = list(chassis_body_ids)
        self.qvel_adrs = list(qvel_adrs)
        self.rocket_speed = float(rocket_speed)
        self.blast_radius = float(blast_radius)
        self.cooldown_s = float(cooldown_s)
        self.gravity = float(-model.opt.gravity[2])
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # geom id -> owning car index (-1 = world)
        self.geom_owner = np.full(model.ngeom, -1, dtype=np.int32)
        for i, ids in enumerate(body_subtrees):
            for g in range(model.ngeom):
                if model.geom_bodyid[g] in ids:
                    self.geom_owner[g] = i
        self.rockets: list[Rocket] = []
        self.blasts: list[Blast] = []
        self.launchers = [Launcher(i) for i in range(self.n_cars)]
        self._events: list[dict] = []
        self._geomid = np.zeros(1, dtype=np.int32)

    # ------------------------------------------------------------------ state

    def ready(self, i: int) -> bool:
        L = self.launchers[i]
        return L.cooldown_left <= 0.0 and L.dead_for < 0.0

    def is_dead(self, i: int) -> bool:
        return self.launchers[i].dead_for >= 0.0

    def revive(self, i: int) -> None:
        L = self.launchers[i]
        L.dead_for = -1.0
        L.cooldown_left = 0.0

    def reset(self) -> None:
        self.rockets.clear()
        self.blasts.clear()
        self._events.clear()
        for L in self.launchers:
            L.cooldown_left = 0.0
            L.last_dir = None
            L.shots = L.kills = L.deaths = 0
            L.dead_for = -1.0

    def pop_events(self) -> list[dict]:
        out, self._events = self._events, []
        return out

    def car_pos(self, data, i: int) -> np.ndarray:
        return data.xpos[self.chassis_body_ids[i]].copy()

    def car_vel(self, data, i: int) -> np.ndarray:
        adr = self.qvel_adrs[i]
        return data.qvel[adr : adr + 3].copy()

    def car_yaw(self, data, i: int) -> float:
        return yaw_from_quat(data.xquat[self.chassis_body_ids[i]])

    # ------------------------------------------------------------------ firing

    def fire(self, data, i: int, az: float, el: float) -> Rocket | None:
        """Launch from car i if ready. `az` is relative to the car's heading."""
        if not self.ready(i):
            return None
        el = float(np.clip(el, 0.0, MAX_ELEVATION))
        direction = aim_direction(self.car_yaw(data, i), az, el)
        pos = self.car_pos(data, i) + np.array([0.0, 0.0, LAUNCH_HEIGHT])
        vel = self.car_vel(data, i) + self.rocket_speed * direction
        rocket = Rocket(owner=i, pos=pos, vel=vel)
        rocket.trail.append(pos.copy())
        self.rockets.append(rocket)
        L = self.launchers[i]
        L.cooldown_left = self.cooldown_s
        L.last_dir = direction
        L.shots += 1
        self._events.append({"type": "shot", "car": i})
        return rocket

    # ------------------------------------------------------------------ stepping

    def step(self, model, data, dt: float) -> None:
        for L in self.launchers:
            if L.cooldown_left > 0.0:
                L.cooldown_left -= dt
            if L.dead_for >= 0.0:
                L.dead_for += dt
        for rocket in self.rockets:
            if rocket.alive:
                self._advance(model, data, rocket, dt)
        self.rockets = [r for r in self.rockets if r.alive]
        self.blasts = [b for b in self.blasts if data.time - b.time <= BLAST_KEEP_S]

    def _advance(self, model, data, rocket: Rocket, dt: float) -> None:
        rocket.vel[2] -= self.gravity * dt
        start = rocket.pos.copy()
        seg = rocket.vel * dt
        seg_len = float(np.linalg.norm(seg))
        if seg_len > 1e-12:
            direction = seg / seg_len
            hit = self._first_hit(model, data, start, direction, seg_len, rocket)
            if hit is not None:
                rocket.alive = False
                self.detonate(model, data, hit, rocket.owner)
                return
        rocket.pos = start + seg
        rocket.traveled += seg_len
        rocket.age += dt
        rocket.trail.append(rocket.pos.copy())
        if rocket.age >= MAX_FLIGHT_S:
            rocket.alive = False  # fizzles mid-air, no blast

    def _first_hit(self, model, data, pnt, direction, seg_len, rocket) -> np.ndarray | None:
        """Nearest geom on the swept segment, skipping the owner's own body while the
        rocket is still within ARMING_M of launch."""
        traveled = 0.0
        for _ in range(4):
            d = mujoco.mj_ray(
                model, data, pnt + traveled * direction, direction, None, 1, -1, self._geomid
            )
            if d < 0 or traveled + d > seg_len:
                return None
            owner = int(self.geom_owner[self._geomid[0]])
            if owner == rocket.owner and rocket.traveled + traveled + d < ARMING_M:
                traveled += d + 0.01  # march past own geometry
                continue
            return pnt + (traveled + d) * direction
        return None

    # ------------------------------------------------------------------ blast

    def detonate(self, model, data, pos, owner: int) -> Blast:
        pos = np.asarray(pos, dtype=np.float64)
        victims = []
        for i in range(self.n_cars):
            if self.is_dead(i):
                continue
            if np.linalg.norm(self.car_pos(data, i) - pos) <= self.blast_radius:
                victims.append(i)
        blast = Blast(pos=pos, owner=owner, time=float(data.time), victims=victims)
        self.blasts.append(blast)
        for v in victims:
            self._kill(data, v, pos)
            if v == owner:
                self._events.append({"type": "self_kill", "car": owner})
            else:
                self.launchers[owner].kills += 1
                self._events.append({"type": "kill", "shooter": owner, "victim": v})
        return blast

    def _kill(self, data, i: int, blast_pos) -> None:
        L = self.launchers[i]
        L.dead_for = 0.0
        L.deaths += 1
        adr = self.qvel_adrs[i]
        away = self.car_pos(data, i) - blast_pos
        away[2] = 0.0
        n = np.linalg.norm(away)
        away = away / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        data.qvel[adr : adr + 3] += KILL_IMPULSE_RADIAL * away + np.array([0.0, 0.0, KILL_IMPULSE_UP])
        data.qvel[adr + 3 : adr + 6] += self.rng.uniform(-KILL_SPIN, KILL_SPIN, size=3)
