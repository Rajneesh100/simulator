# Rocket Fire-Fight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--firearms rocket` mode where race cars fire gravity-obeying rockets, train `rally6` to shoot rivals and dodge, and animate rockets/blasts in `robosim race`, with every existing command unchanged when the flag is absent.

**Architecture:** Rockets are Python-simulated point projectiles (`combat/rockets.py`) stepped alongside `mj_step`, using `mj_ray` for impact and a blast sphere for kills; a scripted intercept solver (`combat/gunner.py`) arms opponents. The policy interface grows a 16-float combat block in the obs and 3 extra actions `[fire, az, el]`; `VersusRaceEnv` and `ui/race.py` share the same `RocketSystem`.

**Tech Stack:** Python 3.11, MuJoCo 3.x (`mj_ray`, `mjv_connector`), Gymnasium, Stable-Baselines3 PPO, numpy, pytest. Run everything with `uv run` from `simulator/`.

**Spec:** `docs/superpowers/specs/2026-08-30-rocket-firefight-design.md`

## Global Constraints

- Default `firearms="none"` must leave obs/action spaces, rewards, CLI behavior, and all existing tests byte-for-byte unchanged.
- Control rate 20 Hz (`physics.CONTROL_HZ`), physics 2 ms (`physics.TIMESTEP`); rockets advance every physics substep.
- Constants (verbatim from spec): `ROCKET_SPEED=40.0`, `BLAST_RADIUS=2.0`, `COOLDOWN_S=2.0`, `ARMING_M=0.5`, `MAX_FLIGHT_S=4.0`, `LAUNCH_HEIGHT=0.2`, `KILL_BONUS=15.0`, `DEATH_PENALTY=20.0`, `SHOT_COST=0.3`, `COMBAT_NORM_DIST=20.0`, `COMBAT_K=3`, combat block = 16 floats, `MAX_ELEVATION = 45°`, race-mode `RESPAWN_S=4.0`.
- HUD/overlay strings ASCII only (existing test enforces).
- `envs/obs.py` remains the single definition of the observation layout.
- Commit after every task on branch `fire-fight`; commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Tests: `uv run pytest tests/test_combat.py -v` for new code; full `uv run pytest -q` at the end of Tasks 4, 6, 7.

---

### Task 1: Rocket projectile system (`combat/rockets.py`)

**Files:**
- Create: `src/robosim/combat/__init__.py` (empty)
- Create: `src/robosim/combat/rockets.py`
- Test: `tests/test_combat.py`

**Interfaces:**
- Produces:
  - `aim_direction(yaw: float, az: float, el: float) -> np.ndarray(3)` unit vector.
  - `action_to_aim(az_cmd: float, el_cmd: float) -> tuple[float, float]` maps `[-1,1]` to `az ∈ [-π, π]`, `el ∈ [0, π/4]`.
  - `yaw_from_quat(quat_wxyz) -> float`.
  - `class RocketSystem(model, chassis_body_ids: list[int], body_subtrees: list[frozenset[int]], qvel_adrs: list[int], rocket_speed=40.0, blast_radius=2.0, cooldown_s=2.0)` with:
    - `.rockets: list[Rocket]`, `.blasts: list[Blast]`, `.launchers: list[Launcher]`
    - `.ready(i) -> bool`, `.is_dead(i) -> bool`
    - `.fire(data, i, az, el) -> Rocket | None` (az relative to car heading)
    - `.step(model, data, dt) -> None` (gravity, impact, blasts, cooldown, dead timers)
    - `.detonate(model, data, pos, owner) -> Blast`
    - `.revive(i)`, `.reset()`, `.pop_events() -> list[dict]`
  - `Rocket` fields: `owner, pos, vel, age, trail (deque), alive`; `Blast` fields: `pos, owner, time, victims`; `Launcher` fields: `car, cooldown_left, last_dir, shots, kills, deaths, dead_for` (`dead_for < 0` = alive).
  - Events: `{"type": "shot", "car": i}`, `{"type": "kill", "shooter": s, "victim": v}`, `{"type": "self_kill", "car": i}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_combat.py
"""Rocket fire-fight: ballistics, impact, blast, gunner, obs block, env, warm start."""

import mujoco
import numpy as np
import pytest

from robosim.core import physics
from robosim.core.world import build_multi_world
from robosim.sensors.lidar import body_subtree_ids

G = 9.81


def _make_world(n_cars=2, scene="racetrack:obstacles=0"):
    world = build_multi_world(scene, "racecar:max_speed=12,max_torque=1.2", n_cars=n_cars)
    return world


def _rocket_system(world, **kw):
    from robosim.combat.rockets import RocketSystem

    chassis = [world.model.body(n).id for n in world.root_body_names]
    subtrees = [body_subtree_ids(world.model, c) for c in chassis]
    return RocketSystem(world.model, chassis, subtrees, world.qvel_adrs, **kw)


def _place(world, i, xy, yaw, z=None):
    """Teleport car i to xy with heading yaw (rad), zero velocity."""
    adr = world.qpos_adrs[i]
    if z is None:
        z = world.data.qpos[adr + 2]
    half = yaw / 2.0
    world.data.qpos[adr : adr + 3] = [xy[0], xy[1], z]
    world.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0, 0, np.sin(half)]
    vadr = world.qvel_adrs[i]
    world.data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(world.model, world.data)


def _fly(world, system, seconds):
    dt = world.model.opt.timestep
    for _ in range(int(seconds / dt)):
        mujoco.mj_step(world.model, world.data)
        system.step(world.model, world.data, dt)


def test_aim_direction_and_action_mapping():
    from robosim.combat.rockets import action_to_aim, aim_direction

    d = aim_direction(0.0, 0.0, 0.0)
    assert np.allclose(d, [1, 0, 0])
    d = aim_direction(0.0, np.pi / 2, 0.0)  # left of heading
    assert np.allclose(d, [0, 1, 0], atol=1e-9)
    d = aim_direction(0.0, 0.0, np.pi / 4)
    assert np.isclose(d[2], np.sin(np.pi / 4)) and np.isclose(np.linalg.norm(d), 1.0)
    az, el = action_to_aim(-1.0, -1.0)
    assert np.isclose(az, -np.pi) and np.isclose(el, 0.0)
    az, el = action_to_aim(0.5, 1.0)
    assert np.isclose(az, np.pi / 2) and np.isclose(el, np.pi / 4)


def test_rocket_follows_parabola_to_analytic_range():
    """Fired at 30 deg from a stationary car on flat ground, the rocket lands near
    v^2 sin(2*el)/g (+ the small extra from launch height)."""
    world = _make_world(1, scene="flat")
    system = _rocket_system(world, rocket_speed=20.0)
    _place(world, 0, (0.0, 0.0), 0.0)
    el = np.deg2rad(30)
    rocket = system.fire(world.data, 0, az=0.0, el=el)
    assert rocket is not None
    launch = rocket.pos.copy()
    _fly(world, system, 3.0)
    assert not rocket.alive and len(system.blasts) == 1
    landed = system.blasts[0].pos
    flat_range = 20.0**2 * np.sin(2 * el) / G  # 35.3 m
    dist = np.hypot(*(landed[:2] - launch[:2]))
    assert flat_range < dist < flat_range + 1.5  # launch height adds a little


def test_rocket_inherits_platform_velocity():
    """A sideways shot from a car moving forward at 10 m/s keeps that forward
    component: the rocket's world velocity is car_vel + speed*dir."""
    world = _make_world(1, scene="flat")
    system = _rocket_system(world, rocket_speed=40.0)
    _place(world, 0, (0.0, 0.0), 0.0)
    vadr = world.qvel_adrs[0]
    world.data.qvel[vadr : vadr + 3] = [10.0, 0.0, 0.0]
    rocket = system.fire(world.data, 0, az=np.pi / 2, el=0.0)
    assert np.allclose(rocket.vel, [10.0, 40.0, 0.0], atol=1e-6)


def test_rocket_kills_car_ahead_and_shooter_survives():
    world = _make_world(2, scene="flat")
    system = _rocket_system(world)
    _place(world, 0, (0.0, 0.0), 0.0)
    _place(world, 1, (8.0, 0.0), 0.0)
    assert system.fire(world.data, 0, az=0.0, el=0.0) is not None
    _fly(world, system, 1.0)
    assert system.is_dead(1) and not system.is_dead(0)
    events = system.pop_events()
    assert {"type": "shot", "car": 0} in events
    assert {"type": "kill", "shooter": 0, "victim": 1} in events
    assert system.launchers[0].kills == 1 and system.launchers[1].deaths == 1
    # the kill impulse launched the victim
    assert world.data.qvel[world.qvel_adrs[1] + 2] > 1.0


def test_blast_kills_shooter_inside_radius():
    world = _make_world(2, scene="flat")
    system = _rocket_system(world)
    _place(world, 0, (0.0, 0.0), 0.0)
    _place(world, 1, (1.2, 0.0), 0.0)
    pos = world.data.xpos[world.model.body(world.root_body_names[1]).id].copy()
    blast = system.detonate(world.model, world.data, pos, owner=0)
    assert set(blast.victims) == {0, 1}
    assert system.is_dead(0) and system.is_dead(1)
    events = system.pop_events()
    assert {"type": "self_kill", "car": 0} in events
    assert {"type": "kill", "shooter": 0, "victim": 1} in events


def test_cooldown_and_dead_cannot_fire():
    world = _make_world(2, scene="flat")
    system = _rocket_system(world, cooldown_s=2.0)
    _place(world, 0, (0.0, 0.0), 0.0)
    _place(world, 1, (30.0, 0.0), 0.0)
    assert system.ready(0)
    assert system.fire(world.data, 0, 0.0, 0.0) is not None
    assert not system.ready(0)
    assert system.fire(world.data, 0, 0.0, 0.0) is None
    _fly(world, system, 2.1)
    assert system.ready(0)
    system.detonate(world.model, world.data, world.data.xpos[world.model.body(world.root_body_names[0]).id], owner=1)
    assert system.is_dead(0) and system.fire(world.data, 0, 0.0, 0.0) is None
    system.revive(0)
    assert not system.is_dead(0) and system.ready(0)


def test_rocket_does_not_detonate_on_own_launcher():
    world = _make_world(1, scene="flat")
    system = _rocket_system(world)
    _place(world, 0, (0.0, 0.0), 0.0)
    system.fire(world.data, 0, az=np.pi, el=0.0)  # backwards, over own body
    _fly(world, system, 0.05)  # 2 m of flight
    assert not system.is_dead(0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_combat.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'robosim.combat'`

- [ ] **Step 3: Implement `combat/rockets.py`**

```python
# src/robosim/combat/__init__.py
"""Rockets, blasts, and scripted gunners for --firearms rocket."""
```

```python
# src/robosim/combat/rockets.py
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
        data.qvel[adr : adr + 3] += KILL_IMPULSE_RADIAL * away + [0.0, 0.0, KILL_IMPULSE_UP]
        data.qvel[adr + 3 : adr + 6] += self.rng.uniform(-KILL_SPIN, KILL_SPIN, size=3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_combat.py -v`
Expected: 7 PASS. If `test_rocket_kills_car_ahead_and_shooter_survives` fails because the rocket flies over the target and lands beyond blast range, lower `LAUNCH_HEIGHT` to `0.15` (keep it above the cabin; check `racecar.py` cabin geom size) and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/robosim/combat tests/test_combat.py
git commit -m "feat(combat): rocket projectiles with gravity, mj_ray impact, blast kills

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Intercept solver + scripted gunner (`combat/gunner.py`)

**Files:**
- Create: `src/robosim/combat/gunner.py`
- Test: `tests/test_combat.py` (append)

**Interfaces:**
- Consumes: `RocketSystem`, `aim_direction`, `MAX_ELEVATION` from Task 1.
- Produces:
  - `solve_intercept(shooter_pos, shooter_vel, target_pos, target_vel, speed, g, t_max=3.0) -> tuple[float, float, float] | None` returning `(heading_world, el, t_flight)`.
  - `class ScriptedGunner(gun_range=20.0, sigma_deg=2.0, rng=None)` with `.decide(system, data, shooter: int) -> tuple[float, float] | None` returning `(az_rel, el)` or None.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_combat.py


def _simulate_flight(pos, vel, g, t, dt=0.002):
    p, v = pos.astype(float).copy(), vel.astype(float).copy()
    for _ in range(int(t / dt)):
        v[2] -= g * dt
        p += v * dt
    return p


def test_solve_intercept_hits_crossing_target():
    from robosim.combat.gunner import solve_intercept
    from robosim.combat.rockets import aim_direction

    shooter_pos = np.array([0.0, 0.0, 0.3])
    shooter_vel = np.array([8.0, 0.0, 0.0])
    target_pos = np.array([12.0, -4.0, 0.1])
    target_vel = np.array([0.0, 8.0, 0.0])  # crossing left to right ahead
    sol = solve_intercept(shooter_pos, shooter_vel, target_pos, target_vel, 40.0, G)
    assert sol is not None
    heading, el, t = sol
    assert 0.0 <= el <= np.pi / 4 and 0.1 < t < 1.0
    vel = shooter_vel + 40.0 * aim_direction(0.0, heading, el)
    rocket_at_t = _simulate_flight(shooter_pos, vel, G, t)
    target_at_t = target_pos + target_vel * t
    assert np.linalg.norm(rocket_at_t - target_at_t) < 0.3


def test_solve_intercept_misses_target_that_turns_hard():
    """The dodge principle: the solution assumes constant velocity; a target that
    brakes and turns after launch is not where the rocket goes."""
    from robosim.combat.gunner import solve_intercept
    from robosim.combat.rockets import aim_direction

    shooter_pos = np.array([0.0, 0.0, 0.3])
    shooter_vel = np.zeros(3)
    target_pos = np.array([15.0, 0.0, 0.1])
    target_vel = np.array([0.0, 10.0, 0.0])
    heading, el, t = solve_intercept(shooter_pos, shooter_vel, target_pos, target_vel, 40.0, G)
    vel = 40.0 * aim_direction(0.0, heading, el)
    rocket_at_t = _simulate_flight(shooter_pos, vel, G, t)
    dodged = target_pos + np.array([0.0, -10.0, 0.0]) * t  # reversed direction
    assert np.linalg.norm(rocket_at_t - dodged) > 2.0  # outside the blast radius


def test_solve_intercept_none_when_out_of_reach():
    from robosim.combat.gunner import solve_intercept

    assert solve_intercept(np.zeros(3), np.zeros(3), np.array([500.0, 0, 0]),
                           np.zeros(3), 40.0, G) is None


def test_scripted_gunner_fires_at_target_in_range_and_holds_at_point_blank():
    from robosim.combat.gunner import ScriptedGunner

    world = _make_world(2, scene="flat")
    system = _rocket_system(world)
    gunner = ScriptedGunner(gun_range=20.0, sigma_deg=0.0, rng=np.random.default_rng(0))
    _place(world, 0, (0.0, 0.0), 0.0)
    _place(world, 1, (10.0, 3.0), 0.0)
    aim = gunner.decide(system, world.data, 0)
    assert aim is not None
    az, el = aim
    assert np.isclose(az, np.arctan2(3.0, 10.0), atol=0.05)  # stationary: aim straight at it
    _place(world, 1, (1.0, 0.0), 0.0)  # inside its own blast radius
    assert gunner.decide(system, world.data, 0) is None
    _place(world, 1, (40.0, 0.0), 0.0)  # beyond gun range
    assert gunner.decide(system, world.data, 0) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_combat.py -k "intercept or gunner" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'robosim.combat.gunner'`

- [ ] **Step 3: Implement `combat/gunner.py`**

```python
# src/robosim/combat/gunner.py
"""Scripted rocket aiming: the exact moving-platform / moving-target intercept.

For a flight time t the rocket must leave the shooter with world velocity
    v_req(t) = (target_xy + target_vel_xy*t - shooter_xy) / t   (horizontal)
               (dz + 0.5*g*t^2) / t                              (vertical)
and the muzzle vector is m = v_req - shooter_vel. The physical constraint is
|m| = rocket_speed; f(t) = |m(t)| - speed is monotone-ish decreasing for the low
arc, so bisection on t finds the solution (or proves there is none in range).

Used for opponents in versus training (the frozen driver never held a launcher)
and for weaponless checkpoints in `robosim race`. The learner gets none of this:
it must discover the lead and the elevation from its own observation.
"""

from __future__ import annotations

import numpy as np

from robosim.combat.rockets import LAUNCH_HEIGHT, MAX_ELEVATION, RocketSystem


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
    if el < 0.0 or el > MAX_ELEVATION:
        return None
    heading = float(np.arctan2(m[1], m[0]))
    return heading, el, t


class ScriptedGunner:
    """Fires at the nearest live rival within `gun_range` whenever an intercept exists
    and the predicted impact is outside the shooter's own blast radius."""

    def __init__(self, gun_range: float = 20.0, sigma_deg: float = 2.0,
                 rng: np.random.Generator | None = None):
        self.gun_range = float(gun_range)
        self.sigma = np.deg2rad(sigma_deg)
        self.rng = rng if rng is not None else np.random.default_rng()

    def decide(self, system: RocketSystem, data, shooter: int) -> tuple[float, float] | None:
        if not system.ready(shooter):
            return None
        sp = system.car_pos(data, shooter) + np.array([0.0, 0.0, LAUNCH_HEIGHT])
        sv = system.car_vel(data, shooter)
        yaw = system.car_yaw(data, shooter)
        best = None
        for i in range(system.n_cars):
            if i == shooter or system.is_dead(i):
                continue
            tp = system.car_pos(data, i)
            dist = float(np.linalg.norm(tp - sp))
            if dist > self.gun_range:
                continue
            if best is None or dist < best[0]:
                best = (dist, i, tp)
        if best is None:
            return None
        _, i, tp = best
        tv = system.car_vel(data, i)
        sol = solve_intercept(sp, sv, tp, tv, system.rocket_speed, system.gravity)
        if sol is None:
            return None
        heading, el, t = sol
        impact = tp + tv * t
        if np.linalg.norm(impact - system.car_pos(data, shooter)) <= system.blast_radius + 0.5:
            return None  # would take myself out
        az = heading - yaw
        az = float(np.arctan2(np.sin(az), np.cos(az)))  # wrap to [-pi, pi]
        if self.sigma > 0:
            az += float(self.rng.normal(0.0, self.sigma))
            el = float(np.clip(el + self.rng.normal(0.0, self.sigma), 0.0, MAX_ELEVATION))
        return az, el
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_combat.py -v`
Expected: 11 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/robosim/combat/gunner.py tests/test_combat.py
git commit -m "feat(combat): exact intercept solver and scripted gunner

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Combat observation block (`envs/obs.py`)

**Files:**
- Modify: `src/robosim/envs/obs.py`
- Test: `tests/test_combat.py` (append)

**Interfaces:**
- Produces:
  - `COMBAT_K = 3`, `COMBAT_DIM = 16`, `COMBAT_NORM_DIST = 20.0`, `COMBAT_SPEED_NORM = 12.0`, `THREAT_CONE_DEG = 30.0`.
  - `obs_dim(n_lidar_beams, action_dim, combat: bool = False) -> int`.
  - `combat_block(me_xy, me_quat_wxyz, me_vel_xy, others: list[tuple[xy, vel_xy]], weapon_ready: bool, rockets: list[tuple[pos3, vel3]]) -> np.ndarray(16, float32)`.
  - `build_obs(..., combat: np.ndarray | None = None)` — appends `combat` when given.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_combat.py


def test_combat_block_layout_and_ordering():
    from robosim.envs.obs import COMBAT_DIM, combat_block, obs_dim

    assert COMBAT_DIM == 16
    assert obs_dim(48, 5, combat=True) == 48 + 3 + 2 + 3 + 5 + 16
    assert obs_dim(48, 2) == 58  # unchanged default
    me_xy = np.array([0.0, 0.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])  # heading +x
    me_vel = np.array([5.0, 0.0])
    others = [
        (np.array([10.0, 0.0]), np.array([0.0, 0.0])),  # ahead, 10 m, closing at 5
        (np.array([0.0, 4.0]), np.array([0.0, 0.0])),  # left, 4 m
    ]
    block = combat_block(me_xy, quat, me_vel, others, weapon_ready=True, rockets=[])
    assert block.shape == (16,)
    # nearest first: the car 4 m to the left
    assert np.isclose(block[0], 4.0 / 20.0)
    assert np.isclose(block[1], 1.0, atol=1e-6) and np.isclose(block[2], 0.0, atol=1e-6)
    # second: car ahead at 10 m, closing speed 5 m/s
    assert np.isclose(block[4], 0.5) and np.isclose(block[6], 1.0, atol=1e-6)
    assert np.isclose(block[7], 5.0 / 12.0)
    # third slot empty
    assert np.allclose(block[8:12], 0.0)
    assert block[12] == 1.0  # weapon ready
    assert np.allclose(block[13:16], 0.0)  # no threat


def test_combat_block_threat_detection():
    from robosim.envs.obs import combat_block

    me_xy = np.array([0.0, 0.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    incoming = (np.array([6.0, 0.0, 0.3]), np.array([-40.0, 0.0, 0.0]))  # toward me
    passing = (np.array([3.0, 0.0, 0.3]), np.array([0.0, 40.0, 0.0]))  # sideways
    block = combat_block(me_xy, quat, np.zeros(2), [], False, rockets=[passing, incoming])
    assert block[12] == 0.0
    assert np.isclose(block[13], 6.0 / 20.0)
    assert np.isclose(block[15], 1.0, atol=1e-6)  # bearing 0 -> cos 1


def test_build_obs_appends_combat_block():
    from robosim.envs.obs import build_obs

    base = build_obs(np.ones(4, np.float32), np.zeros(2), np.array([1.0, 0, 0, 0]),
                     (1.0, 0.0), np.zeros(5), np.zeros(2))
    assert base.shape == (4 + 3 + 5 + 2,)
    with_combat = build_obs(np.ones(4, np.float32), np.zeros(2), np.array([1.0, 0, 0, 0]),
                            (1.0, 0.0), np.zeros(5), np.zeros(5), combat=np.zeros(16, np.float32))
    assert with_combat.shape == (4 + 3 + 5 + 5 + 16,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_combat.py -k "combat_block or appends" -v`
Expected: FAIL with `ImportError: cannot import name 'COMBAT_DIM'`

- [ ] **Step 3: Implement in `envs/obs.py`**

Replace the module docstring's layout line and add the block. Final file content:

```python
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
    `rockets` = [(pos3, vel3), ...] of rockets in flight (any owner)."""
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
```

(The threat bearing is the direction *to the rocket* from me, so the policy knows which side the shot is coming from: `-to_me` points from me to the rocket.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_combat.py tests/test_envs.py tests/test_sensors.py -q`
Expected: all PASS (existing obs users unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/robosim/envs/obs.py tests/test_combat.py
git commit -m "feat(obs): optional 16-float combat block (rivals, weapon ready, threat)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Versus env with firearms (`envs/versus_env.py`, `core/world.py`)

**Files:**
- Modify: `src/robosim/core/world.py` (add `MultiWorld.car_velocity`)
- Modify: `src/robosim/envs/versus_env.py`
- Modify: `src/robosim/envs/base_env.py:54-65` (accept and reject `firearms`)
- Test: `tests/test_combat.py` (append)

**Interfaces:**
- Consumes: `RocketSystem`, `action_to_aim`, `ScriptedGunner`, `combat_block`, `obs_dim`.
- Produces: `VersusRaceEnv(..., firearms: str = "none", rocket_speed: float = 40.0)`; attribute `env.rockets: RocketSystem | None`; constants `KILL_BONUS = 15.0`, `DEATH_PENALTY = 20.0`, `SHOT_COST = 0.3`; `info["kills"]`, `info["shots"]`, `info["killed"]`. `RoboSimEnv(..., firearms="none", rocket_speed=40.0)` raises `ValueError` if `firearms != "none"`.
- `MultiWorld.car_velocity(i) -> np.ndarray(3)` world-frame linear velocity.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_combat.py

RALLY = "runs/rally3/best_model.zip"


def _versus_env(**kw):
    pytest.importorskip("stable_baselines3")
    from robosim.envs.versus_env import VersusRaceEnv

    base = dict(
        scene="racetrack:obstacles=0", robot="racecar:max_speed=12,max_torque=1.2",
        task="race:laps=2,lookahead=2.5", opponent_checkpoint=RALLY, opponents=2,
        n_lidar_beams=48, lidar_fov_deg=360.0, lidar_max_range=12.0,
        max_episode_steps=100, firearms="rocket",
    )
    base.update(kw)
    return VersusRaceEnv(**base)


def test_versus_firearms_spaces_and_check_env():
    from gymnasium.utils.env_checker import check_env

    env = _versus_env()
    assert env.action_space.shape == (5,)
    assert env.observation_space.shape == (48 + 3 + 2 + 3 + 5 + 16,)
    check_env(env, skip_render_check=True)


def test_versus_without_firearms_unchanged():
    env = _versus_env(firearms="none")
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (58,)
    assert env.rockets is None


def test_versus_learner_kill_reward_and_shot_cost():
    from robosim.envs.versus_env import KILL_BONUS, SHOT_COST

    env = _versus_env(opponents=1)
    env.reset(seed=0)
    # park the opponent 8 m straight ahead of the learner, both stationary
    lpos, lquat = env.world.car_pose(0)
    from robosim.combat.rockets import yaw_from_quat

    yaw = yaw_from_quat(lquat)
    ahead = lpos[:2] + 8.0 * np.array([np.cos(yaw), np.sin(yaw)])
    adr = env.world.qpos_adrs[1]
    env.world.data.qpos[adr : adr + 2] = ahead
    env.world.data.qpos[adr + 3 : adr + 7] = lquat
    env.world.data.qvel[env.world.qvel_adrs[1] : env.world.qvel_adrs[1] + 6] = 0.0
    mujoco.mj_forward(env.world.model, env.world.data)
    env.opp_model = _NoOpModel()  # opponent sits still
    fire = np.array([0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float32)  # fire, az 0, el 0
    hold = np.array([0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    _, r0, *_ = env.step(fire)
    total = r0
    killed = False
    for _ in range(20):  # 1 s of flight
        _, r, term, trunc, info = env.step(hold)
        total += r
        if info.get("kills"):
            killed = True
            break
    assert killed and env.rockets.is_dead(1)
    assert total > KILL_BONUS - SHOT_COST - 2.0  # progress/time terms are small here


class _NoOpModel:
    def predict(self, batch, deterministic=False, **kw):
        return np.zeros((len(batch), 2), dtype=np.float32), None


def test_versus_learner_death_terminates_with_penalty():
    from robosim.envs.versus_env import DEATH_PENALTY

    env = _versus_env(opponents=1)
    env.reset(seed=0)
    pos = env.world.car_pose(0)[0]
    env.rockets.detonate(env.world.model, env.world.data, pos, owner=1)
    _, r, term, trunc, info = env.step(np.zeros(5, dtype=np.float32))
    assert term and info.get("killed") is True
    assert r <= -DEATH_PENALTY + 1.0


def test_versus_opponents_shoot_and_respawn():
    """Scripted gunners on the frozen opponents put rockets in the air."""
    env = _versus_env(opponents=2)
    env.reset(seed=1)
    shots = 0
    for _ in range(100):
        _, _, term, trunc, _ = env.step(np.array([0.3, 0.0, 0.0, 0.0, -1.0], np.float32))
        shots = max(shots, sum(L.shots for L in env.rockets.launchers[1:]))
        if term or trunc:
            break
    assert shots >= 1


def test_base_env_rejects_firearms():
    from robosim.envs.base_env import RoboSimEnv

    with pytest.raises(ValueError):
        RoboSimEnv(scene="flat", robot="racecar", task="race", firearms="rocket")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_combat.py -k "versus or base_env" -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'firearms'`

- [ ] **Step 3: Add `car_velocity` to `MultiWorld` in `core/world.py`**

After `car_speed` (line ~105):

```python
    def car_velocity(self, i: int) -> np.ndarray:
        """World-frame linear velocity of car i's chassis (free-joint qvel)."""
        adr = self.qvel_adrs[i]
        return self.data.qvel[adr : adr + 3].copy()
```

- [ ] **Step 4: Make `RoboSimEnv` reject firearms explicitly (`base_env.py`)**

Add two parameters at the end of the `__init__` signature (before `render_mode`):

```python
        firearms: str = "none",
        rocket_speed: float = 40.0,
```

and as the first statement after `super().__init__()`:

```python
        if firearms != "none":
            raise ValueError("--firearms needs opponents to shoot: train with --versus")
```

- [ ] **Step 5: Wire firearms into `VersusRaceEnv`**

Edits to `src/robosim/envs/versus_env.py`:

1. Imports — add after the existing `from robosim.sensors.lidar import ...`:

```python
from robosim.combat.gunner import ScriptedGunner
from robosim.combat.rockets import RocketSystem, action_to_aim
from robosim.envs.obs import build_obs, combat_block, obs_dim
```
(and drop the old `from robosim.envs.obs import build_obs, obs_dim` line).

2. Constants — add after `MAX_DS_PER_STEP = 1.0`:

```python
KILL_BONUS = 15.0  # opponent destroyed by a rocket the learner fired
DEATH_PENALTY = 20.0  # learner destroyed (any rocket, own blast included) -> episode over
SHOT_COST = 0.3  # per launch, so it aims instead of spraying
OPP_GUN_RANGE = 20.0
OPP_AIM_SIGMA_DEG = 2.0
```

3. Docstring — append a paragraph:

```
With `firearms="rocket"` every car carries a launcher (`combat/rockets.py`). The
action grows to [throttle, steer, fire, az, el] and the observation gains the combat
block from `envs/obs.py`. Reward adds KILL_BONUS per rival destroyed by the learner's
rocket, SHOT_COST per launch, and DEATH_PENALTY + termination when the learner is
destroyed — by any rocket, its own blast included. Opponents fire through a scripted
intercept gunner (the frozen driver never held a launcher), so incoming fire exists to
dodge. Destroyed opponents respawn like wrecked ones.
```

4. `__init__` signature — add before `render_mode`:

```python
        firearms: str = "none",
        rocket_speed: float = 40.0,
```

5. In `__init__`, replace the two lines that build `action_dim` / `obs_size` / spaces with:

```python
        self.firearms = firearms
        self.rocket_speed = float(rocket_speed)
        drive_dim = self.world.robots[0].action_dim()
        self.drive_dim = drive_dim
        action_dim = drive_dim + 3 if firearms == "rocket" else drive_dim
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        obs_size = obs_dim(n_lidar_beams, action_dim, combat=(firearms == "rocket"))
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_size,), dtype=np.float32)
        for car in self.cars:
            car.prev_action = np.zeros(action_dim, dtype=np.float32)

        self.rockets: RocketSystem | None = None
        self._gunners: list[ScriptedGunner | None] = [None] * len(self.cars)
        if firearms == "rocket":
            self.rockets = RocketSystem(
                self.world.model,
                [c.chassis_id for c in self.cars],
                [c.body_ids for c in self.cars],
                list(self.world.qvel_adrs),
                rocket_speed=self.rocket_speed,
            )
            for c in self.cars[1:]:
                self._gunners[c.index] = ScriptedGunner(OPP_GUN_RANGE, OPP_AIM_SIGMA_DEG)
        elif firearms != "none":
            raise ValueError(f"unknown firearms '{firearms}' (none | rocket)")
```

6. `_obs_for` — replace its final `return build_obs(...)` with:

```python
        combat = None
        if self.rockets is not None:
            me_xy = pos[:2]
            me_vel = self.world.car_velocity(car.index)[:2]
            others = [
                (self.world.car_pose(c.index)[0][:2], self.world.car_velocity(c.index)[:2])
                for c in self.cars if c is not car and not self.rockets.is_dead(c.index)
            ]
            rockets = [(r.pos, r.vel) for r in self.rockets.rockets if r.owner != car.index]
            combat = combat_block(me_xy, quat, me_vel, others, self.rockets.ready(car.index), rockets)
        return build_obs(ranges, pos[:2], quat, goal, proprio, car.prev_action, combat=combat)
```

7. `_respawn` — append at the end:

```python
        if self.rockets is not None:
            self.rockets.revive(car.index)
```

8. `reset` — after `self.world.reset()` add:

```python
        if self.rockets is not None:
            self.rockets.reset()
            self.rockets.rng = rng
            for g in self._gunners:
                if g is not None:
                    g.rng = rng
```

and change `car.prev_action[:] = 0.0` (already fine — array is the right size).

9. `_opponent_actions` — the frozen model predicts on the full obs but outputs `drive_dim` actions; opponents' `prev_action` must stay the learner's action size. Replace the loop body's start:

```python
        for car, a in zip(self.cars[1:], acts):
            a = np.asarray(a, dtype=np.float32)[: self.drive_dim]
```
and the frozen model must see the obs layout **it** trained with. The frozen checkpoint (rally5) has no combat block and a 2-wide prev_action, so build its rows with a plain obs: change `rows = [self._obs_for(c) for c in self.cars[1:]]` to `rows = [self._opp_obs(c) for c in self.cars[1:]]` and add:

```python
    def _opp_obs(self, car: _Car) -> np.ndarray:
        """Observation in the FROZEN opponent's own layout (no combat block, 2-wide
        prev_action) — a brain must see the obs it trained on."""
        pos, quat = self.world.car_pose(car.index)
        ranges = car.lidar.sense(
            self.world.model, self.world.data,
            exclude_body_id=car.chassis_id, exclude_body_ids=car.body_ids,
            rng=self.np_random,
        )
        proprio = self.world.robots[car.index].read_proprio(
            self.world.model, self.world.data, prefix=self.world.prefixes[car.index]
        )
        la = self.lookahead
        if self.lookahead_time > 0:
            la = max(la, self._car_speed(car) * self.lookahead_time)
        goal = _point_at(self._pts, self._cum, self._total, car.s + la)
        return build_obs(ranges, pos[:2], quat, goal, proprio, car.prev_action[: self.drive_dim])
```
Also keep `car.prev_action` for opponents at `drive_dim` width: in `_opponent_actions` the assignment `car.prev_action = a` already stores the 2-wide array (fine; only `_opp_obs` reads it).

10. `step` — replace the first block up to the physics loop with:

```python
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        opp_actions = self._opponent_actions()
        self.world.set_ctrl(0, self.world.robots[0].action_to_ctrl(action[: self.drive_dim]))
        self.learner.prev_action = action
        for car, a in zip(self.cars[1:], opp_actions):
            self.world.set_ctrl(car.index, self.world.robots[car.index].action_to_ctrl(a))
            car.prev_action = a

        shots = 0
        if self.rockets is not None:
            if action[self.drive_dim] >= 0.5 and self.rockets.ready(0):
                az, el = action_to_aim(action[self.drive_dim + 1], action[self.drive_dim + 2])
                if self.rockets.fire(self.world.data, 0, az, el) is not None:
                    shots += 1
            for car in self.cars[1:]:
                g = self._gunners[car.index]
                if g is not None and not self.rockets.is_dead(car.index):
                    aim = g.decide(self.rockets, self.world.data, car.index)
                    if aim is not None:
                        self.rockets.fire(self.world.data, car.index, *aim)
            # destroyed cars coast: no drive
            for car in self.cars:
                if self.rockets.is_dead(car.index):
                    self.world.set_ctrl(car.index, self.world.robots[car.index].action_to_ctrl(
                        np.zeros(self.drive_dim, dtype=np.float32)))

        sub_dt = self.world.model.opt.timestep
        for _ in range(self._steps_per_action):
            if self._movers:
                update_movers(self.world.data, self._movers, self.world.data.time)
            mujoco.mj_step(self.world.model, self.world.data)
            if self.rockets is not None:
                self.rockets.step(self.world.model, self.world.data, sub_dt)
            self._learner_contacts()
```

11. In `step`, in the opponent loop, treat a rocket death as a wreck. Replace the `newly_wrecked = (...)` expression with:

```python
            destroyed = self.rockets is not None and self.rockets.is_dead(opp.index)
            newly_wrecked = (
                opp.wrecked_for == 0.0
                and (destroyed or opp.flipped_for >= OPP_WRECK_FLIP_S or off_track
                     or opp.no_progress_for >= OPP_NO_PROGRESS_S)
            )
```

12. In `step`, after the `reward += TAKEDOWN_BONUS * takedowns` block, add:

```python
        if self.rockets is not None:
            kills = 0
            killed = False
            for ev in self.rockets.pop_events():
                if ev["type"] == "kill" and ev["shooter"] == 0:
                    kills += 1
                elif ev["type"] == "kill" and ev["victim"] == 0:
                    killed = True
                elif ev["type"] == "self_kill" and ev["car"] == 0:
                    killed = True
            reward += KILL_BONUS * kills - SHOT_COST * shots
            if kills:
                info["kills"] = kills
            if shots:
                info["shots"] = shots
            if killed or self.rockets.is_dead(0):
                reward -= DEATH_PENALTY
                terminated = True
                info["killed"] = True
```

13. Keep the wall-crash check but not for an already-destroyed learner: change `if self._learner_wall_crash() or _uprightness(lquat) < 0.3:` to `if not terminated and (self._learner_wall_crash() or _uprightness(lquat) < 0.3):`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_combat.py tests/test_racing.py -v`
Expected: all PASS. `check_env` may warn about obs bounds — warnings are fine, failures are not.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: all PASS (existing versus smoke test still passes with 2-dim actions).

- [ ] **Step 8: Commit**

```bash
git add src/robosim/core/world.py src/robosim/envs/base_env.py src/robosim/envs/versus_env.py tests/test_combat.py
git commit -m "feat(env): VersusRaceEnv firearms=rocket — 5-dim action, combat obs, kill/death reward, scripted opponent gunners

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Training plumbing — CLI, run config, warm-start expansion

**Files:**
- Modify: `src/robosim/train/run_config.py`
- Modify: `src/robosim/train/bc.py` (add `expand_policy_state_dict`)
- Modify: `src/robosim/train/train.py:89-160, 196-202, 226-240`
- Modify: `src/robosim/cli.py:69-132` (train command)
- Test: `tests/test_combat.py` (append)

**Interfaces:**
- Produces:
  - `expand_policy_state_dict(sd: dict, extra_obs: int, extra_act: int, fire_bias: float = -1.0) -> dict`.
  - `run_config._ENV_KEYS` gains `"firearms"`, `"rocket_speed"`; `env_kwargs_from_run` drops `firearms` when it is `"none"`/None so `RoboSimEnv` callers are unaffected.
  - `run_train(..., firearms: str = "none", rocket_speed: float = 40.0)`; config.yaml records both.
  - CLI: `robosim train --firearms none|rocket --rocket-speed 40`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_combat.py


def test_expand_policy_state_dict_preserves_original_outputs():
    import torch

    from robosim.train.bc import expand_policy_state_dict

    torch.manual_seed(0)
    obs_old, act_old, hidden = 58, 2, 64
    sd = {
        "mlp_extractor.policy_net.0.weight": torch.randn(hidden, obs_old),
        "mlp_extractor.policy_net.0.bias": torch.randn(hidden),
        "mlp_extractor.policy_net.2.weight": torch.randn(hidden, hidden),
        "mlp_extractor.policy_net.2.bias": torch.randn(hidden),
        "mlp_extractor.value_net.0.weight": torch.randn(hidden, obs_old),
        "mlp_extractor.value_net.0.bias": torch.randn(hidden),
        "mlp_extractor.value_net.2.weight": torch.randn(hidden, hidden),
        "mlp_extractor.value_net.2.bias": torch.randn(hidden),
        "action_net.weight": torch.randn(act_old, hidden),
        "action_net.bias": torch.randn(act_old),
        "value_net.weight": torch.randn(1, hidden),
        "value_net.bias": torch.randn(1),
        "log_std": torch.randn(act_old),
    }
    new = expand_policy_state_dict(sd, extra_obs=19, extra_act=3)
    assert new["mlp_extractor.policy_net.0.weight"].shape == (hidden, 77)
    assert new["mlp_extractor.value_net.0.weight"].shape == (hidden, 77)
    assert new["action_net.weight"].shape == (5, hidden)
    assert new["log_std"].shape == (5,)
    assert new["action_net.bias"][2].item() == -1.0  # fire stays off until learned
    obs = torch.randn(4, obs_old)
    padded = torch.cat([obs, torch.zeros(4, 19)], dim=1)

    def forward(s, x):
        h = torch.tanh(x @ s["mlp_extractor.policy_net.0.weight"].T + s["mlp_extractor.policy_net.0.bias"])
        h = torch.tanh(h @ s["mlp_extractor.policy_net.2.weight"].T + s["mlp_extractor.policy_net.2.bias"])
        return h @ s["action_net.weight"].T + s["action_net.bias"]

    assert torch.allclose(forward(sd, obs), forward(new, padded)[:, :2], atol=1e-6)


def test_run_config_recovers_firearms(tmp_path):
    import yaml

    from robosim.train.run_config import env_kwargs_from_run

    run = tmp_path / "rally6"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({
        "n_lidar_beams": 48, "firearms": "rocket", "rocket_speed": 35.0}))
    ckpt = run / "best_model.zip"
    ckpt.write_bytes(b"")
    kw = env_kwargs_from_run(str(ckpt))
    assert kw["firearms"] == "rocket" and kw["rocket_speed"] == 35.0
    (run / "config.yaml").write_text(yaml.safe_dump({"n_lidar_beams": 48, "firearms": "none"}))
    kw = env_kwargs_from_run(str(ckpt))
    assert "firearms" not in kw and "rocket_speed" not in kw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_combat.py -k "expand or run_config" -v`
Expected: FAIL (`ImportError` for `expand_policy_state_dict`; `KeyError: 'firearms'`).

- [ ] **Step 3: Implement `expand_policy_state_dict` in `train/bc.py`**

Append to `src/robosim/train/bc.py`:

```python
def expand_policy_state_dict(sd: dict, extra_obs: int, extra_act: int,
                             fire_bias: float = -1.0) -> dict:
    """Grow an SB3 MlpPolicy state dict to a wider obs/action space without changing
    what it computes on the old inputs: new input columns get zero weight (the combat
    block is ignored until learned), new action rows get zero weight, bias `fire_bias`
    for the fire channel (stays holstered) and 0 for the aim channels, and log_std 0
    (unit exploration) for the new rows. Used to warm-start rally6 from rally5.

    The new obs columns must be APPENDED to the old layout (they are: prev_action[2:]
    and the combat block both come after every column the old policy read)."""
    import torch

    out = dict(sd)
    for key in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        if key in out and extra_obs > 0:
            w = out[key]
            out[key] = torch.cat([w, torch.zeros(w.shape[0], extra_obs, dtype=w.dtype)], dim=1)
    if extra_act > 0:
        w = out["action_net.weight"]
        out["action_net.weight"] = torch.cat(
            [w, torch.zeros(extra_act, w.shape[1], dtype=w.dtype)], dim=0)
        b = out["action_net.bias"]
        new_b = torch.zeros(extra_act, dtype=b.dtype)
        new_b[0] = fire_bias
        out["action_net.bias"] = torch.cat([b, new_b], dim=0)
        if "log_std" in out:
            ls = out["log_std"]
            out["log_std"] = torch.cat([ls, torch.zeros(extra_act, dtype=ls.dtype)], dim=0)
    return out
```

- [ ] **Step 4: Update `train/run_config.py`**

```python
_ENV_KEYS = ("n_lidar_beams", "lidar_fov_deg", "lidar_max_range", "max_episode_steps",
             "firearms", "rocket_speed")
```
and in `env_kwargs_from_run`, after building `kwargs`:

```python
    if kwargs.get("firearms") in (None, "none"):
        kwargs.pop("firearms", None)
        kwargs.pop("rocket_speed", None)
```

- [ ] **Step 5: Update `train/train.py`**

1. Signature: add after `versus_opponents: int = 2,`:
```python
    firearms: str = "none",
    rocket_speed: float = 40.0,
```
2. In `env_kwargs = dict(...)` add `firearms=firearms, rocket_speed=rocket_speed,`. (RoboSimEnv accepts and rejects non-"none"; VersusRaceEnv uses them.)
3. In the `if versus_checkpoint:` block, after the print, add:
```python
        if firearms != "none":
            print(f"[train] FIREARMS: {firearms} at {rocket_speed} m/s — opponents use the scripted gunner")
```
and before that block add a guard:
```python
    if firearms != "none" and not versus_checkpoint:
        raise ValueError("--firearms needs opponents to shoot: add --versus <checkpoint>")
```
4. Replace the `bc_init` loading block (lines ~196-202) with:
```python
    if bc_init:
        try:
            state_dict = torch.load(bc_init, map_location="cpu")
            target = model.policy.state_dict()
            key_in = "mlp_extractor.policy_net.0.weight"
            if key_in in state_dict and key_in in target:
                extra_obs = target[key_in].shape[1] - state_dict[key_in].shape[1]
                extra_act = target["action_net.weight"].shape[0] - state_dict["action_net.weight"].shape[0]
                if extra_obs > 0 or extra_act > 0:
                    from robosim.train.bc import expand_policy_state_dict

                    state_dict = expand_policy_state_dict(state_dict, extra_obs, extra_act)
                    print(f"[train] expanded bc_init by +{extra_obs} obs / +{extra_act} actions")
            model.policy.load_state_dict(state_dict, strict=False)
            print(f"[train] loaded BC-pretrained weights from {bc_init}")
        except Exception as e:
            print(f"[train] WARNING: failed to load bc_init '{bc_init}': {e}")
```
5. In the `config = {...}` dict add `"firearms": firearms, "rocket_speed": rocket_speed,`.

- [ ] **Step 6: Update the `train` command in `cli.py`**

Add options after `versus_opponents`:
```python
    firearms: str = typer.Option("none", help="none | rocket — arm every car (versus mode only)"),
    rocket_speed: float = typer.Option(40.0, help="Rocket muzzle speed m/s (range = v^2 sin(2 el)/g)"),
```
and pass `firearms=firearms, rocket_speed=rocket_speed,` to `run_train`.

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_combat.py tests/test_training.py -q`
Expected: all PASS.

- [ ] **Step 8: Smoke-train 2k steps headless to prove the recipe wires up end to end**

Run:
```bash
uv run robosim train --task "race:laps=2,lookahead=2.5,lookahead_time=0.6" \
  --scene "racetrack:layout=gp,scale=25,seed=11,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --steps 2000 --n-envs 1 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 --max-episode-steps 200 \
  --versus runs/rally5/best_model.zip --versus-opponents 2 \
  --firearms rocket --bc-init runs/rally5/policy_init.pt --run-name rocket_smoke
```
Expected: prints `[train] expanded bc_init by +19 obs / +3 actions`, finishes, writes `runs/rocket_smoke/config.yaml` containing `firearms: rocket`. (If `runs/rally5/policy_init.pt` is missing, create it first: `uv run python -c "import torch; from stable_baselines3 import PPO; torch.save(PPO.load('runs/rally5/best_model.zip', device='cpu').policy.state_dict(), 'runs/rally5/policy_init.pt')"`.) Then `rm -rf runs/rocket_smoke`.

- [ ] **Step 9: Commit**

```bash
git add src/robosim/train/bc.py src/robosim/train/run_config.py src/robosim/train/train.py src/robosim/cli.py tests/test_combat.py
git commit -m "feat(train): --firearms/--rocket-speed, config recovery, warm-start expansion from weaponless checkpoints

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Race mode — rockets, kills, animation, `--firearms` on `robosim race`

**Files:**
- Modify: `src/robosim/ui/overlay.py` (add `draw_rocket`, `draw_blast`, `draw_aim`)
- Modify: `src/robosim/ui/race.py`
- Modify: `src/robosim/cli.py:153-185` (race command)
- Test: `tests/test_combat.py` (append), `tests/test_ui.py` (append)

**Interfaces:**
- Consumes: `RocketSystem`, `ScriptedGunner`, `action_to_aim`, `combat_block`, `env_kwargs_from_run` (now returns `firearms`).
- Produces:
  - `overlay.draw_rocket(scn, rocket, rgba)`, `overlay.draw_blast(scn, blast, age_s, radius)`, `overlay.draw_aim(scn, origin, direction, rgba)`.
  - `run_race(..., firearms: str = "none", rocket_speed: float = 40.0)`.
  - CLI `robosim race --firearms rocket --rocket-speed 40`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_ui.py


def test_rocket_and_blast_overlays_draw_headless():
    import mujoco
    import numpy as np

    from robosim.combat.rockets import Blast, Rocket
    from robosim.core.world import build_world
    from robosim.ui import overlay

    world = build_world("flat", "racecar")
    scn = mujoco.MjvScene(world.model, maxgeom=2000)
    scn.ngeom = 0
    rocket = Rocket(owner=0, pos=np.array([1.0, 0.0, 0.5]), vel=np.array([40.0, 0.0, -1.0]))
    for k in range(12):
        rocket.trail.append(np.array([1.0 - 0.08 * k, 0.0, 0.5]))
    overlay.draw_rocket(scn, rocket, (1.0, 0.2, 0.2, 0.9))
    n_after_rocket = scn.ngeom
    assert n_after_rocket >= 3  # body capsule + nose + trail
    blast = Blast(pos=np.array([3.0, 0.0, 0.1]), owner=0, time=0.0, victims=[])
    overlay.draw_blast(scn, blast, age_s=0.2, radius=2.0)
    assert scn.ngeom > n_after_rocket
    overlay.draw_blast(scn, blast, age_s=0.9, radius=2.0)  # past the animation: no-op
    overlay.draw_aim(scn, np.array([0.0, 0.0, 0.3]), np.array([1.0, 0.0, 0.0]), (1, 0, 0, 1))
    assert scn.ngeom <= scn.maxgeom
```

```python
# append to tests/test_combat.py


def test_race_module_exposes_firearms_option():
    import inspect

    from robosim.ui.race import run_race

    params = inspect.signature(run_race).parameters
    assert params["firearms"].default == "none"
    assert params["rocket_speed"].default == 40.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui.py::test_rocket_and_blast_overlays_draw_headless tests/test_combat.py::test_race_module_exposes_firearms_option -v`
Expected: FAIL (`AttributeError: module 'robosim.ui.overlay' has no attribute 'draw_rocket'`, `KeyError: 'firearms'`).

- [ ] **Step 3: Add the draw functions to `ui/overlay.py`**

Append after `draw_goal`:

```python
RGBA_BLAST_CORE = np.array([1.0, 0.55, 0.1, 0.9], dtype=np.float32)
RGBA_BLAST_RING = np.array([1.0, 0.3, 0.05, 0.8], dtype=np.float32)
RGBA_EXHAUST = np.array([1.0, 0.8, 0.4, 0.5], dtype=np.float32)
BLAST_ANIM_S = 0.5
ROCKET_LEN = 0.18
ROCKET_RADIUS = 0.03


def _add_capsule(scn, from_, to, rgba, radius) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=_ZERO3, pos=_ZERO3, mat=_EYE3,
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
        np.asarray(from_, dtype=np.float64), np.asarray(to, dtype=np.float64),
    )
    scn.ngeom += 1


def draw_rocket(scn, rocket, rgba) -> None:
    """Capsule body along the velocity in the owner's color, bright nose, and an
    exhaust trail of fading spheres sampled from the last positions."""
    v = np.asarray(rocket.vel, dtype=np.float64)
    n = np.linalg.norm(v)
    d = v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    nose = np.asarray(rocket.pos, dtype=np.float64)
    tail = nose - d * ROCKET_LEN
    _add_capsule(scn, tail, nose, rgba, ROCKET_RADIUS)
    _add_sphere(scn, nose, np.array([1.0, 1.0, 0.8, 1.0], dtype=np.float32), ROCKET_RADIUS * 1.2)
    pts = list(rocket.trail)[::-5][1:9]  # every 5th substep, newest first, 8 puffs
    for k, p in enumerate(pts):
        fade = 1.0 - k / 8.0
        rgba_k = RGBA_EXHAUST.copy()
        rgba_k[3] *= fade
        _add_sphere(scn, p, rgba_k, 0.02 + 0.03 * (1 - fade))


def draw_blast(scn, blast, age_s: float, radius: float) -> None:
    """Expanding translucent fireball (0.3 m -> radius, orange -> dark red, fading)
    plus a flash ring on the ground marking the damage circle."""
    if age_s < 0.0 or age_s > BLAST_ANIM_S:
        return
    f = age_s / BLAST_ANIM_S
    r = 0.3 + (radius - 0.3) * f
    rgba = RGBA_BLAST_CORE.copy()
    rgba[1] *= 1.0 - 0.7 * f
    rgba[3] *= 1.0 - f
    _add_sphere(scn, blast.pos, rgba, r)
    ring = RGBA_BLAST_RING.copy()
    ring[3] *= 1.0 - f
    z = float(blast.pos[2])
    n_seg = 24
    for k in range(n_seg):
        a0 = 2 * np.pi * k / n_seg
        a1 = 2 * np.pi * (k + 1) / n_seg
        p0 = (blast.pos[0] + radius * np.cos(a0), blast.pos[1] + radius * np.sin(a0), z)
        p1 = (blast.pos[0] + radius * np.cos(a1), blast.pos[1] + radius * np.sin(a1), z)
        _add_line(scn, p0, p1, ring, 3.0)


def draw_aim(scn, origin, direction, rgba) -> None:
    """Short line showing a launcher's last firing direction (while reloading)."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    _add_line(scn, o, o + 0.8 * d, rgba, 2.0)
```

- [ ] **Step 4: Wire firearms into `ui/race.py`**

1. Imports — add:
```python
from robosim.combat.gunner import ScriptedGunner
from robosim.combat.rockets import RocketSystem, action_to_aim
from robosim.envs.obs import build_obs, combat_block
```
(replace the existing `from robosim.envs.obs import build_obs`).

2. Constants — add after `NO_PROGRESS_TRIGGER_S`:
```python
RESPAWN_S = 4.0  # destroyed car sits burning this long before the marshal rebuilds it
CHARCOAL = (0.15, 0.15, 0.15)
```

3. `_CarState.__init__` — add fields:
```python
        self.armed = False  # this car's checkpoint trained with firearms (5-dim action)
        self.gunner: ScriptedGunner | None = None
        self.rgb = rgb
```

4. Docstring — append:
```
`--firearms rocket` arms every car (`combat/rockets.py`). Cars whose checkpoint trained
with firearms fire from their own policy action; cars driven by weaponless checkpoints
use the scripted intercept gunner, so any checkpoint can fight. Rockets fly real
parabolas, detonate on the first thing they touch, and destroy everything within the
blast radius — shooter included. A destroyed car turns charcoal, tumbles, and is
rebuilt by the marshal after RESPAWN_S. The leaderboard shows K/D.
```

5. `run_race` signature — add `firearms: str = "none", rocket_speed: float = 40.0,`.

6. Per-model settings loop — record whether each model is armed. Change the loop to:
```python
    lidar_kwargs_per_model, la_base, la_time, armed_per_model = [], [], [], []
    for c in checkpoints:
        env_kwargs = env_kwargs_from_run(c)
        ...  # existing lidar/lookahead lines unchanged
        armed_per_model.append(env_kwargs.get("firearms") == "rocket")
    if firearms == "none" and any(armed_per_model):
        raise SystemExit("this checkpoint trained with --firearms rocket: run race with --firearms rocket")
```

7. After `_tint_cars(world)` and the car_states loop, build the system:
```python
    rockets: RocketSystem | None = None
    if firearms == "rocket":
        rockets = RocketSystem(
            world.model, [c.chassis_id for c in car_states], [c.body_ids for c in car_states],
            list(world.qvel_adrs), rocket_speed=rocket_speed, rng=rng,
        )
        for car in car_states:
            car.armed = armed_per_model[car.model_idx]
            if not car.armed:
                car.gunner = ScriptedGunner(
                    gun_range=20.0, sigma_deg=2.0 + 6.0 * v * rng.uniform(), rng=rng)
        print(f"  firearms: rocket at {rocket_speed:g} m/s — "
              f"{sum(c.armed for c in car_states)} policy gunners, "
              f"{sum(c.gunner is not None for c in car_states)} scripted", flush=True)
    elif firearms != "none":
        raise SystemExit(f"unknown --firearms '{firearms}' (none | rocket)")
```
and in the car_states loop set `car.prev_action = np.zeros(5 if armed_per_model[m] else 2, dtype=np.float32)` right after constructing `car`.

8. `marshal_respawn` — at the end add:
```python
        if rockets is not None:
            rockets.revive(car.index)
            _tint_car(world, car.index, car.rgb)
```
and add a module-level helper next to `_tint_cars`:
```python
def _tint_car(world, i: int, rgb) -> None:
    for geom_name in world.robots[i].collision_geom_names():
        geom_id = world.model.geom(world.prefixes[i] + geom_name).id
        world.model.geom_rgba[geom_id][:3] = rgb
```

9. `reset_race` — add `if rockets is not None: rockets.reset()` and `_tint_car(world, car.index, car.rgb)` for every car; also `car.prev_action[:] = 0.0` is already there.

10. Obs rows — replace the `obs_rows.append(build_obs(...))` with:
```python
                    combat = None
                    if rockets is not None and car.armed:
                        others = [
                            (world.car_pose(o.index)[0][:2], world.car_velocity(o.index)[:2])
                            for o in car_states if o is not car and not rockets.is_dead(o.index)
                        ]
                        in_flight = [(r.pos, r.vel) for r in rockets.rockets if r.owner != car.index]
                        combat = combat_block(pos[:2], quat, world.car_velocity(car.index)[:2],
                                              others, rockets.ready(car.index), in_flight)
                    obs_rows.append(
                        build_obs(ranges, pos[:2], quat, goal, proprio, car.prev_action, combat=combat)
                    )
```

11. Action application loop — replace the body with:
```python
                for car, action in zip(car_states, actions):
                    action = np.asarray(action, dtype=np.float32)
                    drive = action[:2].copy()
                    drive[0] *= car.aggression
                    if car.jitter > 0:
                        drive = drive + rng.normal(0.0, car.jitter, size=2)
                    drive = np.clip(drive, -1.0, 1.0).astype(np.float32)
                    dead = rockets is not None and rockets.is_dead(car.index)
                    if dead:
                        drive = np.zeros(2, dtype=np.float32)
                    else:
                        car.update_stall(world.car_speed(car.index), control_dt, rng)
                        if car.recovering:  # scripted un-wedge: back out with a twist
                            drive = np.array([-0.6, car.recovery_steer], dtype=np.float32)
                    world.set_ctrl(car.index, world.robots[car.index].action_to_ctrl(drive))
                    if rockets is not None and not dead:
                        if car.armed:
                            if action[2] >= 0.5:
                                az, el = action_to_aim(action[3], action[4])
                                rockets.fire(world.data, car.index, az, el)
                        elif car.gunner is not None:
                            aim = car.gunner.decide(rockets, world.data, car.index)
                            if aim is not None:
                                rockets.fire(world.data, car.index, *aim)
                    if car.armed:
                        car.prev_action = np.concatenate([drive, action[2:5]]).astype(np.float32)
                    else:
                        car.prev_action = drive
```

12. Physics loop — add rocket stepping:
```python
                sub_dt = world.model.opt.timestep
                for _ in range(steps_per_action):
                    if movers:
                        update_movers(world.data, movers, world.data.time)
                    mujoco.mj_step(world.model, world.data)
                    if rockets is not None:
                        rockets.step(world.model, world.data, sub_dt)
                race_time += control_dt
                if rockets is not None:
                    for ev in rockets.pop_events():
                        if ev["type"] == "kill":
                            s, vv = car_states[ev["shooter"]], car_states[ev["victim"]]
                            _tint_car(world, vv.index, CHARCOAL)
                            print(f"[race] KILL: {s.label} -> {vv.label}", flush=True)
                        elif ev["type"] == "self_kill":
                            c = car_states[ev["car"]]
                            _tint_car(world, c.index, CHARCOAL)
                            print(f"[race] SELF-KILL: {c.label}", flush=True)
```

13. Watchdog loop — skip marshal watchdogs for dead cars and rebuild after `RESPAWN_S`. At the top of the `for car in car_states:` progress loop body (after `car.laps = ...`), add:
```python
                    if rockets is not None and rockets.is_dead(car.index):
                        car.progress_anchor = car.progress
                        car.progress_timer = 0.0
                        if rockets.launchers[car.index].dead_for >= RESPAWN_S:
                            marshal_respawn(car, "destroyed")
                        continue
```

14. Drawing — inside `with viewer.lock():` after the lidar block add:
```python
                if rockets is not None:
                    for r in rockets.rockets:
                        overlay.draw_rocket(scn, r, car_states[r.owner].rgba)
                    for b in rockets.blasts:
                        overlay.draw_blast(scn, b, world.data.time - b.time, rockets.blast_radius)
                    for car in car_states:
                        L = rockets.launchers[car.index]
                        if L.cooldown_left > 0 and L.last_dir is not None and L.dead_for < 0:
                            pos, _ = world.car_pose(car.index)
                            overlay.draw_aim(scn, pos + [0, 0, 0.2], L.last_dir, car.rgba)
```

15. Leaderboard — extend `board_right` per car with K/D and DEAD:
```python
                    def status(car):
                        if rockets is not None and rockets.is_dead(car.index):
                            return " DEAD"
                        return (" FLIP" if car.flipped_for > 0
                                else (" OFF" if car.off_track_for > 0
                                      else (" rec" if car.recovering else "")))

                    def kd(car):
                        if rockets is None:
                            return ""
                        L = rockets.launchers[car.index]
                        return f" K{L.kills}/D{L.deaths}"

                    board_right = "\n".join(
                        f"{car.label} L{min(car.laps, laps)}/{laps} "
                        f"{car.progress:6.1f}m {world.car_speed(car.index):3.1f}m/s"
                        + kd(car) + status(car)
                        for car in standings
                    )
```
and in the final standings print add `f" K{rockets.launchers[car.index].kills}/D{rockets.launchers[car.index].deaths}" if rockets else ""`.

16. Startup print: append ` · rockets: {firearms}` when firearms != "none".

- [ ] **Step 5: Add `--firearms` / `--rocket-speed` to the `race` command in `cli.py`**

```python
    firearms: str = typer.Option("none", help="none | rocket — arm every car; weaponless "
                                 "checkpoints get a scripted gunner"),
    rocket_speed: float = typer.Option(40.0, help="Rocket muzzle speed m/s (sets range)"),
```
and pass `firearms=firearms, rocket_speed=rocket_speed,` to `run_race`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_ui.py tests/test_combat.py -q`
Expected: all PASS.

- [ ] **Step 7: Headless race smoke (no window): drive the race loop pieces for 3 s**

Write and run a throwaway script in the scratchpad (not committed) that imports `run_race`'s dependencies and does what the loop does without a viewer: build world with 3 cars, `RocketSystem`, `ScriptedGunner` per car, step 60 control steps with the rally5 policy driving, and assert at least one rocket launched and no exception. Simplest form:

```bash
uv run python - <<'EOF'
import numpy as np, mujoco
from robosim.core.world import build_multi_world
from robosim.combat.rockets import RocketSystem
from robosim.combat.gunner import ScriptedGunner
from robosim.sensors.lidar import body_subtree_ids
from robosim.core import physics
w = build_multi_world("racetrack:layout=gp,scale=18,seed=11,obstacles=0,road_width=3.5",
                      "racecar:max_speed=12,max_torque=1.2", n_cars=3)
ch = [w.model.body(n).id for n in w.root_body_names]
rs = RocketSystem(w.model, ch, [body_subtree_ids(w.model, c) for c in ch], list(w.qvel_adrs))
g = [ScriptedGunner(rng=np.random.default_rng(i)) for i in range(3)]
dt = w.model.opt.timestep
for step in range(200):
    for i in range(3):
        w.set_ctrl(i, w.robots[i].action_to_ctrl(np.array([0.5, 0.0])))
        aim = g[i].decide(rs, w.data, i)
        if aim: rs.fire(w.data, i, *aim)
    for _ in range(physics.physics_steps_per_control_step()):
        mujoco.mj_step(w.model, w.data); rs.step(w.model, w.data, dt)
print("shots", [L.shots for L in rs.launchers], "deaths", [L.deaths for L in rs.launchers], "events", rs.pop_events()[:5])
EOF
```
Expected: prints non-zero shots; typically at least one death (cars start bunched on the grid).

- [ ] **Step 8: Live check (macOS window)**

Run: `uv run robosim race --checkpoint runs/rally5/best_model.zip --scene "racetrack:layout=gp,scale=18,seed=11,obstacles=0,road_width=3.5" --robot racecar:max_speed=12,max_torque=1.2 --cars 5 --laps 5 --variation 0.8 --firearms rocket`
Expected: rockets visible as colored capsules with exhaust puffs, orange fireballs with a ground ring on impact, `KILL:` lines in the terminal, charcoal cars tumbling then respawning, `K0/D1` on the leaderboard. Press `p` for a screenshot and inspect it. (If this session cannot open a window, record that the live check was skipped and rely on Step 7.)

- [ ] **Step 9: Commit**

```bash
git add src/robosim/ui/overlay.py src/robosim/ui/race.py src/robosim/cli.py tests/test_ui.py tests/test_combat.py
git commit -m "feat(race): --firearms rocket — policy/scripted gunners, kills with respawn, rocket + blast animation, K/D board

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Docs, decision log, full verification

**Files:**
- Modify: `README.md` (racing section, after the rally5 recipe ~line 343)
- Modify: `CLAUDE.md` (decision log, append at bottom)
- Modify: `UI_CHECKLIST.md` (add a firearms walkthrough item)

- [ ] **Step 1: README — add the fire-fight section after the rally5 block**

```markdown
### Fire-fight: `--firearms rocket`

Every car gets a rocket launcher (`src/robosim/combat/`). Rockets leave with the car's
own velocity plus `--rocket-speed` (40 m/s) along the aim, fall under gravity (real
parabola: range = v^2 sin(2 el) / g, ~1.8 deg of elevation for a 10 m shot), detonate
on the first thing they touch, and destroy every car within 2 m — the shooter included.
A destroyed car turns charcoal, tumbles, and is rebuilt by the marshal after 4 s.

Watch it today with any checkpoint (weaponless brains get a scripted intercept gunner):

```bash
uv run robosim race --checkpoint runs/rally5/best_model.zip \
  --scene "racetrack:layout=gp,scale=18,seed=11,obstacles=0,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 \
  --cars 5 --laps 5 --variation 0.8 --firearms rocket
```

Train a brain that shoots and dodges itself — rally6 (action grows to
[throttle, steer, fire, az, el]; obs gains a 16-float combat block, see `envs/obs.py`;
rally5's weights are expanded in place so it starts out driving like rally5):

```bash
uv run python -c "import torch; from stable_baselines3 import PPO; torch.save(PPO.load('runs/rally5/best_model.zip', device='cpu').policy.state_dict(), 'runs/rally5/policy_init.pt')"
uv run robosim train --task "race:laps=2,lookahead=2.5,lookahead_time=0.6" \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --steps 1000000 --n-envs 8 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 --max-episode-steps 1500 \
  --versus runs/rally5/best_model.zip --versus-opponents 2 \
  --firearms rocket --rocket-speed 40 \
  --bc-init runs/rally5/policy_init.pt --run-name rally6

# then: rally6 (policy gunner) vs rally5 (scripted gunner)
uv run robosim race --checkpoint runs/rally6/best_model.zip --checkpoint runs/rally5/best_model.zip \
  --scene "racetrack:layout=gp,scale=18,seed=11,obstacles=0,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --cars 6 --laps 5 --firearms rocket
```

Reward in versus+firearms: +15 per kill, -20 and episode over when destroyed (own blast
included), -0.3 per shot, on top of the usual progress/overtake/takedown/win terms.
```

- [ ] **Step 2: CLAUDE.md decision log entry (append at the bottom of the log)**

```markdown
- **2026-08-30 — Rockets are Python projectiles, not MuJoCo bodies (`combat/rockets.py`).**
  A rocket is (pos, vel, owner) stepped every 2 ms substep with the model's gravity and
  an `mj_ray` sweep for impact; a blast sphere (2 m) kills every chassis inside, shooter
  included. No spec changes, so every pre-existing checkpoint/scene/command is untouched
  when `--firearms none` (default). Policy interface with firearms: action
  [throttle, steer, fire, az, el]; obs += 16-float combat block (3 nearest rivals'
  range/bearing/closing speed, weapon ready, nearest incoming rocket) — still defined
  only in `envs/obs.py`. The learner gets NO ballistic solver; opponents (and weaponless
  checkpoints in race mode) use `combat/gunner.py`'s exact intercept (bisection on
  flight time for |v_req(t) - v_shooter| = speed), so incoming fire is well aimed and
  dodging is worth learning. Warm start across the space change:
  `bc.expand_policy_state_dict` zero-pads new obs columns and action rows (fire bias -1),
  reproducing the old policy exactly on old inputs — `train --bc-init` does it
  automatically when shapes mismatch. Frozen opponents keep their own obs layout
  (`VersusRaceEnv._opp_obs`) — a brain must see the obs it trained on.
```

- [ ] **Step 3: UI_CHECKLIST.md — add under the race section**

```markdown
- [ ] `--firearms rocket`: colored capsules with exhaust puffs fly real arcs, orange
      fireball + ground ring on impact, `KILL:` in the terminal, hit car goes charcoal
      and tumbles, respawns after ~4 s, leaderboard shows `K/D` and `DEAD`.
```

- [ ] **Step 4: Full test suite**

Run: `uv run pytest -q`
Expected: all PASS, zero failures. Paste the summary line into the commit message.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md UI_CHECKLIST.md
git commit -m "docs: fire-fight recipe, rally6 training, decision log

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review notes

- Spec coverage: physics (T1), aiming/gunner (T2), obs (T3), training env + reward + opponent gunners + respawn (T4), warm start + config + CLI train (T5), race mode + animation + K/D + CLI race (T6), docs/tests (T7). Non-goals untouched.
- Names used consistently: `RocketSystem.fire/step/detonate/ready/is_dead/revive/reset/pop_events/launchers/rockets/blasts`, `action_to_aim`, `combat_block`, `obs_dim(..., combat=)`, `expand_policy_state_dict`, `MultiWorld.car_velocity`, `draw_rocket/draw_blast/draw_aim`.
- Open risk flagged in T1 step 4 (launch height vs. cabin) with the concrete fallback.
