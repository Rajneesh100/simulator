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
    speed = 10.0  # keeps the landing inside the flat scene's finite ground plane
    system = _rocket_system(world, rocket_speed=speed)
    _place(world, 0, (0.0, 0.0), 0.0)
    el = np.deg2rad(30)
    rocket = system.fire(world.data, 0, az=0.0, el=el)
    assert rocket is not None
    launch = rocket.pos.copy()
    landed = None
    for _ in range(300):  # 3 s in 10 ms slices; blasts are pruned after 1 s
        _fly(world, system, 0.01)
        if system.blasts:
            landed = system.blasts[0].pos.copy()
            break
    assert not rocket.alive and landed is not None
    flat_range = speed**2 * np.sin(2 * el) / G  # 8.8 m
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
    for _ in range(100):  # up to 1 s, in 10 ms slices so we can catch the kill moment
        _fly(world, system, 0.01)
        if system.is_dead(1):
            break
    assert system.is_dead(1) and not system.is_dead(0)
    # the kill impulse launched the victim (checked right after the blast)
    assert world.data.qvel[world.qvel_adrs[1] + 2] > 1.0
    events = system.pop_events()
    assert {"type": "shot", "car": 0} in events
    assert {"type": "kill", "shooter": 0, "victim": 1} in events
    assert system.launchers[0].kills == 1 and system.launchers[1].deaths == 1


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
    system.detonate(
        world.model, world.data,
        world.data.xpos[world.model.body(world.root_body_names[0]).id], owner=1,
    )
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
