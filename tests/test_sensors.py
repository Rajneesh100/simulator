import mujoco
import numpy as np

from robosim.core.world import build_world
from robosim.sensors.lidar import Lidar
from robosim.sensors.proprio import quat_to_pitch_roll


def test_lidar_returns_sane_ranges_on_flat_scene():
    world = build_world("flat", "diffdrive")
    mujoco.mj_forward(world.model, world.data)
    lidar = Lidar(n_beams=8)
    chassis_id = world.model.body(world.root_body_name).id
    ranges = lidar.sense(world.model, world.data, exclude_body_id=chassis_id)
    assert ranges.shape == (8,)
    assert np.all(ranges >= 0.0) and np.all(ranges <= 1.0)
    # empty flat scene: all beams should read max range (no hits)
    assert np.allclose(ranges, 1.0)


def test_lidar_noise_and_dropout():
    world = build_world("flat", "diffdrive")
    mujoco.mj_forward(world.model, world.data)
    lidar = Lidar(n_beams=32, noise_std=0.1, dropout_prob=1.0)
    chassis_id = world.model.body(world.root_body_name).id
    rng = np.random.default_rng(0)
    ranges = lidar.sense(world.model, world.data, exclude_body_id=chassis_id, rng=rng)
    # dropout_prob=1.0 forces every beam to max range regardless of noise
    assert np.allclose(ranges, 1.0)


def test_proprio_dim_and_values():
    world = build_world("flat", "diffdrive")
    mujoco.mj_forward(world.model, world.data)
    proprio = world.robot.read_proprio(world.model, world.data)
    assert proprio.shape == (5,)


def test_quat_to_pitch_roll_identity():
    pitch, roll = quat_to_pitch_roll(np.array([1.0, 0.0, 0.0, 0.0]))
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6
