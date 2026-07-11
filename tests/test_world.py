import mujoco
import numpy as np

from robosim.core.registry import list_robots, list_scenes
from robosim.core.world import build_world


def test_flat_scene_and_diffdrive_load():
    world = build_world("flat", "diffdrive")
    assert world.model.nq > 0
    assert world.model.nu == 2


def test_registry_lists_flat_and_diffdrive():
    assert "flat" in list_scenes()
    assert "diffdrive" in list_robots()


def test_offscreen_render_produces_frame():
    world = build_world("flat", "diffdrive")
    renderer = mujoco.Renderer(world.model, height=120, width=160)
    renderer.update_scene(world.data)
    frame = renderer.render()
    assert frame.shape == (120, 160, 3)
    assert frame.std() > 0


def test_driving_forward_moves_forward_without_sliding():
    world = build_world("flat", "diffdrive")
    action = np.array([1.0, 0.0], dtype=np.float32)
    ctrl_map = world.robot.action_to_ctrl(action)
    names = [world.model.actuator(i).name for i in range(world.model.nu)]
    for _ in range(500):
        for n in names:
            world.data.actuator(n).ctrl[0] = ctrl_map[n]
        mujoco.mj_step(world.model, world.data)
    pos, _ = world.robot_pose()
    assert pos[0] > 0.3  # moved forward substantially
    assert abs(pos[1]) < 0.05  # negligible lateral drift/sliding


def test_pure_rotation_does_not_translate():
    world = build_world("flat", "diffdrive")
    action = np.array([0.0, 1.0], dtype=np.float32)
    ctrl_map = world.robot.action_to_ctrl(action)
    names = [world.model.actuator(i).name for i in range(world.model.nu)]
    for _ in range(500):
        for n in names:
            world.data.actuator(n).ctrl[0] = ctrl_map[n]
        mujoco.mj_step(world.model, world.data)
    pos, _ = world.robot_pose()
    assert np.hypot(pos[0], pos[1]) < 0.05


def test_reset_returns_to_spawn():
    world = build_world("flat", "diffdrive")
    action = np.array([1.0, 0.0], dtype=np.float32)
    ctrl_map = world.robot.action_to_ctrl(action)
    names = [world.model.actuator(i).name for i in range(world.model.nu)]
    for _ in range(100):
        for n in names:
            world.data.actuator(n).ctrl[0] = ctrl_map[n]
        mujoco.mj_step(world.model, world.data)
    world.reset()
    pos, _ = world.robot_pose()
    np.testing.assert_allclose(pos, world.spawn_pos, atol=1e-6)
