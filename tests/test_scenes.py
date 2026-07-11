import mujoco
import numpy as np
import pytest

from robosim.core.registry import list_scenes
from robosim.core.world import build_world

ALL_SCENES = ["flat", "stairs", "slope", "obstacles", "rough_terrain"]


def test_all_scenes_registered():
    for name in ALL_SCENES:
        assert name in list_scenes()


@pytest.mark.parametrize("scene", ALL_SCENES)
def test_scene_loads_and_renders(scene):
    world = build_world(scene, "diffdrive")
    assert world.model.nq > 0
    renderer = mujoco.Renderer(world.model, height=100, width=140)
    renderer.update_scene(world.data)
    frame = renderer.render()
    assert frame.std() > 0


@pytest.mark.parametrize("scene", ALL_SCENES)
def test_scene_physics_stable(scene):
    world = build_world(scene, "diffdrive")
    for _ in range(300):
        world.data.ctrl[:] = [3.0, 3.0]
        mujoco.mj_step(world.model, world.data)
    assert np.all(np.isfinite(world.data.qpos))
    assert np.all(np.isfinite(world.data.qvel))


def test_stairs_parametric_options():
    w1 = build_world("stairs:height=0.05,count=4", "diffdrive")
    w2 = build_world("stairs:height=0.05,count=10", "diffdrive")
    assert w1.model.ngeom != w2.model.ngeom


def test_obstacles_seeded_reproducible():
    w1 = build_world("obstacles:seed=42,count=5", "diffdrive")
    w2 = build_world("obstacles:seed=42,count=5", "diffdrive")
    np.testing.assert_array_equal(w1.model.geom_pos, w2.model.geom_pos)


def test_obstacles_maze_variant():
    w = build_world("obstacles:maze=1,seed=1", "diffdrive")
    assert w.model.nq > 0


def test_rough_terrain_has_hfield():
    w = build_world("rough_terrain", "diffdrive")
    assert w.model.nhfield == 1
