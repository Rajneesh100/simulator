"""Headless tests for the Isaac-feel UI layer: theme assets, overlay drawing, HUD
timing, and key/camera state machines. Nothing here opens a window."""

import mujoco
import numpy as np

from robosim.core import theme
from robosim.core.world import build_world
from robosim.ui import overlay


def _themed_render(world, width=320, height=240, scene_option=None):
    renderer = mujoco.Renderer(world.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(world.model, cam)
    cam.azimuth = theme.CAMERA_AZIMUTH
    cam.elevation = theme.CAMERA_ELEVATION
    cam.distance = theme.CAMERA_DISTANCE
    renderer.update_scene(world.data, camera=cam, scene_option=scene_option)
    return renderer


def test_theme_assets_present_on_world():
    world = build_world("flat", "diffdrive")
    assert world.model.ntex >= 2  # skybox + grid
    assert world.model.nmat >= 1
    assert world.model.geom("ground").matid[0] >= 0
    assert world.model.vis.global_.azimuth == theme.CAMERA_AZIMUTH
    assert world.model.vis.global_.offwidth == theme.OFFSCREEN_WIDTH


def test_render_is_dark_but_not_blank():
    world = build_world("flat", "diffdrive")
    renderer = _themed_render(world)
    frame = renderer.render()
    assert frame.mean() < 70  # dark viewport
    assert frame.std() > 3  # but grid/robot/sky are visible


def test_terrain_scene_keeps_own_look_and_renders():
    world = build_world("rough_terrain", "diffdrive")
    assert world.model.geom("terrain").matid[0] == -1  # hfield not grid-themed
    renderer = _themed_render(world)
    assert renderer.render().std() > 3


def test_hud_timer_math():
    t = overlay.HudTimer(window=10.0)
    assert t.rtf() == 0.0 and t.steps_per_sec() == 0.0
    t.update(0.0, 0, now=100.0)
    t.update(0.5, 250, now=100.5)
    t.update(1.0, 250, now=101.0)
    assert abs(t.rtf() - 1.0) < 1e-9
    assert abs(t.steps_per_sec() - 500.0) < 1e-9
    # half-speed sim: 0.25s of sim time per 0.5s wall
    t2 = overlay.HudTimer(window=10.0)
    t2.update(0.0, 0, now=0.0)
    t2.update(0.25, 125, now=0.5)
    assert abs(t2.rtf() - 0.5) < 1e-9


def test_speed_ladder_clamps():
    state = overlay.OverlayState(speed=1.0)
    for expected in [2.0, 5.0, 5.0]:
        overlay.speed_up(state)
        assert state.speed == expected
    for expected in [2.0, 1.0, 0.5, 0.25, 0.1, 0.1]:
        overlay.speed_down(state)
        assert state.speed == expected


def test_handle_common_key_state_machine():
    state = overlay.OverlayState()
    assert overlay.handle_common_key(state, ord(" "))
    assert state.paused
    assert overlay.handle_common_key(state, ord("."))
    assert state.paused and state.single_step
    assert overlay.handle_common_key(state, ord("C"))
    assert state.follow
    assert overlay.handle_common_key(state, ord("L"))
    assert state.show_lidar
    assert overlay.handle_common_key(state, ord("X"))
    assert state.show_contacts
    assert overlay.handle_common_key(state, ord("G"))
    assert not state.show_goal
    assert overlay.handle_common_key(state, ord("H"))
    assert not state.show_hud
    assert overlay.handle_common_key(state, ord("P"))
    assert state.screenshot_requested
    assert overlay.handle_common_key(state, ord("="))
    assert state.speed == 2.0
    assert overlay.handle_common_key(state, ord("-"))
    assert state.speed == 1.0
    assert not overlay.handle_common_key(state, ord("Q"))  # unbound key not consumed
    state.trail.append(np.zeros(3))
    assert overlay.handle_common_key(state, ord("R"), world=None)
    assert len(state.trail) == 0


def test_trail_toggle_clears_trail():
    state = overlay.OverlayState()
    state.trail.append(np.zeros(3))
    overlay.handle_common_key(state, ord("T"))
    assert not state.show_trail and len(state.trail) == 0


def test_apply_camera_follow_and_restore():
    world = build_world("flat", "diffdrive")
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    state = overlay.OverlayState(follow=True)
    overlay.apply_camera(cam, world.model, world.root_body_name, state)
    assert cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert cam.trackbodyid == world.model.body(world.root_body_name).id
    state.follow = False
    overlay.apply_camera(cam, world.model, world.root_body_name, state)
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert cam.trackbodyid == -1


def test_apply_vis_flags():
    opt = mujoco.MjvOption()
    state = overlay.OverlayState(show_contacts=True)
    overlay.apply_vis_flags(opt, state)
    assert opt.frame == mujoco.mjtFrame.mjFRAME_WORLD
    assert opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]
    assert opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]
    state.show_contacts = False
    overlay.apply_vis_flags(opt, state)
    assert not opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]


def test_draw_helpers_add_geoms_and_render():
    world = build_world("flat", "diffdrive")
    renderer = _themed_render(world)
    scn = renderer.scene
    base = scn.ngeom

    overlay.draw_goal(scn, (3.0, 0.0))
    assert scn.ngeom == base + 2  # beacon line + sphere

    origin = np.array([0.0, 0.0, 0.1])
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ranges_m = np.array([2.0, 5.0])  # first hits (< max_range), second misses
    overlay.draw_lidar(scn, origin, dirs, ranges_m, max_range=5.0)
    assert scn.ngeom == base + 2 + 3  # hit: line+sphere, miss: line only

    state = overlay.OverlayState()
    for p in [(0, 0, 0.05), (0.1, 0, 0.05), (0.2, 0, 0.05)]:
        overlay.push_trail(state, np.array(p))
    overlay.draw_trail(scn, state.trail)
    assert scn.ngeom == base + 5 + 2  # 3 points -> 2 segments

    assert renderer.render().std() > 3


def test_push_trail_min_spacing():
    state = overlay.OverlayState()
    overlay.push_trail(state, np.array([0.0, 0.0, 0.0]))
    overlay.push_trail(state, np.array([0.005, 0.0, 0.0]))  # below spacing threshold
    assert len(state.trail) == 1
    overlay.push_trail(state, np.array([0.1, 0.0, 0.0]))
    assert len(state.trail) == 2


def test_hud_text_builders():
    state = overlay.OverlayState(paused=True, speed=0.5)
    timer = overlay.HudTimer()
    texts = overlay.hud_status_texts(
        "view", 1.23, state, timer, np.array([0.1, -0.2, 0.05]), 45.0, 0.3
    )
    assert all(len(t) == 4 for t in texts)
    hud_values = texts[0][3]
    assert "PAUSED" in hud_values and "0.5x" in hud_values
    toggles = overlay.overlay_toggle_line(state)
    assert "lidar" in toggles[2]


def test_screenshot_saves_png(tmp_path):
    world = build_world("flat", "diffdrive")
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(world.model, cam)
    path = overlay.save_screenshot(world.model, world.data, cam, out_dir=str(tmp_path))
    assert path.exists()
    import imageio

    img = imageio.imread(path)
    assert img.shape == (theme.OFFSCREEN_HEIGHT, theme.OFFSCREEN_WIDTH, 3)
