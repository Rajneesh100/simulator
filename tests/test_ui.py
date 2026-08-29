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
    assert overlay.handle_common_key(state, ord("1"))
    assert state.show_lidar
    assert overlay.handle_common_key(state, ord("X"))
    assert state.show_contacts
    assert overlay.handle_common_key(state, ord("2"))
    assert not state.show_goal
    assert overlay.handle_common_key(state, ord("H"))
    assert not state.show_hud
    assert overlay.handle_common_key(state, ord("P"))
    assert state.screenshot_requested
    assert overlay.handle_common_key(state, ord("="))
    assert state.speed == 2.0
    assert overlay.handle_common_key(state, ord("-"))
    assert state.speed == 1.0
    # L and G must stay UNBOUND: their native viewer effects (additive blending, fog)
    # are internal render-scene flags we cannot pin back.
    assert not overlay.handle_common_key(state, ord("L"))
    assert not overlay.handle_common_key(state, ord("G"))
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


def test_apply_vis_flags_pins_native_key_side_effects():
    """MuJoCo's native viewer toggles these via letter/digit shortcuts that fire
    alongside our key callback; apply_vis_flags must revert them every frame."""
    opt = mujoco.MjvOption()
    state = overlay.OverlayState()
    # Simulate native shortcut damage: D hid static geometry, X killed textures,
    # T made everything transparent, A drew autoconnect lines, '0' hid geom group 0.
    opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = True
    opt.geomgroup[0] = 0
    overlay.apply_vis_flags(opt, state)
    assert opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC]
    assert opt.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE]
    assert not opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]
    assert not opt.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT]
    assert not opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]
    assert all(opt.geomgroup)


def test_teleop_drive_keys():
    from robosim.ui.teleop import (
        KEY_DOWN,
        KEY_LEFT,
        KEY_RIGHT,
        KEY_UP,
        DriveState,
        handle_drive_key,
    )

    drive = DriveState()
    assert handle_drive_key(drive, KEY_UP)
    assert drive.v == 0.25
    for _ in range(5):
        handle_drive_key(drive, KEY_UP)
    assert drive.v == 1.0  # clamped
    assert handle_drive_key(drive, KEY_LEFT)
    assert drive.omega == 0.25
    for _ in range(6):
        handle_drive_key(drive, KEY_RIGHT)
    assert drive.omega == -1.0  # clamped
    assert handle_drive_key(drive, KEY_DOWN)
    assert drive.v == 0.75
    assert handle_drive_key(drive, ord(" "))
    assert drive.v == 0.0 and drive.omega == 0.0
    # letters must NOT drive (they collide with native render shortcuts)
    assert not handle_drive_key(drive, ord("W"))
    assert not handle_drive_key(drive, ord("D"))


def test_hud_strings_are_pure_ascii():
    """mujoco's overlay bitmap font renders non-ASCII as garbage glyphs."""
    state = overlay.OverlayState(paused=True)
    timer = overlay.HudTimer()
    texts = overlay.hud_status_texts(
        "teleop", 92.75, state, timer, np.array([0.88, 0.0, 0.07]), -0.8, 0.0
    )
    texts.append(overlay.overlay_toggle_line(state))
    for _, _, col1, col2 in texts:
        for s in (col1, col2):
            if s:
                assert all(ord(ch) < 128 for ch in s), f"non-ascii in HUD: {s!r}"
    for line in overlay.KEY_HINTS:
        assert all(ord(ch) < 128 for ch in line)


def test_draw_trail_subsamples_long_trails():
    world = build_world("flat", "diffdrive")
    renderer = _themed_render(world)
    scn = renderer.scene
    base = scn.ngeom
    state = overlay.OverlayState()
    for i in range(overlay.TRAIL_MAX_POINTS):
        state.trail.append(np.array([i * 0.05, 0.0, 0.05]))
    overlay.draw_trail(scn, state.trail)
    assert scn.ngeom - base <= overlay.TRAIL_MAX_SEGMENTS


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
    n_after_blast = scn.ngeom
    overlay.draw_blast(scn, blast, age_s=0.9, radius=2.0)  # past the animation: no-op
    assert scn.ngeom == n_after_blast
    overlay.draw_aim(scn, np.array([0.0, 0.0, 0.3]), np.array([1.0, 0.0, 0.0]), (1, 0, 0, 1))
    assert scn.ngeom <= scn.maxgeom
