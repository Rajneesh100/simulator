"""Built-in video+audio recording renders offscreen frames at the sim clock and muxes."""

import mujoco
import numpy as np
import pytest

pytest.importorskip("imageio_ffmpeg")
pytest.importorskip("PIL")


def test_recorder_writes_mp4_with_video_and_audio(tmp_path):
    from robosim.core.world import build_multi_world
    from robosim.ui import overlay
    from robosim.ui.recorder import VideoRecorder, probe_streams

    world = build_multi_world("flat", "racecar", n_cars=2)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = world.model.body(world.root_body_names[0]).id
    cam.distance, cam.azimuth, cam.elevation = 4.0, 90.0, -30.0
    rec = VideoRecorder(tmp_path / "clip.mp4", world.model, n_cars=2, fps=10,
                        width=320, height=180)

    def draw(scn):
        overlay._add_sphere(scn, (1.0, 0.0, 0.5), (1.0, 0.2, 0.2, 0.9), 0.1)

    dt = world.model.opt.timestep
    captured = 0
    for _ in range(int(1.0 / dt)):  # 1 s of sim time -> 10 frames at 10 fps
        mujoco.mj_step(world.model, world.data)
        if rec.due(world.data.time):
            cars = [
                (world.car_pose(i)[0], world.car_velocity(i), world.car_speed(i), 0.5, True)
                for i in range(2)
            ]
            events = [("launch", world.car_pose(1)[0])] if captured == 2 else []
            rec.capture(world.data, cam, lookat=world.car_pose(0)[0], draw_overlays=draw,
                        hud_lines=["P1 RED  12.3m", "P2 GREEN 10.1m"], audio_cars=cars,
                        events=events)
            captured += 1
    assert captured == 10 and rec.frames == 10
    out = rec.close()
    assert out.exists() and out.stat().st_size > 5_000
    streams = probe_streams(out)
    assert streams == {"video": True, "audio": True}
    assert not (tmp_path / "clip.video.mp4").exists()  # temps cleaned up


def test_burn_hud_returns_same_shape():
    from robosim.ui.recorder import burn_hud

    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    out = burn_hud(frame, ["hello", "world"])
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert out.max() > 0  # something was drawn
    assert burn_hud(frame, []) is frame
