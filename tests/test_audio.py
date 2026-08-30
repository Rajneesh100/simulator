"""Race audio: spatialization math, engine mapping, headless mixer rendering."""

import numpy as np

from robosim.ui import audio


class _Cam:
    def __init__(self, lookat=(0, 0, 0), distance=5.0, azimuth=90.0, elevation=-30.0):
        self.lookat = np.array(lookat, dtype=float)
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation


def test_listener_from_camera_sits_behind_lookat_along_view_direction():
    cam = _Cam(lookat=(10, 0, 0), distance=4.0, azimuth=0.0, elevation=0.0)
    pos, forward, right = audio.listener_from_camera(cam)
    assert np.allclose(forward, [1, 0, 0])
    assert np.allclose(pos, [6, 0, 0])  # 4 m behind the lookat, looking +x
    assert np.allclose(right, [0, -1, 0])  # +y is to the LEFT when looking +x
    pos2, _, _ = audio.listener_from_camera(cam, lookat=(20, 0, 0))
    assert np.allclose(pos2, [16, 0, 0])  # explicit lookat (tracking camera) wins


def test_spatialize_pan_gain_and_doppler():
    lpos = np.zeros(3)
    right = np.array([0.0, -1.0, 0.0])  # looking +x
    gain_ahead, pan_ahead, dop = audio.spatialize([5, 0, 0], [0, 0, 0], lpos, right)
    assert abs(pan_ahead) < 1e-9 and np.isclose(dop, 1.0)
    _, pan_right, _ = audio.spatialize([0, -5, 0], [0, 0, 0], lpos, right)
    _, pan_left, _ = audio.spatialize([0, 5, 0], [0, 0, 0], lpos, right)
    assert pan_right > 0.99 and pan_left < -0.99
    gain_far, _, _ = audio.spatialize([20, 0, 0], [0, 0, 0], lpos, right)
    assert gain_far < gain_ahead < 1.0
    _, _, approaching = audio.spatialize([10, 0, 0], [-12, 0, 0], lpos, right)
    _, _, receding = audio.spatialize([10, 0, 0], [12, 0, 0], lpos, right)
    assert approaching > 1.0 > receding
    assert np.isclose(approaching, 343 / (343 - 12))


def test_engine_pitch_rises_with_speed():
    assert audio.engine_hz(0.0) == audio.IDLE_HZ
    assert audio.engine_hz(12.0) > audio.engine_hz(6.0) > audio.engine_hz(0.0)


def test_mixer_renders_headless_block_with_voices_and_events():
    mixer = audio.AudioMixer(n_cars=3)
    cam = _Cam(lookat=(0, 0, 0), distance=6.0, azimuth=0.0, elevation=-20.0)
    mixer.set_listener(cam)
    silent = mixer.render_block(512)
    assert silent.shape == (512, 2) and np.allclose(silent, 0.0)  # nothing moving yet
    mixer.update([
        (np.array([0.0, 0.0, 0.1]), np.array([8.0, 0, 0]), 8.0, 0.9, True),
        (np.array([3.0, 4.0, 0.1]), np.zeros(3), 2.0, 0.3, True),
        (np.array([1.0, -2.0, 0.1]), np.zeros(3), 5.0, 1.0, False),  # dead: silent
    ])
    block = mixer.render_block(1024)
    assert block.shape == (1024, 2) and block.dtype == np.float32
    assert np.all(np.isfinite(block)) and np.max(np.abs(block)) <= 1.0
    assert np.max(np.abs(block)) > 0.01
    rms_before = float(np.sqrt(np.mean(block**2)))
    mixer.play("blast", [0.5, 0.0, 0.1])
    loud = mixer.render_block(1024)
    assert float(np.sqrt(np.mean(loud**2))) > rms_before
    for _ in range(60):  # one-shots finish and are dropped
        mixer.render_block(1024)
    assert mixer._oneshots == []


def test_mixer_start_degrades_gracefully_without_device(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import fails
    mixer = audio.AudioMixer(n_cars=1)
    assert mixer.start() is False
    mixer.stop()
