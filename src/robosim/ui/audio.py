"""Procedural, spatialized race audio for the viewer (`robosim race --sound`).

No audio files: every sound is synthesized in numpy.
- ENGINE per car: harmonic-rich buzz whose pitch follows speed (idle ~40 Hz, +25 Hz per
  m/s) and whose volume follows throttle. Dead cars fall silent.
- ONE-SHOTS: rocket launch (whoosh: decaying noise with a falling low-pass), blast (loud
  low-passed noise burst + 60 Hz thump), dud (short thunk).
- SPATIAL, per view frame: the listener is the viewer camera. Each source is attenuated
  by distance (inverse-square with a 3 m knee), panned left/right by its bearing relative
  to the camera's look direction, and Doppler-shifted by its radial velocity (343 m/s),
  so the car you follow is loud and centered and an overtaking car sweeps across.

Output goes through `sounddevice` (PortAudio) when available; `render_block` is pure
numpy so the mixer is unit-testable headless and degrades to silence without a device.
"""

from __future__ import annotations

import threading

import numpy as np

SAMPLE_RATE = 44_100
BLOCK = 1024
SPEED_OF_SOUND = 343.0
ATTEN_KNEE_M = 3.0  # distance at which a source is at -3 dB
IDLE_HZ = 40.0
HZ_PER_MPS = 25.0
MASTER = 0.5
N_HARMONICS = 6


# ---------------------------------------------------------------------------
# listener + spatialization (pure functions)


def listener_from_camera(cam, lookat=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position, forward, right) of the viewer camera. MuJoCo places the camera at
    lookat - distance * forward, with forward from azimuth/elevation. Pass `lookat`
    explicitly when the camera is tracking a body (the viewer updates it at render time)."""
    az = np.deg2rad(float(cam.azimuth))
    el = np.deg2rad(float(cam.elevation))
    forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    la = np.asarray(cam.lookat if lookat is None else lookat, dtype=np.float64)
    pos = la - float(cam.distance) * forward
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(right)
    right = right / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    return pos, forward, right


def spatialize(source_pos, source_vel, listener_pos, listener_right) -> tuple[float, float, float]:
    """(gain, pan, doppler) for one source: gain in (0, 1], pan in [-1 left, +1 right],
    doppler = pitch multiplier (> 1 approaching)."""
    d = np.asarray(source_pos, dtype=np.float64) - np.asarray(listener_pos, dtype=np.float64)
    dist = float(np.linalg.norm(d))
    gain = 1.0 / (1.0 + (dist / ATTEN_KNEE_M) ** 2)
    if dist < 1e-6:
        return 1.0, 0.0, 1.0
    direction = d / dist
    pan = float(np.clip(np.dot(direction, listener_right), -1.0, 1.0))
    radial = float(np.dot(np.asarray(source_vel, dtype=np.float64), direction))  # + = away
    doppler = SPEED_OF_SOUND / max(SPEED_OF_SOUND + radial, 50.0)
    return gain, pan, doppler


def engine_hz(speed_mps: float) -> float:
    return IDLE_HZ + HZ_PER_MPS * max(0.0, float(speed_mps))


def _pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power stereo panning."""
    theta = (pan + 1.0) * np.pi / 4.0
    return float(np.cos(theta)), float(np.sin(theta))


# ---------------------------------------------------------------------------
# voices


class EngineVoice:
    def __init__(self):
        self.phase = 0.0
        self.freq = IDLE_HZ
        self.gain = 0.0
        self.pan = 0.0
        self.target = (IDLE_HZ, 0.0, 0.0)

    def render(self, n: int) -> np.ndarray:
        f1, g1, p1 = self.target
        # linear ramps from the last block's values: no clicks on speed/volume steps
        freq = np.linspace(self.freq, f1, n, endpoint=False)
        gain = np.linspace(self.gain, g1, n, endpoint=False)
        phase = self.phase + np.cumsum(freq) / SAMPLE_RATE
        wave = np.zeros(n)
        for k in range(1, N_HARMONICS + 1):  # sawtooth-ish: 1/k harmonics
            wave += np.sin(2 * np.pi * k * phase) / k
        wave += 0.5 * np.sin(2 * np.pi * 0.5 * phase)  # sub-octave rumble
        wave *= gain / 2.0
        self.phase = float(phase[-1] % 1.0)
        self.freq, self.gain, self.pan = f1, g1, p1
        left, right = _pan_gains(p1)
        return np.stack([wave * left, wave * right], axis=1)


def _one_pole_lowpass(x: np.ndarray, cutoff_hz) -> np.ndarray:
    """Simple first-order low-pass; `cutoff_hz` may be an array (time-varying)."""
    cutoff = np.broadcast_to(np.asarray(cutoff_hz, dtype=np.float64), x.shape)
    alpha = 1.0 - np.exp(-2 * np.pi * cutoff / SAMPLE_RATE)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += alpha[i] * (x[i] - acc)
        y[i] = acc
    return y


def _make_oneshots(rng: np.random.Generator) -> dict[str, np.ndarray]:
    t = lambda s: np.arange(int(s * SAMPLE_RATE)) / SAMPLE_RATE  # noqa: E731
    tt = t(0.45)
    launch = rng.normal(0, 1, len(tt)) * np.exp(-tt / 0.12)
    launch = _one_pole_lowpass(launch, 4000 * np.exp(-tt / 0.15) + 300)
    tt = t(0.7)
    blast = rng.normal(0, 1, len(tt)) * np.exp(-tt / 0.22)
    blast = _one_pole_lowpass(blast, 900.0) * 3.0
    blast += 1.2 * np.sin(2 * np.pi * 60 * tt) * np.exp(-tt / 0.18)  # thump
    tt = t(0.15)
    dud = _one_pole_lowpass(rng.normal(0, 1, len(tt)) * np.exp(-tt / 0.03), 500.0) * 1.5
    out = {"launch": launch, "blast": blast, "dud": dud}
    return {k: (v / max(np.max(np.abs(v)), 1e-9)).astype(np.float64) for k, v in out.items()}


class OneShot:
    def __init__(self, buf: np.ndarray, gain: float, pan: float):
        self.buf = buf
        self.pos = 0
        self.gain = gain
        self.left, self.right = _pan_gains(pan)

    def render(self, n: int) -> np.ndarray | None:
        if self.pos >= len(self.buf):
            return None
        chunk = self.buf[self.pos : self.pos + n] * self.gain
        self.pos += n
        out = np.zeros((n, 2))
        out[: len(chunk), 0] = chunk * self.left
        out[: len(chunk), 1] = chunk * self.right
        return out


# ---------------------------------------------------------------------------
# mixer


class AudioMixer:
    """Owns the voices; the race loop calls `update`/`play` each frame, the audio
    thread calls `render_block`. Safe to use without a device (tests, no PortAudio)."""

    EVENT_GAIN = {"launch": 0.6, "blast": 1.0, "dud": 0.5}

    def __init__(self, n_cars: int, master: float = MASTER, seed: int = 0):
        self.voices = [EngineVoice() for _ in range(n_cars)]
        self.master = float(master)
        self._oneshots: list[OneShot] = []
        self._bufs = _make_oneshots(np.random.default_rng(seed))
        self._lock = threading.Lock()
        self._stream = None
        self.listener = (np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))

    # ---- from the render/physics thread

    def set_listener(self, cam, lookat=None) -> None:
        self.listener = listener_from_camera(cam, lookat)

    def update(self, cars) -> None:
        """cars = [(pos3, vel3, speed_mps, throttle, alive), ...] in car-index order."""
        lpos, _, lright = self.listener
        with self._lock:
            for voice, (pos, vel, speed, throttle, alive) in zip(self.voices, cars):
                if not alive:
                    voice.target = (voice.target[0], 0.0, voice.target[2])
                    continue
                gain, pan, doppler = spatialize(pos, vel, lpos, lright)
                vol = gain * (0.25 + 0.75 * min(1.0, abs(float(throttle))))
                voice.target = (engine_hz(speed) * doppler, vol, pan)

    def play(self, kind: str, pos) -> None:
        lpos, _, lright = self.listener
        gain, pan, _ = spatialize(pos, np.zeros(3), lpos, lright)
        shot = OneShot(self._bufs[kind], gain * self.EVENT_GAIN[kind], pan)
        with self._lock:
            self._oneshots.append(shot)

    # ---- audio thread

    def render_block(self, n: int = BLOCK) -> np.ndarray:
        out = np.zeros((n, 2))
        with self._lock:
            for voice in self.voices:
                out += voice.render(n)
            alive = []
            for shot in self._oneshots:
                chunk = shot.render(n)
                if chunk is not None:
                    out += chunk
                    alive.append(shot)
            self._oneshots = alive
        return np.tanh(out * self.master).astype(np.float32)  # soft clip

    def _callback(self, outdata, frames, time_info, status) -> None:
        outdata[:] = self.render_block(frames)

    def start(self) -> bool:
        """Open the output stream. Returns False (silent mode) if sounddevice or an
        output device is unavailable."""
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, blocksize=BLOCK, dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            return True
        except Exception as e:  # missing module, no device, PortAudio error
            print(f"[audio] sound disabled: {e}")
            self._stream = None
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
