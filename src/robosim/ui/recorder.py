"""Built-in race recording (`robosim race --record-video out.mp4`): video AND audio.

Frames are rendered OFFSCREEN at the simulation clock (every 1/fps of sim time, sampled
inside the physics loop) through a copy of the viewer camera, with the same overlays the
window shows (rockets, blasts, labels, lidar, trails) drawn into the offscreen scene and
a leaderboard burned onto the pixels. The audio track is rendered OFFLINE by the same
`AudioMixer` the live sound uses, one block per video frame, so picture and sound are in
exact lockstep regardless of how fast the machine ran the race. At close, video and WAV
are muxed into one H.264/AAC .mp4 with the ffmpeg binary bundled by `imageio-ffmpeg`
(`uv sync --extra video`; no system ffmpeg needed).
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import mujoco
import numpy as np

from robosim.ui.audio import SAMPLE_RATE, AudioMixer

HUD_FONT_PX = 18


def burn_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw text lines onto the top-right of an RGB frame (Pillow)."""
    if not lines:
        return frame
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("Menlo.ttc", HUD_FONT_PX)
    except Exception:
        try:
            font = ImageFont.load_default(size=HUD_FONT_PX)
        except TypeError:  # older Pillow: no size kwarg
            font = ImageFont.load_default()
    pad, gap = 8, 4
    widths = [draw.textlength(t, font=font) for t in lines]
    w = int(max(widths)) + 2 * pad
    h = len(lines) * (HUD_FONT_PX + gap) + 2 * pad
    x0, y0 = img.width - w - 12, 12
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(0, 0, 0, 150))
    for i, text in enumerate(lines):
        draw.text((x0 + pad, y0 + pad + i * (HUD_FONT_PX + gap)), text,
                  fill=(255, 255, 255, 255), font=font)
    return np.asarray(img)


class VideoRecorder:
    def __init__(self, path, model, n_cars: int, fps: int = 30, width: int = 1280,
                 height: int = 720, audio: bool = True, seed: int = 0):
        import imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        self.mixer = AudioMixer(n_cars, seed=seed) if audio else None
        self._tmp_video = self.path.with_name(self.path.stem + ".video.mp4")
        self._tmp_wav = self.path.with_name(self.path.stem + ".audio.wav")
        self._writer = imageio.get_writer(
            str(self._tmp_video), fps=self.fps, codec="libx264", quality=8,
            macro_block_size=None,
        )
        self._wav = None
        if self.mixer is not None:
            self._wav = wave.open(str(self._tmp_wav), "wb")
            self._wav.setnchannels(2)
            self._wav.setsampwidth(2)
            self._wav.setframerate(SAMPLE_RATE)
        self.frames = 0
        self._samples_written = 0
        self._t0: float | None = None
        self.closed = False

    # ---- scheduling on the sim clock

    def due(self, sim_time: float) -> bool:
        if self._t0 is None:
            return True
        return sim_time >= self._t0 + self.frames / self.fps - 1e-9

    # ---- one frame

    def capture(self, data, cam_src, lookat=None, draw_overlays=None, hud_lines=None,
                audio_cars=None, events=()) -> None:
        """Render one video frame (+ its slice of audio).
        cam_src: the viewer's MjvCamera (copied, so tracking cameras track here too);
        lookat: followed car position for the audio listener under a tracking camera;
        draw_overlays(scn): draws user geoms into the offscreen scene;
        hud_lines: text burned top-right; audio_cars/events: see AudioMixer."""
        if self._t0 is None:
            self._t0 = float(data.time)
        self.cam.type = cam_src.type
        self.cam.azimuth = cam_src.azimuth
        self.cam.elevation = cam_src.elevation
        self.cam.distance = cam_src.distance
        self.cam.lookat[:] = cam_src.lookat if lookat is None else lookat
        self.cam.trackbodyid = cam_src.trackbodyid
        self.cam.fixedcamid = cam_src.fixedcamid
        self.renderer.update_scene(data, camera=self.cam, scene_option=self.opt)
        if draw_overlays is not None:
            draw_overlays(self.renderer.scene)
        frame = self.renderer.render()
        if hud_lines:
            frame = burn_hud(frame, list(hud_lines))
        self._writer.append_data(frame)
        self.frames += 1

        if self.mixer is not None:
            self.mixer.set_listener(cam_src, lookat)
            if audio_cars is not None:
                self.mixer.update(audio_cars)
            for kind, pos in events:
                self.mixer.play(kind, pos)
            # cumulative rounding keeps audio and video clocks locked at any fps
            target = int(round(self.frames * SAMPLE_RATE / self.fps))
            n = target - self._samples_written
            if n > 0:
                block = self.mixer.render_block(n)
                self._wav.writeframes((np.clip(block, -1, 1) * 32767).astype("<i2").tobytes())
                self._samples_written = target

    # ---- finish

    def close(self) -> Path:
        if self.closed:
            return self.path
        self.closed = True
        self._writer.close()
        self.renderer.close()
        if self._wav is not None:
            self._wav.close()
            import imageio_ffmpeg

            cmd = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                "-i", str(self._tmp_video), "-i", str(self._tmp_wav),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(self.path),
            ]
            subprocess.run(cmd, check=True)
            self._tmp_video.unlink(missing_ok=True)
            self._tmp_wav.unlink(missing_ok=True)
        else:
            self._tmp_video.replace(self.path)
        return self.path


def probe_streams(path) -> dict[str, bool]:
    """{'video': bool, 'audio': bool} from the bundled ffmpeg's stream listing."""
    import imageio_ffmpeg

    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True,
    ).stderr
    return {"video": "Video:" in out, "audio": "Audio:" in out}
