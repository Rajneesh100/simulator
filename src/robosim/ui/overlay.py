"""Shared viewer chrome for view/teleop/play — the Isaac-Sim-feel layer.

Everything here rides on mujoco.viewer's native passive handle: HUD boxes via
`viewer.set_texts` (two-column text overlays, mujoco >= 3.3), debug overlays drawn into
`viewer.user_scn`, contact/frame visualization via `viewer.opt` flags, and camera
follow via `viewer.cam`. No custom GUI toolkit.

Key handling, camera logic, and timing stats are plain functions on a plain state
object so they're unit-testable without opening a window.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

SPEED_LADDER = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

RGBA_TRAIL = np.array([0.25, 0.62, 1.0, 0.85], dtype=np.float32)
RGBA_LIDAR_HIT = np.array([1.0, 0.25, 0.2, 0.9], dtype=np.float32)
RGBA_LIDAR_MISS = np.array([0.45, 0.5, 0.55, 0.25], dtype=np.float32)
RGBA_GOAL = np.array([0.15, 0.95, 0.35, 0.9], dtype=np.float32)

TRAIL_MAX_POINTS = 400
TRAIL_MIN_SPACING = 0.02  # meters between stored trail points
TRAIL_MAX_SEGMENTS = 120  # drawn per frame (subsampled) — keeps the viewer lock cheap

_EYE3 = np.eye(3).flatten()
_ZERO3 = np.zeros(3)

FONT_HUD = mujoco.mjtFontScale.mjFONTSCALE_150
FONT_HINTS = mujoco.mjtFontScale.mjFONTSCALE_100
GRID = mujoco.mjtGridPos

# HUD strings must be pure ASCII — mujoco's overlay bitmap font renders anything else
# as garbage glyphs (a real bug we hit with degree signs and omega).
KEY_HINTS = (
    "space pause   .  step   r reset   -/=  speed",
    "c follow   1 lidar   x contacts   t trail   2 goal",
    "p screenshot   h hud   tab left panel   shift-tab right panel   F1 help",
)


@dataclass
class OverlayState:
    """Runtime UI state shared by the viewer loops. Mutated by key callbacks."""

    paused: bool = False
    single_step: bool = False
    speed: float = 1.0
    follow: bool = False
    show_lidar: bool = False
    show_trail: bool = True
    show_goal: bool = True
    show_contacts: bool = False
    show_hud: bool = True
    screenshot_requested: bool = False
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_MAX_POINTS))


def speed_up(state: OverlayState) -> None:
    for s in SPEED_LADDER:
        if s > state.speed + 1e-9:
            state.speed = s
            return
    state.speed = SPEED_LADDER[-1]


def speed_down(state: OverlayState) -> None:
    for s in reversed(SPEED_LADDER):
        if s < state.speed - 1e-9:
            state.speed = s
            return
    state.speed = SPEED_LADDER[0]


def handle_common_key(state: OverlayState, keycode: int, world=None) -> bool:
    """Shared key bindings for all three viewer modes. Returns True if consumed.

    GLFW delivers letter keys as uppercase ASCII codes.

    Key choice is constrained by MuJoCo's built-in viewer: EVERY letter A-Z is a native
    visualization shortcut that fires alongside our callback and cannot be suppressed
    from the passive API. Letters whose native effect is an mjvOption flag are safe —
    apply_vis_flags() pins those flags back every frame. Letters whose native effect
    is an internal render-scene flag (unreachable) are avoided entirely: that's why
    lidar is '1' (L = additive blending) and goal is '2' (G = fog). Digit keys natively
    toggle geom-group visibility, which apply_vis_flags() also pins.
    """
    char = chr(keycode).lower() if 0 < keycode < 256 else ""
    if char == " ":
        state.paused = not state.paused
    elif char == ".":
        state.paused = True
        state.single_step = True
    elif char == "r":
        if world is not None:
            world.reset()
        state.trail.clear()
    elif char == "c":
        state.follow = not state.follow
    elif char == "1":
        state.show_lidar = not state.show_lidar
    elif char == "t":
        state.show_trail = not state.show_trail
        state.trail.clear()
    elif char == "2":
        state.show_goal = not state.show_goal
    elif char == "x":
        state.show_contacts = not state.show_contacts
    elif char == "h":
        state.show_hud = not state.show_hud
    elif char == "p":
        state.screenshot_requested = True
    elif char == "=" or char == "+":
        speed_up(state)
    elif char == "-":
        speed_down(state)
    else:
        return False
    return True


def init_camera(cam, lookat) -> None:
    """Frame the spawn point with the theme's default orbit angles."""
    from robosim.core import theme

    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = theme.CAMERA_AZIMUTH
    cam.elevation = theme.CAMERA_ELEVATION
    cam.distance = theme.CAMERA_DISTANCE
    cam.lookat[:] = [lookat[0], lookat[1], lookat[2] + 0.1]


def apply_camera(cam, model, root_body_name: str, state: OverlayState) -> None:
    """Idempotent follow-camera handling: tracking when on, restore free when off."""
    if state.follow:
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = model.body(root_body_name).id
        cam.fixedcamid = -1
    elif cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.trackbodyid = -1


# mjvOption flags pinned to a fixed value every frame. MuJoCo's native viewer binds
# every letter key to a visualization toggle and fires them alongside our key callback
# (unsuppressible from the passive API) — e.g. D hides all static geometry, X kills
# textures, T makes everything transparent, A draws autoconnect lines. Pinning reverts
# any such native toggle within one frame. Cost: the native panel checkboxes for these
# specific flags won't stick — our own keys are the control surface for them.
_PINNED_VIS_FLAGS = (
    (mujoco.mjtVisFlag.mjVIS_STATIC, True),  # native 'D' (teleop had used it to drive!)
    (mujoco.mjtVisFlag.mjVIS_TEXTURE, True),  # native 'X' — our contacts key
    (mujoco.mjtVisFlag.mjVIS_TRANSPARENT, False),  # native 'T' — our trail key
    (mujoco.mjtVisFlag.mjVIS_AUTOCONNECT, False),  # native 'A'
    (mujoco.mjtVisFlag.mjVIS_CONVEXHULL, False),  # native 'H' — our hud key
    (mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT, False),  # native 'P' — our screenshot key
)


def apply_vis_flags(opt, state: OverlayState) -> None:
    """World-axes gizmo always on; contact points/forces toggleable at runtime; every
    other flag we depend on is pinned against native-shortcut interference."""
    opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = state.show_contacts
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = state.show_contacts
    for flag, value in _PINNED_VIS_FLAGS:
        opt.flags[flag] = value
    # Native digit keys toggle geom/site group visibility — keep all groups visible
    # (our '1'/'2' overlay keys land on these natively).
    opt.geomgroup[:] = 1
    opt.sitegroup[:] = 1


class HudTimer:
    """Real-time factor and physics steps/sec over a sliding wall-clock window."""

    def __init__(self, window: float = 1.0):
        self.window = window
        self._samples: deque = deque()
        self._cum_steps = 0

    def update(self, sim_time: float, steps: int, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self._cum_steps += steps
        self._samples.append((now, sim_time, self._cum_steps))
        while len(self._samples) > 2 and now - self._samples[0][0] > self.window:
            self._samples.popleft()

    def _span(self):
        if len(self._samples) < 2:
            return None
        w0, s0, c0 = self._samples[0]
        w1, s1, c1 = self._samples[-1]
        if w1 - w0 <= 0:
            return None
        return w1 - w0, s1 - s0, c1 - c0

    def rtf(self) -> float:
        span = self._span()
        return span[1] / span[0] if span else 0.0

    def steps_per_sec(self) -> float:
        span = self._span()
        return span[2] / span[0] if span else 0.0


def quat_to_yaw_deg(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


# ---------------------------------------------------------------------------
# user_scn drawing


def _add_line(scn, from_, to, rgba, width_px=2.0) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=_ZERO3,
        pos=_ZERO3,
        mat=_EYE3,
        rgba=rgba,
    )
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_LINE,
        width_px,
        np.asarray(from_, dtype=np.float64),
        np.asarray(to, dtype=np.float64),
    )
    scn.ngeom += 1


def _add_sphere(scn, pos, rgba, radius) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, radius, radius]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=_EYE3,
        rgba=rgba,
    )
    scn.ngeom += 1


def draw_lidar(scn, origin, dirs, ranges_m, max_range: float) -> None:
    """Rays from the sensor origin: red to hit point when the beam hits something,
    faint gray full-length when it doesn't."""
    for d, r in zip(dirs, ranges_m):
        hit = r < max_range - 1e-6
        end = origin + d * r
        _add_line(scn, origin, end, RGBA_LIDAR_HIT if hit else RGBA_LIDAR_MISS, 2.0)
        if hit:
            _add_sphere(scn, end, RGBA_LIDAR_HIT, 0.012)


def push_trail(state: OverlayState, pos) -> None:
    p = np.asarray(pos[:3], dtype=np.float64)
    if state.trail and np.linalg.norm(p - state.trail[-1]) < TRAIL_MIN_SPACING:
        return
    state.trail.append(p)


def draw_trail(scn, trail) -> None:
    pts = list(trail)
    # Subsample long trails: this runs every frame while holding the viewer lock, so
    # cap the connector count instead of drawing all 400 stored points.
    stride = max(1, -(-len(pts) // TRAIL_MAX_SEGMENTS))  # ceil division
    pts = pts[::stride]
    for a, b in zip(pts, pts[1:]):
        _add_line(scn, a, b, RGBA_TRAIL, 3.0)


def draw_goal(scn, goal_xy, ground_z: float = 0.0) -> None:
    """Waypoint beacon: vertical line + floating sphere, Isaac-style."""
    x, y = float(goal_xy[0]), float(goal_xy[1])
    _add_line(scn, (x, y, ground_z), (x, y, ground_z + 0.6), RGBA_GOAL, 3.0)
    _add_sphere(scn, (x, y, ground_z + 0.6), RGBA_GOAL, 0.035)


# ---------------------------------------------------------------------------
# HUD text


def hud_status_texts(
    mode: str,
    sim_time: float,
    state: OverlayState,
    timer: HudTimer,
    pos,
    yaw_deg: float,
    speed_mps: float,
) -> list[tuple]:
    speed_str = f"{state.speed:g}x" + ("  PAUSED" if state.paused else "")
    labels = "mode\ntime\nspeed\nrtf\nphysics\npos\nyaw\nvel"
    values = (
        f"{mode}\n"
        f"{sim_time:.2f} s\n"
        f"{speed_str}\n"
        f"{timer.rtf():.2f}\n"
        f"{timer.steps_per_sec():,.0f} steps/s\n"
        f"({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})\n"
        f"{yaw_deg:+.0f} deg\n"
        f"{speed_mps:.2f} m/s"
    )
    texts = [(FONT_HUD, GRID.mjGRID_TOPLEFT, labels, values)]
    texts.append((FONT_HINTS, GRID.mjGRID_BOTTOMLEFT, "\n".join(KEY_HINTS), None))
    return texts


def overlay_toggle_line(state: OverlayState) -> tuple:
    def mark(on: bool) -> str:
        return "on" if on else "off"

    labels = "lidar\ncontacts\ntrail\ngoal\nfollow"
    values = (
        f"{mark(state.show_lidar)}\n{mark(state.show_contacts)}\n"
        f"{mark(state.show_trail)}\n{mark(state.show_goal)}\n{mark(state.follow)}"
    )
    return (FONT_HINTS, GRID.mjGRID_BOTTOMRIGHT, labels, values)


def update_hud(viewer, texts: list[tuple]) -> bool:
    """Push HUD boxes to the viewer. Returns False if this mujoco has no set_texts
    (caller may fall back to console prints)."""
    if not hasattr(viewer, "set_texts"):
        return False
    if texts:
        viewer.set_texts(texts)
    elif hasattr(viewer, "clear_texts"):
        viewer.clear_texts()
    return True


# ---------------------------------------------------------------------------
# Screenshot (nice-to-have): offscreen re-render of the current viewpoint


def save_screenshot(model, data, cam_src, out_dir: str = "runs/screenshots") -> Path:
    """Render the current camera view offscreen and save a PNG. Uses a throwaway
    Renderer so it doesn't touch the GUI's GL context."""
    import imageio

    from robosim.core import theme

    renderer = mujoco.Renderer(model, height=theme.OFFSCREEN_HEIGHT, width=theme.OFFSCREEN_WIDTH)
    try:
        cam = mujoco.MjvCamera()
        cam.type = cam_src.type
        cam.azimuth = cam_src.azimuth
        cam.elevation = cam_src.elevation
        cam.distance = cam_src.distance
        cam.lookat[:] = cam_src.lookat
        cam.trackbodyid = cam_src.trackbodyid
        cam.fixedcamid = cam_src.fixedcamid
        opt = mujoco.MjvOption()
        opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        renderer.update_scene(data, camera=cam, scene_option=opt)
        frame = renderer.render()
    finally:
        renderer.close()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"robosim_{time.strftime('%Y%m%d_%H%M%S')}.png"
    imageio.imwrite(path, frame)
    return path
