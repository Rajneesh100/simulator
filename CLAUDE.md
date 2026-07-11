# RoboSim — Claude Code project memory

## Project overview

RoboSim is a minimal, Isaac-Lab-style local physics simulator for training robot control
policies (RL / imitation learning), built on MuJoCo + Gymnasium + stable-baselines3. It is
phase 1 of a larger WALL-E-style wheeled robot project: policies trained here are meant to
transfer to real hardware later. Full spec: `../idea.md` (one directory up, not part of this
package — kept only for reference; do not assume it ships with a zip/clone of `simulator/`).

Everything needed to work on this project — code, deps, Claude Code config — lives inside
this `simulator/` folder. No absolute paths, no `~/.claude` writes, no global installs.

## Setup / running things

```bash
./setup.sh                 # one-time: installs uv if missing, uv sync
uv run pytest              # run test suite
uv run robosim scenes      # list registered scenes
uv run robosim view --scene stairs --robot diffdrive
uv run robosim teleop --scene flat --record runs/demo1
uv run robosim train --task goal_reach --scene flat --steps 200_000
uv run robosim play --checkpoint runs/<run>/best_model.zip
```

Python is pinned to 3.11 via `.python-version` (torch/SB3/mujoco wheel compatibility on
Apple Silicon). Dependencies are managed by `uv` — see `pyproject.toml` / `uv.lock`.

## Scene plugin pattern

A new scene is **one file** in `src/robosim/scenes/`, nothing else changes:

```python
from robosim.core.registry import register_scene
from robosim.scenes.base import Scene

@register_scene("my_scene")
class MySceneSpec(Scene):
    name = "my_scene"

    def __init__(self, **params):
        self.foo = float(params.get("foo", 1.0))  # parse + default every param here

    def build(self, spec_builder):
        # add geoms/bodies via spec_builder (a mujoco.MjSpec), e.g.:
        # spec_builder.worldbody.add_geom(...)
        ...

    def spawn_pose(self):
        return (x, y, z), yaw_radians

    def task_info(self):
        return {"goal": (gx, gy)}  # whatever the task layer needs
```

`world.py` discovers scenes purely through the `@register_scene` decorator — core code is
never edited to add a scene. CLI params come in as `--scene name:key=val,key2=val2` and are
passed as `**params` (all strings — parse/cast inside `__init__`, never in core code).

## Physics centralization rule

**All physics constants (timestep, gravity, solver iterations, default friction, contact
params) live in `src/robosim/core/physics.py` and nowhere else.** Scenes and robots place
geometry and set per-geom friction/material only when a scene's premise requires it (e.g.
an icy patch); they never touch global solver/gravity/timestep settings. If a scene needs
different physics behavior, that's a signal it should be a parameter, not a physics override.

## Observation space

Canonical layout defined once in `src/robosim/envs/obs.py`:
`[lidar ranges (n_beams, normalized) | goal in robot frame (distance, sin, cos bearing) |
wheel velocities | IMU (pitch, roll, yaw rate) | previous action]`. Every task builds its
observation from this same function so the real robot can emit an identical vector later.

## Decision log

- **2026-07-11 — Python 3.11, not 3.14 (system default).** Pinned via `uv python pin 3.11`
  for the safest torch/stable-baselines3/mujoco wheel compatibility on Apple Silicon at
  build time. Revisit if/when SB3 and torch confirm 3.13+ support.
- **2026-07-11 — `hatchling` build backend, `src/` layout.** Simple, standard, works cleanly
  with `uv` and `pyproject.toml` `[project.scripts]` for the `robosim` CLI entrypoint.
- **2026-07-11 — Scenes built via `mujoco.MjSpec` (programmatic model editing), not Jinja2
  XML templating.** `MjSpec` is MuJoCo's native Python API for composing models from
  Python — it avoids string-templating fragility and lets `world.py` merge a scene's
  bodies/geoms with the base world and robot spec directly in memory.
- **2026-07-11 — `typer` for the CLI**, per idea.md's suggestion; `[project.scripts]`
  entrypoint `robosim = "robosim.cli:app"`.
- **2026-07-11 — UI built on `mujoco.viewer.launch_passive` + key callbacks**, not a custom
  GUI. Debug overlays (lidar rays, contacts, trajectory trail) are drawn via the viewer's
  `user_scn` scene extras. No dearpygui side-panel needed yet — the passive viewer's
  built-in stats/overlay is sufficient for the required HUD fields; revisit only if a
  control that MuJoCo's viewer truly lacks becomes necessary.
- **2026-07-11 — Wheel mount offset increased (`WHEEL_MOUNT_Z` -0.01 → -0.03) and
  "collision" excludes the scene's own ground/terrain geom.** Original clearance (7mm)
  let the chassis graze the ground under any real driving dynamics, firing spurious
  collision penalties ~15-18% of steps even on an empty flat scene with a random policy.
  Real fix is semantic, not just more clearance: `Scene.ground_geom_names()` (default
  `["ground", "terrain"]`) is excluded from `RoboSimEnv._collided()` — touching the floor
  during a turn isn't a crash, hitting a wall/obstacle is. Increased clearance (~2.7cm)
  on top of that for physical realism.
- **2026-07-11 — Training callback reads `model.ep_info_buffer` directly, not
  `logger.name_to_value["rollout/ep_rew_mean"]`.** SB3 only populates that logger key
  *after* `on_rollout_end` fires, so it's always one iteration stale (empty on the very
  first). `ep_info_buffer` is updated live during rollout collection, so it's correct
  immediately.
- **2026-07-11 — ONNX export forces the legacy TorchScript exporter (`dynamo=False`).**
  torch 2.13's default dynamo-based exporter combined with `dynamic_axes` produced a
  corrupted graph (shape-inference error, unrelated dims). The legacy exporter handles
  this small MLP policy correctly; revisit if a future torch drops the legacy path.
- **2026-07-11 — Curriculum rebuilds the scene via `RoboSimEnv.set_scene()` +
  `VecEnv.env_method`, not a lighter-weight in-place param change.** Scene geometry is
  fixed at MjSpec-compile time, so "harder" (more obstacles, etc.) genuinely needs a
  recompile. This is fine since curriculum stages are spaced thousands of steps apart,
  not per-episode. `train --curriculum "0:flat;50000:obstacles:count=3"` schedules it.
- **2026-07-11 — Moving obstacles (T7) use MuJoCo mocap bodies, driven by a per-step
  callback in `RoboSimEnv.step()`**, not joints+actuators. Mocap bodies have no
  dynamics — their pose is just written directly (`data.mocap_pos`) — which is simpler
  and exactly matches "scripted" motion; `--scene obstacles:moving=N` places N of them.
- **2026-07-11 — BC net (`train/bc.py`) must use `Tanh` activations, not `ReLU`, to
  match SB3 `ActorCriticPolicy`'s default `activation_fn`.** Initially built the BC net
  with ReLU; state dict *shapes* matched SB3's policy fine (`load_state_dict` didn't
  error), but the network computed a different function at inference, so BC-init showed
  zero improvement over a random policy. After switching to Tanh, loading BC weights
  into an untrained PPO policy jumped deterministic rollout reward from -3.04 (random)
  to +11.64 (immediate success) with zero PPO steps — confirms the load path is correct.
  Lesson: matching a state dict's shapes is not sufficient to reuse another framework's
  weights; the module graph (activations included) must match exactly.
- **2026-07-11 — No real teleop recording available for BC in this environment (no
  interactive display for a human to drive WASD).** `train/bc.py` includes
  `generate_scripted_demo()`, a proportional "aim at the goal" controller that produces
  `.npz` files in the exact same format `teleop --record` writes, used to validate the
  BC pipeline end-to-end. It is a test fixture, not a replacement for real demonstrations
  — swap in actual `teleop --record` output for real BC training.
- **2026-07-11 — `cli.py`'s `view`/`teleop`/`play` transparently re-exec under `mjpython`
  on macOS.** `mujoco.viewer.launch_passive` raises `RuntimeError` unless the process was
  launched via mujoco's own `mjpython` launcher (it dispatches the GUI to the main thread,
  which plain `python`/`uv run` doesn't do). `_ensure_mjpython()` checks
  `platform.system() == "Darwin"`, finds `mjpython` on PATH, and `os.execvp`s into
  `mjpython -m robosim.cli <original argv>` (guarded by an env var against re-triggering
  post-relaunch) — so `uv run robosim view ...` just works without the user needing to
  know about mjpython. Verified live: `view`, `teleop`, and `play` all open a real GUI
  window on this machine and run the physics/render loop at real CPU load; `play` on the
  M4 checkpoint completed a full episode (reward -0.01 → +11.64, success) with a live HUD.
- **2026-07-11 — HUD `print(..., end="\r")` needs `flush=True`.** Found while verifying
  `play` live: Python block-buffers stdout when it's not a TTY (and even interactively,
  `\r`-only lines don't trigger line-buffering's flush-on-`\n`), so the HUD never
  actually appeared without an explicit flush. Fixed in `ui/view.py` and `train/play.py`.
- **2026-07-11 — Isaac-feel UI pass: theme lives in `core/theme.py`, chrome in
  `ui/overlay.py`.** The dark skybox/grid are MjSpec assets fixed at compile time, so
  the theme is applied by `core/world.py` — placing it under `core/` keeps the import
  graph acyclic (core never imports ui). It is presentation-only by contract: nothing in
  theme.py may touch option/solver/friction. Grid = builtin checker texture with
  `mark=edge` on a plane material (texuniform, 0.5 m cells); ground planes are themed by
  name via `Scene.ground_geom_names()` — heightfield terrain keeps its scene-chosen
  look. Envs compile the same themed model (adds ~1.5 MB of texture per env; negligible,
  and `render_mode="rgb_array"` frames match the viewer look).
- **2026-07-11 — HUD uses the passive viewer's native `set_texts` overlay API**
  (mujoco ≥ 3.3; present in our 3.10): two-column text boxes at grid positions — no
  dearpygui side window needed for anything in idea.md's must-have list. Debug overlays
  (lidar rays, trail, goal beacon) draw `mjGEOM_LINE` connectors into `viewer.user_scn`
  (`mjv_initGeom` + `mjv_connector`; line width is in pixels); contact points/forces are
  just `viewer.opt.flags[mjVIS_CONTACTPOINT/CONTACTFORCE]`. World-axes gizmo is
  `opt.frame = mjFRAME_WORLD`. All shared across view/teleop/play via `ui/overlay.py`,
  whose key/camera/timing logic is plain functions on a plain state object —
  unit-testable without a window (tests/test_ui.py, 13 tests).
- **2026-07-11 — Viewer loops repaced to a wall-clock budget.** Old loops did one
  mj_step + sleep(timestep) per iteration → a 500 Hz Python loop that couldn't hold
  real-time once overlay work was added, and sim-speed changes distorted it. New loop:
  each ~60 fps UI frame runs `int(wall_dt × speed / timestep)` physics steps (capped),
  so UI rate and physics rate are decoupled, runtime speed 0.1x–5x (`-`/`=` keys) is
  honest, and the HUD's real-time factor is measured, not assumed. Teleop records obs
  per physics step inside the batch, unchanged .npz format.
- **2026-07-11 — Screenshot key ('p') re-renders offscreen instead of grabbing the GUI
  framebuffer.** The passive handle exposes no framebuffer read, so a throwaway
  `mujoco.Renderer` copies `viewer.cam` and renders 1280×720 to
  `runs/screenshots/` (offwidth/offheight raised in the theme). Creating a second GL
  context while the GUI runs worked in offscreen tests but is guarded try/except in the
  loops — worst case it prints a warning instead of killing the session.
- (Further decisions get appended here as they're made — newest at the bottom.)
