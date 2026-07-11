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
- (Further decisions get appended here as they're made — newest at the bottom.)
