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
- **2026-07-12 — Teleop drives with ARROW KEYS, not WASD (deliberate deviation from
  idea.md), and overlay keys avoid letters whose native effect is unrecoverable.**
  User-reported bug: driving with WASD turned the scene into wireframe and hid the
  stairs/floor. Root cause: MuJoCo's built-in viewer binds EVERY letter A-Z to a
  visualization shortcut, and in passive mode those fire alongside our key_callback
  with no way to suppress them (verified: `dir(mujoco._simulate.Simulate)` exposes no
  scene/render-flag access, and `launch_passive` has no suppression option). W toggled
  wireframe, S shadows, A autoconnect, D hid all static geometry — while driving.
  Two-part fix: (1) native effects that live in `mjvOption` (shared with us) are pinned
  back every frame in `overlay.apply_vis_flags` — static/texture/transparent/
  autoconnect/convexhull/contactsplit flags plus geom/site group visibility; (2) native
  effects in the internal render scene (wireframe W, shadow S, additive L, fog G,
  reflection R) are unreachable, so nothing may be bound to those letters: driving
  moved to arrow keys (bonus: auto-repeat = hold-to-drive), lidar moved l→'1', goal
  g→'2' (digit keys only touch pinned geom groups). 'r' reset keeps a known cosmetic
  quirk (native reflection toggle). Lesson: when embedding mujoco's viewer, treat the
  letter keyspace as reserved; check a candidate key against mjVISSTRING/mjRNDSTRING
  native shortcuts before binding it.
- **2026-07-12 — HUD overlay strings must be pure ASCII.** mujoco's `mjr_overlay`
  bitmap font has no glyphs beyond ASCII — the degree sign and omega we used rendered
  as garbage boxes in the user's screenshot. All `set_texts` content is ASCII-only now
  ("deg", "turn"), enforced by a test; fancy unicode stays in terminal prints only.
- **2026-07-12 — Native simulate side panels start hidden** (`show_left_ui/right_ui=
  False` on launch_passive; Tab/Shift+Tab restore them). Our HUD carries the info the
  panels duplicated, and the cleaner viewport matches the Isaac reference.
- **2026-07-12 — Racing stack added as pure plugins: `robots/racecar.py`,
  `scenes/racetrack.py`, `envs/tasks/race.py`.** The racecar is a 4-wheel
  Ackermann-steered car whose action stays 2-dim — (throttle, steer) instead of
  (v, omega) — so every env/train/teleop path works unchanged; `action_to_ctrl` does the
  Ackermann per-wheel steer angles and acts as an open differential on the rear axle.
  Gotcha found live: MuJoCo's automatic parent-child contact exclusion does NOT cover
  grandchildren, and the front wheels (chassis → steering knuckle → wheel) collided with
  the chassis, pinning the car at ~0 m/s under full throttle. Fixed with explicit
  `spec.add_exclude(chassis, front_*_wheel)` pairs — these DO survive
  `frame.attach_body` into the world spec. Racecar also normalizes wheel velocities in
  `read_proprio` (its ~50 rad/s raw wheel speeds would blow past every other robot's
  obs scale).
- **2026-07-12 — Scripted movers extracted to `core/movers.py` with a new waypoint-loop
  path type; view/teleop now animate them too.** The original axis-sine mover update
  lived inside `RoboSimEnv`, so mocap obstacles only moved during env-driven runs
  (train/play) and stood still in `view`/`teleop`. The shared module supports the
  original axis-sine spec unchanged plus `{"path": "loop", "waypoints": [...], "period",
  "phase", "z"}` — constant-speed travel around a closed polyline with yaw following the
  tangent — used by racetrack opponent cars. Movers advance per physics step in all four
  loops (env, view, teleop, play-via-env).
- **2026-07-12 — Race task rewards progress-along-centerline, not distance-to-goal.**
  The racetrack scene publishes its centerline waypoints in `task_info()`; the task
  projects the robot onto that polyline each step and rewards Δarclength (so speed IS
  the reward), with wrap handling at the start/finish line, a per-step ds clamp
  (projection glitches must not mint reward), a centering penalty, crash-terminates on
  collision, and a lap-completion bonus/success. The obs "goal" slot is repointed every
  step to a lookahead point 2.5 m ahead on the centerline — the standard obs vector
  doubles as "where does the track go" with zero layout changes.
- **2026-07-12 — Lidar config and episode length are now per-run settings, recovered
  automatically at play/evaluate time.** `train` grew `--lidar-beams/--lidar-fov/
  --lidar-range/--max-episode-steps` (obs size changes with beam count), records them in
  the run's config.yaml, and `play`/`evaluate` read that config.yaml (checkpoint dir or
  its parent, via `train/run_config.py`) to rebuild a matching env — a checkpoint is
  only usable with the obs space it was trained on. Racing default: 48 beams / 360 deg /
  12 m ("fast lidar"); old runs without the keys fall back to env defaults.
- **2026-07-12 — Overlapping visual road strips z-fight; alternate their heights by
  0.8 mm.** The racetrack road is drawn as overlapping non-colliding box segments along
  the centerline; coplanar overlaps rendered as bright striping. Alternating segment
  z by (i % 2) · 0.0008 is invisible at viewing distance and kills the artifact.
- **2026-07-12 — Lidar self-hit fix: `bodyexclude` excludes ONE body, but a robot is a
  body tree.** On the racecar with 360-degree FOV, 12 of 48 beams permanently read
  0.11–0.40 m on an empty scene — side/rear rays hitting its own wheels/cabin (separate
  bodies from the excluded chassis). `Lidar.sense` now takes `exclude_body_ids`
  (`body_subtree_ids()` of the robot root); beams landing on a self body are re-cast
  from just past the hit until they reach the world. All sense/debug_rays callers pass
  the subtree. NOTE: checkpoints trained before this fix (e.g. race1) learned with
  polluted side beams — retrain for best results.
- **2026-07-12 — Tasks take CLI params exactly like scenes/robots: `--task race:laps=N`.**
  `RoboSimEnv` now routes the task arg through `_parse_scene_arg`. Race uses it for lap
  count: laps=N terminates with success after N laps; laps=0 is endless demo mode (lap
  bonuses keep coming, episode ends only on crash or the step cap — `play` grew
  `--max-steps` to raise that cap for continuous lapping).
- **2026-07-12 — Racetrack layouts are just centerline generators; added `layout=gp`.**
  Everything in the scene (walls, road strips, obstacles, opponent lanes, spawn, task
  waypoints) derives from `self._center`, so the procedural grand-prix circuit is one
  function: a radius-modulated circle (harmonics k=2..5, amplitudes damped 1/k so
  curvature stays drivable at road width), resampled uniform-in-arclength.
  `--scene racetrack:layout=gp,scale=12,seed=7` — scale is mean radius; each seed is a
  different circuit.
- **2026-07-12 — `robosim race`: N copies of ONE policy racing in ONE world
  (`core/world.py build_multi_world` + `ui/race.py`).** Cars attach with prefixes
  (`c0_`, `c1_`, ...) onto an F1-style grid (rows of two along the track behind the
  start line); `MjSpec` excludes and all names survive prefixed attach. Each car gets
  its own Lidar (prefixed site, subtree self-exclusion — cars DO see each other),
  prefixed proprio (`read_proprio` grew a `prefix` param), and its own progress/lap
  tracker reusing race-task helpers; actions come from ONE `model.predict` batch call.
  Palette-tinted chassis (RED/GREEN/BLUE/YELLOW/WHITE/...), live leaderboard HUD,
  follow-cam tracks the current leader, first to `--laps` wins, standings print at the
  finish. Verified headless: 4 race1-driven cars, leaders lapping at ~3.2 m/s. Caveat:
  solo-trained policies never saw other cars — early-race contact can strand cars
  (training ends episodes on collision, so there's no recovery behavior). Multi-car
  *training* (self-play in the env) is future work; race mode is currently a viewer.
- **2026-07-12 — Race mode: identical deterministic brains from a fixed grid = the same
  race every run, with side-by-side pairs wedging each other at turn one.** User-observed:
  every restart replayed identically; paired grid slots stalled both partners (solo-trained
  policies have no contact recovery — collision *ended* training episodes). Fixes, all
  race-mode-level: (1) grid is now staggered single-file (one car per row, alternating
  lanes); (2) actions are sampled stochastically by default (`--deterministic` restores
  replay mode, `--seed` makes variety reproducible); (3) anti-stall: >1.5 s at ~zero speed
  triggers a 0.9 s scripted reverse-and-turn ("rec" on the leaderboard); (4) per-car
  DRIVER PERSONALITY (`--variation`, default 0.5): seeded offsets in throttle aggression
  (x0.94–1.03), cornering lookahead (1.6–3.75 m), and action jitter — same checkpoint,
  genuinely different racing lines and overtakes. Every car draws a tracer in its palette
  color (`draw_trail` grew an rgba param; `push_trail` duck-types onto any object with a
  `.trail`). Verified headless: 3 seeds → 3 different winners, incl. a 146.1 m vs 146.0 m
  finish. Real fix remains multi-car training/self-play.
- (Further decisions get appended here as they're made — newest at the bottom.)
