# RoboSim

A minimal, Isaac-Lab-style local physics simulator for training robot control policies —
MuJoCo for physics, Gymnasium for the env API, stable-baselines3 for RL. Phase 1 of a
larger WALL-E-style wheeled robot project: policies trained here are meant to transfer to
real hardware later. Full design spec: see `../idea.md` (kept one level up as the
original brief — not required at runtime).

Everything needed to work on this project — code, dependencies, Claude Code config —
lives inside this folder. No absolute paths, no writes to `~/.claude`, no global installs.

## Setup

```bash
./setup.sh          # installs uv if missing, then `uv sync`
uv run pytest        # should be all green
```

Requires macOS/Linux with Python available; `uv` pins the project to Python 3.11 (see
`.python-version`) for the safest torch/stable-baselines3/mujoco wheel compatibility on
Apple Silicon.

## CLI

```bash
uv run robosim scenes                                          # list scenes/robots/tasks
uv run robosim view   --scene stairs --robot diffdrive          # watch the world
uv run robosim view   --scene flat --debug-lidar                # + lidar ray overlay
uv run robosim teleop --scene obstacles --record runs/demo1     # WASD drive, optionally record
uv run robosim train  --task goal_reach --scene flat --steps 300000
uv run robosim play   --checkpoint runs/<run>/best_model.zip
uv run robosim evaluate --checkpoint runs/<run>/best_model.zip --episodes 50
uv run robosim export --checkpoint runs/<run>/best_model.zip --out policy.onnx --obs-dim 26
uv run robosim bc-train --records runs/demo1.npz --out bc.pt
```

`view`/`teleop`/`play` open MuJoCo's native interactive viewer with an Isaac-Sim-style
dark theme: gradient-dark viewport, 0.5 m ground grid, world-axes gizmo, and an on-screen
HUD (sim time, speed, real-time factor, physics steps/sec, robot pose/velocity — plus
live reward terms in `play` and drive command + REC counter in `teleop`).

Mouse orbits/pans/zooms (native). Keys, shared across all three modes:

| Key | Action | | Key | Action |
|---|---|---|---|---|
| `space` | pause / resume (`teleop`: stop driving) | | `1` | lidar rays overlay |
| `.` | single-step while paused | | `x` | contact points + forces |
| `r` | reset robot to spawn | | `t` | trajectory trail |
| `-` / `=` | sim speed 0.1x – 5x | | `2` | goal beacon |
| `c` | camera follow-robot | | `p` | screenshot → `runs/screenshots/` |
| `h` | hide/show HUD | | `Tab` / `Shift+Tab` | native left / right panel |
| | | | `F1` | native help overlay |

`teleop` drives with the **arrow keys** (`↑`/`↓` throttle, `←`/`→` turn, tap-to-set
±0.25 per press, hold for auto-repeat).

> **Why arrows and digit keys?** MuJoCo's built-in viewer binds *every* letter A–Z to a
> visualization shortcut that fires alongside our key handler (e.g. `W` = wireframe,
> `D` = hide static geometry) and can't be suppressed. RoboSim pins the recoverable
> flags back every frame, and avoids the unrecoverable ones (`W S L G R`...) in its own
> bindings. If you press a stray letter and the rendering changes, press it again to
> undo — it's a MuJoCo native toggle, not a bug in the scene.

`UI_CHECKLIST.md` is a 2-minute interactive walkthrough of all of the above.

### Scenes

`flat` (default), `stairs`, `slope`, `obstacles` (supports `maze=1` for an occluded-goal
corridor, `moving=N` for scripted mocap obstacles), `rough_terrain`. Pass params as
`--scene name:key=val,key2=val2`, e.g. `--scene stairs:height=0.05,count=12`.

### Tasks

`goal_reach` (T1), `obstacle_nav` (T2), `corridor` (T3), `slope_climb` (T4),
`stair_climb` (T5), `explore` (T6), `moving_obstacle` (T7) — see idea.md's task ladder.
`stair_climb` is expected to be hard for the wheeled `diffdrive` robot; that's a finding,
not a bug (a wheel can't lift over a step riser — motivates a legged robot later).

### Policies

`--policy p1` (MLP, default), `p2` (1D-CNN over lidar beams), `p3` (recurrent/LSTM via
sb3-contrib RecurrentPPO), `p4` (frame-stacked MLP). `--algo ppo` (default) or `sac`.

### Curriculum & domain randomization

```bash
uv run robosim train --curriculum "0:flat;50000:obstacles:count=3;150000:obstacles:count=10"
uv run robosim train --dr   # friction/mass/lidar-noise/lidar-dropout randomized per episode
```

## Architecture

```
src/robosim/
├── core/       # physics.py (ALL global physics constants), registry.py (scene/robot/
│               # task auto-registration), world.py (composes scene+robot MjSpecs)
├── scenes/     # one file per scene, @register_scene, Scene ABC in base.py
├── robots/     # one file per robot, @register_robot, Robot ABC in base.py
├── sensors/    # lidar.py (mj_multiRay raycasts), proprio.py (wheel vel + IMU)
├── envs/       # obs.py (canonical obs layout), base_env.py (Gymnasium Env),
│               # tasks/ (one file per task, @register_task)
├── ui/         # view.py, teleop.py — built on mujoco.viewer
├── train/      # train.py, play.py, evaluate.py, export.py, bc.py, policies.py
└── cli.py      # typer entrypoint: robosim ...
```

### Adding a new scene (no core edits)

```python
# src/robosim/scenes/my_scene.py
from robosim.core.registry import register_scene
from robosim.scenes.base import Scene

@register_scene("my_scene")
class MySceneSpec(Scene):
    def __init__(self, **params):
        self.foo = float(params.get("foo", 1.0))

    def build(self, spec):
        ...  # add geoms/bodies to spec.worldbody

    def spawn_pose(self):
        return (0.0, 0.0, 0.0), 0.0  # (x, y, terrain_z), yaw
```

That's it — `core/registry.py` auto-imports every module under `scenes/`, so
`robosim view --scene my_scene` works immediately. See `CLAUDE.md` for the full pattern
(including the physics-centralization rule) and the project decision log.

## Testing

`uv run pytest` — covers model builds for every scene, offscreen rendering, no-slide
driving physics, Gymnasium `check_env` + seed-determinism, lidar sanity, the full task
ladder, PPO/SAC training smoke runs across all four policy architectures, ONNX export +
int8 quantization, curriculum/domain-randomization/behavior-cloning pipelines end to end.

See `PROGRESS.md` for what's been verified against idea.md's milestones and acceptance
criteria, plus known limitations.
