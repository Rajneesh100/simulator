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

## The model — what's actually driving

The default policy (`--policy p1`) is a PPO **actor-critic**: two small independent
MLPs that share nothing but the input vector. It's deliberately tiny (~16k parameters,
~64 KB as float32) — it has to run at 20 Hz on a Raspberry Pi 5 later, and for
reactive driving from lidar, small is enough.

### The input vector (observation)

One flat float32 vector, built identically every control step (`envs/obs.py`):

| Slice | Size | Meaning |
|---|---|---|
| lidar | `n_beams` | distances, normalized 0–1 (1 = nothing within max range) |
| goal | 3 | distance to goal ÷ 10 m, sin(bearing), cos(bearing) — robot frame |
| wheel velocities | 2 | left/right wheel speed (normalized on `racecar`) |
| IMU | 3 | pitch, roll, yaw-rate |
| previous action | 2 | what the net commanded last step |

Size = `n_beams + 10`. The diffdrive default (16 beams) → **26**; the racecar's race
config (48 beams) → **58**. For the race task, "goal" is repointed every step to a spot
2.5 m ahead on the track centerline, so it doubles as "where the road goes".

### Layer-by-layer (race1 checkpoint: 58-dim obs, 2-dim action)

```
                       observation (58)
                      /                \
            ACTOR ("what to do")     CRITIC ("how good is this state")
        ┌───────────────────────┐  ┌───────────────────────┐
        │ Linear 58 → 64   tanh │  │ Linear 58 → 64   tanh │   (3,776 + 3,776 weights)
        │ Linear 64 → 64   tanh │  │ Linear 64 → 64   tanh │   (4,160 + 4,160)
        │ Linear 64 → 2  (mean) │  │ Linear 64 → 1  (value)│   (130 + 65)
        └───────────┬───────────┘  └───────────────────────┘
                    │ + log_std (2 learned params, state-independent)
                    ▼
        action = (throttle, steer) ∈ [-1, 1]²
```

| Component | Layer | Neurons | Params (w + b) |
|---|---|---|---|
| actor | hidden 1 | 64, tanh | 58·64 + 64 = 3,776 |
| actor | hidden 2 | 64, tanh | 64·64 + 64 = 4,160 |
| actor | action head | 2, linear | 64·2 + 2 = 130 |
| actor | log_std | 2 (free params) | 2 |
| critic | hidden 1 | 64, tanh | 58·64 + 64 = 3,776 |
| critic | hidden 2 | 64, tanh | 64·64 + 64 = 4,160 |
| critic | value head | 1, linear | 64·1 + 1 = 65 |
| **total** | | | **16,069** |

During **training**, the actor's output is the *mean* of a Gaussian; PPO samples around
it (spread = `exp(log_std)`) to explore, and the critic's value estimate is used to
compute advantages. During **play/race** (`deterministic=True`) the mean is used
directly and the critic is dead weight — only the ~8k-param actor actually drives.

The action then goes through the robot's `action_to_ctrl()` (pure math, not learned):
diffdrive maps `(v, ω)` to two wheel speeds; racecar maps `(throttle, steer)` to
Ackermann steering angles + differential rear-wheel speeds. The whole loop:

```
 lidar rays ─┐
 IMU        ─┼→ obs vector → actor MLP → (throttle, steer) → action_to_ctrl()
 encoders   ─┤     ↑ 20 Hz                                        │
 prev action┘                                                     ▼
             MuJoCo physics @ 500 Hz  ←  4 actuator targets (velocity/position servos)
```

### The other architectures (`--policy p2/p3/p4`)

| | Idea | Shape |
|---|---|---|
| `p2` | 1D-CNN over the lidar slice (learns spatial patterns like "gap between obstacles") | Conv1d 1→8 (k=5) → Conv1d 8→16 (k=3) → pool → 32 features, + 16-unit MLP for the rest → concat 48 → one 64 hidden layer |
| `p3` | LSTM memory (remembers what it saw around the corner) | 64-unit LSTM + 64 hidden, via RecurrentPPO |
| `p4` | frame stacking (cheap short-term memory) | same MLP as p1 but input = last 4 obs concatenated (4× input size) |

Inspect any checkpoint's exact layers yourself:

```bash
uv run python -c "from stable_baselines3 import PPO; print(PPO.load('runs/race1/best_model.zip', device='cpu').policy)"
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



## Racing

```bash
# look around / drive it yourself (arrow keys)
uv run robosim view   --scene "racetrack:obstacles=6,opponents=2" --robot racecar
uv run robosim teleop --scene "racetrack:obstacles=6,opponents=2" --robot racecar

# F1-style procedural circuit with hills (every seed = a different track;
# hills=H meters of elevation — the road becomes a real pitched surface)
uv run robosim view --scene "racetrack:layout=gp,scale=12,seed=7,hills=1.2" --robot racecar

# train the driver (start ~500k steps; racing is harder than goal_reach).
# racecar:max_speed=6 makes the car grip-limited: flat-out cornering physically
# fails, so the policy has to LEARN braking points — real racing behavior.
uv run robosim train --task race \
  --scene "racetrack:layout=gp,scale=12,seed=7,hills=1.2,obstacles=4" \
  --robot racecar:max_speed=6 --steps 800000 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 \
  --max-episode-steps 1200 --run-name race_gp

# watch it lap forever (laps=0 = endless; needs the same --robot it trained with)
uv run robosim play --checkpoint runs/race_gp/best_model.zip \
  --scene "racetrack:layout=gp,scale=12,seed=7,hills=1.2,obstacles=4" \
  --robot racecar:max_speed=6 --task race:laps=0 --max-steps 100000 --episodes 1

# THE RACE: N colored cars (RED/GREEN/BLUE/YELLOW/WHITE/...), all driven by the
# same checkpoint, grid start, live leaderboard, first to --laps wins.
# --variation gives each car a driver personality (aggression/lookahead/jitter);
# stochastic by default so every race is different (--seed for reproducible,
# --deterministic for exact replays). Flipped cars get marshaled back upright.
uv run robosim race --checkpoint runs/race_gp/best_model.zip \
  --scene "racetrack:layout=gp,scale=12,seed=7,hills=1.2" \
  --robot racecar:max_speed=6 --cars 5 --laps 20
```

Racetrack params: `layout` (stadium | gp), `scale` (gp mean radius, m), `seed`,
`hills` (elevation amplitude in METERS — bounded by `max_grade` x lap length; if the
request doesn't fit, it prints what it built), `max_grade` (default 0.12, rally ~0.3),
`hill_freq` (harmonics; 3 = smooth GP, 8 = rally bumps), `straight`/`radius` (stadium),
`road_width`, `obstacles`, `opponents`, `opponent_speed`. Tracks come dressed: red/white
kerb walls, dashed centerline, checkered start/finish line.

### Rally training recipes (the lineage, in order of results)

All on the same rally track — bumpy, steep, jumpy. Train and race with the same
`--robot` string, always.

```bash
# rally1 — the baseline that works: plain MLP, 1M steps, straight onto the hills.
uv run robosim train --task race \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --steps 1000000 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 \
  --max-episode-steps 1500 --run-name rally1

# rally2 (NEGATIVE result, kept for the record): p3 LSTM + curriculum lost badly to
# rally1 — RecurrentPPO was unstable at this budget, and curriculum broke best_model
# selection (see CLAUDE.md decision log). Don't repeat without more steps.

# rally3 — the current champion: same recipe as rally1, DOUBLE the budget.
# First checkpoint to lap the rally crash-free (~22 s laps, zero marshal rescues).
uv run robosim train --task race \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --steps 2000000 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 \
  --max-episode-steps 1500 --run-name rally3

# rally4 — rally3 + the course-correction fixes: speed-scaled lookahead (sees hairpins
# earlier -> earlier braking) and random_heading (30% of episodes start facing a random
# direction, teaching it to U-turn back onto the track after a spin).
uv run robosim train --task "race:lookahead=2.5,lookahead_time=0.6,random_heading=0.3" \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --steps 2000000 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 \
  --max-episode-steps 1500 --run-name rally4

# race any of them (task lookahead params are auto-recovered from the run config):
uv run robosim race --checkpoint runs/rally3/best_model.zip \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --cars 5 --laps 10
```

Lessons encoded above (details in CLAUDE.md's decision log): scale the recipe that
works before trying a cleverer architecture; with curriculum runs never trust
best_model.zip (pick from checkpoints/ via `robosim evaluate`); lookahead params
change what the net sees, so they live in the task string and are recorded in the
run config.

### Competitive training — train to WIN, not just to lap

`--versus <checkpoint>` switches training to league mode: the learner races opponent
cars driven by a FROZEN champion, and the reward pays for racecraft — track progress,
signed overtake bonuses (losing places hurts), Burnout-style TAKEDOWNS (+8 when a rival
wrecks within 1.5 s of contact with you), and win/lose bonuses for finishing first
to `laps`. Car-to-car contact is free; walls still end your race — so blocking and
smashing must be aimed. The learner starts from a shuffled grid slot every episode
(sometimes ahead, sometimes buried in traffic). Output is a normal checkpoint that
plugs straight into `robosim race` duels.

```bash
# rally5 — racecraft school. Opponents = frozen rally3 (the fastest clean racer =
# hardest rivals). Warm start = rally4's weights: it has the traffic-relevant skills
# (contact recovery, wrong-way U-turns from random_heading training) AND it trained
# with the same speed-scaled lookahead this task string uses, so the obs semantics
# match exactly. (Versus physics is ~7x slower than solo: 3 cars + opponent
# inference; the warm start is what makes 1M steps enough.)
uv run robosim train --task "race:laps=2,lookahead=2.5,lookahead_time=0.6" \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --steps 1000000 --n-envs 8 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 --max-episode-steps 1500 \
  --versus runs/rally3/best_model.zip --versus-opponents 2 \
  --bc-init runs/rally4/policy_init.pt --run-name rally5

# then the grudge match: the killer vs its own teacher
uv run robosim race \
  --checkpoint runs/rally5/best_model.zip --checkpoint runs/rally3/best_model.zip \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --cars 6 --laps 10 --variation 0.2
```

(`runs/rally4/policy_init.pt` is that run's raw policy weights — regenerate with
`PPO.load(...).policy.state_dict()` -> `torch.save` after any retrain. Head-to-head
context: rally3 out-paces rally4 in clean solo racing (450 m vs 385 m per race-minute)
but rally4 carries the recovery skills that matter in traffic — hence opponent=rally3,
warm-start=rally4. For a stronger league, repeat: train rally6 versus frozen rally5,
and so on.)

### Fire-fight: `--firearms rocket`

Every car gets a rocket launcher (`src/robosim/combat/`). Rockets leave with the car's
own velocity plus `--rocket-speed` (40 m/s) along the aim, fall under gravity (real
parabola: range = v^2 sin(2 el) / g, ~1.8 deg of elevation for a 10 m shot; launchers
depress to -15 deg for close targets), detonate on the first thing they touch, and
destroy every car within 2 m — the shooter included. Like a real RPG fuze, a rocket only
ARMS after 4 m of flight: anything it hits sooner is a dud (no blast), so point-blank
shots can't be suicide. A destroyed car turns charcoal, tumbles, and is rebuilt by the
marshal after 4 s. Kills/deaths show on the leaderboard.

Watch it today with any checkpoint (weaponless brains get a scripted intercept gunner
that solves the moving-platform / moving-target ballistics exactly, plus a little aim
noise scaled by `--variation`):

```bash
uv run robosim race --checkpoint runs/rally5/best_model.zip \
  --scene "racetrack:layout=gp,scale=18,seed=11,obstacles=0,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 \
  --cars 5 --laps 5 --variation 0.8 --firearms rocket
```

Train a brain that shoots and dodges itself — rally6. The action grows to
`[throttle, steer, fire, az, el]` and the obs gains a 16-float combat block (3 nearest
rivals' range / bearing / closing speed, weapon ready, nearest incoming rocket — see
`envs/obs.py`). The learner gets NO ballistic solver: it has to learn the lead and the
elevation from those numbers. Opponents are frozen rally5 drivers with the scripted
gunner, so there is real incoming fire to dodge. rally5's weights are expanded in place
(zero-padded new obs columns / action rows, fire bias -1) so it starts out driving
exactly like rally5 and only has to learn to shoot and survive:

```bash
uv run python -c "import torch; from stable_baselines3 import PPO; torch.save(PPO.load('runs/rally5/best_model.zip', device='cpu').policy.state_dict(), 'runs/rally5/policy_init.pt')"
uv run robosim train --task "race:laps=2,lookahead=2.5,lookahead_time=0.6" \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --steps 1000000 --n-envs 8 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 --max-episode-steps 1500 \
  --versus runs/rally5/best_model.zip --versus-opponents 2 \
  --firearms rocket --rocket-speed 40 \
  --bc-init runs/rally5/policy_init.pt --run-name rally6

# then: rally6 (policy gunner) vs rally5 (scripted gunner)
uv run robosim race --checkpoint runs/rally6/best_model.zip --checkpoint runs/rally5/best_model.zip \
  --scene "racetrack:layout=gp,scale=18,seed=11,obstacles=0,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --cars 6 --laps 5 --firearms rocket
```

**How cars see each other** (`--combat-sensor`, default `los`): no physics oracle. A
rival enters the combat block only if it is within `--view-range` (default = lidar range,
12 m) AND a line of sight from the observer's eye point reaches it — walls, hill crests
and cars in between hide it (360 deg: front, sides, mirrors). Its velocity is not read
from the engine: the observer differences its last two sightings ("two glances"), so a
first sighting has velocity 0 and a rival reappearing from behind a wall restarts the
estimate. Rockets in flight are seen the same way (exhaust visible, not through walls).
Scripted gunners look through the same eyes, so nobody shoots through a wall and their
intercept solutions inherit the estimation error. `--combat-sensor oracle` is the
physics-truth ablation for debugging. The 48-beam lidar stays for driving; it cannot
resolve a car past ~3 m (7.5 deg between beams), which is why sight is modeled as eyes.

Reward in versus+firearms: +15 per kill, -20 and episode over when destroyed (-30 if
it was your own blast), -0.3 per shot, on top of the usual progress/overtake/takedown/win
terms. Scripted gunners hold fire when the predicted impact is inside the arming
distance or within blast radius + 1 m of where THEY will be at impact time.
`firearms`/`rocket_speed` are recorded in the run's config.yaml and recovered by
`play`/`race` like the lidar settings; a checkpoint trained with rockets refuses to race
without `--firearms rocket`. For a stronger league, train rally7 versus frozen rally6.

MODEL DUELS — repeat `--checkpoint` to pit different trained models against each other
(cars assigned round-robin; roof tags and the leaderboard show `COLOR:runname`; each
model automatically gets the lidar/lookahead settings it trained with):

```bash
uv run robosim race \
  --checkpoint runs/rally1/best_model.zip --checkpoint runs/rally3/best_model.zip \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --cars 6 --laps 10 --variation 0.2
```

Speed ceiling: ~15 m/s on this car scale (already ~30 body-lengths/s, a scale ~350 km/h).
Beyond that, cars cross more than a wall thickness per 2 ms physics step and contacts
tunnel — for faster racing scale the whole track up instead.

Racecar params: `max_speed` (m/s), `max_steer` (rad), `max_torque` (N·m per driven
wheel). The motors are velocity servos: with the default 2.5 N·m they have ~7x the
force any legal grade exerts, so they hold max_speed both up- AND downhill (gravity
is simulated, the motor cancels it — like an EV on cruise control with regen).
Lower it (`racecar:max_speed=6,max_torque=0.6`) and gravity becomes part of the
race: climbs sap speed, steep descents push the car PAST max_speed (measured:
commanded 3 m/s, 15-degree slope, 0.5 N·m -> 10.5 m/s). Train and race with the
SAME robot params — the checkpoint learns the car it was given.
