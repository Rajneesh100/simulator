# Rocket fire-fight: `--firearms rocket` — design

Branch: `fire-fight` (from `main`). Date: 2026-08-30.

## Goal

Give race cars a rocket launcher, train a `rally6` policy that shoots rivals down and
dodges incoming fire, and animate it in `robosim race`. Everything is gated behind a new
`--firearms rocket` flag on `train` and `race`; with the flag absent (default `none`)
every existing checkpoint, scene, command, and test behaves exactly as today.

## Physics model (the part that must be "real")

- A rocket is a point projectile simulated in Python: `pos (3)`, `vel (3)`, `owner`,
  `age`. It is **not** a MuJoCo body (no model changes, no pre-allocated slots).
- **Launch**: `vel = car_velocity_world + rocket_speed * dir(az, el)` where `az` is the
  launch azimuth relative to the car's heading and `el` the elevation above horizontal.
  Launch point is 0.12 m above the chassis center so the rocket clears the car.
  Inheriting the platform velocity is what makes a sideways shot from a fast car skew
  forward — the moving-platform difficulty.
- **Flight**: every physics substep (2 ms) `vel.z -= g * dt`, `pos += vel * dt` — a true
  parabola. `g` is read from the MuJoCo model (`model.opt.gravity`).
- **Impact**: the swept segment `pos_prev -> pos` is ray-cast with `mj_ray` (all geoms,
  excluding the owner's body subtree for the first `ARMING_M = 0.5` m of flight so the
  rocket doesn't detonate on its own launcher). First hit (car, wall, road, terrain)
  detonates at the hit point. Rockets older than `MAX_FLIGHT_S = 4 s` self-destruct
  (mid-air, no blast).
- **Range**: with muzzle speed `v` the flat-ground range is `v² sin(2 el) / g`, max at
  45°. `--rocket-speed` (default 40 m/s) is therefore the range parameter: 40 m/s
  reaches 163 m at 45°, and needs only ~1.8° of elevation for a 10 m target — fast,
  nearly flat flight (0.25 s to 10 m, 0.4 s to 15 m) against cars doing ≤ 12 m/s.
- **Blast**: sphere of `BLAST_RADIUS = 2.0 m` around the detonation point. Every car
  whose chassis center is inside is destroyed — **including the shooter** (point-blank
  shots are suicide). Destroyed cars receive a physics impulse (`qvel += up 4 m/s +
  radial 3 m/s + random spin`) so they visibly tumble.
- **Reload**: `COOLDOWN_S = 2.0` per car. Decisions happen at the 20 Hz control rate
  (50 ms), which is the reaction-time floor for shooter and target alike.

All of this lives in `src/robosim/combat/rockets.py` (`RocketSystem`) with no viewer or
gym dependency, so the env and the viewer share one implementation.

## Aiming: who computes the ballistics

- **The learner does not get a solver.** Its action is `[throttle, steer, fire, az, el]`
  (see below); from the target's range, bearing and closing speed in its observation it
  must learn to choose the azimuth lead and the elevation for the range itself.
- **Scripted gunner** (`src/robosim/combat/gunner.py`) — used for opponents in training
  and for cars driven by weaponless checkpoints in race mode. It solves the intercept:
  iterate `t_flight` 4 times: predict target `p + v_target * t`, compute the launch
  direction whose horizontal muzzle component (after subtracting the shooter's velocity)
  reaches that range with the low-arc elevation `el = 0.5 * asin(g R / v²)`, recompute
  `t = R / v_horizontal`. Fires when a target is within `gun_range` (default 20 m), the
  solution exists, the blast would not include the shooter, and cooldown is over. Aim
  noise `sigma_deg` (default 2°) keeps it beatable. Exposed helper
  `solve_intercept(shooter_pos, shooter_vel, target_pos, target_vel, speed, g)` is unit-
  tested by simulating the rocket and checking it lands on the moving target.

## Policy interface (only when firearms enabled)

**Action** `Box(-1, 1, (5,))`: `[throttle, steer, fire, az, el]`.
- `fire >= 0.5` launches if the weapon is ready (else ignored).
- `az` maps `[-1, 1] -> [-180°, +180°]` relative to heading (0 = straight ahead,
  +90° = left, ±180° = backwards).
- `el` maps `[-1, 1] -> [0°, 45°]` (low-arc family only; the high arc is never useful
  against cars).

**Observation**: `build_obs(..., combat=None)` in `envs/obs.py` appends a combat block
when given — `obs.py` stays the single definition of the layout:

```
[ lidar | goal (3) | proprio (5) | prev_action (5) | combat (16) ]
combat = for k in 3 nearest cars, sorted by distance (missing -> zeros):
             [ dist/COMBAT_NORM_DIST, sin bearing, cos bearing, closing_speed/12 ]
         [ weapon_ready (0/1) ]
         [ threat dist/COMBAT_NORM_DIST, sin, cos ]  # nearest rocket whose velocity
                                                       # points within 30° of me;
                                                       # zeros when none
```
`COMBAT_NORM_DIST = 20 m`. Closing speed is positive when the range is shrinking.
`obs_dim` gains `+16` and `action_dim` becomes 5 in firearms mode; both are recorded in
the run's `config.yaml` as `firearms: rocket` / `rocket_speed: 40.0` and recovered by
`run_config.env_kwargs_from_run` so `play`/`race` rebuild a matching env.

## Training: `VersusRaceEnv(firearms="rocket", rocket_speed=40.0)`

Learner = car 0, opponents = frozen checkpoint drivers **plus the scripted gunner** (the
frozen rally5 never held a launcher; without scripted incoming fire the learner would
never learn to dodge). Opponents fire through the same `RocketSystem`.

Reward additions on top of the existing versus reward:
- `KILL_BONUS = +15` for each opponent destroyed by a rocket the learner fired.
- `DEATH_PENALTY = -20` and `terminated=True` when the learner is destroyed — by any
  rocket, including its own blast. Survival is the dodge incentive.
- `SHOT_COST = -0.3` per launch (no spraying).
- Everything else (progress, overtake, takedown, wall crash, win/lose) unchanged.
- Opponents killed respawn after `OPP_RESPAWN_S = 3 s` via the existing `_respawn`.
- `info` gains `kills`, `shots`, `killed_by` for evaluation.

`train.py` passes `firearms`/`rocket_speed` through to the env and config; `--firearms`
requires `--versus` (solo race has nothing to shoot).

**Warm start**: `train/bc.py::expand_policy_state_dict(sd, extra_obs, extra_act)`
zero-pads the first layer of the policy and value MLP extractors (new obs columns) and
the `action_net` (new action rows, weight 0 and bias 0 -> aims at 0/0 and never fires
until learned) plus `log_std`. On the original inputs the expanded net reproduces rally5's
driving outputs exactly (unit-tested). `train --bc-init` accepts the rally5
`policy_init.pt` directly: if shapes mismatch and firearms is on, it expands
automatically and logs it.

Recipe (documented in README):
```
uv run robosim train --task "race:laps=2,lookahead=2.5,lookahead_time=0.6" \
  --scene "racetrack:layout=gp,scale=25,seed=11,hills=6,max_grade=0.3,hill_freq=8,road_width=3.5" \
  --robot racecar:max_speed=12,max_torque=1.2 --steps 1000000 --n-envs 8 \
  --lidar-beams 48 --lidar-fov 360 --lidar-range 12 --max-episode-steps 1500 \
  --versus runs/rally5/best_model.zip --versus-opponents 2 \
  --firearms rocket --rocket-speed 40 \
  --bc-init runs/rally5/policy_init.pt --run-name rally6
```

## Race mode: `robosim race ... --firearms rocket [--rocket-speed 40]`

- Every car gets a launcher. Cars whose checkpoint trained with firearms fire from
  their policy's action; cars driven by weaponless checkpoints use the scripted gunner
  (so rally5 vs rally5 with rockets works today, and rally6 vs rally5 is a fair duel).
- A destroyed car: charcoal tint, actuators zeroed, tumbles under the impulse, `DEAD`
  on the leaderboard; after `RESPAWN_S = 4 s` the existing marshal respawns it at its
  current `s` (progress kept). Race still ends on laps.
- Leaderboard grows `K/D`; terminal prints `KILL: RED -> BLUE` and `SELF-KILL: RED`.
- Personality (`--variation`) also scales gunner aim noise for scripted shooters.

### Animation (`ui/overlay.py`, drawn into `user_scn` every frame)
- `draw_rocket`: a 0.18 m capsule along the velocity vector in the owner's color, a
  bright nose sphere, and an exhaust trail of 8 fading translucent spheres sampled from
  the last positions.
- `draw_blast`: over 0.5 s an expanding sphere (0.3 m -> 2.0 m, orange -> dark red,
  alpha 0.9 -> 0) plus a thin flash ring of line segments at `BLAST_RADIUS` on the
  ground so the damage circle is visible.
- Aim marker: while a car is reloading, a short red line shows its last launch
  direction (helps read what happened).

## Files

New: `src/robosim/combat/__init__.py`, `combat/rockets.py`, `combat/gunner.py`,
`tests/test_combat.py`.
Modified: `envs/obs.py`, `envs/versus_env.py`, `ui/race.py`, `ui/overlay.py`,
`train/train.py`, `train/run_config.py`, `train/bc.py`, `cli.py`, `README.md`,
`CLAUDE.md` (decision log), `tests/test_racing.py` (versus+firearms check_env).

## Testing

- Ballistics: a rocket fired horizontally at 40 m/s from a stationary car in a flat
  world hits the ground at the analytic range; a sideways shot from a car moving at
  10 m/s carries the 10 m/s forward component.
- Impact: a rocket aimed at a stationary car 8 m ahead detonates on it and destroys
  it; the shooter survives at 8 m and dies at 1.5 m.
- Intercept: `solve_intercept` against a target crossing at 8 m/s lands within 0.3 m of
  the target's actual position at impact when the target holds course; a target that
  turns hard (perpendicular acceleration) after launch is missed — the dodge principle.
- Obs: combat block layout, ordering by distance, threat detection cone.
- Env: `check_env` passes for `VersusRaceEnv(firearms="rocket")`; a scripted "fire
  straight at the car in front" learner earns `KILL_BONUS`; self-blast terminates with
  `DEATH_PENALTY`.
- Warm start: expanded state dict reproduces the original action means on padded obs.
- Overlay draw functions run headless without exceeding `maxgeom`.

## Non-goals

Ammo limits, homing rockets, armor/health points, damage falloff, simultaneous
multi-agent self-play (league iteration stays: rally7 trains vs frozen rally6).
