# Progress

Built autonomously against `../idea.md`'s spec, milestone by milestone, with a commit
after each. ~3,100 lines of source+tests across `src/robosim` and `tests/`.

## Milestone status

| Milestone | Status | Evidence |
|---|---|---|
| M1 — World + viewer | Done | `robosim view/teleop` build on `mujoco.viewer`; diffdrive robot drives straight without lateral slide, rotates in place without translating (tests/test_world.py) |
| M2 — Scene system | Done | flat, stairs, slope, obstacles, rough_terrain all registered, load, offscreen-render, and stay physically stable under driven steps (tests/test_scenes.py) |
| M3 — Sensors + env | Done | lidar (mj_multiRay, configurable noise/dropout) returns correct hits on obstacles and all-max-range on empty flat; Gymnasium `check_env` passes; seed-determinism verified with and without domain randomization (tests/test_sensors.py, tests/test_envs.py) |
| M4 — Training | Done | Real 300k-step PPO run on goal_reach/flat: `ep_rew_mean` climbs -5.17 → 11.6 (converged), episode length drops 500 → ~113 as the policy learns the direct path; `play`-equivalent rollout hits the goal 10/10 deterministically; ONNX export matches torch to 1e-7, int8-quantized model produces finite output |
| M5 — Experiment suite | Done | Full T1-T7 task ladder; P1-P4 policies (MLP/CNN/LSTM/frame-stack) all smoke-trained; BC-init verified to jump a policy from -3.04 (random) to +11.64 (immediate success) with zero PPO steps; curriculum and domain-randomization run end-to-end inside real training; `evaluate.py` reports success/collision rate, time-to-goal, path efficiency |
| M6 — Polish | In progress | This file + README; `.claude`/CLAUDE.md conventions in place; fresh-clone test below |

`pytest`: **50 passed**, 0 failed.

## The M4 acceptance run in detail

`runs/m4_goal_reach_flat/` (gitignored, stays local — regenerate with the command below):

```bash
uv run robosim train --task goal_reach --scene flat --steps 300000 --n-envs 8
```

- Throughput: ~4,700 env-steps/sec aggregate on 8 `SubprocVecEnv` workers → ~118k raw
  MuJoCo physics steps/sec (25 substeps per control step), well above idea.md's 10k/sec
  target, measured on Apple Silicon (M-series).
- `learning_curve.csv`: -5.17 (start) → 11.6 (converged by ~150k steps), monotonic.
- Deterministic evaluation (10 episodes, `evaluate.py`): success_rate=1.0,
  collision_rate=0.0, mean_time_to_goal=113 steps, mean_path_efficiency=1.0.
- `policy.onnx` / `policy.int8.onnx` both exported and sanity-checked against the torch
  model (max diff 1.19e-07) and for finite int8-quantized output.

## Known limitations / honest gaps

- **`view`/`teleop`/`play` were verified live, not just unit-tested.** `mujoco.viewer.
  launch_passive` requires macOS processes to run under mujoco's own `mjpython`
  launcher, not plain `python` — `cli.py` transparently re-execs into it
  (`_ensure_mjpython()`) so `uv run robosim view ...` just works. Confirmed by actually
  launching all three: `view` and `teleop` opened a real GUI window and ran the
  physics/render loop at real CPU load (killed manually since they run until the window
  is closed); `play` on the M4 checkpoint ran a full episode to completion with a live
  HUD (reward -0.01 → +11.64, success=True) and exited cleanly on its own. What's *not*
  verified is a human actually operating the mouse camera or WASD keys — that still
  needs a person at the machine.
- **BC training used a scripted stand-in, not a real teleop recording.** Driving WASD
  interactively requires a human at the keyboard, which this session couldn't do itself.
  `train/bc.py`'s pipeline is verified correct end-to-end (see M5 evidence above); swap
  in real `teleop --record` output for genuine human demonstrations.
- **`stair_climb` (T5) is expected to plateau, not succeed** — idea.md calls this out
  explicitly: a wheeled robot has no way to lift over a step riser. That's the intended
  finding, not a defect; it motivates a legged robot in a later phase.
- **ONNX export supports P1/P2/P4 (non-recurrent) policies only.** P3's LSTM hidden
  state isn't threaded through the export wrapper; exporting a recurrent policy needs
  additional state-input/output plumbing not yet built.
- **Curriculum + P4 (frame-stack) combination is untested.** `VecFrameStack` wraps the
  vec env after creation; `env_method` proxying through that wrapper for `set_scene`
  should work (VecEnvWrapper forwards `env_method`) but hasn't been explicitly verified
  in combination.
- **`--algo sac` is wired but not run end-to-end in this session** — PPO was the
  primary path per idea.md's "PPO (primary)" note; SAC shares the same policy_kwargs
  path and should work, but hasn't been smoke-tested here the way PPO was.

## UI polish pass (post-M6)

Isaac-Sim-feel viewer chrome added on top of the milestones above: dark-themed viewport
(gradient skybox + 0.5 m grid floor, `core/theme.py`), on-screen HUD via the passive
viewer's native `set_texts` (time / speed / RTF / physics-rate / pose / velocity, reward
terms in play, drive+REC in teleop), runtime-toggleable debug overlays (lidar rays,
contact points/forces, trail, goal beacon), camera follow, 0.1x–5x runtime speed keys,
and a screenshot key. Verified headlessly (every scene + every overlay rendered
offscreen and inspected; 13 new unit tests) plus live process launches of all three
modes — `play` ran a full episode to success under the new chrome. Human-in-the-loop
items live in `UI_CHECKLIST.md` (2-minute walkthrough).

First live user test caught a real one: driving with WASD wireframed the scene and hid
the static world, because MuJoCo's native viewer binds every letter A–Z to a render
shortcut that fires alongside our key handler and can't be suppressed. Fixed by pinning
all recoverable `mjvOption` flags every frame, moving teleop to arrow-key driving, and
rebinding lidar/goal to `1`/`2` (details in the CLAUDE.md decision log). The same
screenshot exposed non-ASCII glyphs garbling in MuJoCo's overlay font — HUD text is
ASCII-only now, enforced by test.

## Next steps

1. Walk through `UI_CHECKLIST.md` (~2 min) — the only untested surface is literal
   mouse/keyboard input in the live window.
2. Record a real teleop session and re-run `bc-train` on it instead of the scripted demo.
3. Longer PPO runs on `obstacle_nav`/`corridor` to compare P1 vs P2 vs P3/P4 per idea.md's
   task-ladder table.
4. If `stair_climb` is worth pursuing further, that's the signal to start on a legged
   robot (`robots/legged.py`), per idea.md's "out of scope for now" note.
