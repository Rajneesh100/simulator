# UI walkthrough — 2 minutes

Everything below launches, runs, and renders correctly per automated checks (offscreen
renders of every scene + overlays, process-level GUI launches, 17 headless UI tests).
What automation can't do is press keys or move the mouse in the live window — that's
this list. Items are ordered so one pass through covers everything.

**Heads-up on stray letter keys:** MuJoCo's native viewer binds every letter A–Z to a
render toggle (`W` wireframe, `S` shadows, …). RoboSim's own bindings avoid the
dangerous ones and auto-repair the rest, but if you mash a random letter and the
rendering changes, press the same key again to undo — that's MuJoCo, not a scene bug.

## 1. `uv run robosim view --scene stairs` (~60s)

- [ ] Viewport is dark (near-black gradient sky), ground shows a subtle 0.5 m grid,
      RGB world-axes triad at the origin, robot/goal/stairs clearly lit. Native side
      panels are hidden (Tab / Shift+Tab bring them back).
- [ ] HUD top-left: mode/time/speed/rtf/physics-rate/pos/yaw/vel, updating live, no
      garbled characters. Key hints bottom-left, toggle states bottom-right,
      scene+robot top-right.
- [ ] Mouse: left-drag orbits, right-drag pans, scroll zooms (MuJoCo native).
- [ ] `space` pauses (HUD shows PAUSED, rtf → 0), `space` resumes, `.` single-steps
      while paused, `r` resets robot to spawn.
- [ ] `-` / `=` walk speed through 0.1x → 5x; HUD `speed` and `rtf` track it.
- [ ] `1` lidar rays (red = hit with endpoint dot, faint gray = miss),
      `x` contact points/forces at the wheels, `t` trajectory trail (drive first),
      `2` goal beacon (green line + sphere) — each toggles on AND off, and the scene
      keeps its textures/static geometry throughout (no vanishing floor).
- [ ] `c` follow-cam locks camera to the robot; `c` again returns to free orbit.
- [ ] `p` prints "screenshot saved: runs/screenshots/…" in the terminal; PNG opens fine.
- [ ] `h` hides all HUD boxes; `h` restores.

## 2. `uv run robosim teleop --scene obstacles --record runs/uidemo` (~40s)

- [ ] Bottom-center hint shows the drive keys; top-right HUD shows `throttle` /
      `turn` / `record: REC n steps` with the counter climbing.
- [ ] **Arrow keys** drive: `↑`/`↓` bump throttle ±0.25, `←`/`→` bump turn,
      `space` stops dead — robot obeys, wheels grip (no ice-skating), and the
      rendering stays intact while driving (this was the old WASD-wireframe bug).
- [ ] Drive near a box with `1` on: rays turn red toward the obstacle.
- [ ] Close the window (don't Ctrl-C): terminal prints "Saved N steps to
      runs/uidemo.npz" and the file exists.

## 3. `uv run robosim play --checkpoint runs/m4_goal_reach_flat/best_model.zip` (~30s)

- [ ] Robot drives itself to the goal beacon; top-right HUD shows episode/reward/
      return/dist/success updating each control step (return ends ≈ +11.6).
- [ ] `space` freezes the policy mid-run, `.` advances one control step, `c` follow-cam
      chases the robot, `-`/`=` change playback speed.
- [ ] Per-episode summary lines print in the terminal; final "Done: n/5 successful".

Known cosmetic quirk: `r` (reset) also toggles the native floor-reflection flag —
subtle, and pressing `r` twice restores it; MuJoCo offers no way to suppress it.

Anything that fails: note the item number — the mapping to code is
1→`ui/view.py`+`core/theme.py`, 2→`ui/teleop.py`, 3→`train/play.py`, shared chrome in
`ui/overlay.py`.
