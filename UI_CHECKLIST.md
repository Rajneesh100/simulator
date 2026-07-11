# UI walkthrough — 2 minutes

Everything below launches, runs, and renders correctly per automated checks (offscreen
renders of every scene + overlays, process-level GUI launches, 13 headless UI tests).
What automation can't do is press keys or move the mouse in the live window — that's
this list. Items are ordered so one pass through covers everything.

## 1. `uv run robosim view --scene stairs` (~60s)

- [ ] Viewport is dark (near-black gradient sky), ground shows a subtle 0.5 m grid,
      RGB world-axes triad at the origin, robot/goal/stairs clearly lit.
- [ ] HUD top-left: mode/time/speed/rtf/physics-rate/pos/vel, updating live.
      Key hints bottom-left, overlay toggle states bottom-right, scene+robot top-right.
- [ ] Mouse: left-drag orbits, right-drag pans, scroll zooms (MuJoCo native).
- [ ] `space` pauses (HUD shows PAUSED, rtf → 0), `space` resumes, `.` single-steps
      while paused, `r` resets robot to spawn.
- [ ] `-` / `=` walk speed through 0.1x → 5x; HUD `speed` and `rtf` track it.
- [ ] `l` lidar rays (red = hit with endpoint dot, faint gray = miss),
      `x` contact points/forces at the wheels, `t` trajectory trail (drive first),
      `g` goal beacon (green line + sphere) — each toggles on AND off.
- [ ] `c` follow-cam locks camera to the robot; `c` again returns to free orbit.
- [ ] `p` prints "screenshot saved: runs/screenshots/…" in the terminal; PNG opens fine.
- [ ] `h` hides all HUD boxes; `h` restores. `Tab` opens MuJoCo's native side panel.

## 2. `uv run robosim teleop --scene obstacles --record runs/uidemo` (~40s)

- [ ] Top-right HUD shows `drive v` / `drive ω` / `record: REC n steps` with the step
      counter climbing.
- [ ] `w`/`s` bump speed ±0.25, `a`/`d` bump turn, `space` stops dead — robot obeys,
      wheels grip (no ice-skating), trail paints the path.
- [ ] Drive near a box with `l` on: rays turn red toward the obstacle.
- [ ] Close the window (don't Ctrl-C): terminal prints "Saved N steps to
      runs/uidemo.npz" and the file exists.

## 3. `uv run robosim play --checkpoint runs/m4_goal_reach_flat/best_model.zip` (~30s)

- [ ] Robot drives itself to the goal beacon; top-right HUD shows episode/reward/
      return/dist/success updating each control step (return ends ≈ +11.6).
- [ ] `space` freezes the policy mid-run, `.` advances one control step, `c` follow-cam
      chases the robot, `-`/`=` change playback speed.
- [ ] Per-episode summary lines print in the terminal; final "Done: n/5 successful".

Anything that fails: note the item number — the mapping to code is
1→`ui/view.py`+`core/theme.py`, 2→`ui/teleop.py`, 3→`train/play.py`, shared chrome in
`ui/overlay.py`.
