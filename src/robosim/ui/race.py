"""Multi-car race: N identically-policied cars on one grid, first to `laps` wins.

Every car runs the SAME trained checkpoint — same brain, different starting slot — so
differences come purely from grid position and car-to-car interaction (they see each
other on lidar and can collide). A live leaderboard ranks cars by race distance; the race
ends when the leader completes the lap target, and final standings print to the terminal.

Two race-mode behaviors compensate for policies trained solo with crash-ends-episode
(they never learned contact recovery, and a deterministic policy from a fixed grid makes
every race an identical replay):

- stochastic driving (default): actions are sampled from the policy's own Gaussian
  instead of taking the mean, so every race unfolds differently. `--deterministic`
  restores exact-replay mode.
- anti-stall recovery: a car stopped for >1.5 s gets a short scripted reverse-and-turn
  (like an RC game respawn nudge), then the policy takes over again. 'rec' shows on the
  leaderboard while it runs.

Cars are tinted from a fixed palette (RED, GREEN, BLUE, YELLOW, WHITE, ...) so the HUD
names match what you see. Follow-cam ('c') tracks the current leader.
Identical brains produce identical racing lines ("copy-cat laps"), so each car also gets
a seeded DRIVER PERSONALITY — small per-car offsets in aggression (throttle scale),
cornering lookahead, and action jitter (`--variation`, 0 disables). Same checkpoint,
different driving styles: that's what makes overtakes happen. Every car leaves a tracer
in its own color ('t' toggles).
"""

from __future__ import annotations

import time
from collections import deque

import mujoco.viewer
import numpy as np

from robosim.core import physics
from robosim.core.movers import derive_movers, update_movers
from robosim.core.world import build_multi_world
from robosim.envs.obs import build_obs
from robosim.envs.tasks.race import LOOKAHEAD_M, _point_at, _project
from robosim.sensors.lidar import Lidar, body_subtree_ids
from robosim.train.run_config import env_kwargs_from_run
from robosim.ui import overlay

HUD_PERIOD = 0.1
STALL_SPEED = 0.12  # m/s under which a car counts as stopped
STALL_TRIGGER_S = 1.5  # stopped this long -> start recovery
RECOVERY_S = 0.9  # scripted reverse duration
FLIP_UP_THRESHOLD = 0.3  # world-z of the body's up axis; below = on its side or roof
FLIP_TRIGGER_S = 2.5  # tipped this long -> marshal puts it back on its wheels

PALETTE = [
    ("RED", (0.85, 0.10, 0.10)),
    ("GREEN", (0.10, 0.75, 0.25)),
    ("BLUE", (0.15, 0.35, 0.90)),
    ("YELLOW", (0.95, 0.80, 0.10)),
    ("WHITE", (0.92, 0.92, 0.92)),
    ("ORANGE", (0.95, 0.50, 0.10)),
    ("PURPLE", (0.60, 0.20, 0.85)),
    ("CYAN", (0.10, 0.80, 0.80)),
]


class _CarState:
    def __init__(self, world, i: int, lidar_kwargs: dict):
        prefix = world.prefixes[i]
        self.index = i
        self.name, rgb = PALETTE[i % len(PALETTE)]
        self.rgba = (*rgb, 0.85)
        self.lidar = Lidar(site_name=prefix + "lidar_site", **lidar_kwargs)
        self.chassis_id = world.model.body(world.root_body_names[i]).id
        self.body_ids = body_subtree_ids(world.model, self.chassis_id)
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.s = 0.0
        self.progress = 0.0
        self.laps = 0
        self.stalled_for = 0.0
        self.recovery_left = 0.0
        self.recovery_steer = 1.0
        self.trail: deque = deque(maxlen=400)
        self.flipped_for = 0.0
        # driver personality — filled in by run_race from the race seed
        self.aggression = 1.0
        self.lookahead = LOOKAHEAD_M
        self.jitter = 0.0

    def race_distance(self) -> float:
        return self.progress

    def update_stall(self, speed: float, dt: float, rng: np.random.Generator) -> None:
        """Anti-stall: after STALL_TRIGGER_S at ~zero speed, take over with a short
        reverse-and-turn so a wedged car rejoins the race."""
        if self.recovery_left > 0.0:
            self.recovery_left -= dt
            return
        if speed < STALL_SPEED:
            self.stalled_for += dt
            if self.stalled_for >= STALL_TRIGGER_S:
                self.stalled_for = 0.0
                self.recovery_left = RECOVERY_S
                self.recovery_steer = float(rng.choice([-1.0, 1.0]))
        else:
            self.stalled_for = 0.0

    @property
    def recovering(self) -> bool:
        return self.recovery_left > 0.0


def _uprightness(quat_wxyz) -> float:
    """World-z component of the body's up axis: 1 upright, 0 on its side, -1 on roof."""
    w, x, y, z = quat_wxyz
    return float(1.0 - 2.0 * (x * x + y * y))


def _tint_cars(world) -> None:
    for i, robot in enumerate(world.robots):
        _, rgb = PALETTE[i % len(PALETTE)]
        for geom_name in robot.collision_geom_names():
            geom_id = world.model.geom(world.prefixes[i] + geom_name).id
            world.model.geom_rgba[geom_id][:3] = rgb


def run_race(
    checkpoint: str,
    scene: str = "racetrack:obstacles=0",
    robot: str = "racecar",
    cars: int = 4,
    laps: int = 3,
    speed: float = 1.0,
    deterministic: bool = False,
    seed: int | None = None,
    variation: float = 0.5,
) -> None:
    from robosim.train.play import _load_model

    model, is_recurrent = _load_model(checkpoint)
    rng = np.random.default_rng(seed)
    if seed is not None:  # stochastic action sampling goes through torch's RNG
        import torch

        torch.manual_seed(seed)
    env_kwargs = env_kwargs_from_run(checkpoint)
    lidar_kwargs = dict(
        n_beams=env_kwargs.get("n_lidar_beams", 16),
        fov_deg=env_kwargs.get("lidar_fov_deg", 270.0),
        max_range=env_kwargs.get("lidar_max_range", 5.0),
    )

    world = build_multi_world(scene, robot, n_cars=cars)
    info = world.scene.task_info()
    if "waypoints" not in info:
        raise SystemExit("robosim race needs a track scene (waypoints in task_info), e.g. racetrack")
    pts = np.asarray(info["waypoints"], dtype=np.float64)
    nxt = np.roll(pts, -1, axis=0)
    seg_len = np.hypot(*(nxt - pts).T)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
    total = float(info["track_length"])

    _tint_cars(world)
    movers = derive_movers(world.model, world.scene)
    car_states = [_CarState(world, i, lidar_kwargs) for i in range(cars)]
    v = max(0.0, variation)
    for car in car_states:
        pos, _ = world.car_pose(car.index)
        car.s, _ = _project(pts, cum, pos[:2])
        # driver personality: same brain, different style — this is what breaks the
        # copy-cat procession and produces real overtakes
        car.aggression = 1.0 + v * rng.uniform(-0.12, 0.06)
        car.lookahead = LOOKAHEAD_M * (1.0 + v * rng.uniform(-0.35, 0.5))
        car.jitter = v * rng.uniform(0.02, 0.12)
        print(f"  {car.name:7s} aggression {car.aggression:.2f}  "
              f"lookahead {car.lookahead:.2f} m  jitter {car.jitter:.3f}", flush=True)

    state = overlay.OverlayState(speed=max(0.1, min(5.0, speed)))
    timer = overlay.HudTimer()
    steps_per_action = physics.physics_steps_per_control_step()
    control_dt = world.model.opt.timestep * steps_per_action
    lstm_states = None
    episode_starts = np.ones(cars, dtype=bool)
    winner: _CarState | None = None
    race_time = 0.0

    def marshal_respawn(car: _CarState) -> None:
        """Set a flipped car back on its wheels on the centerline at its current s —
        no distance gained, velocities zeroed, small lateral nudge to avoid landing
        exactly on whoever flipped it."""
        p = np.array(_point_at(pts, cum, total, car.s))
        ahead = np.array(_point_at(pts, cum, total, car.s + 0.3))
        d = ahead - p
        yaw = float(np.arctan2(d[1], d[0]))
        normal = np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1e-9)
        p = p + normal * rng.uniform(-0.3, 0.3)
        adr = world.qpos_adrs[car.index]
        z = world.spawn_poses[car.index][0][2]
        half = yaw / 2.0
        world.data.qpos[adr : adr + 3] = [p[0], p[1], z]
        world.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        vadr = world.qvel_adrs[car.index]
        world.data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(world.model, world.data)
        car.flipped_for = 0.0
        car.stalled_for = 0.0
        car.recovery_left = 0.0
        print(f"[race] marshal: {car.name} back on its wheels", flush=True)

    def reset_race():
        nonlocal lstm_states, episode_starts, winner, race_time
        world.reset()
        for car in car_states:
            pos, _ = world.car_pose(car.index)
            car.s, _ = _project(pts, cum, pos[:2])
            car.progress = 0.0
            car.laps = 0
            car.prev_action[:] = 0.0
            car.stalled_for = 0.0
            car.recovery_left = 0.0
            car.flipped_for = 0.0
            car.trail.clear()
        lstm_states = None
        episode_starts = np.ones(cars, dtype=bool)
        winner = None
        race_time = 0.0
        state.trail.clear()

    def key_callback(keycode):
        if 0 < keycode < 256 and chr(keycode).lower() == "r":
            reset_race()
        else:
            overlay.handle_common_key(state, keycode, world=None)

    print(f"robosim race — {cars} cars, same policy, first to {laps} laps wins")
    print("keys: space pause · -/= speed · c follow leader · x contacts · t trail · "
          "r restart · p screenshot · h hud", flush=True)

    with mujoco.viewer.launch_passive(
        world.model,
        world.data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        overlay.init_camera(viewer.cam, world.spawn_poses[0][0])
        last_hud = 0.0

        while viewer.is_running():
            frame_start = time.perf_counter()

            stepped = False
            if winner is None and (not state.paused or state.single_step):
                state.single_step = False

                obs_rows = []
                for car in car_states:
                    pos, quat = world.car_pose(car.index)
                    ranges = car.lidar.sense(
                        world.model, world.data,
                        exclude_body_id=car.chassis_id,
                        exclude_body_ids=car.body_ids,
                    )
                    proprio = world.robots[car.index].read_proprio(
                        world.model, world.data, prefix=world.prefixes[car.index]
                    )
                    goal = _point_at(pts, cum, total, car.s + car.lookahead)
                    obs_rows.append(
                        build_obs(ranges, pos[:2], quat, goal, proprio, car.prev_action)
                    )
                obs_batch = np.stack(obs_rows)

                if is_recurrent:
                    actions, lstm_states = model.predict(
                        obs_batch, state=lstm_states, episode_start=episode_starts,
                        deterministic=deterministic,
                    )
                    episode_starts[:] = False
                else:
                    actions, _ = model.predict(obs_batch, deterministic=deterministic)

                for car, action in zip(car_states, actions):
                    action = np.asarray(action, dtype=np.float32)
                    action[0] *= car.aggression
                    if car.jitter > 0:
                        action = action + rng.normal(0.0, car.jitter, size=2)
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)
                    car.update_stall(world.car_speed(car.index), control_dt, rng)
                    if car.recovering:  # scripted un-wedge: back out with a twist
                        action = np.array([-0.6, car.recovery_steer], dtype=np.float32)
                    world.set_ctrl(
                        car.index, world.robots[car.index].action_to_ctrl(action)
                    )
                    car.prev_action = action

                for _ in range(steps_per_action):
                    if movers:
                        update_movers(world.data, movers, world.data.time)
                    mujoco.mj_step(world.model, world.data)
                race_time += control_dt

                for car in car_states:
                    pos, _ = world.car_pose(car.index)
                    s_new, _ = _project(pts, cum, pos[:2])
                    ds = s_new - car.s
                    if ds < -total / 2:
                        ds += total
                    elif ds > total / 2:
                        ds -= total
                    car.s = s_new
                    car.progress += float(np.clip(ds, -1.0, 1.0))
                    car.laps = max(car.laps, int(car.progress // total))
                    # flipped? (real physics — cars CAN end up on their roof)
                    _, quat = world.car_pose(car.index)
                    if _uprightness(quat) < FLIP_UP_THRESHOLD:
                        car.flipped_for += control_dt
                        if car.flipped_for >= FLIP_TRIGGER_S:
                            marshal_respawn(car)
                    else:
                        car.flipped_for = 0.0
                    if car.laps >= laps and winner is None:
                        winner = car

                if winner is not None:
                    standings = sorted(car_states, key=lambda c: -c.race_distance())
                    print(f"\nWINNER: {winner.name} in {race_time:.1f} s", flush=True)
                    for rank, car in enumerate(standings, 1):
                        print(f"  P{rank} {car.name:7s} laps {car.laps} "
                              f"({car.progress:.1f} m)", flush=True)
                stepped = True

            timer.update(world.data.time, steps_per_action if stepped else 0)
            standings = sorted(car_states, key=lambda c: -c.race_distance())
            leader = standings[0]

            with viewer.lock():
                overlay.apply_camera(
                    viewer.cam, world.model, world.root_body_names[leader.index], state
                )
                overlay.apply_vis_flags(viewer.opt, state)
                scn = viewer.user_scn
                scn.ngeom = 0
                if state.show_trail:  # every car leaves a tracer in its own color
                    for car in car_states:
                        pos, _ = world.car_pose(car.index)
                        overlay.push_trail(car, pos)  # duck-typed: uses car.trail
                        overlay.draw_trail(scn, car.trail, rgba=car.rgba)

            if frame_start - last_hud > HUD_PERIOD:
                last_hud = frame_start
                if state.show_hud:
                    board_left = "\n".join(
                        f"P{rank}" for rank in range(1, len(standings) + 1)
                    )
                    board_right = "\n".join(
                        f"{car.name} L{min(car.laps, laps)}/{laps} "
                        f"{car.progress:6.1f}m {world.car_speed(car.index):3.1f}m/s"
                        + (" FLIP" if car.flipped_for > 0 else (" rec" if car.recovering else ""))
                        for car in standings
                    )
                    texts = [
                        (
                            overlay.FONT_HUD,
                            overlay.GRID.mjGRID_TOPLEFT,
                            f"race\ntime\nlaps\ncars",
                            f"{'FINISHED' if winner else 'running'}\n"
                            f"{race_time:.1f} s\n{laps}\n{cars}",
                        ),
                        (overlay.FONT_HUD, overlay.GRID.mjGRID_TOPRIGHT, board_left, board_right),
                    ]
                    if winner is not None:
                        texts.append(
                            (
                                overlay.FONT_HUD,
                                overlay.GRID.mjGRID_BOTTOM,
                                f"WINNER: {winner.name} ({race_time:.1f} s) -- press r to race again",
                                None,
                            )
                        )
                    texts.append(overlay.overlay_toggle_line(state))
                    overlay.update_hud(viewer, texts)
                else:
                    overlay.update_hud(viewer, [])

            viewer.sync()

            if state.screenshot_requested:
                state.screenshot_requested = False
                try:
                    path = overlay.save_screenshot(world.model, world.data, viewer.cam)
                    print(f"\n[race] screenshot saved: {path}")
                except Exception as e:
                    print(f"\n[race] screenshot failed: {e}")

            target_dt = control_dt / state.speed if stepped else 1.0 / 60.0
            elapsed = time.perf_counter() - frame_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
