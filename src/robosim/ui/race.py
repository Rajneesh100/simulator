"""Multi-car race: N policy-driven cars on one grid, first to `laps` wins.

Pass ONE checkpoint and every car runs the same brain (differences come from grid slot,
driver personality, and car-to-car interaction), or SEVERAL checkpoints (repeat
--checkpoint) for a model-vs-model duel: cars are assigned round-robin, each car's roof
tag and leaderboard entry carries its model's run name, and each model gets the lidar
config and lookahead settings it was trained with (read from its own run config). A live
leaderboard ranks cars by race distance; the race ends when the leader completes the lap
target, and final standings print to the terminal.

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

`--firearms rocket` arms every car (`combat/rockets.py`). Cars whose checkpoint trained
with firearms fire from their own policy action; cars driven by weaponless checkpoints
use the scripted intercept gunner, so any checkpoint can fight. Rockets fly real
parabolas, detonate on the first thing they touch, and destroy everything within the
blast radius — shooter included. A destroyed car turns charcoal, tumbles, and is
rebuilt by the marshal after RESPAWN_S. The leaderboard shows K/D.

Follow-cam picks a car: `--follow leader|RED|BLUE|3` (color, name, or grid index);
in the viewer `]` / `[` cycle cars, `0` snaps back to the leader, `c` toggles follow.
`--sound` adds procedural, spatialized audio (`ui/audio.py`): engines pitched by speed,
launches, blasts — attenuated and panned relative to the camera you are looking through.
`--record-video out.mp4` records the race — picture AND sound — offscreen at the sim
clock with the same camera, overlays, and a burned-in leaderboard (`ui/recorder.py`).
"""

from __future__ import annotations

import time
from collections import deque

import mujoco.viewer
import numpy as np

from robosim.combat.gunner import ScriptedGunner
from robosim.combat.perception import Perception
from robosim.combat.rockets import RocketSystem, action_to_aim
from robosim.core import physics
from robosim.core.movers import derive_movers, update_movers
from robosim.core.world import build_multi_world
from robosim.envs.obs import build_obs, combat_block
from robosim.envs.tasks.race import LOOKAHEAD_M, _point_at, _project
from robosim.sensors.lidar import Lidar, body_subtree_ids
from robosim.train.run_config import env_kwargs_from_run, task_params_from_run
from robosim.ui import overlay
from robosim.ui.audio import AudioMixer
from robosim.ui.recorder import VideoRecorder

HUD_PERIOD = 0.1
STALL_SPEED = 0.12  # m/s under which a car counts as stopped
STALL_TRIGGER_S = 1.5  # stopped this long -> start recovery
RECOVERY_S = 0.9  # scripted reverse duration
FLIP_UP_THRESHOLD = 0.3  # world-z of the body's up axis; below = on its side or roof
FLIP_TRIGGER_S = 2.5  # tipped this long -> marshal puts it back on its wheels
OFFTRACK_MARGIN = 0.4  # meters beyond the road edge that counts as escaped
OFFTRACK_TRIGGER_S = 1.5  # escaped this long -> marshal brings it back
WRONGWAY_TRIGGER_S = 2.5  # heading against the track this long -> marshal turns it around
NO_PROGRESS_EPS = 0.5  # must gain this many meters of race distance...
NO_PROGRESS_TRIGGER_S = 6.0  # ...within this window, else marshal (catch-all)
RESPAWN_S = 4.0  # destroyed car sits burning this long before the marshal rebuilds it
CHARCOAL = (0.15, 0.15, 0.15)

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
        base_name, rgb = PALETTE[i % len(PALETTE)]
        # colors repeat past 8 cars — RED2, GREEN2, ... keep names unique
        lap_around = i // len(PALETTE)
        self.name = base_name if lap_around == 0 else f"{base_name}{lap_around + 1}"
        self.rgba = (*rgb, 0.85)
        self.rgb = rgb
        self.armed = False  # this car's checkpoint trained with firearms (5-dim action)
        self.gunner: ScriptedGunner | None = None
        self.eyes: Perception | None = None  # how THIS car sees rivals (los | oracle)
        self.model_idx = 0
        self.tag = ""  # run name of the model driving this car (multi-model duels)
        self.label = self.name  # display name; gains ":tag" in duels
        self.la_time = 0.0
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
        self.off_track_for = 0.0
        self.wrong_way_for = 0.0
        self.progress_anchor = 0.0
        self.progress_timer = 0.0
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


def _tint_car(world, i: int, rgb) -> None:
    for geom_name in world.robots[i].collision_geom_names():
        geom_id = world.model.geom(world.prefixes[i] + geom_name).id
        world.model.geom_rgba[geom_id][:3] = rgb


def _tint_cars(world) -> None:
    for i in range(len(world.robots)):
        _, rgb = PALETTE[i % len(PALETTE)]
        _tint_car(world, i, rgb)


def resolve_follow(spec: str, car_states) -> int | None:
    """'leader' -> None; a color/name/label ('BLUE', 'blue:rally6') or grid index -> car index."""
    key = (spec or "leader").strip().lower()
    if key in ("leader", "lead", "first", ""):
        return None
    if key.isdigit():
        i = int(key)
        if 0 <= i < len(car_states):
            return i
        raise SystemExit(f"--follow {spec}: grid index out of range (0..{len(car_states) - 1})")
    for car in car_states:
        if key in (car.name.lower(), car.label.lower()):
            return car.index
    names = ", ".join(c.label for c in car_states)
    raise SystemExit(f"--follow {spec}: no such car (leader, a grid index, or one of {names})")


def _run_tag(checkpoint: str) -> str:
    """Short label for a checkpoint: its run directory name (rally1, rally3, ...)."""
    from pathlib import Path

    d = Path(checkpoint).resolve().parent
    if d.name == "checkpoints":
        d = d.parent
    return d.name


def run_race(
    checkpoint: str | list[str],
    scene: str = "racetrack:obstacles=0",
    robot: str = "racecar",
    cars: int = 4,
    laps: int = 3,
    speed: float = 1.0,
    deterministic: bool = False,
    seed: int | None = None,
    variation: float = 0.5,
    lookahead: float = LOOKAHEAD_M,
    lookahead_time: float = 0.0,
    firearms: str = "none",
    rocket_speed: float = 40.0,
    combat_sensor: str = "los",
    view_range: float = 0.0,
    follow: str = "leader",
    sound: bool = False,
    record_video: str | None = None,
    record_fps: int = 30,
    record_seconds: float = 0.0,
) -> None:
    from robosim.train.play import _load_model

    checkpoints = [checkpoint] if isinstance(checkpoint, str) else list(checkpoint)
    models = [_load_model(c) for c in checkpoints]  # [(model, is_recurrent), ...]
    tags = [_run_tag(c) for c in checkpoints]
    n_models = len(models)
    rng = np.random.default_rng(seed)
    # stochastic action sampling goes through torch's GLOBAL RNG, and SB3's load()
    # just re-seeded it from the checkpoint's saved seed — so without reseeding here,
    # every launch would sample the identical action stream.
    import torch

    torch.manual_seed(seed if seed is not None
                      else int(np.random.SeedSequence().generate_state(1)[0]))
    # per-model sensing + lookahead, each from its OWN run config (CLI values are the
    # fallback for runs that didn't record them) — a brain must see the obs it trained on
    lidar_kwargs_per_model, la_base, la_time, armed_per_model = [], [], [], []
    sensor_per_model, view_per_model = [], []
    for c in checkpoints:
        env_kwargs = env_kwargs_from_run(c)
        armed_per_model.append(env_kwargs.get("firearms") == "rocket")
        sensor_per_model.append(env_kwargs.get("combat_sensor", combat_sensor))
        view_per_model.append(float(env_kwargs.get("view_range", 0.0) or 0.0))
        lidar_kwargs_per_model.append(dict(
            n_beams=env_kwargs.get("n_lidar_beams", 16),
            fov_deg=env_kwargs.get("lidar_fov_deg", 270.0),
            max_range=env_kwargs.get("lidar_max_range", 5.0),
        ))
        task_params = task_params_from_run(c)
        la_base.append(float(task_params.get("lookahead", lookahead)))
        la_time.append(float(task_params.get("lookahead_time", lookahead_time)))
    if firearms == "none" and any(armed_per_model):
        raise SystemExit("this checkpoint trained with --firearms rocket: "
                         "run race with --firearms rocket")
    if firearms not in ("none", "rocket"):
        raise SystemExit(f"unknown --firearms '{firearms}' (none | rocket)")

    world = build_multi_world(scene, robot, n_cars=cars)
    info = world.scene.task_info()
    if "waypoints" not in info:
        raise SystemExit("robosim race needs a track scene (waypoints in task_info), e.g. racetrack")
    pts3 = np.asarray(info["waypoints"], dtype=np.float64)
    pts = pts3[:, :2]
    zs = pts3[:, 2] if pts3.shape[1] >= 3 else np.zeros(len(pts3))
    nxt = np.roll(pts, -1, axis=0)
    seg_len = np.hypot(*(nxt - pts).T)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]
    total = float(info["track_length"])

    def z_at(s: float) -> float:
        """Road-surface height at arclength s (linear between waypoints)."""
        s = s % total
        i = int(np.searchsorted(cum, s, side="right") - 1)
        j = (i + 1) % len(zs)
        seg = seg_len[i] if seg_len[i] > 1e-9 else 1.0
        frac = (s - cum[i]) / seg
        return float(zs[i] + frac * (zs[j] - zs[i]))

    _tint_cars(world)
    movers = derive_movers(world.model, world.scene)
    car_states = []
    v = max(0.0, variation)
    for i in range(cars):
        m = i % n_models  # round-robin model assignment
        car = _CarState(world, i, lidar_kwargs_per_model[m])
        car.armed = firearms == "rocket" and armed_per_model[m]
        car.prev_action = np.zeros(5 if car.armed else 2, dtype=np.float32)
        car.model_idx = m
        car.tag = tags[m]
        car.label = car.name if n_models == 1 else f"{car.name}:{car.tag}"
        pos, _ = world.car_pose(car.index)
        car.s, _ = _project(pts, cum, pos[:2])
        # driver personality: same brain, different style — this is what breaks the
        # copy-cat procession and produces real overtakes
        car.aggression = 1.0 + v * rng.uniform(-0.12, 0.06)
        car.lookahead = la_base[m] * (1.0 + v * rng.uniform(-0.35, 0.5))
        car.la_time = la_time[m]
        car.jitter = v * rng.uniform(0.02, 0.12)
        car_states.append(car)
        print(f"  {car.label:15s} aggression {car.aggression:.2f}  "
              f"lookahead {car.lookahead:.2f} m  jitter {car.jitter:.3f}", flush=True)
    group_indices = [[c.index for c in car_states if c.model_idx == m] for m in range(n_models)]

    rockets: RocketSystem | None = None
    if firearms == "rocket":
        rockets = RocketSystem(
            world.model, [c.chassis_id for c in car_states], [c.body_ids for c in car_states],
            list(world.qvel_adrs), rocket_speed=rocket_speed, rng=rng,
        )
        # eyes: an armed brain sees the way it trained (recorded in its run config);
        # scripted gunners see the way the race flag says. One Perception per mode.
        eyes_by_mode: dict[str, Perception] = {}

        def eyes_for(mode: str, rng_m: float) -> Perception:
            if mode not in eyes_by_mode:
                vr = view_range if view_range > 0 else (rng_m or max(
                    k["max_range"] for k in lidar_kwargs_per_model))
                eyes_by_mode[mode] = Perception(world.model, rockets, vr, mode)
            return eyes_by_mode[mode]

        for car in car_states:
            m = car.model_idx
            if car.armed:
                car.eyes = eyes_for(sensor_per_model[m], view_per_model[m])
            else:  # weaponless brain: the scripted gunner pulls the trigger
                car.gunner = ScriptedGunner(
                    gun_range=20.0, sigma_deg=2.0 + 6.0 * v * rng.uniform(), rng=rng)
                car.eyes = eyes_for(combat_sensor, 0.0)
        modes = sorted({c.eyes.mode for c in car_states})
        print(f"  firearms: rocket at {rocket_speed:g} m/s -- "
              f"{sum(c.armed for c in car_states)} policy gunners, "
              f"{sum(c.gunner is not None for c in car_states)} scripted; "
              f"sight: {'/'.join(modes)}, "
              f"{max(e.view_range for e in eyes_by_mode.values()):g} m", flush=True)

    follow_idx: int | None = resolve_follow(follow, car_states)
    mixer: AudioMixer | None = None
    if sound:
        mixer = AudioMixer(n_cars=cars, seed=0 if seed is None else seed)
        if not mixer.start():
            mixer = None
    seen_blasts: set[int] = set()
    recorder: VideoRecorder | None = None
    if record_video:
        recorder = VideoRecorder(record_video, world.model, n_cars=cars, fps=record_fps,
                                 seed=0 if seed is None else seed)
        print(f"[race] recording video+audio to {record_video} at {record_fps} fps (sim clock)")
    pending_sounds: list[tuple[str, np.ndarray]] = []  # for the recorder's next frame

    def car_status(car) -> str:
        if rockets is not None and rockets.is_dead(car.index):
            return " DEAD"
        return (" FLIP" if car.flipped_for > 0
                else (" OFF" if car.off_track_for > 0
                      else (" rec" if car.recovering else "")))

    def car_kd(car) -> str:
        if rockets is None:
            return ""
        L = rockets.launchers[car.index]
        return f" K{L.kills}/D{L.deaths}"

    def audio_cars():
        return [
            (world.car_pose(c.index)[0], world.car_velocity(c.index),
             world.car_speed(c.index), float(c.prev_action[0]),
             rockets is None or not rockets.is_dead(c.index))
            for c in car_states
        ]

    def draw_overlays(scn) -> None:
        """Everything the window draws on top of the physics — shared with the recorder."""
        if state.show_trail:  # every car leaves a tracer in its own color
            for car in car_states:
                overlay.draw_trail(scn, car.trail, rgba=car.rgba)
        for rank, car in enumerate(sorted(car_states, key=lambda c: -c.race_distance()), 1):
            pos, _ = world.car_pose(car.index)
            overlay.draw_label(scn, (pos[0], pos[1], pos[2] + 0.35), f"P{rank} {car.label}", car.rgba)
        if state.show_lidar:  # every car's rays, tinted in its own color
            for car in car_states:
                origin, dirs, ranges_m = car.lidar.debug_rays(
                    world.model, world.data, exclude_body_id=car.chassis_id,
                    exclude_body_ids=car.body_ids,
                )
                overlay.draw_lidar(scn, origin, dirs, ranges_m, car.lidar.max_range, rgba_hit=car.rgba)
        if rockets is not None:
            for r in rockets.rockets:
                overlay.draw_rocket(scn, r, car_states[r.owner].rgba)
            for b in rockets.blasts:
                overlay.draw_blast(scn, b, world.data.time - b.time, rockets.blast_radius)
            for car in car_states:
                L = rockets.launchers[car.index]
                if L.cooldown_left > 0 and L.last_dir is not None and L.dead_for < 0:
                    pos, _ = world.car_pose(car.index)
                    overlay.draw_aim(scn, pos + np.array([0, 0, 0.2]), L.last_dir, car.rgba)

    state = overlay.OverlayState(speed=max(0.1, min(5.0, speed)))
    if follow_idx is not None:  # naming a car means "follow it", not just "select it"
        state.follow = True
    timer = overlay.HudTimer()
    steps_per_action = physics.physics_steps_per_control_step()
    control_dt = world.model.opt.timestep * steps_per_action
    lstm_states = [None] * n_models
    episode_starts = [np.ones(len(g), dtype=bool) for g in group_indices]
    winner: _CarState | None = None
    race_time = 0.0

    half_width = float(info["road_half_width"])

    def marshal_respawn(car: _CarState, reason: str = "flip") -> None:
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
        z = z_at(car.s) + world.robots[car.index].spawn_z_offset()
        half = yaw / 2.0
        world.data.qpos[adr : adr + 3] = [p[0], p[1], z]
        world.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        vadr = world.qvel_adrs[car.index]
        world.data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(world.model, world.data)
        car.flipped_for = 0.0
        car.stalled_for = 0.0
        car.recovery_left = 0.0
        car.off_track_for = 0.0
        car.wrong_way_for = 0.0
        car.progress_anchor = car.progress
        car.progress_timer = 0.0
        if rockets is not None:
            rockets.revive(car.index)
            _tint_car(world, car.index, car.rgb)
        print(f"[race] marshal: {car.label} back on track ({reason})", flush=True)

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
            car.off_track_for = 0.0
            car.wrong_way_for = 0.0
            car.progress_anchor = 0.0
            car.progress_timer = 0.0
            car.trail.clear()
            _tint_car(world, car.index, car.rgb)
        if rockets is not None:
            rockets.reset()
            for car in car_states:
                car.eyes.reset()
        lstm_states = [None] * n_models
        episode_starts = [np.ones(len(g), dtype=bool) for g in group_indices]
        winner = None
        race_time = 0.0
        state.trail.clear()

    def key_callback(keycode):
        nonlocal follow_idx
        char = chr(keycode).lower() if 0 < keycode < 256 else ""
        if char == "r":
            reset_race()
        elif char == "]":  # next car (leader -> grid 0 -> 1 -> ...)
            follow_idx = 0 if follow_idx is None else (follow_idx + 1) % cars
            state.follow = True
        elif char == "[":
            follow_idx = cars - 1 if follow_idx is None else (follow_idx - 1) % cars
            state.follow = True
        elif char == "0":
            follow_idx = None
            state.follow = True
        else:
            overlay.handle_common_key(state, keycode, world=None)

    if n_models == 1:
        print(f"robosim race — {cars} cars, same policy, first to {laps} laps wins")
    else:
        print(f"robosim race — MODEL DUEL: {' vs '.join(tags)} across {cars} cars, "
              f"first to {laps} laps wins")
    print("keys: space pause · -/= speed · c follow on/off · ]/[ next/prev car · 0 leader · "
          "x contacts · t trail · r restart · p screenshot · h hud"
          + (" · rockets: ON" if firearms == "rocket" else "")
          + (" · sound: ON" if mixer is not None else "")
          + (f" · recording: {record_video}" if recorder is not None else ""), flush=True)

    try:
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
                    la = car.lookahead
                    if car.la_time > 0:  # speed-scaled, matching this car's training
                        la = max(la, world.car_speed(car.index) * car.la_time)
                    goal = _point_at(pts, cum, total, car.s + la)
                    combat = None
                    if rockets is not None and car.armed:
                        now = world.data.time
                        others = [
                            (p[:2], v_[:2]) for _, p, v_ in
                            car.eyes.rivals(world.model, world.data, car.index, now)
                        ]
                        in_flight = car.eyes.rockets(world.model, world.data, car.index)
                        combat = combat_block(
                            pos[:2], quat, world.car_velocity(car.index)[:2],
                            others, rockets.ready(car.index), in_flight,
                        )
                    obs_rows.append(
                        build_obs(ranges, pos[:2], quat, goal, proprio, car.prev_action,
                                  combat=combat)
                    )

                actions = [None] * cars  # each model drives its own group of cars
                for m, (model, is_recurrent) in enumerate(models):
                    batch = np.stack([obs_rows[i] for i in group_indices[m]])
                    if is_recurrent:
                        acts, lstm_states[m] = model.predict(
                            batch, state=lstm_states[m],
                            episode_start=episode_starts[m],
                            deterministic=deterministic,
                        )
                        episode_starts[m][:] = False
                    else:
                        acts, _ = model.predict(batch, deterministic=deterministic)
                    for i, a in zip(group_indices[m], acts):
                        actions[i] = a

                for car, action in zip(car_states, actions):
                    action = np.asarray(action, dtype=np.float32)
                    drive = action[:2].copy()
                    drive[0] *= car.aggression
                    if car.jitter > 0:
                        drive = drive + rng.normal(0.0, car.jitter, size=2)
                    drive = np.clip(drive, -1.0, 1.0).astype(np.float32)
                    dead = rockets is not None and rockets.is_dead(car.index)
                    if dead:  # a wreck coasts
                        drive = np.zeros(2, dtype=np.float32)
                    else:
                        car.update_stall(world.car_speed(car.index), control_dt, rng)
                        if car.recovering:  # scripted un-wedge: back out with a twist
                            drive = np.array([-0.6, car.recovery_steer], dtype=np.float32)
                    world.set_ctrl(
                        car.index, world.robots[car.index].action_to_ctrl(drive)
                    )
                    if rockets is not None and not dead:
                        if car.armed:
                            if action[2] >= 0.5:
                                az, el = action_to_aim(action[3], action[4])
                                rockets.fire(world.data, car.index, az, el)
                        elif car.gunner is not None:
                            aim = car.gunner.decide(
                                rockets, world.data, car.index, perception=car.eyes,
                                model=world.model, now=world.data.time,
                            )
                            if aim is not None:
                                rockets.fire(world.data, car.index, *aim)
                    if car.armed:
                        car.prev_action = np.concatenate([drive, action[2:5]]).astype(np.float32)
                    else:
                        car.prev_action = drive

                sub_dt = world.model.opt.timestep
                for _ in range(steps_per_action):
                    if movers:
                        update_movers(world.data, movers, world.data.time)
                    mujoco.mj_step(world.model, world.data)
                    if rockets is not None:
                        rockets.step(world.model, world.data, sub_dt)
                    if (recorder is not None and record_seconds > 0
                            and recorder.frames >= record_seconds * recorder.fps):
                        out = recorder.close()  # self-finalizing clip; the race goes on
                        print(f"[race] saved {recorder.frames} frames "
                              f"({recorder.frames / recorder.fps:.1f} s) with audio to {out}",
                              flush=True)
                        recorder = None
                    if recorder is not None and recorder.due(world.data.time):
                        standings_now = sorted(car_states, key=lambda c: -c.race_distance())
                        target_now = (standings_now[0] if follow_idx is None
                                      else car_states[follow_idx])
                        hud = [f"t {race_time:5.1f}s  lap target {laps}  follow "
                               f"{'leader' if follow_idx is None else target_now.label}"] + [
                            f"P{rank} {car.label:14s} L{min(car.laps, laps)}/{laps} "
                            f"{car.progress:6.1f}m{car_kd(car)}{car_status(car)}"
                            for rank, car in enumerate(standings_now, 1)
                        ]
                        recorder.capture(
                            world.data, viewer.cam,
                            lookat=world.car_pose(target_now.index)[0] if state.follow else None,
                            draw_overlays=draw_overlays, hud_lines=hud,
                            audio_cars=audio_cars(), events=pending_sounds,
                        )
                        pending_sounds = []
                race_time += control_dt
                if rockets is not None:
                    for ev in rockets.pop_events():
                        if ev["type"] in ("shot", "dud"):
                            kind = "launch" if ev["type"] == "shot" else "dud"
                            at = world.car_pose(ev["car"])[0]
                            if mixer is not None:
                                mixer.play(kind, at)
                            if recorder is not None:
                                pending_sounds.append((kind, at))
                        if ev["type"] == "kill":
                            s_, v_ = car_states[ev["shooter"]], car_states[ev["victim"]]
                            _tint_car(world, v_.index, CHARCOAL)
                            print(f"[race] KILL: {s_.label} -> {v_.label}", flush=True)
                        elif ev["type"] == "self_kill":
                            c_ = car_states[ev["car"]]
                            _tint_car(world, c_.index, CHARCOAL)
                            print(f"[race] SELF-KILL: {c_.label}", flush=True)

                for car in car_states:
                    pos, quat = world.car_pose(car.index)
                    s_new, lateral = _project(pts, cum, pos[:2])
                    ds = s_new - car.s
                    if ds < -total / 2:
                        ds += total
                    elif ds > total / 2:
                        ds -= total
                    car.s = s_new
                    car.progress += float(np.clip(ds, -1.0, 1.0))
                    car.laps = max(car.laps, int(car.progress // total))
                    if rockets is not None and rockets.is_dead(car.index):
                        car.progress_anchor = car.progress
                        car.progress_timer = 0.0
                        if rockets.launchers[car.index].dead_for >= RESPAWN_S:
                            marshal_respawn(car, "destroyed")
                        continue

                    # marshal watchdogs — physics stays real, but a car that stops
                    # earning race distance gets brought back to the racing surface:
                    # flipped (roof/side), escaped over a wall (hill-crest launches),
                    # or the catch-all: no progress for a while, whatever the cause.
                    if _uprightness(quat) < FLIP_UP_THRESHOLD:
                        car.flipped_for += control_dt
                    else:
                        car.flipped_for = 0.0
                    if lateral > half_width + OFFTRACK_MARGIN:
                        car.off_track_for += control_dt
                    else:
                        car.off_track_for = 0.0
                    # wrong way: heading opposes the track tangent while moving
                    w_, x_, y_, z_ = quat
                    yaw = np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
                    ahead = np.array(_point_at(pts, cum, total, car.s + 1.0))
                    here = np.array(_point_at(pts, cum, total, car.s))
                    tangent = ahead - here
                    heading_dot = float(np.cos(yaw) * tangent[0] + np.sin(yaw) * tangent[1])
                    if heading_dot < -0.2 and world.car_speed(car.index) > 0.3:
                        car.wrong_way_for += control_dt
                    else:
                        car.wrong_way_for = 0.0
                    if car.progress - car.progress_anchor >= NO_PROGRESS_EPS:
                        car.progress_anchor = car.progress
                        car.progress_timer = 0.0
                    else:
                        car.progress_timer += control_dt

                    if car.flipped_for >= FLIP_TRIGGER_S:
                        marshal_respawn(car, "flip")
                    elif car.off_track_for >= OFFTRACK_TRIGGER_S:
                        marshal_respawn(car, "off track")
                    elif car.wrong_way_for >= WRONGWAY_TRIGGER_S:
                        marshal_respawn(car, "wrong way")
                    elif car.progress_timer >= NO_PROGRESS_TRIGGER_S:
                        marshal_respawn(car, "no progress")
                    if car.laps >= laps and winner is None:
                        winner = car

                if winner is not None:
                    standings = sorted(car_states, key=lambda c: -c.race_distance())
                    print(f"\nWINNER: {winner.label} in {race_time:.1f} s", flush=True)
                    for rank, car in enumerate(standings, 1):
                        kd = ""
                        if rockets is not None:
                            L = rockets.launchers[car.index]
                            kd = f"  K{L.kills}/D{L.deaths}"
                        print(f"  P{rank} {car.label:15s} laps {car.laps} "
                              f"({car.progress:.1f} m){kd}", flush=True)
                stepped = True

            timer.update(world.data.time, steps_per_action if stepped else 0)
            standings = sorted(car_states, key=lambda c: -c.race_distance())
            leader = standings[0]
            target = leader if follow_idx is None else car_states[follow_idx]

            if mixer is not None:
                lookat = world.car_pose(target.index)[0] if state.follow else None
                mixer.set_listener(viewer.cam, lookat)
                mixer.update(audio_cars())
            if rockets is not None and (mixer is not None or recorder is not None):
                for b in rockets.blasts:
                    if id(b) not in seen_blasts:
                        seen_blasts.add(id(b))
                        if mixer is not None:
                            mixer.play("blast", b.pos)
                        if recorder is not None:
                            pending_sounds.append(("blast", b.pos.copy()))
                if len(seen_blasts) > 256:
                    seen_blasts = {id(b) for b in rockets.blasts}

            with viewer.lock():
                overlay.apply_camera(
                    viewer.cam, world.model, world.root_body_names[target.index], state
                )
                overlay.apply_vis_flags(viewer.opt, state)
                scn = viewer.user_scn
                scn.ngeom = 0
                if state.show_trail:
                    for car in car_states:
                        overlay.push_trail(car, world.car_pose(car.index)[0])  # uses car.trail
                draw_overlays(scn)

            if frame_start - last_hud > HUD_PERIOD:
                last_hud = frame_start
                if state.show_hud:
                    board_left = "\n".join(
                        f"P{rank}" for rank in range(1, len(standings) + 1)
                    )
                    board_right = "\n".join(
                        (">" if car is target else " ")
                        + f"{car.label} L{min(car.laps, laps)}/{laps} "
                        f"{car.progress:6.1f}m {world.car_speed(car.index):3.1f}m/s"
                        + car_kd(car) + car_status(car)
                        for car in standings
                    )
                    follow_txt = ("leader" if follow_idx is None else target.label) + (
                        "" if state.follow else " (off)")
                    texts = [
                        (
                            overlay.FONT_HUD,
                            overlay.GRID.mjGRID_TOPLEFT,
                            f"race\ntime\nlaps\ncars\nfollow",
                            f"{'FINISHED' if winner else 'running'}\n"
                            f"{race_time:.1f} s\n{laps}\n{cars}\n{follow_txt}",
                        ),
                        (overlay.FONT_HUD, overlay.GRID.mjGRID_TOPRIGHT, board_left, board_right),
                    ]
                    if winner is not None:
                        texts.append(
                            (
                                overlay.FONT_HUD,
                                overlay.GRID.mjGRID_BOTTOM,
                                f"WINNER: {winner.label} ({race_time:.1f} s) -- press r to race again",
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
    except KeyboardInterrupt:
        print("\n[race] interrupted", flush=True)
    finally:  # Ctrl-C or window close: the recording must still be muxed and saved
        if mixer is not None:
            mixer.stop()
        if recorder is not None and not recorder.closed:
            out = recorder.close()
            print(f"[race] saved {recorder.frames} frames ({recorder.frames / recorder.fps:.1f} s) "
                  f"with audio to {out}", flush=True)
