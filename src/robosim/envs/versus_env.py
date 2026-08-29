"""Competitive racing env: the learner races OPPONENT cars driven by a frozen checkpoint.

League training, not just lap training: the reward pays for racecraft —
- progress along the track (as in the race task), minus a time penalty;
- OVERTAKES: net position changes against the opponents, signed (losing places hurts);
- TAKEDOWNS: an opponent wrecking (flip/pinned/off-track) within a short window after
  contact with the learner earns a bonus — bumping rivals into walls is a strategy;
- WIN/LOSE: first to `laps` ends the episode with a big bonus; an opponent finishing
  first ends it with a penalty.

Car-to-car contact is free (that's racing); hitting a WALL or obstacle, or flipping,
still terminates the learner with a crash penalty — aggression has to be aimed.

Opponents run the frozen policy stochastically with small personality offsets, plus the
race-mode survival aids (anti-stall reverse, marshal respawn) so the learner always has
live rivals to fight. The learner gets NO aids — it must learn to keep itself racing.

Observation/action spaces are identical to the solo race task, so a versus-trained
checkpoint plugs directly into `robosim play` / `robosim race` (including duels against
the very champion it trained against). Lookahead settings come from the same task string
(`--task race:lookahead=2.5,lookahead_time=0.6`), so run configs record them as usual.

With `firearms="rocket"` every car carries a launcher (`combat/rockets.py`). The
action grows to [throttle, steer, fire, az, el] and the observation gains the combat
block from `envs/obs.py`. Reward adds KILL_BONUS per rival destroyed by the learner's
rocket, SHOT_COST per launch, and DEATH_PENALTY + termination when the learner is
destroyed — by any rocket, its own blast included. Opponents fire through a scripted
intercept gunner (the frozen driver never held a launcher), so incoming fire exists to
dodge. Destroyed opponents respawn like wrecked ones. The frozen opponents keep seeing
the obs layout THEY trained with (`_opp_obs`: no combat block, 2-wide prev_action).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robosim.core import physics
from robosim.core.movers import derive_movers, update_movers
from robosim.combat.gunner import ScriptedGunner
from robosim.combat.perception import Perception
from robosim.combat.rockets import RocketSystem, action_to_aim
from robosim.core.world import _parse_scene_arg, build_multi_world
from robosim.envs.obs import build_obs, combat_block, obs_dim
from robosim.envs.tasks.race import _point_at, _project
from robosim.sensors.lidar import Lidar, body_subtree_ids

TIME_PENALTY = 0.02
WALL_CRASH_PENALTY = 8.0  # learner hits wall/obstacle or flips -> episode over
OVERTAKE_BONUS = 2.0  # per net position gained (negative when overtaken)
TAKEDOWN_BONUS = 8.0  # opponent wrecks within TAKEDOWN_WINDOW_S of touching the learner
TAKEDOWN_WINDOW_S = 1.5
WIN_BONUS = 40.0
LOSE_PENALTY = 15.0
MAX_DS_PER_STEP = 1.0
KILL_BONUS = 15.0  # opponent destroyed by a rocket the learner fired
DEATH_PENALTY = 20.0  # learner destroyed (any rocket, own blast included) -> episode over
SHOT_COST = 0.3  # per launch, so it aims instead of spraying
OPP_GUN_RANGE = 20.0
OPP_AIM_SIGMA_DEG = 2.0

OPP_STALL_SPEED = 0.12
OPP_STALL_TRIGGER_S = 1.5
OPP_RECOVERY_S = 0.9
OPP_WRECK_FLIP_S = 1.0  # flipped this long counts as wrecked (takedown check)
OPP_RESPAWN_S = 3.0  # wrecked opponents rejoin after this long
OPP_NO_PROGRESS_S = 6.0


def _load_frozen(checkpoint: str):
    """Load an SB3 checkpoint without importing any viewer modules (train workers)."""
    try:
        from stable_baselines3 import PPO

        return PPO.load(checkpoint, device="cpu"), False
    except Exception:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(checkpoint, device="cpu"), True


class _Car:
    def __init__(self, world, i: int, lidar_kwargs: dict):
        self.index = i
        self.lidar = Lidar(site_name=world.prefixes[i] + "lidar_site", **lidar_kwargs)
        self.chassis_id = world.model.body(world.root_body_names[i]).id
        self.body_ids = body_subtree_ids(world.model, self.chassis_id)
        self.chassis_geom_ids = {
            world.model.geom(world.prefixes[i] + n).id
            for n in world.robots[i].collision_geom_names()
        }
        self.geom_ids = {
            g for g in range(world.model.ngeom)
            if world.model.geom_bodyid[g] in self.body_ids
        }
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.s = 0.0
        self.progress = 0.0
        self.laps = 0
        # opponent aids / wreck bookkeeping
        self.stalled_for = 0.0
        self.recovery_left = 0.0
        self.recovery_steer = 1.0
        self.flipped_for = 0.0
        self.no_progress_for = 0.0
        self.progress_anchor = 0.0
        self.wrecked_for = 0.0
        self.last_learner_contact = -1e9
        self.aggression = 1.0
        self.jitter = 0.0


def _uprightness(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(1.0 - 2.0 * (x * x + y * y))


class VersusRaceEnv(gym.Env):
    """Gymnasium env: car 0 is the learner; cars 1..N run the frozen opponent policy."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene: str = "racetrack:obstacles=0",
        robot: str = "racecar",
        task: str = "race",
        opponent_checkpoint: str = "",
        opponents: int = 2,
        n_lidar_beams: int = 16,
        lidar_fov_deg: float = 270.0,
        lidar_max_range: float = 5.0,
        max_episode_steps: int = 1500,
        domain_randomization=None,  # accepted for train.py symmetry; not applied
        firearms: str = "none",
        rocket_speed: float = 40.0,
        combat_sensor: str = "los",
        view_range: float = 0.0,  # 0 -> lidar_max_range
        render_mode: str | None = None,
    ):
        super().__init__()
        if not opponent_checkpoint:
            raise ValueError("versus training needs --versus <checkpoint> (frozen opponents)")
        self.scene_arg = scene
        self.robot_arg = robot
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.n_opponents = opponents

        task_params = _parse_scene_arg(task)[1]
        self.laps = int(task_params.get("laps", 2) or 2)
        self.lookahead = float(task_params.get("lookahead", 2.5))
        self.lookahead_time = float(task_params.get("lookahead_time", 0.0))

        self.world = build_multi_world(scene, robot, n_cars=opponents + 1)
        info = self.world.scene.task_info()
        if "waypoints" not in info:
            raise ValueError("versus training needs a track scene (waypoints in task_info)")
        pts3 = np.asarray(info["waypoints"], dtype=np.float64)
        self._pts = pts3[:, :2]
        seg = np.hypot(*(np.roll(self._pts, -1, axis=0) - self._pts).T)
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
        self._seg_len = seg
        self._zs = pts3[:, 2] if pts3.shape[1] >= 3 else np.zeros(len(pts3))
        self._total = float(info["track_length"])
        self._half_width = float(info["road_half_width"])

        self.opp_model, self.opp_recurrent = _load_frozen(opponent_checkpoint)
        self._opp_lstm = None

        lidar_kwargs = dict(
            n_beams=n_lidar_beams, fov_deg=lidar_fov_deg, max_range=lidar_max_range
        )
        self.cars = [_Car(self.world, i, lidar_kwargs) for i in range(opponents + 1)]
        self.learner = self.cars[0]
        self._exempt_geom_ids = set()
        for n in self.world.scene.collision_exempt_geom_names():
            try:
                self._exempt_geom_ids.add(self.world.model.geom(n).id)
            except KeyError:
                pass
        self._all_car_geom_ids = set().union(*(c.geom_ids for c in self.cars))

        self.firearms = firearms
        self.rocket_speed = float(rocket_speed)
        drive_dim = self.world.robots[0].action_dim()
        self.drive_dim = drive_dim
        action_dim = drive_dim + 3 if firearms == "rocket" else drive_dim
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        obs_size = obs_dim(n_lidar_beams, action_dim, combat=(firearms == "rocket"))
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_size,), dtype=np.float32)
        self.learner.prev_action = np.zeros(action_dim, dtype=np.float32)

        self.rockets: RocketSystem | None = None
        self.perception: Perception | None = None
        self.combat_sensor = combat_sensor
        self.view_range = float(view_range) if view_range > 0 else float(lidar_max_range)
        self._gunners: list[ScriptedGunner | None] = [None] * len(self.cars)
        if firearms == "rocket":
            self.rockets = RocketSystem(
                self.world.model,
                [c.chassis_id for c in self.cars],
                [c.body_ids for c in self.cars],
                list(self.world.qvel_adrs),
                rocket_speed=self.rocket_speed,
            )
            # everyone — learner and scripted gunners — looks through the same eyes
            self.perception = Perception(
                self.world.model, self.rockets, self.view_range, combat_sensor
            )
            for c in self.cars[1:]:
                self._gunners[c.index] = ScriptedGunner(OPP_GUN_RANGE, OPP_AIM_SIGMA_DEG)
        elif firearms != "none":
            raise ValueError(f"unknown firearms '{firearms}' (none | rocket)")

        self._steps_per_action = physics.physics_steps_per_control_step()
        self._control_dt = self.world.model.opt.timestep * self._steps_per_action
        self._movers = derive_movers(self.world.model, self.world.scene)
        self._elapsed = 0
        self._rank_behind = 0  # opponents currently behind the learner
        self._renderer = None

    # ------------------------------------------------------------------ helpers

    def _z_at(self, s: float) -> float:
        s = s % self._total
        i = int(np.searchsorted(self._cum, s, side="right") - 1)
        j = (i + 1) % len(self._zs)
        seg = self._seg_len[i] if self._seg_len[i] > 1e-9 else 1.0
        frac = (s - self._cum[i]) / seg
        return float(self._zs[i] + frac * (self._zs[j] - self._zs[i]))

    def _car_speed(self, car: _Car) -> float:
        return self.world.car_speed(car.index)

    def _obs_for(self, car: _Car) -> np.ndarray:
        pos, quat = self.world.car_pose(car.index)
        ranges = car.lidar.sense(
            self.world.model, self.world.data,
            exclude_body_id=car.chassis_id, exclude_body_ids=car.body_ids,
            rng=self.np_random,
        )
        proprio = self.world.robots[car.index].read_proprio(
            self.world.model, self.world.data, prefix=self.world.prefixes[car.index]
        )
        la = self.lookahead
        if self.lookahead_time > 0:
            la = max(la, self._car_speed(car) * self.lookahead_time)
        goal = _point_at(self._pts, self._cum, self._total, car.s + la)
        combat = None
        if self.rockets is not None:
            me_vel = self.world.car_velocity(car.index)[:2]
            now = self.world.data.time
            others = [
                (p[:2], v[:2])
                for _, p, v in self.perception.rivals(self.world.model, self.world.data, car.index, now)
            ]
            rockets = self.perception.rockets(self.world.model, self.world.data, car.index)
            combat = combat_block(
                pos[:2], quat, me_vel, others, self.rockets.ready(car.index), rockets
            )
        return build_obs(ranges, pos[:2], quat, goal, proprio, car.prev_action, combat=combat)

    def _opp_obs(self, car: _Car) -> np.ndarray:
        """Observation in the FROZEN opponent's own layout (no combat block, drive-wide
        prev_action) — a brain must see the obs it trained on."""
        pos, quat = self.world.car_pose(car.index)
        ranges = car.lidar.sense(
            self.world.model, self.world.data,
            exclude_body_id=car.chassis_id, exclude_body_ids=car.body_ids,
            rng=self.np_random,
        )
        proprio = self.world.robots[car.index].read_proprio(
            self.world.model, self.world.data, prefix=self.world.prefixes[car.index]
        )
        la = self.lookahead
        if self.lookahead_time > 0:
            la = max(la, self._car_speed(car) * self.lookahead_time)
        goal = _point_at(self._pts, self._cum, self._total, car.s + la)
        return build_obs(ranges, pos[:2], quat, goal, proprio, car.prev_action[: self.drive_dim])

    def _respawn(self, car: _Car) -> None:
        p = np.array(_point_at(self._pts, self._cum, self._total, car.s))
        ahead = np.array(_point_at(self._pts, self._cum, self._total, car.s + 0.3))
        d = ahead - p
        yaw = float(np.arctan2(d[1], d[0]))
        adr = self.world.qpos_adrs[car.index]
        z = self._z_at(car.s) + self.world.robots[car.index].spawn_z_offset()
        half = yaw / 2.0
        self.world.data.qpos[adr : adr + 3] = [p[0], p[1], z]
        self.world.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        vadr = self.world.qvel_adrs[car.index]
        self.world.data.qvel[vadr : vadr + 6] = 0.0
        mujoco.mj_forward(self.world.model, self.world.data)
        car.flipped_for = 0.0
        car.stalled_for = 0.0
        car.recovery_left = 0.0
        car.no_progress_for = 0.0
        car.progress_anchor = car.progress
        car.wrecked_for = 0.0
        if self.rockets is not None:
            self.rockets.revive(car.index)

    def _update_progress(self, car: _Car) -> float:
        pos, _ = self.world.car_pose(car.index)
        s_new, lateral = _project(self._pts, self._cum, pos[:2])
        ds = s_new - car.s
        if ds < -self._total / 2:
            ds += self._total
        elif ds > self._total / 2:
            ds -= self._total
        ds = float(np.clip(ds, -MAX_DS_PER_STEP, MAX_DS_PER_STEP))
        car.s = s_new
        car.progress += ds
        car.laps = max(car.laps, int(car.progress // self._total))
        return lateral

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.world.reset()
        if self.rockets is not None:
            self.rockets.reset()
            self.perception.reset()
            self.rockets.rng = rng
            for g in self._gunners:
                if g is not None:
                    g.rng = rng
        # shuffle grid slots: the learner must practice from the front AND the back
        perm = rng.permutation(len(self.cars))
        for car, slot in zip(self.cars, perm):
            (x, y, z), yaw = self.world.spawn_poses[int(slot)]
            adr = self.world.qpos_adrs[car.index]
            half = yaw / 2.0
            self.world.data.qpos[adr : adr + 3] = [x, y, z]
            self.world.data.qpos[adr + 3 : adr + 7] = [np.cos(half), 0, 0, np.sin(half)]
        mujoco.mj_forward(self.world.model, self.world.data)

        for car in self.cars:
            pos, _ = self.world.car_pose(car.index)
            car.s, _ = _project(self._pts, self._cum, pos[:2])
            car.progress = 0.0
            car.laps = 0
            car.prev_action[:] = 0.0
            car.stalled_for = car.recovery_left = car.flipped_for = 0.0
            car.no_progress_for = car.wrecked_for = 0.0
            car.progress_anchor = 0.0
            car.last_learner_contact = -1e9
            car.aggression = 1.0 + rng.uniform(-0.06, 0.03)
            car.jitter = rng.uniform(0.01, 0.06)
        self._opp_lstm = None
        self._elapsed = 0
        self._rank_behind = sum(
            1 for c in self.cars[1:] if c.progress <= self.learner.progress
        )
        return self._obs_for(self.learner), {}

    def _opponent_actions(self) -> list[np.ndarray]:
        rows = [self._opp_obs(c) for c in self.cars[1:]]
        batch = np.stack(rows)
        if self.opp_recurrent:
            starts = np.array([self._opp_lstm is None] * len(rows))
            acts, self._opp_lstm = self.opp_model.predict(
                batch, state=self._opp_lstm, episode_start=starts, deterministic=False
            )
        else:
            acts, _ = self.opp_model.predict(batch, deterministic=False)
        out = []
        for car, a in zip(self.cars[1:], acts):
            a = np.asarray(a, dtype=np.float32)[: self.drive_dim].copy()
            a[0] *= car.aggression
            a = np.clip(a + self.np_random.normal(0, car.jitter, 2), -1, 1).astype(np.float32)
            # anti-stall reverse so opponents stay racy
            if car.recovery_left > 0:
                car.recovery_left -= self._control_dt
                a = np.array([-0.6, car.recovery_steer], dtype=np.float32)
            elif self._car_speed(car) < OPP_STALL_SPEED:
                car.stalled_for += self._control_dt
                if car.stalled_for >= OPP_STALL_TRIGGER_S:
                    car.stalled_for = 0.0
                    car.recovery_left = OPP_RECOVERY_S
                    car.recovery_steer = float(self.np_random.choice([-1.0, 1.0]))
            else:
                car.stalled_for = 0.0
            out.append(a)
        return out

    def _learner_contacts(self) -> None:
        """Record the sim time of the latest learner-vs-each-opponent touch."""
        data = self.world.data
        t = data.time
        for k in range(data.ncon):
            g1, g2 = data.contact[k].geom1, data.contact[k].geom2
            for opp in self.cars[1:]:
                if (g1 in self.learner.geom_ids and g2 in opp.geom_ids) or (
                    g2 in self.learner.geom_ids and g1 in opp.geom_ids
                ):
                    opp.last_learner_contact = t

    def _learner_wall_crash(self) -> bool:
        data = self.world.data
        for k in range(data.ncon):
            g1, g2 = data.contact[k].geom1, data.contact[k].geom2
            for mine, other in ((g1, g2), (g2, g1)):
                if mine in self.learner.chassis_geom_ids:
                    if other not in self._exempt_geom_ids and other not in self._all_car_geom_ids:
                        return True
        return False

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        opp_actions = self._opponent_actions()
        self.world.set_ctrl(0, self.world.robots[0].action_to_ctrl(action[: self.drive_dim]))
        self.learner.prev_action = action
        for car, a in zip(self.cars[1:], opp_actions):
            self.world.set_ctrl(car.index, self.world.robots[car.index].action_to_ctrl(a))
            car.prev_action = a

        shots = 0
        if self.rockets is not None:
            if action[self.drive_dim] >= 0.5 and self.rockets.ready(0):
                az, el = action_to_aim(action[self.drive_dim + 1], action[self.drive_dim + 2])
                if self.rockets.fire(self.world.data, 0, az, el) is not None:
                    shots += 1
            for car in self.cars[1:]:
                g = self._gunners[car.index]
                if g is not None and not self.rockets.is_dead(car.index):
                    aim = g.decide(
                        self.rockets, self.world.data, car.index,
                        perception=self.perception, model=self.world.model,
                        now=self.world.data.time,
                    )
                    if aim is not None:
                        self.rockets.fire(self.world.data, car.index, *aim)
            # destroyed cars coast: no drive
            for car in self.cars:
                if self.rockets.is_dead(car.index):
                    self.world.set_ctrl(car.index, self.world.robots[car.index].action_to_ctrl(
                        np.zeros(self.drive_dim, dtype=np.float32)))

        sub_dt = self.world.model.opt.timestep
        for _ in range(self._steps_per_action):
            if self._movers:
                update_movers(self.world.data, self._movers, self.world.data.time)
            mujoco.mj_step(self.world.model, self.world.data)
            if self.rockets is not None:
                self.rockets.step(self.world.model, self.world.data, sub_dt)
            self._learner_contacts()

        # learner progress + state
        self._update_progress(self.learner)
        reward = 0.0
        terminated = False
        info: dict[str, Any] = {}
        prev_progress = self.learner.progress  # after update; delta computed via ds below

        # recompute ds from the update (progress delta this step)
        # (kept simple: _update_progress already added ds; reconstruct via anchor)
        ds = self.learner.progress - self.learner.progress_anchor
        self.learner.progress_anchor = self.learner.progress
        reward += ds - TIME_PENALTY

        # opponents: progress, wreck detection, takedowns, respawns
        takedowns = 0
        t_now = self.world.data.time
        for opp in self.cars[1:]:
            lateral = self._update_progress(opp)
            _, quat = self.world.car_pose(opp.index)
            flipped = _uprightness(quat) < 0.3
            opp.flipped_for = opp.flipped_for + self._control_dt if flipped else 0.0
            if opp.progress - opp.progress_anchor >= 0.3:
                opp.progress_anchor = opp.progress
                opp.no_progress_for = 0.0
            else:
                opp.no_progress_for += self._control_dt
            off_track = lateral > self._half_width + 0.4

            destroyed = self.rockets is not None and self.rockets.is_dead(opp.index)
            newly_wrecked = (
                opp.wrecked_for == 0.0
                and (destroyed or opp.flipped_for >= OPP_WRECK_FLIP_S or off_track
                     or opp.no_progress_for >= OPP_NO_PROGRESS_S)
            )
            if newly_wrecked:
                opp.wrecked_for = 1e-9
                if t_now - opp.last_learner_contact <= TAKEDOWN_WINDOW_S:
                    takedowns += 1  # you put them there
            if opp.wrecked_for > 0.0:
                opp.wrecked_for += self._control_dt
                if opp.wrecked_for >= OPP_RESPAWN_S:
                    self._respawn(opp)

        reward += TAKEDOWN_BONUS * takedowns
        if takedowns:
            info["takedowns"] = takedowns

        if self.rockets is not None:
            kills = 0
            killed = False
            for ev in self.rockets.pop_events():
                if ev["type"] == "kill" and ev["shooter"] == 0:
                    kills += 1
                elif ev["type"] == "kill" and ev["victim"] == 0:
                    killed = True
                elif ev["type"] == "self_kill" and ev["car"] == 0:
                    killed = True
            reward += KILL_BONUS * kills - SHOT_COST * shots
            if kills:
                info["kills"] = kills
            if shots:
                info["shots"] = shots
            if killed or self.rockets.is_dead(0):
                reward -= DEATH_PENALTY
                terminated = True
                info["killed"] = True

        # overtakes: net position changes, signed
        behind_now = sum(1 for c in self.cars[1:] if c.progress <= self.learner.progress)
        reward += OVERTAKE_BONUS * (behind_now - self._rank_behind)
        self._rank_behind = behind_now

        # learner crash/flip: aggression must be aimed — walls end your race
        _, lquat = self.world.car_pose(self.learner.index)
        if not terminated and (self._learner_wall_crash() or _uprightness(lquat) < 0.3):
            reward -= WALL_CRASH_PENALTY
            terminated = True
            info["crashed"] = True

        # win / lose
        if self.learner.laps >= self.laps:
            reward += WIN_BONUS
            terminated = True
            info["success"] = True
            info["won"] = True
        elif any(c.laps >= self.laps for c in self.cars[1:]):
            reward -= LOSE_PENALTY
            terminated = True
            info["won"] = False

        self._elapsed += 1
        truncated = self._elapsed >= self.max_episode_steps
        info["rank"] = 1 + (self.n_opponents - behind_now)
        info["progress_m"] = self.learner.progress
        obs = self._obs_for(self.learner)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.world.model, height=480, width=640)
        self._renderer.update_scene(self.world.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
