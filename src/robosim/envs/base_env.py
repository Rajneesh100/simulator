"""Gymnasium Env wrapping a World (scene + robot) + sensors + a Task.

Domain randomization and curriculum difficulty are first-class here (idea.md M5): DR
perturbs the *compiled* model's friction/mass arrays at reset (fast — no MjSpec rebuild),
and difficulty params are forwarded into the scene constructor, which does require a
rebuild since it changes geometry. Both are seeded off the same per-episode RNG so runs
are reproducible.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robosim.core import physics
from robosim.core.registry import get_task
from robosim.core.world import build_world
from robosim.envs.obs import build_obs, obs_dim
from robosim.sensors.lidar import Lidar


class DomainRandomization:
    """Per-episode randomization ranges. All ranges are (low, high); None disables it."""

    def __init__(
        self,
        friction_scale: tuple[float, float] | None = None,
        mass_scale: tuple[float, float] | None = None,
        lidar_noise_std: tuple[float, float] | None = None,
        lidar_dropout_prob: tuple[float, float] | None = None,
    ):
        self.friction_scale = friction_scale
        self.mass_scale = mass_scale
        self.lidar_noise_std = lidar_noise_std
        self.lidar_dropout_prob = lidar_dropout_prob

    def to_dict(self) -> dict[str, Any]:
        return {
            "friction_scale": self.friction_scale,
            "mass_scale": self.mass_scale,
            "lidar_noise_std": self.lidar_noise_std,
            "lidar_dropout_prob": self.lidar_dropout_prob,
        }


class RoboSimEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        scene: str = "flat",
        robot: str = "diffdrive",
        task: str = "goal_reach",
        n_lidar_beams: int = 16,
        lidar_fov_deg: float = 270.0,
        lidar_max_range: float = 5.0,
        max_episode_steps: int = 500,
        domain_randomization: DomainRandomization | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.scene_arg = scene
        self.robot_arg = robot
        self.task_arg = task
        self.max_episode_steps = max_episode_steps
        self.dr = domain_randomization
        self.render_mode = render_mode

        self.world = build_world(scene, robot)
        self.task = get_task(task)()
        self.lidar = Lidar(
            n_beams=n_lidar_beams, fov_deg=lidar_fov_deg, max_range=lidar_max_range
        )
        self._refresh_derived_state()

        action_dim = self.world.robot.action_dim()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        obs_size = obs_dim(n_lidar_beams, action_dim)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_size,), dtype=np.float32)

        self._steps_per_action = physics.physics_steps_per_control_step()
        self._prev_action = np.zeros(action_dim, dtype=np.float32)
        self._task_state: dict[str, Any] = {}
        self._elapsed_steps = 0
        self._renderer: mujoco.Renderer | None = None

    def _refresh_derived_state(self) -> None:
        """(Re)derive ids/arrays that depend on the compiled model — called on init and
        whenever set_scene() rebuilds self.world with a new scene (curriculum hook)."""
        self._chassis_id = self.world.model.body(self.world.root_body_name).id
        self._chassis_geom_ids = {
            self.world.model.geom(n).id for n in self.world.robot.collision_geom_names()
        }
        self._ground_geom_ids = set()
        for n in self.world.scene.ground_geom_names():
            try:
                self._ground_geom_ids.add(self.world.model.geom(n).id)
            except KeyError:
                pass
        self._base_friction = self.world.model.geom_friction.copy()
        self._base_mass = self.world.model.body_mass.copy()
        self._movers = []
        for mover in self.world.scene.moving_bodies():
            mocap_id = self.world.model.body(mover["name"]).mocapid[0]
            axis_vec = np.array([1.0, 0.0, 0.0] if mover["axis"] == "x" else [0.0, 1.0, 0.0])
            self._movers.append({**mover, "mocap_id": mocap_id, "axis_vec": axis_vec})

    def _update_movers(self) -> None:
        t = self.world.data.time
        for m in self._movers:
            offset = m["amplitude"] * np.sin(2 * np.pi * t / m["period"]) * m["axis_vec"]
            self.world.data.mocap_pos[m["mocap_id"]] = np.array(m["base_pos"]) + offset

    def set_scene(self, scene_arg: str) -> None:
        """Rebuild the world with a new scene (same robot) — curriculum hook (idea.md
        M5): training scripts call this via VecEnv.env_method() to advance difficulty
        between stages. Rebuilds are relatively expensive (recompiles the MjSpec), so
        this is meant to be called every N episodes, not every step."""
        self.scene_arg = scene_arg
        self.world = build_world(scene_arg, self.robot_arg)
        self._refresh_derived_state()

    def _apply_domain_randomization(self, rng: np.random.Generator) -> dict[str, Any]:
        applied: dict[str, Any] = {}
        model = self.world.model
        model.geom_friction[:] = self._base_friction
        model.body_mass[:] = self._base_mass
        if self.dr is None:
            return applied
        if self.dr.friction_scale is not None:
            scale = rng.uniform(*self.dr.friction_scale)
            model.geom_friction[:] = self._base_friction * scale
            applied["friction_scale"] = scale
        if self.dr.mass_scale is not None:
            scale = rng.uniform(*self.dr.mass_scale)
            model.body_mass[:] = self._base_mass * scale
            applied["mass_scale"] = scale
        if self.dr.lidar_noise_std is not None:
            self.lidar.noise_std = rng.uniform(*self.dr.lidar_noise_std)
            applied["lidar_noise_std"] = self.lidar.noise_std
        if self.dr.lidar_dropout_prob is not None:
            self.lidar.dropout_prob = rng.uniform(*self.dr.lidar_dropout_prob)
            applied["lidar_dropout_prob"] = self.lidar.dropout_prob
        return applied

    def _collided(self) -> bool:
        data = self.world.data
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in self._chassis_geom_ids and c.geom2 not in self._ground_geom_ids:
                return True
            if c.geom2 in self._chassis_geom_ids and c.geom1 not in self._ground_geom_ids:
                return True
        return False

    def _sense(self) -> np.ndarray:
        """Lidar sense, cached into task_state so tasks (e.g. obstacle_nav's clearance
        shaping) can read the same ranges reward() needs without re-raycasting."""
        ranges = self.lidar.sense(
            self.world.model, self.world.data, exclude_body_id=self._chassis_id, rng=self.np_random
        )
        self._task_state["last_min_lidar"] = float(ranges.min())
        return ranges

    def _compute_obs(self, ranges: np.ndarray) -> np.ndarray:
        pos, quat = self.world.robot_pose()
        proprio = self.world.robot.read_proprio(self.world.model, self.world.data)
        goal = self._task_state.get("goal", (0.0, 0.0))
        return build_obs(ranges, pos[:2], quat, goal, proprio, self._prev_action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        dr_applied = self._apply_domain_randomization(rng)
        self.world.reset()
        self._task_state = self.task.reset(self.world, rng)
        self._prev_action = np.zeros(self.world.robot.action_dim(), dtype=np.float32)
        self._elapsed_steps = 0
        ranges = self._sense()
        obs = self._compute_obs(ranges)
        info = {"domain_randomization": dr_applied}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        ctrl_map = self.world.robot.action_to_ctrl(action)
        for name, value in ctrl_map.items():
            self.world.data.actuator(name).ctrl[0] = value
        for _ in range(self._steps_per_action):
            if self._movers:
                self._update_movers()
            mujoco.mj_step(self.world.model, self.world.data)

        collided = self._collided()
        ranges = self._sense()
        reward, terminated, task_info = self.task.reward(
            self.world, self._task_state, action, collided
        )
        self._prev_action = action
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.max_episode_steps
        obs = self._compute_obs(ranges)
        info = {"collided": collided, **task_info}
        return obs, reward, terminated, truncated, info

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
