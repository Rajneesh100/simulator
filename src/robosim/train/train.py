"""PPO/SAC training: headless, vectorized envs, tensorboard + reproducible run config.

`robosim train --task goal_reach --scene flat --steps 2_000_000`
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from robosim.envs.base_env import DomainRandomization, RoboSimEnv
from robosim.train.policies import policy_kwargs_for, sb3_policy_name_for

FRAME_STACK_K = 4


class LearningCurveCallback(BaseCallback):
    """Appends (timesteps, ep_rew_mean) to runs/<run>/learning_curve.csv every rollout."""

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = csv_path
        self.rows: list[tuple[int, float]] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # Read ep_info_buffer directly rather than the SB3 logger: the logger only
        # records "rollout/ep_rew_mean" *after* on_rollout_end fires, so it's always
        # one iteration stale (and empty on the very first rollout).
        buf = self.model.ep_info_buffer
        if buf:
            ep_rew_mean = float(np.mean([info["r"] for info in buf]))
            self.rows.append((self.num_timesteps, ep_rew_mean))
            with open(self.csv_path, "w") as f:
                f.write("timesteps,ep_rew_mean\n")
                for t, r in self.rows:
                    f.write(f"{t},{r}\n")


def parse_curriculum(spec: str) -> list[tuple[int, str]]:
    """Parse '0:flat;50000:obstacles:count=3;150000:obstacles:count=10' into
    [(0, 'flat'), (50000, 'obstacles:count=3'), (150000, 'obstacles:count=10')]."""
    stages = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        timestep_str, _, scene_arg = chunk.partition(":")
        stages.append((int(timestep_str), scene_arg))
    return sorted(stages, key=lambda s: s[0])


class CurriculumCallback(BaseCallback):
    """Advances scene difficulty on both training and eval envs at fixed timesteps
    (idea.md M5: "training script supports scheduled progression"). Rebuilds the MjSpec
    per sub-env via VecEnv.env_method, so stages should be spaced widely (thousands of
    steps apart), not every rollout.
    """

    def __init__(self, stages: list[tuple[int, str]], eval_env=None):
        super().__init__()
        self.stages = stages
        self.eval_env = eval_env
        self._next_stage_idx = 0

    def _on_step(self) -> bool:
        while (
            self._next_stage_idx < len(self.stages)
            and self.num_timesteps >= self.stages[self._next_stage_idx][0]
        ):
            _, scene_arg = self.stages[self._next_stage_idx]
            self.training_env.env_method("set_scene", scene_arg)
            if self.eval_env is not None:
                self.eval_env.env_method("set_scene", scene_arg)
            print(f"[curriculum] t={self.num_timesteps}: scene -> '{scene_arg}'")
            self._next_stage_idx += 1
        return True


def run_train(
    task: str = "goal_reach",
    scene: str = "flat",
    robot: str = "diffdrive",
    steps: int = 200_000,
    n_envs: int = 8,
    algo: str = "ppo",
    policy: str = "p1",
    run_name: str | None = None,
    seed: int = 0,
    bc_init: str | None = None,
    domain_randomization: DomainRandomization | None = None,
    curriculum: str | None = None,
    n_lidar_beams: int = 16,
    lidar_fov_deg: float = 270.0,
    lidar_max_range: float = 5.0,
    max_episode_steps: int = 500,
    versus_checkpoint: str | None = None,
    versus_opponents: int = 2,
    firearms: str = "none",
    rocket_speed: float = 40.0,
    combat_sensor: str = "los",
    view_range: float = 0.0,
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = policy.lower()
    if run_name is None:
        run_name = f"{task}_{scene}_{algo}_{policy}_{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        scene=scene,
        robot=robot,
        task=task,
        domain_randomization=domain_randomization,
        n_lidar_beams=n_lidar_beams,
        lidar_fov_deg=lidar_fov_deg,
        lidar_max_range=lidar_max_range,
        max_episode_steps=max_episode_steps,
        firearms=firearms,
        rocket_speed=rocket_speed,
        combat_sensor=combat_sensor,
        view_range=view_range,
    )
    env_cls = RoboSimEnv
    if firearms != "none" and not versus_checkpoint:
        raise ValueError("--firearms needs opponents to shoot: add --versus <checkpoint>")
    if versus_checkpoint:
        from robosim.envs.versus_env import VersusRaceEnv

        if curriculum:
            raise ValueError("--curriculum is not supported with --versus training")
        env_cls = VersusRaceEnv
        env_kwargs["opponent_checkpoint"] = versus_checkpoint
        env_kwargs["opponents"] = versus_opponents
        print(f"[train] VERSUS mode: learner races {versus_opponents} frozen opponents "
              f"from {versus_checkpoint}")
        if firearms != "none":
            print(f"[train] FIREARMS: {firearms} at {rocket_speed} m/s — "
                  f"opponents use the scripted gunner; sight = {combat_sensor}"
                  f"{f' {view_range} m' if view_range > 0 else ' (lidar range)'}")
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    try:
        vec_env = make_vec_env(
            env_cls, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=vec_cls
        )
    except Exception as e:  # SubprocVecEnv can fail to spawn in some sandboxes
        print(f"[train] {vec_cls.__name__} failed ({e}); falling back to DummyVecEnv")
        vec_env = make_vec_env(
            env_cls, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv
        )

    eval_env = make_vec_env(
        env_cls, n_envs=1, seed=seed + 10_000, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv
    )

    if policy == "p4":
        vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK_K)
        eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK_K)

    policy_kwargs = policy_kwargs_for(policy, n_lidar_beams=n_lidar_beams)
    sb3_policy_name = sb3_policy_name_for(policy)

    if policy == "p3":
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO(
            sb3_policy_name,
            vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            tensorboard_log=str(run_dir / "tb"),
        )
    elif algo == "sac":
        from stable_baselines3 import SAC

        model = SAC(
            sb3_policy_name,
            vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            tensorboard_log=str(run_dir / "tb"),
        )
    else:
        from stable_baselines3 import PPO

        model = PPO(
            sb3_policy_name,
            vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
            n_steps=max(256, 2048 // n_envs),
            tensorboard_log=str(run_dir / "tb"),
        )

    if bc_init:
        try:
            state_dict = torch.load(bc_init, map_location="cpu")
            target = model.policy.state_dict()
            key_in = "mlp_extractor.policy_net.0.weight"
            if key_in in state_dict and key_in in target and "action_net.weight" in state_dict:
                extra_obs = target[key_in].shape[1] - state_dict[key_in].shape[1]
                extra_act = (target["action_net.weight"].shape[0]
                             - state_dict["action_net.weight"].shape[0])
                if extra_obs > 0 or extra_act > 0:
                    from robosim.train.bc import expand_policy_state_dict

                    state_dict = expand_policy_state_dict(state_dict, extra_obs, extra_act)
                    print(f"[train] expanded bc_init by +{extra_obs} obs / +{extra_act} actions")
            model.policy.load_state_dict(state_dict, strict=False)
            print(f"[train] loaded BC-pretrained weights from {bc_init}")
        except Exception as e:
            print(f"[train] WARNING: failed to load bc_init '{bc_init}': {e}")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(steps // 10, 1000) // n_envs,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ckpt",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(steps // 20, 1000) // n_envs,
        n_eval_episodes=5,
        deterministic=True,
    )
    curve_cb = LearningCurveCallback(run_dir / "learning_curve.csv")
    callbacks = [checkpoint_cb, eval_cb, curve_cb]
    if curriculum:
        stages = parse_curriculum(curriculum)
        callbacks.append(CurriculumCallback(stages, eval_env=eval_env))

    config = {
        "task": task,
        "scene": scene,
        "robot": robot,
        "algo": algo,
        "policy": policy,
        "steps": steps,
        "n_envs": n_envs,
        "seed": seed,
        "n_lidar_beams": n_lidar_beams,
        "lidar_fov_deg": lidar_fov_deg,
        "lidar_max_range": lidar_max_range,
        "max_episode_steps": max_episode_steps,
        "versus_checkpoint": versus_checkpoint,
        "versus_opponents": versus_opponents if versus_checkpoint else None,
        "firearms": firearms,
        "rocket_speed": rocket_speed,
        "combat_sensor": combat_sensor,
        "view_range": view_range,
        "domain_randomization": domain_randomization.to_dict() if domain_randomization else None,
        "bc_init": bc_init,
        "curriculum": curriculum,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    model.learn(total_timesteps=steps, callback=CallbackList(callbacks))
    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"[train] saved final model to {final_path}")
    print(f"[train] best model (by eval) at {run_dir / 'best_model.zip'}")
    return run_dir
