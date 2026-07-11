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

N_LIDAR_BEAMS = 16
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


def _make_env_fn(scene, robot, task, dr, seed_offset, render_mode=None):
    def _fn():
        env = RoboSimEnv(
            scene=scene,
            robot=robot,
            task=task,
            n_lidar_beams=N_LIDAR_BEAMS,
            domain_randomization=dr,
            render_mode=render_mode,
        )
        return env

    return _fn


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
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = policy.lower()
    if run_name is None:
        run_name = f"{task}_{scene}_{algo}_{policy}_{int(time.time())}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(scene=scene, robot=robot, task=task, domain_randomization=domain_randomization)
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    try:
        vec_env = make_vec_env(
            RoboSimEnv, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=vec_cls
        )
    except Exception as e:  # SubprocVecEnv can fail to spawn in some sandboxes
        print(f"[train] {vec_cls.__name__} failed ({e}); falling back to DummyVecEnv")
        vec_env = make_vec_env(
            RoboSimEnv, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv
        )

    eval_env = make_vec_env(
        RoboSimEnv, n_envs=1, seed=seed + 10_000, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv
    )

    if policy == "p4":
        vec_env = VecFrameStack(vec_env, n_stack=FRAME_STACK_K)
        eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK_K)

    policy_kwargs = policy_kwargs_for(policy, n_lidar_beams=N_LIDAR_BEAMS)
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

    config = {
        "task": task,
        "scene": scene,
        "robot": robot,
        "algo": algo,
        "policy": policy,
        "steps": steps,
        "n_envs": n_envs,
        "seed": seed,
        "n_lidar_beams": N_LIDAR_BEAMS,
        "domain_randomization": domain_randomization.to_dict() if domain_randomization else None,
        "bc_init": bc_init,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    model.learn(total_timesteps=steps, callback=CallbackList([checkpoint_cb, eval_cb, curve_cb]))
    final_path = run_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"[train] saved final model to {final_path}")
    print(f"[train] best model (by eval) at {run_dir / 'best_model.zip'}")
    return run_dir
