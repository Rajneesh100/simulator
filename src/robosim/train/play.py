"""Load a trained checkpoint and run it in the interactive viewer."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from robosim.envs.base_env import RoboSimEnv


def _load_model(checkpoint: str):
    try:
        from stable_baselines3 import PPO

        return PPO.load(checkpoint, device="cpu"), False
    except Exception:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(checkpoint, device="cpu"), True


def run_play(
    checkpoint: str,
    scene: str = "flat",
    robot: str = "diffdrive",
    task: str = "goal_reach",
    episodes: int = 5,
) -> None:
    model, is_recurrent = _load_model(checkpoint)
    env = RoboSimEnv(scene=scene, robot=robot, task=task)

    with mujoco.viewer.launch_passive(env.world.model, env.world.data) as viewer:
        for ep in range(episodes):
            if not viewer.is_running():
                break
            obs, info = env.reset(seed=ep)
            lstm_state = None
            episode_start = np.array([True])
            done = False
            total_reward = 0.0
            while not done and viewer.is_running():
                step_start = time.time()
                if is_recurrent:
                    action, lstm_state = model.predict(
                        obs, state=lstm_state, episode_start=episode_start, deterministic=True
                    )
                    episode_start = np.array([False])
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                viewer.sync()
                print(
                    f"ep={ep} t={env.world.data.time:6.2f}s reward={reward:+.3f} "
                    f"total={total_reward:+.2f} dist={info.get('distance_to_goal', float('nan')):.2f}",
                    end="\r",
                )
                dt_left = env.world.model.opt.timestep * env._steps_per_action - (
                    time.time() - step_start
                )
                if dt_left > 0:
                    time.sleep(dt_left)
            print(f"\nEpisode {ep}: total_reward={total_reward:.2f} success={info.get('success', False)}")
