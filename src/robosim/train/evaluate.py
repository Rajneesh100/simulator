"""Run N seeded episodes and report success rate, collision rate, time-to-goal, and path
efficiency — the same protocol for every policy/task so numbers are comparable.
"""

from __future__ import annotations

import numpy as np

from robosim.envs.base_env import RoboSimEnv


def _load_model(checkpoint: str):
    try:
        from stable_baselines3 import PPO

        return PPO.load(checkpoint, device="cpu"), False
    except Exception:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(checkpoint, device="cpu"), True


def evaluate_policy(
    checkpoint: str,
    scene: str = "flat",
    robot: str = "diffdrive",
    task: str = "goal_reach",
    episodes: int = 50,
    seed: int = 0,
    max_episode_steps: int = 500,
) -> dict:
    model, is_recurrent = _load_model(checkpoint)
    env = RoboSimEnv(scene=scene, robot=robot, task=task, max_episode_steps=max_episode_steps)

    successes = 0
    collisions = 0
    times_to_goal = []
    path_efficiencies = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        start_pos, _ = env.world.robot_pose()
        goal = env._task_state.get("goal", (0.0, 0.0))
        straight_line_dist = float(np.hypot(goal[0] - start_pos[0], goal[1] - start_pos[1]))

        lstm_state = None
        episode_start = np.array([True])
        done = False
        path_len = 0.0
        prev_pos = start_pos[:2].copy()
        episode_collided = False
        steps = 0

        while not done:
            if is_recurrent:
                action, lstm_state = model.predict(
                    obs, state=lstm_state, episode_start=episode_start, deterministic=True
                )
                episode_start = np.array([False])
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            pos, _ = env.world.robot_pose()
            path_len += float(np.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]))
            prev_pos = pos[:2].copy()
            episode_collided = episode_collided or info.get("collided", False)

        if info.get("success", False):
            successes += 1
            times_to_goal.append(steps)
            if path_len > 1e-6:
                path_efficiencies.append(min(1.0, straight_line_dist / path_len))
        if episode_collided:
            collisions += 1

    results = {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "mean_time_to_goal_steps": float(np.mean(times_to_goal)) if times_to_goal else None,
        "mean_path_efficiency": float(np.mean(path_efficiencies)) if path_efficiencies else None,
    }
    return results


def run_evaluate(
    checkpoint: str,
    scene: str = "flat",
    robot: str = "diffdrive",
    task: str = "goal_reach",
    episodes: int = 50,
    seed: int = 0,
) -> dict:
    results = evaluate_policy(checkpoint, scene, robot, task, episodes, seed)
    print(f"Evaluated {checkpoint} on task={task} scene={scene} over {episodes} episodes:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results
