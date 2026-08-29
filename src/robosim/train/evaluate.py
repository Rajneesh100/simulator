"""Run N seeded episodes and report success rate, collision rate, time-to-goal, and path
efficiency — the same protocol for every policy/task so numbers are comparable.
"""

from __future__ import annotations

import numpy as np

from robosim.envs.base_env import RoboSimEnv
from robosim.train.run_config import env_kwargs_from_run


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
    env_kwargs = env_kwargs_from_run(checkpoint)
    env_kwargs.setdefault("max_episode_steps", max_episode_steps)
    env = RoboSimEnv(scene=scene, robot=robot, task=task, **env_kwargs)

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


def evaluate_versus(
    checkpoint: str,
    opponent: str,
    scene: str,
    robot: str,
    task: str = "race:laps=2,lookahead=2.5,lookahead_time=0.6",
    episodes: int = 10,
    seed: int = 0,
    deterministic: bool = True,
    opponents: int = 2,
    max_steps: int = 1500,
) -> dict:
    """Fire-fight scorecard: the checkpoint races `opponents` frozen `opponent` cars (with
    scripted gunners when firearms are on) for N seeded episodes. Reports kills, deaths,
    self-kills, shots and hit rate, near-miss distance, wins, crashes, and progress per
    episode — the numbers that say whether a brain can actually shoot and survive."""
    import numpy as np

    from robosim.envs.versus_env import VersusRaceEnv
    from robosim.train.run_config import env_kwargs_from_run, task_params_from_run

    kw = env_kwargs_from_run(checkpoint)
    kw.pop("max_episode_steps", None)
    tp = task_params_from_run(checkpoint)
    if tp and ":" not in task:
        task = "race:" + ",".join(f"{k}={v}" for k, v in tp.items())
    env = VersusRaceEnv(
        scene=scene, robot=robot, task=task, opponent_checkpoint=opponent,
        opponents=opponents, max_episode_steps=max_steps, **kw,
    )
    model, is_recurrent = _load_model(checkpoint)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        state, start = None, np.ones(1, dtype=bool)
        row = dict(kills=0, deaths=0, self_kills=0, shots=0, near_misses=[], won=0,
                   crashed=0, steps=0, progress_m=0.0)
        while True:
            if is_recurrent:
                a, state = model.predict(obs, state=state, episode_start=start,
                                         deterministic=deterministic)
                start[:] = False
            else:
                a, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(a)
            row["kills"] += info.get("kills", 0)
            row["shots"] += info.get("shots", 0)
            if "near_miss_m" in info:
                row["near_misses"].append(info["near_miss_m"])
            if info.get("killed"):
                row["deaths"] += 1
            if info.get("self_kill"):
                row["self_kills"] += 1
            if info.get("crashed"):
                row["crashed"] = 1
            row["steps"] += 1
            if term or trunc:
                row["won"] = int(bool(info.get("won")))
                row["progress_m"] = float(info.get("progress_m", 0.0))
                break
        rows.append(row)
    n = float(episodes)
    shots = sum(r["shots"] for r in rows)
    kills = sum(r["kills"] for r in rows)
    nm = [m for r in rows for m in r["near_misses"]]
    summary = {
        "episodes": episodes,
        "deterministic": deterministic,
        "kills_per_episode": kills / n,
        "deaths_per_episode": sum(r["deaths"] for r in rows) / n,
        "self_kills_per_episode": sum(r["self_kills"] for r in rows) / n,
        "shots_per_episode": shots / n,
        "hit_rate": (kills / shots) if shots else 0.0,
        "median_near_miss_m": float(np.median(nm)) if nm else float("nan"),
        "win_rate": sum(r["won"] for r in rows) / n,
        "crash_rate": sum(r["crashed"] for r in rows) / n,
        "mean_progress_m": sum(r["progress_m"] for r in rows) / n,
        "mean_steps": sum(r["steps"] for r in rows) / n,
    }
    print(f"evaluate-versus {checkpoint} vs {opponent} ({'deterministic' if deterministic else 'sampled'}, "
          f"{episodes} episodes):")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    return summary
