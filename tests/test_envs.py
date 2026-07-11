import numpy as np
from gymnasium.utils.env_checker import check_env

from robosim.envs.base_env import DomainRandomization, RoboSimEnv


def test_check_env_goal_reach():
    env = RoboSimEnv(scene="flat", robot="diffdrive", task="goal_reach", max_episode_steps=50)
    check_env(env, skip_render_check=True)


def test_check_env_explore():
    env = RoboSimEnv(scene="flat", robot="diffdrive", task="explore", max_episode_steps=50)
    check_env(env, skip_render_check=True)


def test_deterministic_under_seed():
    def rollout(seed):
        env = RoboSimEnv(scene="flat", robot="diffdrive", task="goal_reach", max_episode_steps=30)
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(123)
        frames = [obs.copy()]
        for _ in range(30):
            a = rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
            obs, *_ = env.step(a)
            frames.append(obs.copy())
        return np.concatenate(frames)

    assert np.allclose(rollout(7), rollout(7))


def test_domain_randomization_varies_with_seed():
    dr = DomainRandomization(friction_scale=(0.5, 1.5))
    env = RoboSimEnv(scene="flat", robot="diffdrive", task="goal_reach", domain_randomization=dr)
    _, info1 = env.reset(seed=1)
    _, info2 = env.reset(seed=2)
    assert info1["domain_randomization"]["friction_scale"] != info2["domain_randomization"]["friction_scale"]


def test_render_rgb_array():
    env = RoboSimEnv(scene="flat", robot="diffdrive", task="goal_reach", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (480, 640, 3)
    assert frame.std() > 0
    env.close()
