import shutil

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from robosim.envs.base_env import DomainRandomization, RoboSimEnv
from robosim.train.bc import generate_scripted_demo, train_bc
from robosim.train.evaluate import evaluate_policy
from robosim.train.train import parse_curriculum, run_train

TASK_SCENE_CONFIGS = [
    ("goal_reach", "flat"),
    ("obstacle_nav", "obstacles:count=6,seed=1"),
    ("corridor", "obstacles:maze=1,seed=2"),
    ("slope_climb", "slope"),
    ("stair_climb", "stairs"),
    ("explore", "flat"),
    ("moving_obstacle", "obstacles:moving=2,seed=1,count=4"),
]


@pytest.mark.parametrize("task,scene", TASK_SCENE_CONFIGS)
def test_task_scene_combo_runs(task, scene):
    env = RoboSimEnv(scene=scene, robot="diffdrive", task=task, max_episode_steps=30)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    for _ in range(30):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))


def test_moving_obstacle_actually_moves():
    env = RoboSimEnv(scene="obstacles:moving=1,seed=1,count=1", robot="diffdrive", task="moving_obstacle")
    env.reset(seed=0)
    mocap_id = env._movers[0]["mocap_id"]
    positions = []
    for _ in range(50):
        env.step(np.zeros(2, dtype=np.float32))
        positions.append(env.world.data.mocap_pos[mocap_id].copy())
    positions = np.array(positions)
    assert positions[:, 1].std() > 0.01  # y-coordinate actually oscillates


def test_curriculum_parsing():
    stages = parse_curriculum("0:flat;50000:obstacles:count=3;150000:obstacles:count=10")
    assert stages == [(0, "flat"), (50000, "obstacles:count=3"), (150000, "obstacles:count=10")]


def test_set_scene_rebuilds_world():
    env = RoboSimEnv(scene="flat", robot="diffdrive", task="goal_reach")
    env.reset(seed=0)
    n_geom_before = env.world.model.ngeom
    env.set_scene("obstacles:count=10,seed=1")
    env.reset(seed=0)
    assert env.world.model.ngeom != n_geom_before
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


@pytest.mark.parametrize("policy", ["p1", "p2", "p3", "p4"])
def test_policy_architectures_train_without_error(policy):
    run_dir = run_train(
        task="goal_reach",
        scene="flat",
        robot="diffdrive",
        steps=1024,
        n_envs=1,
        policy=policy,
        run_name=f"pytest_policy_{policy}",
    )
    assert (run_dir / "final_model.zip").exists()
    shutil.rmtree(run_dir, ignore_errors=True)


def test_domain_randomization_end_to_end_in_training():
    dr = DomainRandomization(friction_scale=(0.7, 1.3), lidar_noise_std=(0.0, 0.03))
    run_dir = run_train(
        task="goal_reach",
        scene="flat",
        robot="diffdrive",
        steps=1024,
        n_envs=1,
        policy="p1",
        run_name="pytest_dr",
        domain_randomization=dr,
    )
    assert (run_dir / "config.yaml").exists()
    shutil.rmtree(run_dir, ignore_errors=True)


def test_curriculum_end_to_end_in_training():
    run_dir = run_train(
        task="goal_reach",
        scene="flat",
        robot="diffdrive",
        steps=2048,
        n_envs=1,
        policy="p1",
        run_name="pytest_curriculum",
        curriculum="0:flat;1024:obstacles:count=2,seed=1",
    )
    assert (run_dir / "final_model.zip").exists()
    shutil.rmtree(run_dir, ignore_errors=True)


def test_bc_pipeline_end_to_end(tmp_path):
    demo_path = generate_scripted_demo(str(tmp_path / "demo"), episodes=6, max_steps=150)
    bc_path = train_bc([str(demo_path)], str(tmp_path / "bc.pt"), epochs=10)
    assert bc_path.exists()

    run_dir = run_train(
        task="goal_reach",
        scene="flat",
        robot="diffdrive",
        steps=1024,
        n_envs=1,
        policy="p1",
        run_name="pytest_bc",
        bc_init=str(bc_path),
    )
    assert (run_dir / "final_model.zip").exists()
    shutil.rmtree(run_dir, ignore_errors=True)


def test_evaluate_protocol_reports_expected_keys():
    run_dir = run_train(
        task="goal_reach",
        scene="flat",
        robot="diffdrive",
        steps=1024,
        n_envs=1,
        policy="p1",
        run_name="pytest_eval",
    )
    results = evaluate_policy(str(run_dir / "final_model.zip"), scene="flat", task="goal_reach", episodes=3)
    assert set(results.keys()) == {
        "episodes",
        "success_rate",
        "collision_rate",
        "mean_time_to_goal_steps",
        "mean_path_efficiency",
    }
    shutil.rmtree(run_dir, ignore_errors=True)
