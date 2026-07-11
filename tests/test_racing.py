"""Racing stack: racetrack scene, racecar robot, race task, loop movers."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from robosim.core.registry import get_robot, get_scene, get_task
from robosim.core.world import build_world
from robosim.envs.base_env import RoboSimEnv


def test_racing_pieces_registered():
    assert get_scene("racetrack") is not None
    assert get_robot("racecar") is not None
    assert get_task("race") is not None


def test_racetrack_builds_and_reports_track_info():
    world = build_world("racetrack:obstacles=4,seed=1", "racecar")
    info = world.scene.task_info()
    expected_len = 4 * 5.0 + 2 * np.pi * 2.5
    assert abs(info["track_length"] - expected_len) < 1e-6
    assert len(info["waypoints"]) == 96
    assert info["road_half_width"] == 1.3
    # spawn sits on the centerline start, heading along the track
    (x, y, _), yaw = world.scene.spawn_pose()
    assert abs(x - info["waypoints"][0][0]) < 1e-9
    assert abs(y - info["waypoints"][0][1]) < 1e-9
    assert abs(yaw) < 0.1  # bottom straight heads +x


def test_racecar_ackermann_geometry():
    robot = get_robot("racecar")()
    straight = robot.action_to_ctrl(np.array([1.0, 0.0]))
    assert straight["front_left_steer"] == 0.0
    assert straight["rear_left_motor"] == straight["rear_right_motor"] > 0

    left = robot.action_to_ctrl(np.array([0.5, 1.0]))
    # left turn: inner (left) wheel steers sharper, spins slower
    assert left["front_left_steer"] > left["front_right_steer"] > 0
    assert left["rear_left_motor"] < left["rear_right_motor"]

    right = robot.action_to_ctrl(np.array([0.5, -1.0]))
    assert right["front_right_steer"] < right["front_left_steer"] < 0
    assert right["rear_right_motor"] < right["rear_left_motor"]


def test_check_env_race():
    env = RoboSimEnv(
        scene="racetrack:obstacles=0", robot="racecar", task="race", max_episode_steps=50
    )
    check_env(env, skip_render_check=True)


def test_race_rewards_forward_progress():
    env = RoboSimEnv(
        scene="racetrack:obstacles=0", robot="racecar", task="race", max_episode_steps=200
    )
    env.reset(seed=0)
    total_reward = 0.0
    info = {}
    for _ in range(60):  # 3 s of full throttle on the opening straight
        _, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        total_reward += reward
        if terminated or truncated:
            break
    assert info["progress_m"] > 1.0
    assert total_reward > 0.0
    # lookahead goal moved ahead of the spawn along the track
    goal = env._task_state["goal"]
    assert goal[0] > env.world.spawn_pos[0] + 1.0


def test_race_backward_driving_penalized():
    env = RoboSimEnv(
        scene="racetrack:obstacles=0", robot="racecar", task="race", max_episode_steps=100
    )
    env.reset(seed=0)
    total_reward = 0.0
    for _ in range(40):
        _, reward, terminated, truncated, info = env.step(np.array([-1.0, 0.0], dtype=np.float32))
        total_reward += reward
        if terminated or truncated:
            break
    assert total_reward < 0.0


def test_opponent_cars_lap_the_track():
    env = RoboSimEnv(
        scene="racetrack:obstacles=0,opponents=2", robot="racecar", task="race",
        max_episode_steps=50,
    )
    env.reset(seed=0)
    mocap_before = env.world.data.mocap_pos.copy()
    for _ in range(20):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
    moved = np.linalg.norm(env.world.data.mocap_pos - mocap_before, axis=1)
    assert (moved > 0.1).all()  # both opponents progressed along the loop


def test_axis_movers_still_work():
    env = RoboSimEnv(
        scene="obstacles:count=0,moving=1", robot="diffdrive", task="goal_reach",
        max_episode_steps=50,
    )
    env.reset(seed=0)
    before = env.world.data.mocap_pos.copy()
    for _ in range(20):
        env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert np.linalg.norm(env.world.data.mocap_pos - before) > 0.05
