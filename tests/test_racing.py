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


def test_lidar_no_self_hits_at_360_fov():
    import mujoco

    from robosim.sensors.lidar import Lidar, body_subtree_ids

    world = build_world("flat", "racecar")
    mujoco.mj_forward(world.model, world.data)
    lidar = Lidar(n_beams=48, fov_deg=360.0, max_range=12.0)
    chassis = world.model.body(world.root_body_name).id
    ranges = lidar.sense(
        world.model, world.data, exclude_body_id=chassis,
        exclude_body_ids=body_subtree_ids(world.model, chassis),
    ) * 12.0
    # empty flat scene: nothing within a meter in any direction (was 0.11 m on own wheels)
    assert ranges.min() > 1.0


def test_gp_layout_builds_closed_circuit():
    world = build_world("racetrack:layout=gp,scale=9,seed=3,obstacles=0", "racecar")
    info = world.scene.task_info()
    pts = np.asarray(info["waypoints"])[:, :2]
    assert info["track_length"] > 2 * np.pi * 7  # meaningfully bigger than a small circle
    # closed loop: consecutive gaps (incl. wraparound) all small and uniform-ish
    gaps = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)
    assert gaps.max() < 3 * gaps.min()


def test_race_laps_param():
    env = RoboSimEnv(scene="racetrack:obstacles=0", robot="racecar", task="race:laps=0",
                     max_episode_steps=50)
    def drive_across_line(e):
        e._task_state["progress"] = e._task_state["track_length"] - 0.005
        for _ in range(10):  # a few steps to get rolling from standstill
            _, _, terminated, _, info = e.step(np.array([1.0, 0.0], dtype=np.float32))
            if info["laps_done"] >= 1 or terminated:
                return terminated, info
        return terminated, info

    env.reset(seed=0)
    terminated, info = drive_across_line(env)
    assert info["laps_done"] == 1
    assert not terminated  # laps=0: endless, lap bonus but no termination
    assert "success" not in info

    env2 = RoboSimEnv(scene="racetrack:obstacles=0", robot="racecar", task="race",
                      max_episode_steps=50)
    env2.reset(seed=0)
    terminated2, info2 = drive_across_line(env2)
    assert terminated2 and info2.get("success")  # default laps=1 finishes the race


def test_multi_world_grid():
    import mujoco

    from robosim.core.world import build_multi_world

    world = build_multi_world("racetrack:obstacles=0", "racecar", n_cars=4)
    assert world.n_cars == 4
    # every car got its own prefixed actuators and a distinct grid slot
    assert world.model.body("c0_car_chassis").id != world.model.body("c3_car_chassis").id
    positions = [world.car_pose(i)[0][:2] for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.hypot(*(positions[i] - positions[j])) > 0.5
    # the per-car self-collision excludes survived the prefixed attach
    assert world.model.nexclude == 8
    # no car-on-car contact at the grid
    mujoco.mj_forward(world.model, world.data)
    for k in range(world.data.ncon):
        c = world.data.contact[k]
        n1 = world.model.geom(c.geom1).name
        n2 = world.model.geom(c.geom2).name
        assert not ("chassis" in n1 and "chassis" in n2)
    # prefixed proprio reads work per car
    p = world.robots[2].read_proprio(world.model, world.data, prefix=world.prefixes[2])
    assert p.shape == (5,) and np.isfinite(p).all()

    # drive all four forward one control step: everyone should move
    from robosim.core import physics
    before = [world.car_pose(i)[0][:2].copy() for i in range(4)]
    for i in range(4):
        world.set_ctrl(i, world.robots[i].action_to_ctrl(np.array([1.0, 0.0])))
    for _ in range(physics.physics_steps_per_control_step() * 20):
        mujoco.mj_step(world.model, world.data)
    for i in range(4):
        assert np.hypot(*(world.car_pose(i)[0][:2] - before[i])) > 0.3


def test_hilly_track_drivable():
    import mujoco

    world = build_world("racetrack:layout=gp,scale=9,seed=3,hills=0.8,obstacles=0", "racecar")
    info = world.scene.task_info()
    zs = np.asarray(info["waypoints"])[:, 2]
    assert zs.max() - zs.min() > 0.4  # real elevation change
    assert zs.min() > -1e-9  # nothing sinks below the ground plane
    # grade cap respected
    pts = np.asarray(info["waypoints"])[:, :2]
    seg = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T)
    grades = np.abs(np.diff(np.concatenate([zs, zs[:1]]))) / seg
    assert grades.max() < 0.13
    # drive up the first stretch: car must gain ground, not fall through the road
    env = RoboSimEnv(scene="racetrack:layout=gp,scale=9,seed=3,hills=0.8,obstacles=0",
                     robot="racecar", task="race", max_episode_steps=200)
    env.reset(seed=0)
    info_step = {}
    for _ in range(80):  # 4 s of full throttle
        _, _, terminated, truncated, info_step = env.step(np.array([1.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break
    pos, _ = env.world.robot_pose()
    assert info_step["progress_m"] > 3.0
    assert pos[2] > -0.5  # never fell through the world


def test_racecar_max_speed_param():
    fast = get_robot("racecar")(max_speed="6.0")
    slow = get_robot("racecar")()
    a = np.array([1.0, 0.0])
    assert fast.action_to_ctrl(a)["rear_left_motor"] == 2 * slow.action_to_ctrl(a)["rear_left_motor"]
    # ctrl range scales with the requested top speed so the command isn't clipped
    spec = fast.build_spec()
    model_ok = spec.compile() is not None
    assert model_ok


def test_task_params_recovered_from_run_config(tmp_path):
    from robosim.train.run_config import task_params_from_run

    (tmp_path / "config.yaml").write_text("task: race:laps=2,lookahead=4.0,lookahead_time=0.6\n")
    params = task_params_from_run(str(tmp_path / "best_model.zip"))
    assert params == {"laps": "2", "lookahead": "4.0", "lookahead_time": "0.6"}
    # runs trained with a bare task name have no params
    (tmp_path / "config.yaml").write_text("task: race\n")
    assert task_params_from_run(str(tmp_path / "best_model.zip")) == {}


def test_versus_env_smoke():
    import pytest

    pytest.importorskip("stable_baselines3")
    from robosim.envs.versus_env import VersusRaceEnv

    env = VersusRaceEnv(
        scene="racetrack:obstacles=0", robot="racecar",
        task="race:laps=2,lookahead=2.5",
        opponent_checkpoint="runs/rally3/best_model.zip", opponents=2,
        n_lidar_beams=48, lidar_fov_deg=360.0, lidar_max_range=12.0,
        max_episode_steps=100,
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (48 + 10,)
    opp_before = [env.world.car_pose(i)[0][:2].copy() for i in (1, 2)]
    total_reward = 0.0
    for _ in range(40):  # 2 s: learner floors it, opponents race their frozen policy
        obs, r, term, trunc, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
        total_reward += r
        if term or trunc:
            break
    assert "rank" in info and 1 <= info["rank"] <= 3
    moved = [np.hypot(*(env.world.car_pose(i)[0][:2] - opp_before[k]))
             for k, i in enumerate((1, 2))]
    assert all(m > 0.5 for m in moved)  # frozen opponents actually drive
    assert np.isfinite(total_reward)


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
