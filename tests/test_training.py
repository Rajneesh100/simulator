import shutil
from pathlib import Path

import numpy as np
import pytest

from robosim.envs.base_env import RoboSimEnv
from robosim.envs.obs import obs_dim
from robosim.train.export import export_onnx
from robosim.train.train import run_train

RUN_NAME = "pytest_smoke_train"


@pytest.fixture(scope="module")
def smoke_run():
    run_dir = run_train(
        task="goal_reach",
        scene="flat",
        robot="diffdrive",
        steps=2048,
        n_envs=1,
        algo="ppo",
        policy="p1",
        run_name=RUN_NAME,
        seed=0,
    )
    yield run_dir
    shutil.rmtree(run_dir, ignore_errors=True)


def test_train_produces_artifacts(smoke_run):
    assert (smoke_run / "final_model.zip").exists()
    assert (smoke_run / "config.yaml").exists()
    assert (smoke_run / "learning_curve.csv").exists()
    rows = (smoke_run / "learning_curve.csv").read_text().strip().splitlines()
    assert len(rows) >= 2  # header + at least one logged rollout


def test_export_onnx_and_int8_quantize(smoke_run):
    env = RoboSimEnv(scene="flat", robot="diffdrive", task="goal_reach")
    dim = obs_dim(env.lidar.n_beams, env.world.robot.action_dim())
    out = export_onnx(
        str(smoke_run / "final_model.zip"), str(smoke_run / "policy.onnx"), obs_dim=dim
    )
    assert out.exists()
    assert out.with_name(out.stem + ".int8.onnx").exists()
