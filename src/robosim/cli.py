"""Single CLI entrypoint: robosim view|teleop|train|play|scenes|evaluate."""

from __future__ import annotations

import os
import platform
import shutil
import sys

import typer

from robosim.core.registry import list_robots, list_scenes, list_tasks

app = typer.Typer(add_completion=False, help="RoboSim — minimal Isaac-Lab-style physics simulator.")

_RELAUNCH_MARKER = "ROBOSIM_MJPYTHON_RELAUNCHED"


def _ensure_mjpython() -> None:
    """mujoco.viewer.launch_passive requires macOS scripts run under `mjpython` (its
    own launcher that dispatches the GUI to the main thread) instead of plain `python`.
    Transparently re-exec under it so `robosim view/teleop/play` just work regardless of
    how the user invoked the CLI — this is what idea.md's CLI section promises."""
    if platform.system() != "Darwin" or os.environ.get(_RELAUNCH_MARKER):
        return
    mjpython = shutil.which("mjpython")
    if not mjpython:
        return  # fall through; mujoco will raise its own clear error
    os.environ[_RELAUNCH_MARKER] = "1"
    os.execvp(mjpython, [mjpython, "-m", "robosim.cli", *sys.argv[1:]])


@app.command()
def view(
    scene: str = typer.Option("flat", help="Scene name, optionally 'name:key=val,...'"),
    robot: str = typer.Option("diffdrive", help="Robot name"),
    speed: float = typer.Option(1.0, help="Simulation speed multiplier (0.1x - 5x)"),
    debug_lidar: bool = typer.Option(False, "--debug-lidar", help="Draw lidar rays"),
):
    """Watch the world: flat scene + robot renders in the interactive viewer."""
    _ensure_mjpython()
    from robosim.ui.view import run_view

    run_view(scene, robot, speed=speed, debug_lidar=debug_lidar)


@app.command()
def teleop(
    scene: str = typer.Option("flat", help="Scene name"),
    robot: str = typer.Option("diffdrive", help="Robot name"),
    record: str = typer.Option(None, help="Path (no extension) to record trajectory .npz to"),
):
    """Drive manually with WASD in the viewer, optionally recording the trajectory."""
    _ensure_mjpython()
    from robosim.ui.teleop import run_teleop

    run_teleop(scene, robot, record_path=record)


@app.command()
def scenes():
    """List registered scenes, robots, and tasks."""
    typer.echo(f"Scenes: {', '.join(list_scenes())}")
    typer.echo(f"Robots: {', '.join(list_robots())}")
    typer.echo(f"Tasks:  {', '.join(list_tasks())}")


@app.command()
def train(
    task: str = typer.Option("goal_reach", help="Task name"),
    scene: str = typer.Option("flat", help="Scene name"),
    robot: str = typer.Option("diffdrive", help="Robot name"),
    steps: int = typer.Option(200_000, help="Total training timesteps"),
    n_envs: int = typer.Option(8, help="Parallel envs"),
    algo: str = typer.Option("ppo", help="ppo | sac"),
    policy: str = typer.Option("p1", help="Policy architecture: p1 | p2 | p3 | p4"),
    run_name: str = typer.Option(None, help="Run name under runs/ (default: auto)"),
    seed: int = typer.Option(0, help="Random seed"),
    bc_init: str = typer.Option(None, help="Path to a BC-pretrained policy .zip to init from"),
    dr: bool = typer.Option(False, "--dr", help="Enable default domain randomization ranges"),
    curriculum: str = typer.Option(
        None,
        help="Difficulty schedule, e.g. '0:flat;50000:obstacles:count=3;150000:obstacles:count=10'",
    ),
    lidar_beams: int = typer.Option(16, help="Lidar beam count (obs size changes with it)"),
    lidar_fov: float = typer.Option(270.0, help="Lidar field of view in degrees"),
    lidar_range: float = typer.Option(5.0, help="Lidar max range in meters"),
    max_episode_steps: int = typer.Option(500, help="Control steps per episode (20 Hz)"),
    versus: str = typer.Option(
        None,
        help="COMPETITIVE training: checkpoint whose frozen policy drives opponent cars; "
        "reward adds overtakes, takedowns, and win/lose (racetrack scenes only)",
    ),
    versus_opponents: int = typer.Option(2, help="Opponent cars in versus mode"),
    firearms: str = typer.Option("none", help="none | rocket — arm every car (versus mode only)"),
    rocket_speed: float = typer.Option(
        40.0, help="Rocket muzzle speed m/s (range = v^2 sin(2 el) / g)"
    ),
):
    """Train a policy headless with SB3 PPO/SAC."""
    from robosim.envs.base_env import DomainRandomization
    from robosim.train.train import run_train

    domain_randomization = (
        DomainRandomization(
            friction_scale=(0.6, 1.4),
            mass_scale=(0.8, 1.2),
            lidar_noise_std=(0.0, 0.05),
            lidar_dropout_prob=(0.0, 0.05),
        )
        if dr
        else None
    )

    run_train(
        task=task,
        scene=scene,
        robot=robot,
        steps=steps,
        n_envs=n_envs,
        algo=algo,
        policy=policy,
        run_name=run_name,
        seed=seed,
        bc_init=bc_init,
        domain_randomization=domain_randomization,
        curriculum=curriculum,
        n_lidar_beams=lidar_beams,
        lidar_fov_deg=lidar_fov,
        lidar_max_range=lidar_range,
        max_episode_steps=max_episode_steps,
        versus_checkpoint=versus,
        versus_opponents=versus_opponents,
        firearms=firearms,
        rocket_speed=rocket_speed,
    )


@app.command()
def play(
    checkpoint: str = typer.Option(..., help="Path to a trained model .zip"),
    scene: str = typer.Option("flat", help="Scene name"),
    robot: str = typer.Option("diffdrive", help="Robot name"),
    task: str = typer.Option("goal_reach", help="Task name, e.g. 'race:laps=0' for endless"),
    episodes: int = typer.Option(5, help="Number of episodes to play"),
    max_steps: int = typer.Option(None, help="Override episode step cap (e.g. huge for endless laps)"),
):
    """Load a checkpoint and run the policy in the viewer."""
    _ensure_mjpython()
    from robosim.train.play import run_play

    run_play(
        checkpoint, scene=scene, robot=robot, task=task, episodes=episodes,
        max_episode_steps=max_steps,
    )


@app.command()
def race(
    checkpoint: list[str] = typer.Option(
        ...,
        help="Trained model .zip. Repeat the flag to duel models (e.g. "
        "--checkpoint runs/rally1/best_model.zip --checkpoint runs/rally3/best_model.zip); "
        "cars are assigned to models round-robin.",
    ),
    scene: str = typer.Option("racetrack:obstacles=0", help="Track scene (needs waypoints)"),
    robot: str = typer.Option("racecar", help="Robot name"),
    cars: int = typer.Option(4, help="Number of cars on the grid (colors: red, green, blue, ...)"),
    laps: int = typer.Option(3, help="Laps to win the race"),
    speed: float = typer.Option(1.0, help="Playback speed multiplier"),
    deterministic: bool = typer.Option(
        False, "--deterministic", help="Mean actions: every race is an identical replay"
    ),
    seed: int = typer.Option(None, help="Seed the stochastic race (reproducible variety)"),
    variation: float = typer.Option(
        0.5, help="Driver personality spread 0..1: per-car aggression/lookahead/jitter"
    ),
    lookahead: float = typer.Option(2.5, help="Base goal lookahead (m) — match training"),
    lookahead_time: float = typer.Option(
        0.0, help="Speed-scaled lookahead horizon (s) — match training's lookahead_time"
    ),
    firearms: str = typer.Option(
        "none", help="none | rocket — arm every car; weaponless checkpoints get a scripted gunner"
    ),
    rocket_speed: float = typer.Option(40.0, help="Rocket muzzle speed m/s (sets range)"),
):
    """Grid-start N copies of the same policy and race them to a winner."""
    _ensure_mjpython()
    from robosim.ui.race import run_race

    run_race(
        checkpoint, scene=scene, robot=robot, cars=cars, laps=laps, speed=speed,
        deterministic=deterministic, seed=seed, variation=variation,
        lookahead=lookahead, lookahead_time=lookahead_time,
        firearms=firearms, rocket_speed=rocket_speed,
    )


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., help="Path to a trained model .zip"),
    scene: str = typer.Option("flat", help="Scene name"),
    robot: str = typer.Option("diffdrive", help="Robot name"),
    task: str = typer.Option("goal_reach", help="Task name"),
    episodes: int = typer.Option(50, help="Number of seeded evaluation episodes"),
    seed: int = typer.Option(0, help="Base seed for evaluation episodes"),
):
    """Run N seeded episodes and report success rate, collision rate, time-to-goal, path efficiency."""
    from robosim.train.evaluate import run_evaluate

    run_evaluate(checkpoint, scene=scene, robot=robot, task=task, episodes=episodes, seed=seed)


@app.command()
def bc_train(
    records: list[str] = typer.Option(..., help="One or more teleop --record .npz files"),
    out: str = typer.Option(..., help="Output path for BC weights (.pt)"),
    epochs: int = typer.Option(100, help="Training epochs"),
):
    """Train a small MLP via behavior cloning on recorded trajectories, saved for --bc-init."""
    from robosim.train.bc import train_bc

    train_bc(records, out, epochs=epochs)


@app.command()
def export(
    checkpoint: str = typer.Option(..., help="Path to a trained (non-recurrent) model .zip"),
    out: str = typer.Option(..., help="Output .onnx path"),
    obs_dim: int = typer.Option(..., help="Observation vector dimension"),
):
    """Export a trained policy to ONNX, with an int8-quantized inference sanity check."""
    from robosim.train.export import export_onnx

    export_onnx(checkpoint, out, obs_dim)


if __name__ == "__main__":
    app()
