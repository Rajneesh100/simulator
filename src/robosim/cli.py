"""Single CLI entrypoint: robosim view|teleop|train|play|scenes|evaluate."""

from __future__ import annotations

import typer

from robosim.core.registry import list_robots, list_scenes, list_tasks

app = typer.Typer(add_completion=False, help="RoboSim — minimal Isaac-Lab-style physics simulator.")


@app.command()
def view(
    scene: str = typer.Option("flat", help="Scene name, optionally 'name:key=val,...'"),
    robot: str = typer.Option("diffdrive", help="Robot name"),
    speed: float = typer.Option(1.0, help="Simulation speed multiplier (0.1x - 5x)"),
    debug_lidar: bool = typer.Option(False, "--debug-lidar", help="Draw lidar rays"),
):
    """Watch the world: flat scene + robot renders in the interactive viewer."""
    from robosim.ui.view import run_view

    run_view(scene, robot, speed=speed, debug_lidar=debug_lidar)


@app.command()
def teleop(
    scene: str = typer.Option("flat", help="Scene name"),
    robot: str = typer.Option("diffdrive", help="Robot name"),
    record: str = typer.Option(None, help="Path (no extension) to record trajectory .npz to"),
):
    """Drive manually with WASD in the viewer, optionally recording the trajectory."""
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
    )


@app.command()
def play(
    checkpoint: str = typer.Option(..., help="Path to a trained model .zip"),
    scene: str = typer.Option("flat", help="Scene name"),
    robot: str = typer.Option("diffdrive", help="Robot name"),
    task: str = typer.Option("goal_reach", help="Task name"),
    episodes: int = typer.Option(5, help="Number of episodes to play"),
):
    """Load a checkpoint and run the policy in the viewer."""
    from robosim.train.play import run_play

    run_play(checkpoint, scene=scene, robot=robot, task=task, episodes=episodes)


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
