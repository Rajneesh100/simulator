"""Recover env-shaping settings (lidar config, episode length) from a run's config.yaml.

A checkpoint is only usable with the observation space it was trained on, so play/
evaluate look for the config.yaml that `train` wrote next to the checkpoint (or one
directory up, for files under runs/<run>/checkpoints/) and rebuild the env to match.
Runs older than these keys simply fall back to the env defaults.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_ENV_KEYS = ("n_lidar_beams", "lidar_fov_deg", "lidar_max_range", "max_episode_steps",
             "firearms", "rocket_speed", "combat_sensor", "view_range")
_FIREARMS_KEYS = ("firearms", "rocket_speed", "combat_sensor", "view_range")


def _load_config(checkpoint: str) -> dict:
    ckpt = Path(checkpoint).resolve()
    for directory in (ckpt.parent, ckpt.parent.parent):
        cfg_path = directory / "config.yaml"
        if cfg_path.is_file():
            return yaml.safe_load(cfg_path.read_text()) or {}
    return {}


def env_kwargs_from_run(checkpoint: str) -> dict:
    config = _load_config(checkpoint)
    kwargs = {k: config[k] for k in _ENV_KEYS if config.get(k) is not None}
    if kwargs.get("firearms") in (None, "none"):  # weaponless runs: plain env kwargs
        for k in _FIREARMS_KEYS:
            kwargs.pop(k, None)
    if kwargs:
        print(f"[run-config] env settings for {checkpoint}: {kwargs}")
    return kwargs


def task_params_from_run(checkpoint: str) -> dict:
    """Task params the checkpoint trained with (e.g. race lookahead settings), parsed
    from the run config's task string ('race:lookahead=4,lookahead_time=0.6')."""
    from robosim.core.world import _parse_scene_arg

    task = _load_config(checkpoint).get("task") or ""
    return _parse_scene_arg(task)[1] if ":" in task else {}
