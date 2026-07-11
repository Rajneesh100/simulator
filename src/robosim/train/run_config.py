"""Recover env-shaping settings (lidar config, episode length) from a run's config.yaml.

A checkpoint is only usable with the observation space it was trained on, so play/
evaluate look for the config.yaml that `train` wrote next to the checkpoint (or one
directory up, for files under runs/<run>/checkpoints/) and rebuild the env to match.
Runs older than these keys simply fall back to the env defaults.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_ENV_KEYS = ("n_lidar_beams", "lidar_fov_deg", "lidar_max_range", "max_episode_steps")


def env_kwargs_from_run(checkpoint: str) -> dict:
    ckpt = Path(checkpoint).resolve()
    for directory in (ckpt.parent, ckpt.parent.parent):
        cfg_path = directory / "config.yaml"
        if cfg_path.is_file():
            config = yaml.safe_load(cfg_path.read_text()) or {}
            kwargs = {k: config[k] for k in _ENV_KEYS if config.get(k) is not None}
            if kwargs:
                print(f"[run-config] env settings from {cfg_path}: {kwargs}")
            return kwargs
    return {}
