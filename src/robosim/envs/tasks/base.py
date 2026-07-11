"""Task ABC. Each file in envs/tasks/ subclasses this and registers with @register_task.

Tasks consume goal/placement info the *scene* already computed (scenes own seeded
randomization + goal placement per idea.md) and turn it into a reward signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Task(ABC):
    name: str = ""

    def __init__(self, **params: str):
        self.params = params

    def reset(self, world, rng: np.random.Generator) -> dict[str, Any]:
        """Called right after world.reset(). Returns task state, e.g. {'goal': (x, y)}."""
        return dict(world.scene.task_info())

    @abstractmethod
    def reward(
        self,
        world,
        task_state: dict[str, Any],
        action: np.ndarray,
        collided: bool,
    ) -> tuple[float, bool, dict[str, Any]]:
        """Return (reward, terminated, info) for the current step."""

    def max_episode_steps(self) -> int:
        return int(self.params.get("max_episode_steps", 500))
