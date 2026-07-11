"""Robot ABC. Every robot in robots/ subclasses this and registers with @register_robot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Robot(ABC):
    """A robot builds its own MjSpec body tree and knows how to drive/sense itself.

    `build()` returns a fresh mujoco.MjSpec containing just the robot (rooted at its own
    worldbody), which core/world.py attaches into the world spec at the scene's spawn
    pose. Robots expose actuator/sensor names so envs/sensors can look them up on the
    compiled model without hardcoding indices.
    """

    name: str = ""

    def __init__(self, **params: str):
        self.params = params

    @abstractmethod
    def build_spec(self):
        """Return a standalone mujoco.MjSpec containing this robot's body tree."""

    @abstractmethod
    def root_body_name(self) -> str:
        """Name of the free-floating root body (for attach + pose queries)."""

    def spawn_z_offset(self) -> float:
        """Height of the robot's root-body origin above ground level (terrain z=0) when
        resting naturally (e.g. wheels/feet just touching down). Scenes specify terrain
        height only; world.py adds this offset so scenes stay robot-agnostic."""
        return 0.0

    @abstractmethod
    def action_dim(self) -> int:
        ...

    @abstractmethod
    def action_to_ctrl(self, action: np.ndarray) -> dict[str, float]:
        """Map a canonical action vector (e.g. (v, omega)) to {actuator_name: ctrl_value}."""

    def sensor_names(self) -> dict[str, str]:
        """Logical sensor role -> mujoco sensor name, for sensors/ modules to look up."""
        return {}

    def proprio_dim(self) -> int:
        return 0

    def read_proprio(self, model, data, prefix: str = "") -> np.ndarray:
        """`prefix` is the attach prefix in multi-robot worlds ("" in the normal case)."""
        return np.zeros(0, dtype=np.float32)

    def collision_geom_names(self) -> list[str]:
        """Geoms whose contacts count as a "collision" for reward/termination purposes."""
        return []

    def extra_info(self) -> dict[str, Any]:
        return {}
