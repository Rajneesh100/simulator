"""Scene ABC. Every scene in scenes/ subclasses this and registers with @register_scene."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Scene(ABC):
    """A scene adds geometry/bodies to a world's MjSpec and defines spawn/goal info.

    Scenes never touch global physics (see core/physics.py) — they only place geometry,
    and may set per-geom material/friction when the scene's premise requires it (e.g. a
    slick ramp). Constructor receives scene params as keyword strings from the CLI
    (`--scene name:key=val,...`); parse/cast them here with sensible defaults.
    """

    name: str = ""

    def __init__(self, **params: str):
        self.params = params

    @abstractmethod
    def build(self, spec) -> None:
        """Add geoms/bodies to `spec` (a mujoco.MjSpec), rooted at spec.worldbody."""

    @abstractmethod
    def spawn_pose(self) -> tuple[tuple[float, float, float], float]:
        """Return ((x, y, terrain_z), yaw_radians) for where the robot should spawn.

        terrain_z is the ground height at that point, NOT the robot origin height —
        world.py adds the robot's own spawn_z_offset() so scenes stay robot-agnostic.
        """

    def task_info(self) -> dict[str, Any]:
        """Scene-specific info a task layer might use (goal position, bounds, etc.)."""
        return {}

    def ground_geom_names(self) -> list[str]:
        """Geoms that are "ground" for collision purposes — touching them (e.g. a wheel
        grazing the floor, or the chassis briefly scraping flat terrain mid-turn) is
        normal driving, not a collision. Defaults match every built-in scene's ground
        geom name; override if a scene names its ground/terrain geom differently."""
        return ["ground", "terrain"]

    def collision_exempt_geom_names(self) -> list[str]:
        """Geoms whose contact with the robot does NOT count as a collision for
        reward/termination. Defaults to the ground geoms; scenes with a physical
        driving surface (e.g. racetrack hills' road slabs) add it here — scraping
        the road is driving, not crashing. Kept separate from ground_geom_names()
        because that list also drives ground *theming*."""
        return self.ground_geom_names()

    def moving_bodies(self) -> list[dict[str, Any]]:
        """Optional: scripted moving-body specs for tasks like T7 moving_obstacle.

        Each dict describes one scripted mover, e.g.
        {"name": "mover_0", "path": "linear", "amplitude": 1.0, "period": 4.0, "axis": "x"}.
        Default: none. Scenes that support moving obstacles override this.
        """
        return []
