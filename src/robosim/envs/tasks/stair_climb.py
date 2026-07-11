"""T5: climb stairs. reward = height gained + upright bonus.

Use with `--scene stairs`. idea.md flags this as "likely hard for diffdrive — that's the
finding" — a wheeled robot has no way to lift over a step riser, so training here is
expected to plateau well before the goal; that result is itself the motivating case for
a future legged robot. Reuses SlopeClimbTask's reward shape (height progress + upright).
"""

from __future__ import annotations

from robosim.core.registry import register_task
from robosim.envs.tasks.slope_climb import SlopeClimbTask


@register_task("stair_climb")
class StairClimbTask(SlopeClimbTask):
    pass
