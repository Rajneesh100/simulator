"""2D lidar via MuJoCo raycasts, mounted at a robot's 'lidar_site'.

Configurable beam count / FOV / range, with optional Gaussian noise and beam dropout for
sim-to-real robustness training (idea.md: "sensors support configurable noise/dropout").

Self-exclusion: mj_multiRay's `bodyexclude` skips exactly ONE body, but a robot is a
body *tree* — on the racecar, 360-degree side beams hit its own wheels/cabin (separate
bodies) at ~0.12 m forever. Pass `exclude_body_ids` (see `body_subtree_ids`) and beams
that land on any self body are re-cast from just past the hit until they reach the world.
"""

from __future__ import annotations

import mujoco
import numpy as np

_MARCH_EPS = 0.005  # meters to step past a self-hit before re-casting
_MARCH_MAX = 6  # a beam can cross at most a few self geoms (wheel, cabin, wheel)


def body_subtree_ids(model, root_body_id: int) -> frozenset[int]:
    """All body ids in the subtree rooted at root_body_id (the robot's own bodies)."""
    ids = {root_body_id}
    for body_id in range(root_body_id + 1, model.nbody):
        if model.body_parentid[body_id] in ids:
            ids.add(body_id)
    return frozenset(ids)


class Lidar:
    def __init__(
        self,
        n_beams: int = 16,
        fov_deg: float = 270.0,
        max_range: float = 5.0,
        noise_std: float = 0.0,
        dropout_prob: float = 0.0,
        site_name: str = "lidar_site",
    ):
        self.n_beams = n_beams
        self.fov = np.deg2rad(fov_deg)
        self.max_range = max_range
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.site_name = site_name
        angles = np.linspace(-self.fov / 2, self.fov / 2, n_beams)
        self._local_dirs = np.stack(
            [np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1
        )

    def _origin_and_dirs(self, model, data):
        site = data.site(self.site_name)
        origin = site.xpos.copy()
        rotmat = site.xmat.reshape(3, 3)
        dirs = self._local_dirs @ rotmat.T
        return origin, dirs

    def sense(
        self,
        model,
        data,
        exclude_body_id: int = -1,
        rng: np.random.Generator | None = None,
        exclude_body_ids: frozenset[int] | None = None,
    ) -> np.ndarray:
        """Return normalized ranges in [0, 1], where 1.0 means max_range / no hit."""
        origin, dirs = self._origin_and_dirs(model, data)
        geomid = np.zeros(self.n_beams, dtype=np.int32)
        dist = np.zeros(self.n_beams, dtype=np.float64)
        mujoco.mj_multiRay(
            model,
            data,
            pnt=origin,
            vec=dirs.reshape(-1).astype(np.float64),
            geomgroup=None,
            flg_static=1,
            bodyexclude=exclude_body_id,
            geomid=geomid,
            dist=dist,
            normal=None,
            nray=self.n_beams,
            cutoff=self.max_range,
        )
        if exclude_body_ids:
            for i in range(self.n_beams):
                if dist[i] >= 0 and model.geom_bodyid[geomid[i]] in exclude_body_ids:
                    dist[i] = self._march_past_self(
                        model, data, origin, dirs[i], float(dist[i]),
                        exclude_body_id, exclude_body_ids,
                    )
        ranges = np.where(dist < 0, self.max_range, dist).astype(np.float32)
        if rng is not None and self.noise_std > 0:
            ranges = ranges + rng.normal(0.0, self.noise_std, size=ranges.shape).astype(np.float32)
        if rng is not None and self.dropout_prob > 0:
            dropped = rng.random(ranges.shape) < self.dropout_prob
            ranges[dropped] = self.max_range
        ranges = np.clip(ranges, 0.0, self.max_range)
        return (ranges / self.max_range).astype(np.float32)

    def _march_past_self(
        self, model, data, origin, direction, self_hit_dist: float,
        exclude_body_id: int, exclude_body_ids: frozenset[int],
    ) -> float:
        """Re-cast a beam from just past a self-body hit until it lands on the world
        (or nothing). Returns the true distance from the original origin, or -1."""
        traveled = self_hit_dist
        geomid_out = np.zeros(1, dtype=np.int32)
        for _ in range(_MARCH_MAX):
            traveled += _MARCH_EPS
            if traveled >= self.max_range:
                return -1.0
            pnt = origin + traveled * direction
            d = mujoco.mj_ray(
                model, data, pnt, direction, None, 1, exclude_body_id, geomid_out
            )
            if d < 0:
                return -1.0
            if model.geom_bodyid[geomid_out[0]] not in exclude_body_ids:
                return traveled + d
            traveled += d
        return -1.0  # still inside own geometry after max steps; treat as no hit

    def debug_rays(self, model, data, exclude_body_id: int = -1,
                   exclude_body_ids: frozenset[int] | None = None):
        """Return (origin, unit_dirs, actual_ranges_in_meters) for viewer overlays."""
        origin, dirs = self._origin_and_dirs(model, data)
        norm = self.sense(
            model, data, exclude_body_id=exclude_body_id, exclude_body_ids=exclude_body_ids
        ) * self.max_range
        return origin, dirs, norm
