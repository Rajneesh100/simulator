"""2D lidar via MuJoCo raycasts, mounted at a robot's 'lidar_site'.

Configurable beam count / FOV / range, with optional Gaussian noise and beam dropout for
sim-to-real robustness training (idea.md: "sensors support configurable noise/dropout").
"""

from __future__ import annotations

import mujoco
import numpy as np


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

    def sense(self, model, data, exclude_body_id: int = -1, rng: np.random.Generator | None = None) -> np.ndarray:
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
        ranges = np.where(dist < 0, self.max_range, dist).astype(np.float32)
        if rng is not None and self.noise_std > 0:
            ranges = ranges + rng.normal(0.0, self.noise_std, size=ranges.shape).astype(np.float32)
        if rng is not None and self.dropout_prob > 0:
            dropped = rng.random(ranges.shape) < self.dropout_prob
            ranges[dropped] = self.max_range
        ranges = np.clip(ranges, 0.0, self.max_range)
        return (ranges / self.max_range).astype(np.float32)

    def debug_rays(self, model, data, exclude_body_id: int = -1):
        """Return (origin, unit_dirs, actual_ranges_in_meters) for viewer overlays."""
        origin, dirs = self._origin_and_dirs(model, data)
        norm = self.sense(model, data, exclude_body_id=exclude_body_id) * self.max_range
        return origin, dirs, norm
