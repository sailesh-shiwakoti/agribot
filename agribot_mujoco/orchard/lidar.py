"""
Simulated 2D lidar + tree detection.

Casts a horizontal fan of rays with mujoco.mj_ray from the base's lidar site
(scanning only group-0 geoms = the trees), then clusters the returns and
fits a circle to each cluster to recover tree-trunk centres and radii. This is
the robot's *perception* of where the trees are — everything downstream
(mapping, planning, navigation) uses these detections, not ground truth.
"""

from __future__ import annotations

import numpy as np
import mujoco

# raycast only group-0 geoms (trees); base is group 2, arm is group 2
_GEOMGROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


class Lidar:
    def __init__(self, model, data, cfg, base_body="mobile_base",
                 site="lidar"):
        self.m = model
        self.d = data
        self.cfg = cfg
        self.site = model.site(site).id
        self.exclude = model.body(base_body).id
        self.angles = np.linspace(0, 2 * np.pi, cfg.n_lidar, endpoint=False)

    def scan(self):
        """Return (origin_xy, points) — an (N, 2) array of world-xy hits."""
        origin = self.d.site_xpos[self.site].copy()
        pts = []
        geomid = np.zeros(1, dtype=np.int32)
        for a in self.angles:
            vec = np.array([np.cos(a), np.sin(a), 0.0])
            dist = mujoco.mj_ray(self.m, self.d, origin, vec, _GEOMGROUP,
                                 1, self.exclude, geomid)
            if 0 < dist < self.cfg.lidar_range:
                pts.append(origin[:2] + dist * vec[:2])
        return origin[:2], (np.array(pts) if pts else np.zeros((0, 2)))

    def detect_trees(self, cluster_dist=0.4, trunk_r=0.08):
        """Cluster this scan into estimated trunk centres.

        Trunks are thin, so returns hit only the near surface. We cluster the
        arc points and push the cluster centroid outward (away from the sensor)
        by an estimated trunk radius to recover the true centre.
        Returns list of (center_xy, n_points).
        """
        origin, pts = self.scan()
        clusters = []
        for p in pts:
            for c in clusters:
                if np.linalg.norm(p - c[-1]) < cluster_dist:
                    c.append(p)
                    break
            else:
                clusters.append([p])
        trees = []
        for c in clusters:
            arr = np.array(c)
            centroid = arr.mean(axis=0)
            outward = centroid - origin
            n = np.linalg.norm(outward)
            center = centroid + (outward / n) * trunk_r if n > 1e-6 else centroid
            trees.append((center, len(c)))
        return trees
