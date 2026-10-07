"""Ray-sampled isovist metrics, as computed by ray-casting tools.

Tools such as the Grasshopper isovist components or depthmapX cast ``N``
equally spaced rays and describe the isovist by the polygon through the hit
points. :func:`sampled_metrics` reproduces that description from the exact
depth function of an :class:`~pysovist.Isovist`, so the error of any ray count
can be measured against the exact value of the same scene. For a convex room
the polygon area converges as ``N^-2``. Where the boundary has occluding edges
it converges as ``N^-1``, because a ray that straddles a depth jump cuts a
triangle off or adds one.
"""

from __future__ import annotations

import numpy as np

from .results import Isovist


def sampled_metrics(iso: Isovist, n_rays: int, offset: float = 0.0) -> dict:
    """Metrics of the polygon through ``n_rays`` equally spaced hit points.

    ``offset`` is the angle of the first ray for a full turn (absolute radians).
    Radial statistics are population statistics over the ``n_rays`` depths.
    """
    ang, r = iso.radial(n_rays, offset=offset)
    pts = np.c_[r * np.cos(ang), r * np.sin(ang)]
    full = iso.fov >= 2 * np.pi - 1e-12
    if not full:
        pts = np.vstack([np.zeros((1, 2)), pts])
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perimeter = np.hypot(*(np.roll(pts, -1, axis=0) - pts).T).sum()
    mean = r.mean()
    return {
        "area": area,
        "perimeter": perimeter,
        "r_min": r.min(),
        "r_max": r.max(),
        "r_mean": mean,
        "r_var": r.var(),
        "r_std": r.std(),
        "r_mad": np.abs(r - mean).mean(),
        "compactness": 4 * np.pi * area / perimeter**2,
    }
