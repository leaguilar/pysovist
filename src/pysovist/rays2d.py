"""Single rays in a floor plan.

A ray leaves an origin at an angle and ends on the first wall it meets. The
distance, the hit point and the index of the wall are exact, computed with the
same intersection test as the exact isovist kernel, so the distance of a ray
equals the isovist depth ``Isovist.depth`` at its angle. Counting the rays
that end on each wall gives the obstacle visibility of ray-casting tools, such
as the IsovistRay component of DeCodingSpaces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .isovist2d import _first_hit
from .plan import Plan, _points2d

NO_HIT = -1
"""Wall index of a ray that meets no wall within its range."""


@dataclass(frozen=True)
class RayHits:
    """First hits of rays from ``m`` origins, ``k`` rays per origin.

    Attributes
    ----------
    origins : ndarray of shape (m, 2)
        Ray origins.
    angles : ndarray of shape (m, k)
        Ray directions, in radians counter-clockwise from +x.
    distance : ndarray of shape (m, k)
        Distance from the origin to the first wall, ``inf`` when no wall lies
        within ``max_distance``.
    point : ndarray of shape (m, k, 2)
        Hit point, NaN where nothing is hit.
    segment : ndarray of shape (m, k)
        Index of the wall hit in ``Plan.segments``, ``NO_HIT`` (-1) where
        nothing is hit.
    max_distance : float
        Range limit in metres, ``inf`` for none.
    """

    origins: np.ndarray
    angles: np.ndarray
    distance: np.ndarray
    point: np.ndarray
    segment: np.ndarray
    max_distance: float

    @property
    def hit(self) -> np.ndarray:
        """True where the ray ends on a wall within the range."""
        return self.segment != NO_HIT

    @property
    def depth(self) -> np.ndarray:
        """Visible depth along each ray: the distance, capped at ``max_distance``."""
        return np.minimum(self.distance, self.max_distance)


def cast_rays(occluders, origins, angles, *, max_distance: float = np.inf) -> RayHits:
    """First wall hit by each ray, with its distance and hit point.

    Parameters
    ----------
    occluders : Plan or array_like of shape (n, 2, 2)
        Wall segments. Wall indices refer to ``Plan(occluders).segments``,
        from which zero-length segments are removed.
    origins : array_like of shape (m, 2) or (2,)
        Ray origins. Columns after the second are ignored. An origin on a
        wall is allowed: that wall is not hit, see the notes.
    angles : float or array_like of shape (k,) or (m, k)
        Ray directions in radians, counter-clockwise from +x. A scalar or a
        1-D array gives the same rays to every origin. A 2-D array gives row
        ``i`` to origin ``i``: pass ``angles[:, None]`` for one direction per
        origin.
    max_distance : float
        Range limit in metres, positive (``inf`` for none). A wall farther
        than this along the ray is not hit, and a wall at exactly this
        distance is.

    Returns
    -------
    RayHits
        Distance, hit point and wall index of every ray, shape (m, k).

    Raises
    ------
    ValueError
        When an origin or an angle is not finite, when ``max_distance`` is not
        positive (NaN included), or when 2-D angles do not have one row per
        origin.

    Notes
    -----
    Walls are closed segments with no thickness. A ray through the shared
    endpoint of two walls ends there, on the wall at the smaller computed
    distance, and a ray through the free end of a wall hits that wall. A wall
    that lies along the ray is seen edge-on and is not hit. Only hits at a
    positive distance count.
    """
    plan = occluders if isinstance(occluders, Plan) else Plan(occluders)
    o = _points2d(origins, "origins")
    m = len(o)
    ang = np.asarray(angles, dtype=float)
    if not np.isfinite(ang).all():
        raise ValueError("angles must be finite, in radians counter-clockwise from +x")
    if ang.ndim <= 1:
        ang = np.broadcast_to(ang.reshape(1, -1), (m, ang.size))
    elif ang.ndim != 2 or len(ang) != m:
        raise ValueError(f"per-origin angles need one row per origin: got shape {ang.shape} "
                         f"for {m} origins")
    ang = np.ascontiguousarray(ang)
    R = float(max_distance)
    if not R > 0:
        raise ValueError(f"max_distance must be positive (inf for no limit), got {R}")
    k = ang.shape[1]

    distance = np.full((m, k), np.inf)
    segment = np.full((m, k), NO_HIT, dtype=np.int64)
    p0 = plan.segments[:, 0]
    d = plan.segments[:, 1] - p0
    if len(d) and k:
        for i in range(m):
            u = np.c_[np.cos(ang[i]), np.sin(ang[i])]
            # A ray almost parallel to a wall meets its line beyond the float range: a miss.
            with np.errstate(over="ignore"):
                t, j = _first_hit(u, p0 - o[i], d)
            ok = np.isfinite(t) & (t <= R)
            distance[i, ok] = t[ok]
            segment[i, ok] = j[ok]
    unit = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    with np.errstate(invalid="ignore"):
        point = o[:, None, :] + distance[..., None] * unit
    point[segment == NO_HIT] = np.nan
    return RayHits(o, ang, distance, point, segment, R)
