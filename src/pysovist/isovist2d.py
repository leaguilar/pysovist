"""Exact 2D isovists from wall segments ("wedge sweep").

Seen from the observer, the angles of all plan vertices (segment endpoints
and crossing points) and of all crossings between walls and the range circle
cut the full turn into wedges. Inside one wedge no wall starts, ends, crosses
another wall or leaves the range, so the nearest wall is the same for every
ray of the wedge. One ray per wedge identifies it, and every quantity of the
wedge then has a closed form. No result depends on a number of rays.
"""

from __future__ import annotations

import numpy as np

from .metrics2d import nan_metrics, wedge_metrics
from .plan import Plan
from .results import ARC, SEGMENT, Flags, Isovist

TWO_PI = 2 * np.pi
_ANGLE_TOL = 1e-12  # wedges narrower than this are merged into their neighbours
_ON_WALL_POLICIES = ("nan", "raise")


def isovist(
    occluders,
    origin,
    *,
    max_distance: float = np.inf,
    fov: float = TWO_PI,
    direction: float = 0.0,
    on_wall: str = "nan",
    eps: float = 1e-9,
) -> Isovist:
    """Exact isovist of one observer.

    Parameters
    ----------
    occluders : Plan or array-like of shape (n, 2, 2)
        Wall segments.
    origin : (x, y)
        Observer position.
    max_distance : float
        Range limit in metres. Rays that meet no wall within this distance end
        on the range circle. With the default ``inf``, an escaping ray makes
        the isovist unbounded (area ``inf``, ``flags.unbounded``).
    fov : float
        Field of view in radians, centred on ``direction``. Default: full turn.
    direction : float
        Viewing direction in radians, counter-clockwise from +x.
    on_wall : {"nan", "raise"}
        What to do when the observer is closer than ``eps`` to a wall.
    eps : float
        Distance below which the observer counts as standing on a wall.
    """
    if on_wall not in _ON_WALL_POLICIES:
        raise ValueError(f"on_wall must be one of {_ON_WALL_POLICIES}")
    plan = occluders if isinstance(occluders, Plan) else Plan(occluders)
    o = np.asarray(origin, dtype=float)[:2]
    R = float(max_distance)
    fov = float(min(fov, TWO_PI))
    full = fov >= TWO_PI - _ANGLE_TOL
    start = 0.0 if full else float(direction) - fov / 2

    seg = plan.segments - o
    a, d = seg[:, 0], seg[:, 1] - seg[:, 0]
    dist = _point_segment_distance(a, d)
    clearance = float(dist.min()) if len(dist) else np.inf

    def result(theta0, theta1, kind, p, phi, flags, metrics):
        return Isovist(o, R, fov, float(direction), theta0, theta1, kind, p, phi,
                       clearance, flags, metrics)

    if clearance < eps:
        if on_wall == "raise":
            raise ValueError(f"observer {tuple(o)} is {clearance:.3g} m from a wall")
        empty = np.empty(0)
        return result(empty, empty, empty.astype(np.int8), empty, empty,
                      Flags(on_wall=True), nan_metrics())

    # Walls beyond the range never bound the isovist.
    near = dist < R
    a, d = a[near], d[near]
    verts = plan.vertices - o
    if np.isfinite(R):
        verts = verts[np.einsum("ij,ij->i", verts, verts) < R * R]
        verts = np.concatenate([verts, _circle_crossings(a, d, R)])

    # Event angles, measured from the start of the field of view.
    ev = np.mod(np.arctan2(verts[:, 1], verts[:, 0]) - start, TWO_PI)
    ev = ev[(ev > _ANGLE_TOL) & (ev < fov - _ANGLE_TOL)]
    ev = np.unique(np.r_[0.0, ev, fov])
    ev = ev[np.r_[True, np.diff(ev) > _ANGLE_TOL]]
    ev[-1] = fov
    lo, hi = ev[:-1], ev[1:]

    # One ray per wedge identifies its nearest wall.
    mid = start + (lo + hi) / 2
    u = np.c_[np.cos(mid), np.sin(mid)]
    t, j = _first_hit(u, a, d)
    hit = t < R
    if not np.isfinite(R) and not hit.all():
        empty = np.empty(0)
        return result(empty, empty, empty.astype(np.int8), empty, empty,
                      Flags(unbounded=True), nan_metrics(area=np.inf, perimeter=np.inf))

    kind = np.where(hit, SEGMENT, ARC).astype(np.int8)
    p = np.zeros(len(lo))
    phi = np.zeros(len(lo))
    if hit.any():
        # Take the direction of the wall's normal from the wall itself, not from the foot
        # point: the foot point loses precision when the wall is seen almost edge-on.
        jh = j[hit]
        aj, dj = a[jh], d[jh]
        normal = np.c_[dj[:, 1], -dj[:, 0]] / np.hypot(dj[:, 0], dj[:, 1])[:, None]
        side = np.einsum("ij,ij->i", normal, aj)
        normal[side < 0] *= -1
        p[hit] = np.abs(side)
        phi[hit] = np.arctan2(normal[:, 1], normal[:, 0])
    theta0, theta1 = start + lo, start + hi
    metrics = wedge_metrics(theta0, theta1, kind, p, phi, R, fov, full)
    return result(theta0, theta1, kind, p, phi, Flags(), metrics)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _point_segment_distance(a, d):
    """Distance from the origin to segments ``a + s d``, s in [0, 1]."""
    if len(a) == 0:
        return np.empty(0)
    s = np.clip(-np.einsum("ij,ij->i", a, d) / np.einsum("ij,ij->i", d, d), 0.0, 1.0)
    q = a + s[:, None] * d
    return np.hypot(q[:, 0], q[:, 1])


def _circle_crossings(a, d, R):
    """Points where segments cross the circle of radius R around the origin."""
    A = np.einsum("ij,ij->i", d, d)
    B = 2 * np.einsum("ij,ij->i", a, d)
    C = np.einsum("ij,ij->i", a, a) - R * R
    disc = B * B - 4 * A * C
    ok = disc > 0
    out = []
    sq = np.sqrt(np.where(ok, disc, 0.0))
    for sign in (-1.0, 1.0):
        s = (-B + sign * sq) / (2 * A)
        m = ok & (s > 0) & (s < 1)
        out.append(a[m] + s[m, None] * d[m])
    return np.concatenate(out) if out else np.empty((0, 2))


def _first_hit(u, a, d):
    """Nearest positive hit of rays ``t u`` with segments ``a + s d``.

    Returns the distance ``t`` (``inf`` when nothing is hit) and the segment index.
    """
    k = len(u)
    if len(a) == 0:
        return np.full(k, np.inf), np.zeros(k, dtype=int)
    denom = u[:, None, 0] * d[None, :, 1] - u[:, None, 1] * d[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(a, d)[None, :] / denom
        s = (a[None, :, 0] * u[:, None, 1] - a[None, :, 1] * u[:, None, 0]) / denom
    valid = (np.abs(denom) > 1e-300) & (t > 0) & (s >= -1e-12) & (s <= 1 + 1e-12)
    t = np.where(valid, t, np.inf)
    j = np.argmin(t, axis=1)
    return t[np.arange(k), j], j
