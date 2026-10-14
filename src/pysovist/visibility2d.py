"""Obstacle visibility in 2D: which part of every wall an observer sees.

The exact isovist already knows the answer. Its wedge sweep cuts the view
into wedges inside which one wall is the nearest for every ray, so the part of
that wall between the two edge rays of the wedge is seen and nothing behind it
is. Adding up these pieces wall by wall gives the visible length of every
segment, its visible fraction, the angle it fills in the view, and the
visible subintervals along it. No result depends on a number of rays.

Definitions, for a segment from ``a`` to ``b`` of length ``L``:

- ``intervals``: the visible parts as parameter ranges ``[s0, s1]`` along the
  segment, with the point ``a + s (b - a)``. So ``s = 0`` is the first
  endpoint as given and ``s = 1`` the second.
- ``visible_length``: the summed length of the intervals, in metres.
- ``visible_fraction``: ``visible_length / L``.
- ``visible_angle``: the angle the visible parts fill at the observer, in
  radians. A ray-casting tool with ``n`` rays over the full turn hits the
  segment about ``n * visible_angle / (2 pi)`` times.
- ``visible``: ``visible_length > 0``. A wall seen exactly edge-on, or only
  at one point, is not visible.

The range circle and the field of view cut walls exactly, as in
[`isovist`][pysovist.isovist]. A ray that meets no wall within range adds
nothing, so an open scene with an infinite range is allowed. Where two walls
overlap (a wall drawn twice, or collinear walls that share a stretch), the
shared stretch counts once, for the wall listed first. Two walls overlap
where their distances from the observer agree to a relative 1e-9.

[`segment_ray_hits`][pysovist.visibility2d.segment_ray_hits] counts ray hits
per segment instead, as the Isovist component of the DeCodingSpaces
Grasshopper toolbox does (its output OV).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .isovist2d import (
    _ANGLE_TOL,
    TWO_PI,
    _angular_candidates,
    _circle_crossings,
    _point_segment_distance,
)
from .plan import Plan
from .results import Flags

_ON_WALL_POLICIES = ("nan", "raise")
_LAYOUTS = ("even", "decodingspaces")
_DIRECT_WORK = 200_000  # rays x segments below which every ray tests every segment
_TIE = 1e-9  # relative depth difference below which two walls are equally near
_SWEEP_SLACK = 1e-12  # slack on the segment parameter of a wedge's middle ray
_RAY_SLACK = 1e-9  # slack on the segment parameter of a sampled ray, so corner rays see both walls
_SNAP = 1e-12  # interval ends this close to 0 or 1 are set to 0 or 1
_DS_MIN_RAYS, _DS_MAX_RAYS = 10, 3000  # DeCodingSpaces uses its ray count only strictly between
_DS_FALLBACK_STEP = np.pi / 180  # and casts one ray per degree otherwise


@dataclass(frozen=True)
class SegmentVisibility:
    """Visible part of every segment, seen from one observer.

    All arrays follow the order of the input segments, including segments of
    zero length, which are never visible. With ``flags.on_wall`` the lengths,
    fractions and angles are NaN and no segment is visible.

    Attributes
    ----------
    origin : ndarray of shape (2,)
        Observer position.
    segments : ndarray of shape (n, 2, 2)
        The input segments, in the (x, y) plane.
    length : ndarray of shape (n,)
        Length of each segment, in metres.
    visible_length : ndarray of shape (n,)
        Visible length of each segment, in metres.
    visible_fraction : ndarray of shape (n,)
        ``visible_length / length``, and 0 for a segment of zero length.
    visible_angle : ndarray of shape (n,)
        Angle that the visible part of each segment fills at the observer, in
        radians.
    visible : ndarray of shape (n,), bool
        ``visible_length > 0``.
    intervals : tuple of n ndarrays of shape (k, 2)
        Visible parameter ranges ``[s0, s1]`` along each segment, sorted and
        disjoint, with ``s`` running from 0 at the first endpoint to 1 at the
        second.
    max_distance, fov, direction : float
        The arguments of the call.
    clearance : float
        Distance from the observer to the nearest segment, ``inf`` with none.
    flags : Flags
        ``on_wall`` when the observer stands closer than ``eps`` to a segment.
    """

    origin: np.ndarray
    segments: np.ndarray
    length: np.ndarray
    visible_length: np.ndarray
    visible_fraction: np.ndarray
    visible_angle: np.ndarray
    visible: np.ndarray
    intervals: tuple
    max_distance: float
    fov: float
    direction: float
    clearance: float
    flags: Flags = field(default_factory=Flags)

    def pieces(self) -> tuple[np.ndarray, np.ndarray]:
        """Visible pieces as coordinates.

        Returns
        -------
        index : ndarray of shape (m,)
            Segment of each piece.
        pieces : ndarray of shape (m, 2, 2)
            Endpoints of each piece, in the direction of its segment.
        """
        index = np.repeat(np.arange(len(self.intervals)), [len(iv) for iv in self.intervals])
        if len(index) == 0:
            return index, np.empty((0, 2, 2))
        s = np.concatenate(self.intervals)
        a = self.segments[index, 0]
        d = self.segments[index, 1] - a
        return index, np.stack([a + s[:, :1] * d, a + s[:, 1:] * d], axis=1)


def segment_visibility(
    occluders,
    origin,
    *,
    max_distance: float = np.inf,
    fov: float = TWO_PI,
    direction: float = 0.0,
    on_wall: str = "nan",
    eps: float = 1e-9,
) -> SegmentVisibility:
    """Exact visible length, fraction, angle and subintervals of every segment.

    Parameters
    ----------
    occluders : Plan or array_like of shape (n, 2, 2)
        Wall segments. The result has one entry per segment, in this order.
    origin : (x, y)
        Observer position.
    max_distance : float
        Range limit in metres. A wall crossing the range circle counts only
        inside it. With the default ``inf`` the range is unlimited, and rays
        that meet no wall add nothing.
    fov : float
        Field of view in radians, centred on ``direction``. Default: full turn.
    direction : float
        Viewing direction in radians, counter-clockwise from +x. Used only when
        ``fov`` is below a full turn.
    on_wall : {"nan", "raise"}
        What to do when the observer is closer than ``eps`` to a segment:
        return NaN lengths with ``flags.on_wall`` set, or raise a
        ``ValueError``.
    eps : float
        Distance below which the observer counts as standing on a wall.

    Returns
    -------
    SegmentVisibility
        Per segment: visible length, fraction, angle, flag and subintervals.

    Notes
    -----
    The wedges are those of [`isovist`][pysovist.isovist] with the same
    arguments, so the visible lengths add up to the part of the isovist
    perimeter that lies on walls.
    """
    if on_wall not in _ON_WALL_POLICIES:
        raise ValueError(f"on_wall must be one of {_ON_WALL_POLICIES}")
    segs, plan, keep = _segments(occluders)
    o = np.asarray(origin, dtype=float)[:2]
    R, fov = float(max_distance), float(min(fov, TWO_PI))
    length = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    n = len(segs)
    clearance, pieces = _visible_pieces(plan, o, R, fov, float(direction), eps, on_wall)

    def result(vis_len, vis_angle, intervals, flags):
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(length > 0, vis_len / np.where(length > 0, length, 1.0), 0.0)
        fraction = np.where(np.isnan(vis_len), np.nan, fraction)
        return SegmentVisibility(o, segs, length, vis_len, fraction, vis_angle,
                                 np.nan_to_num(vis_len, nan=0.0) > 0, intervals, R, fov,
                                 float(direction), clearance, flags)

    if pieces is None:
        nan = np.full(n, np.nan)
        empty = tuple(np.empty((0, 2)) for _ in range(n))
        return result(nan, nan.copy(), empty, Flags(on_wall=True))

    g, s0, s1, angle = pieces
    j = keep[g]
    merged = _merge(j, s0, s1, n)
    vis_len = np.array([(iv[:, 1] - iv[:, 0]).sum() for iv in merged]) * length
    vis_angle = np.bincount(j, weights=angle, minlength=n)
    return result(vis_len, vis_angle, merged, Flags())


def segment_visibility_field(
    occluders,
    origins,
    *,
    per_viewpoint: bool = False,
    n_jobs: int = 1,
    chunk: int = 256,
    **kw,
):
    """Visibility of every segment from many viewpoints, such as a path or a grid.

    Parameters
    ----------
    occluders : Plan or array_like of shape (n, 2, 2)
        Wall segments.
    origins : array_like of shape (m, 2)
        Viewpoints. Columns after the second are ignored.
    per_viewpoint : bool
        Also return the visible length of every segment from every viewpoint.
    n_jobs : int
        Worker processes that share the viewpoints. ``-1`` uses all cores, and
        1 runs in the calling process.
    chunk : int
        Viewpoints per task. For a given ``chunk``, the result does not depend on
        ``n_jobs``.
    **kw
        Keyword arguments of
        [`segment_visibility`][pysovist.visibility2d.segment_visibility],
        applied to every viewpoint.

    Returns
    -------
    DataFrame, or (DataFrame, DataFrame) with ``per_viewpoint``
        The first table has one row per input segment, in input order, with
        the columns ``x0, y0, x1, y1`` and ``length`` of the segment,
        ``seen_by`` (the number of viewpoints from which its visible length is
        positive), ``visible_length_sum`` (summed over the viewpoints),
        ``visible_length_mean`` (that sum divided by the number of viewpoints
        off the walls) and ``visible_fraction_mean`` (the mean divided by the
        length, 0 for a segment of zero length). Viewpoints closer than
        ``eps`` to a segment see nothing and are left out of the mean.
        ``attrs["viewpoints"]`` holds the number of viewpoints used and
        ``attrs["on_wall"]`` the number left out.

        The second table has one row per viewpoint and visible segment, with
        the columns ``viewpoint`` (the row in ``origins``), ``x``, ``y``,
        ``segment`` (the row in the first table), ``visible_length``,
        ``visible_fraction`` and ``visible_angle``.
    """
    segs, plan, keep = _segments(occluders)
    origins = np.asarray(origins, dtype=float)[:, :2]
    on_wall = kw.get("on_wall", "nan")
    if on_wall not in _ON_WALL_POLICIES:
        raise ValueError(f"on_wall must be one of {_ON_WALL_POLICIES}")
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    starts = list(range(0, len(origins), max(int(chunk), 1)))
    parts = [origins[i:i + chunk] for i in starts]
    args = ([plan] * len(parts), parts, starts, [kw] * len(parts), [per_viewpoint] * len(parts))
    if n_jobs <= 1 or len(parts) <= 1:
        results = list(map(_field_chunk, *args))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_field_chunk, *args))

    m = len(plan)
    total, seen = np.zeros(m), np.zeros(m, dtype=np.int64)
    used = on_wall_count = 0
    for part_total, part_seen, part_used, part_on_wall, _ in results:
        total += part_total
        seen += part_seen
        used += part_used
        on_wall_count += part_on_wall

    n = len(segs)
    length = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    vis_sum, seen_by = np.zeros(n), np.zeros(n, dtype=np.int64)
    vis_sum[keep], seen_by[keep] = total, seen
    vis_mean = vis_sum / used if used else np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_mean = np.where(length > 0, vis_mean / np.where(length > 0, length, 1.0), 0.0)
    table = pd.DataFrame({
        "x0": segs[:, 0, 0], "y0": segs[:, 0, 1], "x1": segs[:, 1, 0], "y1": segs[:, 1, 1],
        "length": length, "seen_by": seen_by, "visible_length_sum": vis_sum,
        "visible_length_mean": vis_mean, "visible_fraction_mean": frac_mean,
    })
    table.attrs["viewpoints"] = used
    table.attrs["on_wall"] = on_wall_count
    if not per_viewpoint:
        return table

    rows = [r[4] for r in results if len(r[4][0])]
    if rows:
        vp, g, vis_len, vis_angle = (np.concatenate(c) for c in zip(*rows, strict=True))
    else:
        vp, g = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        vis_len, vis_angle = np.empty(0), np.empty(0)
    j = keep[g]
    pairs = pd.DataFrame({
        "viewpoint": vp, "x": origins[vp, 0], "y": origins[vp, 1], "segment": j,
        "visible_length": vis_len, "visible_fraction": vis_len / length[j],
        "visible_angle": vis_angle,
    })
    return table, pairs


def segment_ray_hits(
    occluders,
    origin,
    n_rays: int = 360,
    *,
    max_distance: float = np.inf,
    fov: float = TWO_PI,
    direction: float = 0.0,
    layout: str = "even",
) -> np.ndarray:
    """Number of sampled rays whose first hit lies on each segment.

    This is the ray-sampled counterpart of
    [`segment_visibility`][pysovist.visibility2d.segment_visibility], and with
    ``layout="decodingspaces"`` it reproduces the obstacle visibility output
    (OV) of the DeCodingSpaces Isovist component.

    Parameters
    ----------
    occluders : Plan or array_like of shape (n, 2, 2)
        Wall segments. The result has one count per segment, in this order.
    origin : (x, y)
        Observer position.
    n_rays : int
        Ray count. Its meaning depends on ``layout``.
    max_distance : float
        Range limit in metres. A ray counts only when its first hit is closer.
    fov : float
        Field of view in radians, centred on ``direction``. Default: full turn.
    direction : float
        Viewing direction in radians, counter-clockwise from +x.
    layout : {"even", "decodingspaces"}
        ``"even"``: ``n_rays`` equally spaced rays, from ``direction`` around
        the full turn, or from one edge of the field of view to the other, as
        in [`Isovist.radial`][pysovist.Isovist.radial].
        ``"decodingspaces"``: the rays of the DeCodingSpaces Isovist
        component. The angular step is ``2 pi / n_rays`` for
        ``10 < n_rays < 3000`` and one degree otherwise. The rays are cast
        clockwise from ``direction + fov / 2`` while the angle, summed step
        by step in floating point as the component sums it, stays short of
        ``direction - fov / 2``. The rounding of that sum decides whether a
        ray lands on the far edge: a full turn at ``direction = pi / 2``
        casts ``n_rays + 1`` rays for 360 and 2,999 (the ray opposite to
        ``direction`` twice) and ``n_rays`` rays for 15 and 100. The
        component's default view direction (0, 1) is ``direction = pi / 2``.

    Returns
    -------
    ndarray of shape (n,), int
        Hits per segment.

    Notes
    -----
    A ray hits the nearest segment it meets. A ray that passes exactly
    through the point where one segment ends and the next starts, as at the
    corner of a closed outline, counts once, for the segment that starts
    there. Other exact ties go to the segment listed first. A ray that runs
    along a segment does not hit it.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}")
    segs, plan, keep = _segments(occluders)
    o = np.asarray(origin, dtype=float)[:2]
    ang = _ray_angles(n_rays, float(min(fov, TWO_PI)), float(direction), layout)
    seg = plan.segments - o
    t, j, _ = _nearest(np.c_[np.cos(ang), np.sin(ang)], seg[:, 0], seg[:, 1] - seg[:, 0],
                       _RAY_SLACK, prefer_start=True)
    hit = t < float(max_distance)
    out = np.zeros(len(segs), dtype=np.int64)
    out[keep] = np.bincount(j[hit], minlength=len(plan))
    return out


# --- internals ------------------------------------------------------------------------------------

def _segments(occluders):
    """Input segments, the plan of those with positive length, and their input rows."""
    if isinstance(occluders, Plan):
        return occluders.segments, occluders, np.arange(len(occluders))
    s = np.asarray(occluders, dtype=float)
    if s.size == 0:
        s = np.empty((0, 2, 2))
    s = np.ascontiguousarray(s.reshape(-1, 2, s.shape[-1])[:, :, :2])
    keep = np.flatnonzero(np.linalg.norm(s[:, 1] - s[:, 0], axis=1) > 1e-12)
    return s, Plan(s[keep]), keep


def _visible_pieces(plan, o, R, fov, direction, eps, on_wall):
    """Wall pieces of the exact isovist of one observer.

    Returns the clearance and, unless the observer stands on a wall, the
    arrays (plan index, s0, s1, angle) of every wedge bounded by a wall within
    range: the wall, the parameter range seen in the wedge and its angle.
    """
    full = fov >= TWO_PI - _ANGLE_TOL
    start = 0.0 if full else direction - fov / 2
    seg = plan.segments - o
    a, d = seg[:, 0], seg[:, 1] - seg[:, 0]
    dist = _point_segment_distance(a, d)
    clearance = float(dist.min()) if len(dist) else np.inf
    if clearance < eps:
        if on_wall == "raise":
            raise ValueError(f"observer {tuple(o)} is {clearance:.3g} m from a wall")
        return clearance, None

    # The events of the isovist's wedge sweep: walls beyond the range never bound it.
    near = np.flatnonzero(dist < R)
    a, d = a[near], d[near]
    verts = plan.vertices - o
    if np.isfinite(R):
        verts = verts[np.einsum("ij,ij->i", verts, verts) < R * R]
        verts = np.concatenate([verts, _circle_crossings(a, d, R)])
    ev = np.mod(np.arctan2(verts[:, 1], verts[:, 0]) - start, TWO_PI)
    ev = ev[(ev > _ANGLE_TOL) & (ev < fov - _ANGLE_TOL)]
    ev = np.unique(np.r_[0.0, ev, fov])
    ev = ev[np.r_[True, np.diff(ev) > _ANGLE_TOL]]
    ev[-1] = fov
    lo, hi = ev[:-1], ev[1:]

    mid = start + (lo + hi) / 2
    t, j, _ = _nearest(np.c_[np.cos(mid), np.sin(mid)], a, d, _SWEEP_SLACK)
    hit = t < R
    jh = j[hit]
    # Neighbouring wedges share an edge ray, so their pieces meet exactly. A full turn closes
    # on the ray it starts with.
    edge = start + ev
    if full:
        edge[-1] = start
    s0 = _param(edge[:-1][hit], a[jh], d[jh])
    s1 = _param(edge[1:][hit], a[jh], d[jh])
    lo_s = np.clip(np.minimum(s0, s1), 0.0, 1.0)
    hi_s = np.clip(np.maximum(s0, s1), 0.0, 1.0)
    lo_s[lo_s < _SNAP] = 0.0
    hi_s[hi_s > 1 - _SNAP] = 1.0
    return clearance, (near[jh], lo_s, hi_s, (hi - lo)[hit])


def _param(theta, a, d):
    """Parameter s of the point where the ray at angle theta meets the line a + s d."""
    ux, uy = np.cos(theta), np.sin(theta)
    return (a[:, 0] * uy - a[:, 1] * ux) / (ux * d[:, 1] - uy * d[:, 0])


def _merge(j, s0, s1, n):
    """Per segment, the union of the parameter ranges [s0, s1] as sorted disjoint intervals."""
    order = np.lexsort((s0, j))
    out = [[] for _ in range(n)]
    for k in order:
        lo, hi = s0[k], s1[k]
        if hi <= lo:
            continue
        cur = out[j[k]]
        if cur and lo <= cur[-1][1]:
            cur[-1][1] = max(cur[-1][1], hi)
        else:
            cur.append([lo, hi])
    return tuple(np.array(c, dtype=float).reshape(-1, 2) for c in out)


def _nearest(u, a, d, slack, prefer_start=False):
    """Nearest positive hit of rays ``t u`` with segments ``a + s d``.

    Returns the distance ``t`` (``inf`` when nothing is hit), the segment index
    and the parameter ``s`` of the hit. Hits within a relative ``_TIE`` of the
    nearest tie. With ``prefer_start``, a tie goes first to a segment on which
    the hit is not the second endpoint, so that a ray through a shared vertex
    counts for the segment that starts there. Remaining ties go to the lowest
    index.
    """
    k, n = len(u), len(a)
    if n == 0:
        return np.full(k, np.inf), np.zeros(k, dtype=np.int64), np.zeros(k)
    if k * n <= _DIRECT_WORK:
        cand = np.broadcast_to(np.arange(n), (k, n))
    else:
        cand = _angular_candidates(np.arctan2(u[:, 1], u[:, 0]), a, d)
    idx = np.maximum(cand, 0)
    aa, dd = a[idx], d[idx]
    denom = u[:, None, 0] * dd[..., 1] - u[:, None, 1] * dd[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (aa[..., 0] * dd[..., 1] - aa[..., 1] * dd[..., 0]) / denom
        s = (aa[..., 0] * u[:, None, 1] - aa[..., 1] * u[:, None, 0]) / denom
    valid = (cand >= 0) & (np.abs(denom) > 1e-300) & (t > 0) & (s >= -slack) & (s <= 1 + slack)
    t = np.where(valid, t, np.inf)
    t_min = t.min(axis=1)
    tie = valid & (t <= t_min[:, None] * (1 + _TIE))
    rank = idx.astype(np.int64)
    if prefer_start:
        rank = rank + n * (s > 1 - _TIE)
    col = np.argmin(np.where(tie, rank, 2 * n + 1), axis=1)
    rows = np.arange(k)
    return t[rows, col], idx[rows, col], s[rows, col]


def _ray_angles(n_rays, fov, direction, layout):
    """Absolute ray angles of a sampled view."""
    full = fov >= TWO_PI - _ANGLE_TOL
    if layout == "even":
        n = int(n_rays)
        if full:
            return direction + TWO_PI * np.arange(n) / n
        if n == 1:
            return np.array([direction])
        return direction - fov / 2 + fov * np.arange(n) / (n - 1)
    step = TWO_PI / n_rays if _DS_MIN_RAYS < n_rays < _DS_MAX_RAYS else _DS_FALLBACK_STEP
    return np.pi / 2 - _ds_sweep(step, fov / 2, np.pi / 2 - direction)


def _ds_sweep(step, half, bearing):
    """Bearings of the DeCodingSpaces rays, in radians clockwise from +y.

    The plugin starts at ``bearing - half``, the left edge of the view, and adds ``step`` to
    the angle while it stays below ``bearing + half``. The sum is rounded at every addition,
    so the rounding decides whether a ray lands on the far edge.
    """
    start, end = bearing - half, bearing + half
    if not (np.isfinite(start) and np.isfinite(end) and step > 0):
        raise ValueError(f"the view needs finite angles and a positive step, got bearing "
                         f"{bearing}, half view angle {half} and step {step}")
    n = int(np.floor((end - start) / step)) + 2
    a = np.add.accumulate(np.r_[start, np.full(n, step)])
    return a[a < end]


def _field_chunk(plan, origins, offset, kw, per_viewpoint):
    """Sums over a block of viewpoints, and the visible pairs if asked."""
    R = float(kw.get("max_distance", np.inf))
    fov = float(min(kw.get("fov", TWO_PI), TWO_PI))
    direction = float(kw.get("direction", 0.0))
    eps = float(kw.get("eps", 1e-9))
    on_wall = kw.get("on_wall", "nan")
    unknown = set(kw) - {"max_distance", "fov", "direction", "eps", "on_wall"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments {sorted(unknown)}")
    m = len(plan)
    plan_length = np.linalg.norm(plan.segments[:, 1] - plan.segments[:, 0], axis=1)
    total, seen = np.zeros(m), np.zeros(m, dtype=np.int64)
    used = on_wall_count = 0
    vp, g_all, len_all, ang_all = [], [], [], []
    for i, o in enumerate(origins):
        _, pieces = _visible_pieces(plan, o, R, fov, direction, eps, on_wall)
        if pieces is None:
            on_wall_count += 1
            continue
        used += 1
        g, s0, s1, angle = pieces
        vis = np.bincount(g, weights=s1 - s0, minlength=m) * plan_length
        total += vis
        seen += vis > 0
        if per_viewpoint:
            nz = np.flatnonzero(vis > 0)
            vp.append(np.full(len(nz), offset + i, dtype=np.int64))
            g_all.append(nz)
            len_all.append(vis[nz])
            ang_all.append(np.bincount(g, weights=angle, minlength=m)[nz])
    if vp:
        pairs = tuple(np.concatenate(c) for c in (vp, g_all, len_all, ang_all))
    else:
        none = np.empty(0, dtype=np.int64)
        pairs = (none, none, np.empty(0), np.empty(0))
    return total, seen, used, on_wall_count, pairs
