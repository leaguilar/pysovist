"""The 2D "Isovist" component of DeCodingSpaces, reproduced from an exact isovist.

DeCodingSpaces (release 2021.10.11) is a Grasshopper toolbox for Rhino. Its 2D
"Isovist" component casts rays from each viewpoint against obstacle lines and
reports 17 Isovist Properties computed from the rays and from the polygon
through their end points. This module samples the exact depth function of
[`isovist`][pysovist.isovist] along the plugin's rays and evaluates the
plugin's formulas, so that a Grasshopper definition and a Python script give
the same numbers.

Rays
:   The Precision input ``Pr`` becomes an angular step of ``2 pi / Pr`` when
    ``10 < Pr < 3000`` and of one degree otherwise, without a warning. A
    non-integer ``Pr`` is used as it is. The rays sweep clockwise, from the
    left edge of the view (view direction plus half the view angle) towards
    the right edge. The plugin adds the step to the angle until the angle
    reaches the right edge, so the rounding of that sum decides whether a ray
    lands on the right edge. A full view at the default direction has
    ``Pr + 1`` rays for ``Pr = 360`` and ``2999`` (the last repeats the first)
    and ``Pr`` rays for ``Pr = 15`` and ``100``. With the default direction
    (0, 1, 0), the first ray of a full view points along -y.

Ray ends
:   A ray ends at the first obstacle line it meets, or at the view range when
    it meets none.

Partial views
:   A view angle below 360 degrees closes the polygon through the viewpoint.
    The viewpoint then also counts as a ray of length 0: it is the first
    vertex of the polygon and enters every statistic over the rays, so Min
    Radial is 0.

Sums over rays
:   Standard Deviation, Variance and Skewness are sums over all rays (a
    repeated ray included) of the absolute, squared and cubed deviation from
    the Mean Radial. They grow with the ray count.

Occlusivity
:   The sum of the polygon edges, in the order of the sweep, that end on a ray
    hitting a line while the previous ray misses every line (or is the
    viewpoint), or hits another line that shares no end point with it and
    lies more than ``DEPTH_STEP_TOL`` (1 cm) nearer or farther. An edge from
    a hit to a miss does not count.

Drift Angle
:   The signed angle at the viewpoint from the view direction, read as a
    point, to the drift vector (centroid minus viewpoint), read as a point,
    counter-clockwise positive, in (-pi, pi]. It depends on where the scene
    lies in the plane. pysovist's ``drift_angle`` is the direction of the
    drift.

Inputs mirror the component (a view angle in degrees, a direction vector),
which departs from the radians used elsewhere in pysovist. Agreement with the
plugin covers full and partial views for view directions with a non-negative
x component, non-integer Precision, T-junctions and walls 1 mm apart. A
direction with a negative x component follows the same rules with the bearing
``atan2(x, y)`` in (-pi, pi].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from ..isovist2d import isovist
from ..plan import Plan
from ..rays2d import cast_rays
from ..visibility2d import _ds_sweep

DS_PROPERTY_NAMES = (
    "Area", "Distance Weighted Area", "Perimeter", "Compactness", "Circularity",
    "Convex Deficiency", "Occlusivity", "Min Radial", "Max Radial", "Mean Radial",
    "Standard Deviation", "Variance", "Skewness", "Dispersion", "Elongation",
    "Drift Magnitude", "Drift Angle",
)
"""The plugin's Isovist Properties (IP 0 to 16), in its order and spelling."""

DS_DEFAULTS = {
    "view_direction": (0.0, 1.0, 0.0),
    "view_range": 1000.0,
    "view_angle_deg": 360.0,
    "precision": 360,
}
"""The component's default inputs, as keyword arguments of
[`ds_isovist`][pysovist.compat.decodingspaces.ds_isovist]."""

ONE_DEGREE = 0.017453292519943295
"""Fallback angular step in radians, as stored in the plugin."""

CONNECT_TOL = 1e-9
"""Distance in metres below which two line end points count as shared (for Occlusivity)."""

DEPTH_STEP_TOL = 0.01
"""Depth step in metres up to which a change between two lines that share no end point
continues the boundary (for Occlusivity). The plugin's own threshold lies between the 2.2 mm
step it ignores at a T-junction and the 7.2 cm step it counts there."""


def ds_angular_step(precision: float) -> float:
    """Angular step between the plugin's rays, in radians.

    Parameters
    ----------
    precision : float
        The component's Precision input ``Pr``.

    Returns
    -------
    float
        ``2 pi / Pr`` when ``10 < Pr < 3000``, otherwise one degree (360 rays
        over a full view), as in the component, for 0, negative and infinite
        values too.

    Raises
    ------
    ValueError
        If ``precision`` is NaN.
    """
    pr = float(precision)
    if math.isnan(pr):
        raise ValueError("precision is NaN: pass the component's Precision, a number of rays "
                         "per turn such as 360")
    return 2 * math.pi / pr if 10.0 < pr < 3000.0 else ONE_DEGREE


def _half_view(view_angle_deg: float) -> float:
    """Half the field of view in radians. Values outside (0, 360] mean a full view."""
    va = float(view_angle_deg)
    if math.isnan(va):
        raise ValueError("view_angle_deg is NaN: pass the component's Viewing Angle in degrees, "
                         "360 for a full view")
    if not 0.0 < va <= 360.0:
        va = 360.0
    return va * math.pi / 180.0 / 2.0


def _is_full(view_angle_deg: float) -> bool:
    va = float(view_angle_deg)
    return not 0.0 < va < 360.0


def _direction(view_direction) -> np.ndarray:
    v = np.asarray(view_direction, dtype=float).ravel()[:2]
    if len(v) < 2 or not np.isfinite(v).all() or not v.any():
        raise ValueError(f"view_direction must be a finite vector with a non-zero x or y "
                         f"component, got {view_direction!r}")
    return v


def ds_ray_angles(precision: float = 360, *, view_angle_deg: float = 360.0,
                  view_direction=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Directions of the plugin's rays, in radians counter-clockwise from +x.

    Parameters
    ----------
    precision : float
        The component's Precision input, turned into a step by
        [`ds_angular_step`][pysovist.compat.decodingspaces.ds_angular_step].
    view_angle_deg : float
        The component's Viewing Angle in degrees. Values outside (0, 360] give
        a full view.
    view_direction : (x, y) or (x, y, z)
        The component's Viewing Direction. Only its angle matters here.

    Raises
    ------
    ValueError
        If ``precision`` or ``view_angle_deg`` is NaN, or ``view_direction``
        is not finite or has no x or y component.

    Returns
    -------
    ndarray
        The angles in the plugin's order: from ``direction + fov / 2`` down in
        steps of the angular step while above ``direction - fov / 2``, with
        the step summed in floating point as the plugin does. The sequence
        decreases monotonically and may leave the range (-pi, pi]. A full view
        at Precision ``Pr`` gives ``Pr`` or ``Pr + 1`` angles, see the module
        notes.
    """
    d = _direction(view_direction)
    bearing = math.atan2(d[0], d[1])  # clockwise from +y, as the plugin measures it
    return math.pi / 2 - _ds_sweep(ds_angular_step(precision), _half_view(view_angle_deg),
                                   bearing)


@dataclass(frozen=True)
class DSIsovist:
    """One viewpoint as the DeCodingSpaces "Isovist" component describes it.

    Attributes
    ----------
    origin : ndarray of shape (2,)
        Viewpoint.
    angles : ndarray of shape (m,)
        Ray directions, radians counter-clockwise from +x, in the plugin's
        order (clockwise from the left edge of the view).
    depths : ndarray of shape (m,)
        Exact visible depth along each ray, capped at the view range. NaN when
        the viewpoint lies on a line.
    lines : ndarray of shape (m,)
        Index into ``Plan(occluders).segments`` of the line each ray hits, -1
        for a ray that reaches the view range.
    points : ndarray of shape (m, 2)
        Ray end points (the plugin's radials end there).
    polygon : ndarray of shape (k, 2)
        The plugin's isovist polygon, as an open ring in the plugin's order.
        For a full view it is ``points``. For a narrower view it starts at the
        viewpoint, which the radial statistics count as a ray of length 0.
    properties : dict
        The 17 Isovist Properties keyed by
        [`DS_PROPERTY_NAMES`][pysovist.compat.decodingspaces.DS_PROPERTY_NAMES].
    """

    origin: np.ndarray
    angles: np.ndarray
    depths: np.ndarray
    lines: np.ndarray
    points: np.ndarray
    polygon: np.ndarray
    properties: dict = field(default_factory=dict)


def ds_isovist(occluders, viewpoint, *, view_direction=(0.0, 1.0, 0.0), view_range: float = 1000.0,
               view_angle_deg: float = 360.0, precision: float = 360) -> DSIsovist:
    """The plugin's rays, polygon and 17 Isovist Properties at one viewpoint.

    Parameters
    ----------
    occluders : Plan or array_like of shape (n, 2, 2)
        Obstacle lines (the component's OL input).
    viewpoint : (x, y) or (x, y, z)
        Vantage point (VP).
    view_direction : (x, y) or (x, y, z)
        Viewing Direction (VD), a vector. The rays are centred on it, and the
        Drift Angle uses it as a point.
    view_range : float
        Viewing Range (VR) in metres. Values of 0 and below mean 1000.
    view_angle_deg : float
        Viewing Angle (VA) in degrees. Values outside (0, 360] mean 360.
    precision : float
        Precision (Pr), see
        [`ds_angular_step`][pysovist.compat.decodingspaces.ds_angular_step].

    Returns
    -------
    DSIsovist
        Rays, depths, hit lines, polygon and properties. A viewpoint closer
        than 1e-9 m to a line gives NaN depths and NaN properties.

    Raises
    ------
    ValueError
        If the viewpoint is not finite, ``view_range``, ``view_angle_deg`` or
        ``precision`` is NaN, ``view_direction`` is not a finite vector with an
        x or y component, a partial view holds a single ray (its polygon has
        no area), or ``view_range`` is infinite and a ray meets no line.
    """
    plan = occluders if isinstance(occluders, Plan) else Plan(occluders)
    o = np.asarray(viewpoint, dtype=float).ravel()[:2]
    if not np.isfinite(o).all():
        raise ValueError(f"viewpoint must be finite, got {viewpoint!r}")
    R = float(view_range)
    if math.isnan(R):
        raise ValueError("view_range is NaN: pass the Viewing Range in metres, 1000 by default")
    R = R if R > 0 else DS_DEFAULTS["view_range"]
    dvec = _direction(view_direction)
    ang = ds_ray_angles(precision, view_angle_deg=view_angle_deg, view_direction=dvec)
    full = _is_full(view_angle_deg)
    if not full and len(ang) < 2:
        raise ValueError(f"a view angle of {float(view_angle_deg):g} degrees holds one ray at "
                         f"precision {precision}, and the polygon through it and the viewpoint "
                         f"has no area: widen view_angle_deg or raise precision")
    u = np.c_[np.cos(ang), np.sin(ang)]

    hits = cast_rays(plan, o, ang, max_distance=R)
    if math.isinf(R):
        if not hits.hit.all():
            raise ValueError("view_range is infinite and a ray meets no line, so the polygon has "
                             "no end there: pass a finite range (the component's default is "
                             "1000)")
        # Every ray ends on a line: any finite range beyond the farthest hit gives the same rays.
        R = 2.0 * float(hits.distance.max()) + 1.0 if len(ang) else 1.0
    iso = isovist(plan, o, max_distance=R)
    if iso.flags.on_wall:
        depths = np.full(len(ang), np.nan)
        lines = np.full(len(ang), -1)
        pts = np.full((len(ang), 2), np.nan)
        poly = pts if full else np.vstack([o, pts])
        return DSIsovist(o, ang, depths, lines, pts, poly, dict.fromkeys(DS_PROPERTY_NAMES, np.nan))

    depths = iso.depth(ang)
    lines = np.where(hits.distance[0] < R, hits.segment[0], -1)
    pts = o + depths[:, None] * u
    if full:
        poly, radii, ring_lines = pts, depths, lines
    else:  # the viewpoint comes first, as a ray of length 0 that hits nothing
        poly, radii, ring_lines = np.vstack([o, pts]), np.r_[0.0, depths], np.r_[-1, lines]
    props = _properties(o, dvec, radii, ring_lines, poly, plan.segments)
    return DSIsovist(o, ang, depths, lines, pts, poly, props)


def ds_isovist_properties(occluders, viewpoints, *, view_direction=(0.0, 1.0, 0.0),
                          view_range: float = 1000.0, view_angle_deg: float = 360.0,
                          precision: float = 360) -> pd.DataFrame:
    """The 17 Isovist Properties for many viewpoints, one row each.

    Parameters
    ----------
    occluders : Plan or array_like of shape (n, 2, 2)
        Obstacle lines.
    viewpoints : array_like of shape (m, 2) or (m, 3)
        Vantage points. Columns after the second are ignored.
    view_direction : array_like of shape (2,), (3,) or (k, 2 or 3)
        One direction for every viewpoint, or a list of directions. As in the
        component, a list as long as ``viewpoints`` gives each viewpoint its
        own direction, and any other list gives every viewpoint the first.
    view_range, view_angle_deg, precision
        As in [`ds_isovist`][pysovist.compat.decodingspaces.ds_isovist].

    Returns
    -------
    DataFrame
        Columns ``x, y`` and the names of
        [`DS_PROPERTY_NAMES`][pysovist.compat.decodingspaces.DS_PROPERTY_NAMES],
        one row per viewpoint in the order of ``viewpoints``.
    """
    plan = occluders if isinstance(occluders, Plan) else Plan(occluders)
    vps = np.atleast_2d(np.asarray(viewpoints, dtype=float))[:, :2]
    dirs = np.asarray(view_direction, dtype=float)
    if dirs.ndim == 1:
        dirs = dirs[None]
    if len(dirs) != len(vps):
        dirs = np.repeat(dirs[:1], len(vps), axis=0)
    rows = []
    for o, d in zip(vps, dirs, strict=True):
        p = ds_isovist(plan, o, view_direction=d, view_range=view_range,
                       view_angle_deg=view_angle_deg, precision=precision).properties
        rows.append([o[0], o[1], *(p[k] for k in DS_PROPERTY_NAMES)])
    return pd.DataFrame(rows, columns=["x", "y", *DS_PROPERTY_NAMES])


def _connected(segments, a, b, tol=CONNECT_TOL) -> np.ndarray:
    """Whether segments ``a[i]`` and ``b[i]`` share an end point."""
    ea, eb = segments[a], segments[b]
    gap = np.linalg.norm(ea[:, :, None, :] - eb[:, None, :, :], axis=-1)
    return gap.reshape(len(a), 4).min(axis=1) < tol


def _properties(o, dvec, r, lines, ring, segments) -> dict:
    """The 17 properties from the ring of the plugin's polygon, its radii and hit lines."""
    x, y = ring[:, 0], ring[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    signed = 0.5 * cross.sum()
    area = abs(signed)
    perimeter = np.hypot(xn - x, yn - y).sum()
    centroid = np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * signed)
    try:
        hull = ConvexHull(ring).volume
    except QhullError:
        hull = np.nan

    # Occluding edges, in the order of the sweep: onto a hit from a miss (or the viewpoint), or
    # from a line that shares no end point with it and lies at another depth.
    a, b = lines[:-1], lines[1:]
    occ = (b >= 0) & (b != a)
    both = occ & (a >= 0)
    if both.any():
        idx = np.nonzero(both)[0]
        step = np.abs(r[idx + 1] - r[idx]) > DEPTH_STEP_TOL
        occ[idx] = step & ~_connected(segments, a[idx], b[idx])
    edges = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    occlusivity = float(edges[occ].sum())

    mean = r.mean()
    dev = r - mean
    sad = np.abs(dev).sum()
    r_max = r.max()

    drift = centroid - o
    p1, p2 = drift - o, dvec - o  # both read as points, seen from the viewpoint
    drift_angle = math.atan2(p2[0] * p1[1] - p2[1] * p1[0], p2 @ p1)

    values = (
        area, area, perimeter, 4 * math.pi * area / perimeter**2, math.pi * mean**2 / area,
        area / hull, occlusivity, r.min(), r_max, mean, sad, (dev**2).sum(), (dev**3).sum(),
        mean - sad, mean / r_max, float(np.hypot(*drift)), drift_angle,
    )
    return {k: float(v) for k, v in zip(DS_PROPERTY_NAMES, values, strict=True)}
