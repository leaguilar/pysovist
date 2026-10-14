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
    ``10 < Pr < 3000`` and of one degree otherwise, without a warning. The rays
    sweep counter-clockwise from the view direction minus half the view angle
    to the view direction plus half the view angle, both ends included. A full
    view therefore has ``Pr + 1`` rays, and the last repeats the first. With
    the default direction (0, 1, 0), the first ray points along -y.

Ray ends
:   A ray ends at the first obstacle line it meets, or at the view range when
    it meets none.

Sums over rays
:   Standard Deviation, Variance and Skewness are sums over all rays (the
    repeated ray included) of the absolute, squared and cubed deviation from
    the Mean Radial. They grow with the ray count.

Occlusivity
:   The sum of the polygon edges from a ray that hits a line to the next ray
    (counter-clockwise) when that ray misses every line, or hits another line
    that shares no end point with the first. An edge from a miss to a hit does
    not count.

Drift Angle
:   The angle at the viewpoint between two points: the drift vector (centroid
    minus viewpoint) read as a point, and the view direction read as a point.
    It depends on where the scene lies in the plane. pysovist's
    ``drift_angle`` is the direction of the drift.

Inputs mirror the component (a view angle in degrees, a direction vector),
which departs from the radians used elsewhere in pysovist. Agreement with the
plugin covers full views under the default direction. Four behaviours extend
the same rules beyond it, without a plugin reference value: the first ray
points opposite any view direction, a view angle below 360 degrees is swept
from edge to edge with the polygon closed through the viewpoint, lines that
touch at a T-junction count as not connected, and the Drift Angle carries no
sign.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from ..isovist2d import _first_hit, isovist
from ..plan import Plan

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
        over a full view).
    """
    pr = float(precision)
    return 2 * math.pi / pr if 10.0 < pr < 3000.0 else ONE_DEGREE


def _view_angle(view_angle_deg: float) -> float:
    """Field of view in radians. Values outside (0, 360] mean a full view."""
    va = float(view_angle_deg)
    return 2 * math.pi * (va / 360.0 if 0.0 < va <= 360.0 else 1.0)


def _direction(view_direction) -> np.ndarray:
    v = np.asarray(view_direction, dtype=float).ravel()
    return v[:2]


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

    Returns
    -------
    ndarray
        Angles from ``direction - fov / 2`` to ``direction + fov / 2`` in steps
        of the angular step, both ends included. A full view at Precision ``Pr``
        gives ``Pr + 1`` angles.
    """
    step = ds_angular_step(precision)
    fov = _view_angle(view_angle_deg)
    d = _direction(view_direction)
    start = math.atan2(d[1], d[0]) - fov / 2
    k = math.floor(fov / step + 1e-9)
    return start + np.arange(k + 1) * step


@dataclass(frozen=True)
class DSIsovist:
    """One viewpoint as the DeCodingSpaces "Isovist" component describes it.

    Attributes
    ----------
    origin : ndarray of shape (2,)
        Viewpoint.
    angles : ndarray of shape (m,)
        Ray directions, radians counter-clockwise from +x.
    depths : ndarray of shape (m,)
        Exact visible depth along each ray, capped at the view range. NaN when
        the viewpoint lies on a line.
    lines : ndarray of shape (m,)
        Index into ``Plan(occluders).segments`` of the line each ray hits, -1
        for a ray that reaches the view range.
    points : ndarray of shape (m, 2)
        Ray end points (the plugin's radials end there).
    polygon : ndarray of shape (k, 2)
        The plugin's isovist polygon, as an open ring. For a full view it is
        ``points``. For a narrower view it starts at the viewpoint.
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
        Rays, depths, hit lines, polygon and properties. A viewpoint on a line
        gives NaN depths and NaN properties.
    """
    plan = occluders if isinstance(occluders, Plan) else Plan(occluders)
    o = np.asarray(viewpoint, dtype=float).ravel()[:2]
    R = float(view_range) if float(view_range) > 0 else DS_DEFAULTS["view_range"]
    dvec = _direction(view_direction)
    ang = ds_ray_angles(precision, view_angle_deg=view_angle_deg, view_direction=dvec)
    full = _view_angle(view_angle_deg) >= 2 * math.pi
    u = np.c_[np.cos(ang), np.sin(ang)]

    iso = isovist(plan, o, max_distance=R)
    if iso.flags.on_wall:
        depths = np.full(len(ang), np.nan)
        lines = np.full(len(ang), -1)
        pts = np.full((len(ang), 2), np.nan)
        poly = pts if full else np.vstack([o, pts])
        return DSIsovist(o, ang, depths, lines, pts, poly, dict.fromkeys(DS_PROPERTY_NAMES, np.nan))

    depths = iso.depth(ang)
    seg = plan.segments - o
    t, j = _first_hit(u, seg[:, 0], seg[:, 1] - seg[:, 0])
    lines = np.where(t < R, j, -1)
    pts = o + depths[:, None] * u
    poly = pts if full else np.vstack([o, pts])
    props = _properties(o, dvec, depths, lines, pts, poly, plan.segments)
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


def _properties(o, dvec, r, lines, pts, poly, segments) -> dict:
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    signed = 0.5 * cross.sum()
    area = abs(signed)
    perimeter = np.hypot(xn - x, yn - y).sum()
    centroid = np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * signed)
    try:
        hull = ConvexHull(poly).volume
    except QhullError:
        hull = np.nan

    # Occluding edges: from a hit to a miss, or to a line that shares no end point.
    a, b = lines[:-1], lines[1:]
    occ = (a >= 0) & (b != a)
    both = occ & (b >= 0)
    if both.any():
        occ[both] = ~_connected(segments, a[both], b[both])
    edges = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    occlusivity = float(edges[occ].sum())

    mean = r.mean()
    dev = r - mean
    sad = np.abs(dev).sum()
    r_max = r.max()

    drift = centroid - o
    p1, p2 = drift - o, dvec - o  # both read as points, seen from the viewpoint
    drift_angle = abs(math.atan2(p1[0] * p2[1] - p1[1] * p2[0], p1 @ p2))

    values = (
        area, area, perimeter, 4 * math.pi * area / perimeter**2, math.pi * mean**2 / area,
        area / hull, occlusivity, r.min(), r_max, mean, sad, (dev**2).sum(), (dev**3).sum(),
        mean - sad, mean / r_max, float(np.hypot(*drift)), drift_angle,
    )
    return {k: float(v) for k, v in zip(DS_PROPERTY_NAMES, values, strict=True)}
