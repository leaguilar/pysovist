"""Exact isovist metrics from wedge pieces.

On a wall piece the visible depth is ``r(theta) = p / cos(u)`` with
``u = theta - phi``, so every integral of a power of ``r`` has a closed form.
On a range arc ``r = R``. All metrics below are computed from these forms,
with no ray sampling.

Definitions (``Theta`` is the field-of-view measure, ``2 pi`` for a full turn):

- ``area``: A = 1/2 int r^2 dtheta (Benedikt 1979).
- ``perimeter``: visible wall length + range-arc length + occluding radial
  edges + field-of-view edges (Benedikt 1979).
- ``occlusivity``: total length of occluding radial edges, where the boundary
  jumps in depth (Benedikt 1979).
- ``r_min``, ``r_max``, ``r_mean``: extremes and mean of r over the field of
  view. ``r_var``: second central moment of r (the population variance).
  ``r_std``: its square root. ``r_skew``: standardised third central moment,
  the third central moment divided by r_std^3, and 0 when r_std = 0.
  ``r_mad``: mean absolute deviation of r. ``dispersion``: r_std / r_mean.
- ``compactness``: 4 pi A / P^2, equal to 1 for a disc (Turner et al. 2001).
- ``jaggedness``: P^2 / A (Wiener and Franz 2005).
- ``drift``, ``drift_angle``: distance and direction from the observer to the
  centroid of the isovist (Conroy Dalton 2001).
- ``elongation``: sqrt(l1 / l2), where l1 >= l2 are the principal second
  moments of area about the centroid, so it is at least 1. Equal to the aspect
  ratio for a rectangle, and ``inf`` when l2 = 0.
- ``convex_deficiency``: (A_hull - A) / A_hull. Range arcs enter the hull as
  polygons with a step of 0.5 degrees.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .results import SEGMENT

METRIC_NAMES = (
    "area", "perimeter", "occlusivity", "r_min", "r_max", "r_mean", "r_var", "r_std",
    "r_skew", "r_mad", "dispersion", "compactness", "jaggedness", "drift", "drift_angle",
    "elongation", "convex_deficiency",
)
"""Names of the exact 2D isovist metrics, in order.

They are the keys of [`Isovist.metrics`][pysovist.Isovist.metrics] and the
metric columns of [`isovist_field`][pysovist.isovist_field].
"""


def nan_metrics(**known) -> dict:
    m = dict.fromkeys(METRIC_NAMES, np.nan)
    m.update(known)
    return m


def _wrap(u):
    return np.mod(u + np.pi, 2 * np.pi) - np.pi


def _int_r_powers(u0, u1, p, kind, R, d_theta):
    """Integrals of r^1..r^4 over each wedge."""
    seg = kind == SEGMENT
    out = np.empty((4, len(p)))
    t0, t1 = np.tan(u0), np.tan(u1)
    sec0, sec1 = 1 / np.cos(u0), 1 / np.cos(u1)
    as0, as1 = np.arcsinh(t0), np.arcsinh(t1)
    with np.errstate(invalid="ignore"):
        out[0] = np.where(seg, p * (as1 - as0), R * d_theta)
        out[1] = np.where(seg, p**2 * (t1 - t0), R**2 * d_theta)
        out[2] = np.where(seg, p**3 / 2 * (sec1 * t1 + as1 - sec0 * t0 - as0), R**3 * d_theta)
        out[3] = np.where(seg, p**4 * (t1 + t1**3 / 3 - t0 - t0**3 / 3), R**4 * d_theta)
    return out


def _int_r_above(c, u0, u1, p, kind, R, d_theta):
    """Integral of (r - c) over the parts of each wedge where r > c."""
    total = 0.0
    for i in range(len(p)):
        if kind[i] != SEGMENT:
            if R > c:
                total += (R - c) * d_theta[i]
            continue
        if p[i] >= c:
            lo_hi = [(u0[i], u1[i])]
        else:
            w = np.arccos(p[i] / c)  # r > c where |u| > w
            lo_hi = [(u0[i], min(u1[i], -w)), (max(u0[i], w), u1[i])]
        for a, b in lo_hi:
            if b > a:
                total += p[i] * (np.arcsinh(np.tan(b)) - np.arcsinh(np.tan(a))) - c * (b - a)
    return total


def wedge_metrics(theta0, theta1, kind, p, phi, R, fov, full) -> dict:
    d_theta = theta1 - theta0
    seg = kind == SEGMENT
    u0 = np.where(seg, _wrap(theta0 - phi), 0.0)
    u1 = u0 + d_theta
    with np.errstate(invalid="ignore", divide="ignore"):
        r0 = np.where(seg, p / np.cos(u0), R)
        r1 = np.where(seg, p / np.cos(u1), R)

    ints = _int_r_powers(u0, u1, p, kind, R, d_theta)
    area = 0.5 * ints[1].sum()
    m1, m2, m3 = (ints[k].sum() / fov for k in range(3))
    r_var = max(m2 - m1 * m1, 0.0)
    r_std = np.sqrt(r_var)
    r_skew = (m3 - 3 * m1 * m2 + 2 * m1**3) / r_std**3 if r_std > 0 else 0.0
    r_mad = 2 * _int_r_above(m1, u0, u1, p, kind, R, d_theta) / fov

    r_min = np.where(seg & (u0 <= 0) & (u1 >= 0), p, np.minimum(r0, r1)).min()
    r_max = np.maximum(r0, r1).max()

    # Boundary: wall pieces, range arcs, depth jumps between wedges, field-of-view edges.
    c0, s0, c1, s1 = np.cos(theta0), np.sin(theta0), np.cos(theta1), np.sin(theta1)
    P0 = np.c_[r0 * c0, r0 * s0]
    P1 = np.c_[r1 * c1, r1 * s1]
    pieces = np.where(seg, np.hypot(*(P1 - P0).T), R * d_theta).sum()
    nxt = np.r_[r0[1:], r0[0]] if full else r0[1:]
    jumps = np.abs(nxt - (r1 if full else r1[:-1]))
    tol = 1e-9 * max(r_max, 1.0)
    occlusivity = jumps[jumps > tol].sum()
    fov_edges = 0.0 if full else r0[0] + r1[-1]
    perimeter = pieces + occlusivity + fov_edges

    # Centroid and second moments: triangles (observer, P0, P1) and circular sectors.
    tri_area = 0.5 * (P0[:, 0] * P1[:, 1] - P0[:, 1] * P1[:, 0])
    sec_area = 0.5 * R**2 * d_theta if np.isfinite(R) else np.zeros_like(d_theta)
    w_area = np.where(seg, tri_area, sec_area)
    half = d_theta / 2
    mid = (theta0 + theta1) / 2
    rho = np.zeros_like(d_theta)
    if np.isfinite(R):
        nz = half > 0
        rho[nz] = 4 * R * np.sin(half[nz]) / (3 * d_theta[nz])
    cen = np.where(seg[:, None], (P0 + P1) / 3, np.c_[rho * np.cos(mid), rho * np.sin(mid)])
    centroid = (w_area[:, None] * cen).sum(axis=0) / area

    ax, ay, bx, by = P0[:, 0], P0[:, 1], P1[:, 0], P1[:, 1]
    R4 = R**4 if np.isfinite(R) else 0.0
    ixx = np.where(seg, tri_area / 6 * (ax * ax + ax * bx + bx * bx),
                   R4 / 4 * (d_theta / 2 + (np.sin(2 * theta1) - np.sin(2 * theta0)) / 4))
    iyy = np.where(seg, tri_area / 6 * (ay * ay + ay * by + by * by),
                   R4 / 4 * (d_theta / 2 - (np.sin(2 * theta1) - np.sin(2 * theta0)) / 4))
    ixy = np.where(seg, tri_area / 12 * (2 * ax * ay + ax * by + bx * ay + 2 * bx * by),
                   R4 / 8 * (np.sin(theta1) ** 2 - np.sin(theta0) ** 2))
    cxx = ixx.sum() - area * centroid[0] ** 2
    cyy = iyy.sum() - area * centroid[1] ** 2
    cxy = ixy.sum() - area * centroid[0] * centroid[1]
    lam = np.linalg.eigvalsh(np.array([[cxx, cxy], [cxy, cyy]]))  # ascending: l2, l1
    elongation = np.sqrt(lam[1] / lam[0]) if lam[0] > 0 else np.inf

    return {
        "area": area,
        "perimeter": perimeter,
        "occlusivity": occlusivity,
        "r_min": float(r_min),
        "r_max": float(r_max),
        "r_mean": m1,
        "r_var": r_var,
        "r_std": r_std,
        "r_skew": r_skew,
        "r_mad": r_mad,
        "dispersion": r_std / m1,
        "compactness": 4 * np.pi * area / perimeter**2,
        "jaggedness": perimeter**2 / area,
        "drift": float(np.hypot(*centroid)),
        "drift_angle": float(np.arctan2(centroid[1], centroid[0])),
        "elongation": float(elongation),
        "convex_deficiency": _convex_deficiency(theta0, theta1, kind, P0, P1, R, area, full),
    }


HULL_ARC_STEP = np.radians(0.5)


def _convex_deficiency(theta0, theta1, kind, P0, P1, R, area, full, step=HULL_ARC_STEP):
    pts = [P0, P1]
    if not full:
        pts.append(np.zeros((1, 2)))
    for t0, t1, k in zip(theta0, theta1, kind, strict=True):
        if k != SEGMENT:
            th = np.linspace(t0, t1, max(2, int(np.ceil((t1 - t0) / step)) + 1))
            pts.append(np.c_[R * np.cos(th), R * np.sin(th)])
    try:
        hull = ConvexHull(np.concatenate(pts))
    except QhullError:
        return np.nan
    return max(hull.volume - area, 0.0) / hull.volume
