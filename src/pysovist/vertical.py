"""Exact isovists in a vertical plane through the eye.

A vertical isovist is the 2D isovist of a vertical section of a building:
the region of one vertical plane through the eye that the eye sees. The plane
contains the eye and a horizontal direction, the azimuth. Cutting a triangle
mesh with it gives a set of segments, the vertical section, and the exact 2D
kernel of [`isovist`][pysovist.isovist] gives the isovist and its metrics in
that plane. Ray-casting tools obtain the same section by turning the model
by 90 degrees and cutting it horizontally.

Plane coordinates are ``(u, z)``: ``u`` is the horizontal distance from the
eye along the azimuth, negative behind the eye, and ``z`` is the height of
the scene. The eye stands at ``(0, z_eye)``. Angles in the plane are
elevations, in radians counter-clockwise from the forward horizontal: 0 looks
ahead, pi/2 up, -pi/2 down and pi behind.
"""

from __future__ import annotations

import numpy as np

from .isovist2d import TWO_PI, isovist
from .plan import Plan
from .results import Isovist


def vertical_section(mesh, eye, azimuth: float) -> Plan:
    """Section of a mesh by the vertical plane through ``eye`` along ``azimuth``.

    Every triangle that crosses the plane gives the segment along which it
    crosses, in plane coordinates ``(u, z)``. A vertex in the plane counts
    as lying on its right side (seen along ``azimuth``), so a solid that
    spans the signed distances ``w0 <= w < w1`` from the plane, positive to
    the left, is cut when ``w0 <= 0 < w1``. Triangles in the plane give no
    segment. This is the rule of ``Mesh.section``, turned upright.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh in metres, z up. Any object with ``vertices`` (n, 3)
        and ``triangles`` (m, 3) works.
    eye : (x, y, z)
        A point of the plane. Only its x and y place the plane, and they
        are the origin of ``u``.
    azimuth : float
        Horizontal direction of the plane, in radians counter-clockwise
        from +x. It is the direction of +u.

    Returns
    -------
    Plan
        One segment per crossing triangle, in plane coordinates.
    """
    v = np.asarray(mesh.vertices, dtype=float)
    t = np.asarray(mesh.triangles, dtype=np.int64).reshape(-1, 3)
    e = np.asarray(eye, dtype=float)
    c, s = np.cos(azimuth), np.sin(azimuth)
    rel = v[:, :2] - e[:2]
    uz = np.c_[rel[:, 0] * c + rel[:, 1] * s, v[:, 2]]
    w = rel[:, 1] * c - rel[:, 0] * s  # signed distance to the plane, positive to the left
    left = w > 0
    n_left = left[t].sum(axis=1)
    cut = (n_left == 1) | (n_left == 2)
    tri, side = t[cut], left[t[cut]]
    # The lone vertex is the one on its own side of the plane. Both crossing edges start there.
    lone = np.where(n_left[cut] == 1, side.argmax(axis=1), side.argmin(axis=1))
    rows = np.arange(len(tri))
    a = tri[rows, lone]
    ends = [_crossing(uz, w, a, tri[rows, (lone + k) % 3], left) for k in (1, 2)]
    return Plan(np.stack(ends, axis=1) if len(tri) else np.empty((0, 2, 2)))


def vertical_isovist(
    mesh,
    eye,
    azimuth: float,
    *,
    elevation: float = 0.0,
    fov: float = TWO_PI,
    max_distance: float = np.inf,
    on_wall: str = "nan",
    eps: float = 1e-9,
) -> Isovist:
    """Exact isovist of an eye in the vertical plane along ``azimuth``.

    The isovist of ``vertical_section(mesh, eye, azimuth)`` seen from
    ``(0, z_eye)``, in plane coordinates ``(u, z)``.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh in metres, z up.
    eye : (x, y, z)
        Eye position. Its z is an absolute coordinate of the scene.
    azimuth : float
        Horizontal direction of the plane, in radians counter-clockwise
        from +x.
    elevation : float
        Centre of the field of view in the plane, in radians above the
        forward horizontal. Used only when ``fov`` is below a full turn.
    fov : float
        Field of view in the plane, in radians. Default: the full turn.
    max_distance : float
        Range limit in metres. A section that is open, for example with no
        ceiling, needs it for finite metrics.
    on_wall : {"nan", "raise"}
        What to do when the eye is closer than ``eps`` to a surface, as in
        ``isovist``.
    eps : float
        Distance below which the eye counts as standing on a surface.

    Returns
    -------
    Isovist
        Wedges, flags and the standard metrics, in plane coordinates. Its
        ``origin`` is ``(0, z_eye)``, its angles are elevations, and
        ``plane_to_world`` maps its points back to the scene.
    """
    e = np.asarray(eye, dtype=float)
    section = vertical_section(mesh, e, azimuth)
    return isovist(section, (0.0, e[2]), max_distance=max_distance, fov=fov,
                   direction=elevation, on_wall=on_wall, eps=eps)


def plane_to_world(points, eye, azimuth: float) -> np.ndarray:
    """Map points ``(u, z)`` of the vertical plane to scene coordinates (k, 3).

    The point ``(u, z)`` lies at ``(x_eye + u cos(azimuth),
    y_eye + u sin(azimuth), z)``.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    e = np.asarray(eye, dtype=float)
    return np.c_[e[0] + p[:, 0] * np.cos(azimuth), e[1] + p[:, 0] * np.sin(azimuth), p[:, 1]]


def _crossing(uz, w, a, b, left):
    """Plane point where each edge ``(a[i], b[i])`` crosses the plane ``w = 0``.

    Each edge is interpolated from its vertex on the right side, so the two
    triangles that share an edge give the same point to the last bit, and a
    vertex in the plane is returned exactly.
    """
    lo = np.where(left[a], b, a)
    hi = np.where(left[a], a, b)
    f = w[lo] / (w[lo] - w[hi])
    return uz[lo] + f[:, None] * (uz[hi] - uz[lo])
