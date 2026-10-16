"""Shape metrics of view volumes: the 3D counterparts of the 2D isovist metrics.

A view volume is star-shaped around the eye. Ray ``i`` leaves the eye in the
unit direction ``u_i``, carries the solid angle ``w_i`` and ends at the depth
``r_i``, after the escape policy has set the depth of the rays that hit
nothing: a clipped ray ends at the range limit. The metrics below are computed
from these three arrays, so they hold alike for equal-area directions, angle
grids and fields of view. ``Omega = sum w_i`` is the solid angle of the view,
``V = 1/3 sum w_i r_i^3`` its volume, and the eye sits at the origin.

Radial statistics, weighted by solid angle like ``r_mean`` and ``r_std``:

- ``r_var``: ``sum w_i (r_i - r_mean)^2 / Omega`` in m2, the square of ``r_std``.
- ``r_skew``: ``sum w_i (r_i - r_mean)^3 / Omega / r_std^3``, and 0 when
  ``r_std`` is 0.
- ``r_mad``: ``sum w_i |r_i - r_mean| / Omega`` in m.
- ``dispersion``: ``r_std / r_mean``.

Centroid and second moments of the visible volume:

- ``drift``: the distance in m from the eye to the centroid
  ``c = (1/V) int r^4 / 4 u domega = (1/V) sum w_i r_i^4 u_i / 4``.
- ``drift_horizontal``: the horizontal length of ``c`` in m.
  ``drift_vertical``: its height above the eye in m, negative below it.
- ``drift_azimuth``: the direction of ``c`` in radians counter-clockwise from
  +x, the counterpart of the 2D ``drift_angle``. ``drift_elevation``: its
  angle above the eye's horizontal plane in radians.
- ``elongation``: ``sqrt(l1 / l3)``, and ``elongation_12``: ``sqrt(l1 / l2)``,
  where ``l1 >= l2 >= l3`` are the principal second moments of the volume about
  its centroid, the eigenvalues of ``int r^5 / 5 u u^T domega - V c c^T``. A
  box of sides ``a >= b >= c`` has elongation ``a / c`` and elongation_12
  ``a / b``: a corridor has both large, a wide low hall a large elongation and
  an elongation_12 near 1.

The boundary:

- ``surface_area``: the area ``S`` in m2 of the boundary of the view volume,
  the counterpart of the perimeter. It holds the visible surfaces, the
  occluding faces and, for a field of view, the side faces from the eye to the
  edges of the field.
- ``occluding_area``: the area in m2 of the occluding faces, the counterpart
  of occlusivity. Where a near object hides a far surface, the boundary runs
  along the rays that graze the object, from the object to the far surface.
  Over an occluding contour of angular length element ``ds`` the face has the
  area ``int (r_far^2 - r_near^2) / 2 ds``.
- ``sphericity``: ``36 pi V^2 / S^3``, 1 for a ball seen from its centre, the
  counterpart of compactness. ``jaggedness``: ``S^3 / V^2``, the counterpart
  of ``P^2 / A``.
- ``convex_deficiency``: ``max(H - V, 0) / H``, with ``H`` the volume of the
  convex hull of the eye and the ray ends. The hull of the ray ends lies
  inside the true hull, so the deficiency runs low, by at most 2e-3 at 65,536
  rays in the reference rooms, and it is clamped at 0 as in 2D.

The surface through the ray ends. The unit directions are triangulated by
their convex hull, which is their Delaunay triangulation on the sphere, and
each triangle is mapped to the triangle through its three ray ends. A plane
``a . x = 1`` is seen at the inverse depth ``1/r = a . u``, so the plane
through the ray ends of a triangle predicts the inverse depth of every ray
that ends on it exactly, at any distance and angle. Neighbouring rays ``i``
and ``j``, with ``i`` the nearer, see a depth jump when

    a . u_j > (1 + k |u_i - u_j|) / r_j  for every plane a of the near side,
    b . u_i < (1 - k |u_i - u_j|) / r_i  for every plane b of the far side,

with ``k`` the ``jump_factor``, 1 by default: ``j`` ends behind the surface
of ``i``, and ``i`` in front of the surface of ``j``, each by more than ``k``
times the angle between the rays in relative depth. The planes of the near
side are those of the triangles around ``i``, without ``j``, whose three
rays lie nearer than the middle of the two inverse depths, and those of the
far side the triangles around ``j``, without ``i``, whose rays lie farther.
A wall, floor or ceiling predicts its own rays exactly, so it never counts as
occluding, however near the eye or grazing. Where two planes meet, one of the
two conditions fails: along a concave edge, such as a corner of a room seen
from inside, ``j`` lies in front of the plane of ``i``, and along a convex
edge, such as a corner of a pillar seen from outside, ``i`` lies behind the
plane of ``j``. The eye's distance to its surroundings does not enter. A side
without such a triangle, as on an object narrower than about two ray
spacings, leaves the decision to the other side, and a flagged edge is
cleared when a triangle around either ray, with no flagged side, still sees
the surface continue across it.

The triangles with a side across a jump form a band along each occluding
contour. In a triangle crossed on two sides, the occluding face is the
quadrilateral between the radial segments on those sides, from the near
surface to the far surface in the direction halfway along each side, and the
pieces of the near and far surfaces on either side of the face are surface.
A triangle crossed on one side is surface, as its third ray is continuous
with both others. In a triangle crossed on all three sides, where three
surfaces meet, each side's segment joins the triangle's centre. The depth of
each surface at a segment is extrapolated from the ray that sees it, by
fitting ``1/r = a . u`` to the ray and its neighbours on the same surface,
which is exact on a plane. The extrapolated depth stays within the range of
the depths the fit used, widened by that range on either side, so a fit
tilted by neighbours on other surfaces (balls of a point cloud seen through
gaps) cannot send a face out to an arbitrary depth. The band follows the
lattice of the rays in steps, so each piece of a face counts with its vector
area projected onto the mean vector area of the pieces within ``N^(-1/4)``
radians that the jump crosses in the same sense, with ``N = 4 pi / mean(w)``.
The steps cancel, and so do the steps of the side faces of a field of view,
which are built from the eye to the outermost rays.

Both areas are estimates through sampled ray ends, like a sampled 2D
perimeter. The surface area falls short where two surfaces meet at an edge,
and its error falls as ``N^(-1/2)``. In the reference rooms the occluding
area comes within 0.6 % of the exact faces from 65,536 rays on, and settles
less than 0.5 % below them, a floor that more rays do not remove. Four prices remain. A face whose
relative depth step is close to ``k`` times the ray spacing is flagged on
some of its edges only, and where every other edge is flagged it counts as
surface. A surface seen so nearly edge-on that it covers a strip one ray wide
has no triangle of its own, and where it meets another surface it can count
as occluding. For a field of view the surface ends at the outermost rays,
half a ray spacing inside the field's edges. For a point cloud, the grooves
between the balls are concave and never count, but where balls are seen at a
grazing angle one ball hides part of the next, and these small faces count
once neighbouring rays come closer than the spacing of the balls.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .directions import fibonacci_sphere, in_field_of_view

SHAPE_METRIC_NAMES = (
    "surface_area", "occluding_area", "r_var", "r_skew", "r_mad", "dispersion", "sphericity",
    "jaggedness", "drift", "drift_horizontal", "drift_vertical", "drift_azimuth",
    "drift_elevation", "elongation", "elongation_12", "convex_deficiency",
)
"""Names of the shape metrics of a view volume, in order.

With ``shape=True`` they follow the [`VOLUME_METRIC_NAMES`][pysovist.VOLUME_METRIC_NAMES] in
[`ViewVolume.metrics`][pysovist.ViewVolume.metrics], and they are the last columns of
[`view_volume_field`][pysovist.view_volume_field].
"""


@dataclass(frozen=True)
class Surface:
    """Triangulation of a set of ray directions, shared by every eye that casts them."""

    triangles: np.ndarray  # (T, 3) ray indices, counter-clockwise seen from outside
    edges: np.ndarray  # (E, 2) ray indices of the triangle sides, each side once
    sides: np.ndarray  # (T, 3) row of ``edges`` of each side of each triangle
    chords: np.ndarray  # (E,) distance between the two unit directions of each edge
    centres: np.ndarray  # (T, 3) unit vectors through the centroids of the triangles
    rim: np.ndarray  # (F, 2) sides on the edge of a field of view, in the order of their triangle
    rim_centres: np.ndarray  # (F, 3) unit vectors through the midpoints of the rim sides
    rim_out: np.ndarray  # (F, 3) direction from the inside of the field across each rim side
    window: float  # chord radius of the window over which occluding faces are smoothed
    fans: np.ndarray  # (n, D) triangles around each ray, -1 after the last
    wings: np.ndarray  # (E, 4) triangles next to the two triangles of each edge, see _wings


_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 3  # triangulations kept: a few MB each at 65,536 rays, 0.3 GB at 2^20
_LOCK = threading.Lock()
GAP_FACTOR = 10.0  # a triangle wider than this many ray spacings is a gap in the directions


def triangulate(d: np.ndarray, w: np.ndarray, field: dict | None = None) -> Surface:
    """Triangles of the unit directions ``d`` (n, 3), each carrying the solid angle ``w``.

    Over the whole sphere the triangles are the facets of the convex hull of the
    directions, their Delaunay triangulation on the sphere. For a field of view
    (``field`` holds ``hfov``, ``vfov``, ``direction`` and ``elevation``) the
    hull also takes a Fibonacci set of the same density outside the field, and
    the triangles with a corner outside are dropped, so the surface ends at
    the outermost rays. The last three triangulations are kept, so that
    repeated calls with one direction set triangulate it once. Calls from
    several threads share the kept triangulations.

    Raises
    ------
    ValueError
        When the directions leave a gap wider than ``GAP_FACTOR`` ray spacings,
        or no triangle lies inside the field of view.
    """
    view = None if field is None else {k: float(v) for k, v in field.items()}
    key = hashlib.blake2b(d.tobytes() + w.tobytes() + repr(view).encode()).hexdigest()
    with _LOCK:
        surface = _CACHE.get(key)
        if surface is not None:
            _CACHE.move_to_end(key)
            return surface
    surface = _triangulate(d, w, view)
    with _LOCK:
        _CACHE[key] = surface
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return surface


def _triangulate(d, w, field) -> Surface:
    n = len(d)
    pts = d
    if field is not None:
        g = fibonacci_sphere(max(int(round(4 * np.pi / w.mean())), 1))
        pts = np.vstack([d, g[~in_field_of_view(g, **field)]])
    try:
        tri = ConvexHull(pts).simplices
    except QhullError:
        raise ValueError("the directions lie on one great circle and enclose no solid angle: "
                         "use directions that cover the sphere or a field of view") from None
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    normal = np.cross(b - a, c - a)
    flip = np.einsum("ij,ij->i", normal, a + b + c) < 0
    tri[flip] = tri[flip][:, ::-1]
    normal[flip] *= -1
    keep = (tri < n).all(axis=1)
    tri, normal = tri[keep], normal[keep]
    if len(tri) == 0:
        raise ValueError("no three neighbouring directions lie inside the field of view, so the "
                         "view has no surface: use more rays or a wider field")
    spacing = np.sqrt(w.mean())
    offset = np.einsum("ij,ij->i", normal, d[tri[:, 0]]) / np.linalg.norm(normal, axis=1)
    widest = float(np.arccos(np.clip(offset, -1.0, 1.0)).max())
    if widest > GAP_FACTOR * spacing:
        raise ValueError(
            f"the directions leave a gap of {np.degrees(widest):.3g} degrees, more than "
            f"{GAP_FACTOR:g} ray spacings: give directions that cover the sphere, or the field "
            "of view they cover (hfov, vfov, direction, elevation)")
    directed = np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1).reshape(-1, 2)
    edges, inverse, count = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True,
                                      return_counts=True)
    sides = inverse.reshape(-1, 3)
    once = count[inverse] == 1
    rim = directed[once]
    third = tri[:, [2, 0, 1]].reshape(-1)[once]  # the corner of the rim side's triangle opposite it
    rim_out = d[rim].sum(axis=1) / 2 - d[third]
    centres = d[tri].sum(axis=1)
    centres /= np.linalg.norm(centres, axis=1)[:, None]
    rim_centres = d[rim].sum(axis=1)
    rim_centres /= np.linalg.norm(rim_centres, axis=1)[:, None]
    rho = (4 * np.pi / w.mean()) ** -0.25
    return Surface(triangles=tri, edges=edges, sides=sides,
                   chords=np.linalg.norm(d[edges[:, 0]] - d[edges[:, 1]], axis=1),
                   centres=centres, rim=rim, rim_centres=rim_centres, rim_out=rim_out,
                   window=2 * np.sin(rho / 2), fans=_fans(tri, n), wings=_wings(tri, edges, sides))


def _fans(tri, n):
    """(n, D) the triangles around each ray, padded with -1."""
    ray = tri.reshape(-1)
    order = np.argsort(ray, kind="stable")
    ray, t = ray[order], np.repeat(np.arange(len(tri)), 3)[order]
    count = np.bincount(ray, minlength=n)
    start = np.r_[0, np.cumsum(count)[:-1]]
    fans = np.full((n, max(int(count.max()), 1)), -1, dtype=np.int32)
    fans[ray, np.arange(len(ray)) - start[ray]] = t
    return fans


def _wings(tri, edges, sides):
    """(E, 4) for each edge (i, j): the triangles across the sides (i, x) and (j, x) of its
    first triangle (i, j, x), then those across (i, y) and (j, y) of its second (j, i, y), -1
    where a triangle is missing."""
    owner = np.full((len(edges), 2), -1)
    s = sides.reshape(-1)
    t = np.repeat(np.arange(len(tri)), 3)
    order = np.argsort(s, kind="stable")
    s, t = s[order], t[order]
    first = np.r_[True, s[1:] != s[:-1]]
    owner[s[first], 0] = t[first]
    owner[s[~first], 1] = t[~first]
    wings = np.full((len(edges), 4), -1, dtype=np.int32)
    for col in (0, 1):
        has = np.flatnonzero(owner[:, col] >= 0)
        tt = owner[has, col]
        corners = tri[tt]
        for side, v in ((0, edges[has, 0]), (1, edges[has, 1])):
            other = edges[has, 1 - side]
            pv = np.argmax(corners == v[:, None], axis=1)
            po = np.argmax(corners == other[:, None], axis=1)
            px = 3 - pv - po  # the third corner
            k = np.where((px - pv) % 3 == 1, pv, px)  # the side from corner k to corner k + 1
            edge = sides[tt, k]
            a, b = owner[edge, 0], owner[edge, 1]
            wings[has, 2 * col + side] = np.where(a == tt, b, a)
    return wings


def shape_metrics(r, d, w, volume, surface: Surface, jump_factor: float) -> dict:
    """Shape metric arrays over the rows of depths ``r`` (m, n).

    ``volume`` (m,) is the view volume of each row. Each row gives the result it
    gives alone, to the last bit.
    """
    m = len(r)
    out = {k: np.full(m, np.nan) for k in SHAPE_METRIC_NAMES}
    total = w.sum()
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        shift = r[:, :1]
        x = r - shift
        mean_x = (w * x).sum(axis=1) / total
        dev = x - mean_x[:, None]
        var = (w * dev**2).sum(axis=1) / total
        std = np.sqrt(var)
        m3 = (w * dev**3).sum(axis=1) / total
        out["r_var"] = var
        out["r_skew"] = np.where(std == 0, 0.0, m3 / std**3)
        out["r_mad"] = (w * np.abs(dev)).sum(axis=1) / total
        out["dispersion"] = std / (shift[:, 0] + mean_x)

        # einsum sums each row in the same order whatever the number of rows, where a matrix
        # product would take another path for one row than for several
        first = np.einsum("mn,ni->mi", w * r**4 / 4, d)
        c = first / volume[:, None]
        horizontal = np.hypot(c[:, 0], c[:, 1])
        out["drift"] = np.linalg.norm(c, axis=1)
        out["drift_horizontal"] = horizontal
        out["drift_vertical"] = c[:, 2]
        out["drift_azimuth"] = np.arctan2(c[:, 1], c[:, 0])
        out["drift_elevation"] = np.arctan2(c[:, 2], horizontal)

    finite = np.isfinite(r).all(axis=1) & np.isfinite(volume)
    for i in np.flatnonzero(finite):
        second = np.einsum("n,ni,nj->ij", w * r[i] ** 5 / 5, d, d)
        central = second - volume[i] * np.outer(c[i], c[i])
        if np.isfinite(central).all():
            l3, l2, l1 = np.linalg.eigvalsh(central)
            out["elongation"][i] = np.sqrt(l1 / l3) if l3 > 0 else np.inf
            out["elongation_12"][i] = np.sqrt(l1 / l2) if l2 > 0 else np.inf
        s, occluding, hull = _surface(r[i], d, surface, jump_factor)
        out["surface_area"][i] = s
        out["occluding_area"][i] = occluding
        out["convex_deficiency"][i] = max(hull - volume[i], 0.0) / hull if hull > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        s = out["surface_area"]
        out["sphericity"] = 36 * np.pi * volume**2 / s**3
        out["jaggedness"] = s**3 / volume**2
    unknown = np.isnan(r).any(axis=1)
    endless = ~unknown & np.isinf(r).any(axis=1)
    for k in SHAPE_METRIC_NAMES:
        out[k][unknown | endless] = np.nan
    out["surface_area"][endless] = np.inf
    return out


def _surface(r, d, surface: Surface, jump_factor: float):
    """Surface area, occluding area and hull volume of one eye."""
    p = d * r[:, None]
    t = surface.triangles
    normals = _cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    jump = _jumps(r, d, surface, normals, jump_factor)
    s, occluding = _boundary(r, d, surface, jump, normals)
    try:
        hull = ConvexHull(np.vstack([p, np.zeros((1, 3))])).volume
    except QhullError:
        hull = np.nan
    return s, occluding, hull


def _cross(u, v):
    """Cross products of the rows of ``u`` and ``v`` (n, 3), faster than np.cross for many rows."""
    out = np.empty_like(u)
    out[:, 0] = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    out[:, 1] = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    out[:, 2] = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return out


# --- the depth-jump test --------------------------------------------------------------------------

def _jumps(r, d, surface: Surface, normals, k: float) -> np.ndarray:
    """Which edges of the surface span a depth jump, as a boolean array over ``surface.edges``.

    A plane ``a . x = 1`` is seen at the inverse depth ``1/r = a . u``. Every triangle carries
    the plane through its three ray ends. Of the two rays of an edge, the near one has the
    larger inverse depth. A triangle around the near ray, without the far ray, whose three rays
    lie nearer than the middle of the two inverse depths, is a plane of the near side; the planes
    of the far side are the triangles around the far ray, without the near ray, whose rays lie
    farther. The edge spans a jump when the far ray ends behind every plane of the near side, and
    the near ray in front of every plane of the far side, each by more than ``k`` times the angle
    between the rays in relative depth. A side without such a plane leaves the decision to the
    other, and an edge with none on either side is calm. A plane predicts its own rays exactly,
    so no plane is flagged at any angle or distance. Where two planes meet along a concave edge
    (a corner of a room seen from inside) the far ray lies in front of the near plane, and along
    a convex edge (a pillar's corner seen from outside) the near ray lies behind the far plane,
    so no edge between two planes is flagged either. A second look clears a flagged edge when a
    triangle around either ray, with no flagged side, still sees the other ray on the side of its
    plane that a continuous surface would. A ray with depth 0 ends at the eye: an edge from it to
    a ray with a positive depth spans a jump.
    """
    tri, e = surface.triangles, surface.edges
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        q = 1.0 / r
        plane = normals / np.einsum("ij,ij->i", normals, d[tri[:, 0]] * r[tri[:, 0], None])[:, None]
    qt = q[tri]
    qmin, qmax = qt.min(axis=1), qt.max(axis=1)
    first_near = q[e[:, 0]] >= q[e[:, 1]]
    near = np.where(first_near, e[:, 0], e[:, 1])
    far = e[:, 0] + e[:, 1] - near
    jump = np.zeros(len(e), dtype=bool)
    zero = np.isinf(q[near])
    jump[zero & np.isfinite(q[far])] = True

    def behind(t, ray, scale):
        """Inverse depth of the planes ``t`` (m,) or (m, D) at ``ray`` (m,) less ``scale`` times
        the ray's own: positive where the ray ends behind the plane by more than ``scale``. NaN
        for a plane through the eye, which decides nothing."""
        with np.errstate(invalid="ignore", over="ignore"):
            if t.ndim == 1:
                return np.einsum("ij,ij->i", plane[t], d[ray]) - q[ray] * scale
            return np.einsum("ijk,ik->ij", plane[t], d[ray]) - (q[ray] * scale)[:, None]

    idx = np.flatnonzero(~zero & np.isfinite(q[far]) & (q[near] > q[far]))
    # a first pass over the triangles next to the edge's own two, the near side's first: one of
    # them that sees the edge continue settles it as calm
    for col_near, near_side in ((0, True), (2, True), (0, False), (2, False)):
        first = first_near[idx]
        # columns 0 and 2 hold the triangles at the edge's first ray, 1 and 3 at its second
        at = first if near_side else ~first
        t = np.where(at, surface.wings[idx, col_near], surface.wings[idx, col_near + 1])
        nr, fr = near[idx], far[idx]
        mid = 0.5 * (q[nr] + q[fr])
        tau = k * surface.chords[idx]
        has = t >= 0
        t = np.where(has, t, 0)
        if near_side:
            calm = has & (qmin[t] > mid) & ~(behind(t, fr, 1 + tau) > 0)
        else:
            calm = has & (qmax[t] < mid) & ~(-behind(t, nr, 1 - tau) > 0)
        idx = idx[~calm]
    nr, fr = near[idx], far[idx]
    mid = 0.5 * (q[nr] + q[fr])
    tau = k * surface.chords[idx]
    if len(idx):
        fans_n, fans_f = surface.fans[nr], surface.fans[fr]
        ok_n = fans_n >= 0
        tn = np.where(ok_n, fans_n, 0)
        ok_n &= (tri[tn] != fr[:, None, None]).all(axis=2) & (qmin[tn] > mid[:, None])
        yes_n = np.where(ok_n, behind(tn, fr, 1 + tau) > 0, True).all(axis=1)
        ok_f = fans_f >= 0
        tf = np.where(ok_f, fans_f, 0)
        ok_f &= (tri[tf] != nr[:, None, None]).all(axis=2) & (qmax[tf] < mid[:, None])
        yes_f = np.where(ok_f, -behind(tf, nr, 1 - tau) > 0, True).all(axis=1)
        jump[idx] = yes_n & yes_f & (ok_n.any(axis=1) | ok_f.any(axis=1))
    # the second look
    idx = np.flatnonzero(jump & ~zero)
    if len(idx):
        band = jump[surface.sides].any(axis=1)
        nr, fr = near[idx], far[idx]
        tau = k * surface.chords[idx]
        clear = np.zeros(len(idx), dtype=bool)
        for ray, other, near_side in ((nr, fr, True), (fr, nr, False)):
            fan = surface.fans[ray]
            ok = fan >= 0
            t = np.where(ok, fan, 0)
            ok &= ~band[t] & (tri[t] != other[:, None, None]).all(axis=2)
            margin = (behind(t, other, 1 + tau) if near_side else -behind(t, other, 1 - tau))
            clear |= (ok & ~(margin > 0)).any(axis=1)
        jump[idx[clear]] = False
    return jump


# --- the boundary through the ray ends --------------------------------------------------------

def _boundary(r, d, surface: Surface, jump, normals=None):
    """Surface area and occluding area of one eye, given the edges that span a depth jump.

    A triangle with no side across a jump, or one, is surface: with one, its third ray is
    continuous with both others. In a triangle crossed on two sides, the occluding face is the
    quadrilateral between the radial segments on those sides, and the pieces of the near and far
    surfaces on either side of it are surface. In a triangle crossed on all three sides, each
    side's segment joins the triangle's centre, where the three surfaces meet, and the piece of
    each surface around its ray is surface.
    """
    p = d * r[:, None]
    tri, sides = surface.triangles, surface.sides
    if normals is None:
        normals = _cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
    area = 0.5 * np.linalg.norm(normals, axis=1)
    crossed = jump[sides]
    count = crossed.sum(axis=1)
    s = float(area[count < 2].sum())
    faces, centres, across = [], [], []
    if (count >= 2).any():
        seg = _Segments(r, d, surface, jump)
        two = np.flatnonzero(count == 2)
        if len(two):
            calm = np.argmin(crossed[two], axis=1)  # side c joins corners c and c + 1
            x = tri[two, (calm + 2) % 3]
            y, z = tri[two, calm], tri[two, (calm + 1) % 3]
            kxy, kxz = seg.index[sides[two, (calm + 2) % 3]], seg.index[sides[two, (calm + 1) % 3]]
            quad = 0.5 * np.cross(seg.hi[kxz] - seg.lo[kxy], seg.lo[kxz] - seg.hi[kxy])
            up = seg.up[kxy] + seg.up[kxz]
            faces.append(_facing(quad, up))
            centres.append(surface.centres[two])
            across.append(up)
            px = 0.5 * np.cross(seg.end(kxy, x) - p[x], seg.end(kxz, x) - p[x])
            pyz = 0.5 * np.cross(p[z] - seg.end(kxy, y), seg.end(kxz, z) - p[y])
            s += float(np.linalg.norm(px, axis=1).sum() + np.linalg.norm(pyz, axis=1).sum())
        three = np.flatnonzero(count == 3)
        if len(three):
            c = surface.centres[three]
            for k in range(3):
                a, b = tri[three, k], tri[three, (k + 1) % 3]
                kab, kca = seg.index[sides[three, k]], seg.index[sides[three, (k + 2) % 3]]
                ca, cb = seg.at(a, c), seg.at(b, c)
                quad = 0.5 * np.cross(cb - seg.end(kab, a), ca - seg.end(kab, b))
                faces.append(_facing(quad, seg.up[kab]))
                centres.append(c)
                across.append(seg.up[kab])
                piece = 0.5 * np.cross(ca - p[a], seg.end(kca, a) - seg.end(kab, a))
                s += float(np.linalg.norm(piece, axis=1).sum())
    occluding = 0.0
    if faces:
        occluding = _projected_area(np.concatenate(faces), np.concatenate(centres),
                                    np.concatenate(across), surface.window)
    rim = surface.rim
    sides_of_field = _projected_area(0.5 * np.cross(p[rim[:, 1]], p[rim[:, 0]]),
                                     surface.rim_centres, surface.rim_out, surface.window)
    return s + occluding + sides_of_field, occluding


def _facing(quad, up):
    """The view lies on the far side of an occluding face: its outward normal points back
    across the jump, to the near side."""
    flip = np.einsum("ij,ij->i", quad, up) > 0
    quad[flip] *= -1
    return quad


class _Segments:
    """The radial segment from the near surface to the far surface on each edge across a jump.

    The segment lies in the direction ``m`` halfway between the two rays. The depth of each
    surface there is extrapolated from the ray that sees it: the inverse depth of a plane is
    linear in the direction, ``1/r = a . u``, and ``a`` is fitted to the ray and its neighbours
    on the same surface. The extrapolated depth stays within the range of the depths the fit
    used, widened by that range on either side: ``m`` lies half a side from the ray and the
    neighbours a full side, so a linear trend in ``1/r`` along a neighbour within 60 degrees of
    the gradient stays inside unless ``1/r`` falls by more than half per side toward the jump,
    while a fit tilted by a few neighbours on different surfaces cannot send the face out to an
    arbitrary depth.
    """

    def __init__(self, r, d, surface: Surface, jump):
        e = surface.edges
        jumps = np.flatnonzero(jump)
        i, j = e[jumps, 0], e[jumps, 1]
        self.r, self.d = r, d
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / r
        self.planes, lo, hi = _inverse_depth_planes(r, inv, d, e, jump, np.unique(e[jumps]))
        span = hi - lo
        self.lo_r, self.hi_r = lo - span, hi + span
        near = np.where(r[i] <= r[j], i, j)
        far = i + j - near
        m = d[i] + d[j]
        m /= np.linalg.norm(m, axis=1)[:, None]
        r_far = self.depth(far, m)
        r_near = np.minimum(self.depth(near, m), r_far)
        self.lo, self.hi = r_near[:, None] * m, r_far[:, None] * m
        self.up = d[far] - d[near]
        self.first_near = near == i
        self.edge_first = i
        self.index = np.full(len(e), -1)
        self.index[jumps] = np.arange(len(jumps))

    def depth(self, v, u):
        """Depth of the surface of ray ``v`` in the direction ``u``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            q = np.einsum("ij,ij->i", self.planes[v], u)
            rv = np.where(np.isfinite(q) & (q > 0), 1 / q, self.r[v])
        return np.clip(rv, self.lo_r[v], self.hi_r[v])

    def at(self, v, u):
        """Point of the surface of ray ``v`` in the direction ``u``."""
        return self.depth(v, u)[:, None] * u

    def end(self, k, v):
        """End of segment ``k`` on the surface of its ray ``v``."""
        v_near = (v == self.edge_first[k]) == self.first_near[k]
        return np.where(v_near[:, None], self.lo[k], self.hi[k])


def _inverse_depth_planes(r, inv, d, e, jump, verts):
    """Least-squares ``a`` with ``1/r = a . u`` around each ray of ``verts``, NaN elsewhere.

    The fit takes the ray and its neighbours across edges without a jump. The
    smallest and largest depth among the rays of each fit come with it.
    """
    n = len(inv)
    index = np.full(n, -1)
    index[verts] = np.arange(len(verts))
    A = np.einsum("ni,nj->nij", d[verts], d[verts])
    with np.errstate(invalid="ignore"):
        b = d[verts] * inv[verts, None]
    lo, hi = r.astype(float), r.astype(float)
    calm = ~jump
    with np.errstate(invalid="ignore"):  # a ray at depth 0 has 1/r = inf
        for s, t in ((e[:, 0], e[:, 1]), (e[:, 1], e[:, 0])):
            sel = calm & (index[s] >= 0)
            np.minimum.at(lo, s[sel], r[t[sel]])
            np.maximum.at(hi, s[sel], r[t[sel]])
            k, t = index[s[sel]], t[sel]
            np.add.at(A, k, np.einsum("ni,nj->nij", d[t], d[t]))
            np.add.at(b, k, d[t] * inv[t, None])
    out = np.full((n, 3), np.nan)
    lam, vec = np.linalg.eigh(A)
    ok = (lam[:, 0] > 1e-10 * lam[:, 2]) & np.isfinite(b).all(axis=1)
    coef = np.einsum("nji,nj->ni", vec[ok], b[ok]) / lam[ok]
    out[verts[ok]] = np.einsum("nij,nj->ni", vec[ok], coef)
    return out, lo, hi


def _projected_area(vec, centres, across, window) -> float:
    """Area of a radial surface given by the vector areas ``vec`` of its pieces.

    The pieces of an occluding face, or of a side face of a field of view,
    follow the lattice of the rays in steps, so the sum of their areas would
    measure the steps. Each piece counts instead with its vector area projected
    onto the mean vector area of the pieces within ``window`` (chord distance
    between ``centres``) that the surface is crossed in the same sense:
    ``across`` points from the near side to the far side of a depth jump, or out
    of a field of view. The two flanks of a thin object thus stay apart, while
    the steps of one flank cancel.
    """
    if len(vec) == 0:
        return 0.0
    i, j = cKDTree(centres).query_pairs(window, output_type="ndarray").T
    same = np.einsum("ij,ij->i", across[i], across[j]) > 0
    i, j = i[same], j[same]
    s = vec.copy()
    for k in range(3):
        s[:, k] += (np.bincount(i, weights=vec[j, k], minlength=len(vec))
                    + np.bincount(j, weights=vec[i, k], minlength=len(vec)))
    norm = np.linalg.norm(s, axis=1)
    ok = norm > 0
    return float((np.einsum("ij,ij->i", vec[ok], s[ok]) / norm[ok]).sum())
