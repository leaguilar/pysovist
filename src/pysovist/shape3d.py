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
each triangle is mapped to the triangle through its three ray ends.
Neighbouring rays ``i`` and ``j`` see a depth jump when

    |1/r_i - 1/r_j| > k |u_i - u_j| / c,

with ``c`` the clearance of the eye (its distance to the nearest occluder) and
``k`` the ``jump_factor``, 1 by default. On a plane at distance ``p`` from the
eye, ``1/r = n . u / p`` for the plane's unit normal ``n``, so between two rays
``1/r`` changes by at most ``|u_i - u_j| / p``. A larger change cannot come from
one plane at least the clearance away, so no wall, floor or ceiling is taken
for an occluding face, however grazing the angle at which it is seen. A jump
from the depth ``r_near`` to ``r_far`` is seen once neighbouring rays are closer
than ``(1/r_near - 1/r_far) c / k`` apart.

The triangles with a side across a jump form a band along each occluding
contour. In each, the occluding face is rebuilt from radial segments from the
near surface to the far surface, in the direction halfway along each side
across the jump. The depth of each surface there is extrapolated from the ray
that sees it, by fitting ``1/r = a . u`` to the ray and its neighbours on the
same surface, which is exact on a plane. The extrapolated depth stays within
the range of the depths the fit used, widened by that range on either side,
so a fit tilted by neighbours on other surfaces (balls of a point cloud seen
through gaps, from an eye close to a surface) cannot send a face out to an
arbitrary depth. The band follows the lattice of the
rays in steps, so each piece of a face counts with its vector area projected
onto the mean vector area of the pieces within ``N^(-1/4)`` radians that the
jump crosses in the same sense, with ``N = 4 pi / mean(w)``. The steps cancel,
and so do the steps of the side faces of a field of view, which are built
from the eye to the outermost rays.

Both areas are estimates through sampled ray ends, like a sampled 2D
perimeter. The surface area falls short where two surfaces meet at an edge,
and its error falls as ``N^(-1/2)``. The occluding area converges with the ray
count once the rays resolve the jump. Three prices remain. A plane that passes
nearer to the eye than the clearance, with its nearest point off the surface
(a wall end or a door jamb seen almost edge-on), can count as occluding. For a
field of view the surface ends at the outermost rays, half a ray spacing
inside the field's edges. For a point cloud, once neighbouring rays come
closer than the spacing of its balls, the grooves between the balls count as
occluding faces.
"""

from __future__ import annotations

import hashlib
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


_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 3  # triangulations kept: a few MB each at 65,536 rays, 0.2 GB at 2^20
GAP_FACTOR = 10.0  # a triangle wider than this many ray spacings is a gap in the directions


def triangulate(d: np.ndarray, w: np.ndarray, field: dict | None = None) -> Surface:
    """Triangles of the unit directions ``d`` (n, 3), each carrying the solid angle ``w``.

    Over the whole sphere the triangles are the facets of the convex hull of the
    directions, their Delaunay triangulation on the sphere. For a field of view
    (``field`` holds ``hfov``, ``vfov``, ``direction`` and ``elevation``) the
    hull also takes a Fibonacci set of the same density outside the field, and
    the triangles with a corner outside are dropped, so the surface ends at
    the outermost rays. The last three triangulations are kept, so that
    repeated calls with one direction set triangulate it once.

    Raises
    ------
    ValueError
        When the directions leave a gap wider than ``GAP_FACTOR`` ray spacings,
        or no triangle lies inside the field of view.
    """
    key = hashlib.blake2b(d.tobytes() + w.tobytes() + repr(field).encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    surface = _triangulate(d, w, field)
    _CACHE[key] = surface
    if len(_CACHE) > _CACHE_SIZE:
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
    once = count[inverse] == 1
    rim = directed[once]
    third = tri[:, [2, 0, 1]].reshape(-1)[once]  # the corner of the rim side's triangle opposite it
    rim_out = d[rim].sum(axis=1) / 2 - d[third]
    centres = d[tri].sum(axis=1)
    centres /= np.linalg.norm(centres, axis=1)[:, None]
    rim_centres = d[rim].sum(axis=1)
    rim_centres /= np.linalg.norm(rim_centres, axis=1)[:, None]
    rho = (4 * np.pi / w.mean()) ** -0.25
    return Surface(triangles=tri, edges=edges, sides=inverse.reshape(-1, 3),
                   chords=np.linalg.norm(d[edges[:, 0]] - d[edges[:, 1]], axis=1),
                   centres=centres, rim=rim, rim_centres=rim_centres, rim_out=rim_out,
                   window=2 * np.sin(rho / 2))


def shape_metrics(r, d, w, volume, clearance, surface: Surface, jump_factor: float) -> dict:
    """Shape metric arrays over the rows of depths ``r`` (m, n).

    ``volume`` (m,) is the view volume of each row and ``clearance`` (m,) the
    distance from its eye to the nearest occluder.
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

        first = (w * r**4 / 4) @ d
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
        s, occluding, hull = _surface(r[i], d, surface, clearance[i], jump_factor)
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


def _surface(r, d, surface: Surface, clearance: float, jump_factor: float):
    """Surface area, occluding area and hull volume of one eye."""
    p = d * r[:, None]
    tri = surface.triangles
    a, b, c = p[tri[:, 0]], p[tri[:, 1]], p[tri[:, 2]]
    vec = 0.5 * np.cross(b - a, c - a)
    area = np.linalg.norm(vec, axis=1)
    e = surface.edges
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / r
        jump = np.abs(inv[e[:, 0]] - inv[e[:, 1]]) > jump_factor * surface.chords / clearance
    band = jump[surface.sides].any(axis=1)
    occluding = 0.0
    if band.any():
        faces, up = _ribbon(r, inv, d, surface, jump, band, vec[band])
        occluding = _projected_area(faces, surface.centres[band], up, surface.window)
    rim = surface.rim
    sides = _projected_area(0.5 * np.cross(p[rim[:, 1]], p[rim[:, 0]]), surface.rim_centres,
                            surface.rim_out, surface.window)
    try:
        hull = ConvexHull(np.vstack([p, np.zeros((1, 3))])).volume
    except QhullError:
        hull = np.nan
    return area[~band].sum() + occluding + sides, occluding, hull


def _ribbon(r, inv, d, surface: Surface, jump, band, vec):
    """Vector areas of the occluding faces inside the band triangles.

    On each edge that spans a jump, the occluding face is the radial segment
    from the near surface to the far surface in the direction ``m`` halfway
    between the two rays. The depth of each surface there is extrapolated from
    the ray that sees it: the inverse depth of a plane is linear in the
    direction, ``1/r = a . u``, and ``a`` is fitted to the ray and its
    neighbours on the same surface. The extrapolated depth stays within the
    range of the depths the fit used, widened by that range on either side:
    ``m`` lies half a side from the ray and the neighbours a full side, so a
    linear trend along any neighbour within 60 degrees of the gradient stays
    inside, while a fit tilted by a few neighbours on different surfaces
    cannot send the face out to an arbitrary depth. A band triangle crossed
    by two such edges holds the quadrilateral between their segments. A triangle crossed by one
    or three keeps its own vector area ``vec``.
    """
    e = surface.edges
    jumps = np.flatnonzero(jump)
    i, j = e[jumps, 0], e[jumps, 1]
    near = np.where(r[i] <= r[j], i, j)
    far = i + j - near
    m = d[i] + d[j]
    m /= np.linalg.norm(m, axis=1)[:, None]
    planes, lo, hi = _inverse_depth_planes(r, inv, d, e, jump, np.unique(e[jumps]))
    with np.errstate(divide="ignore", invalid="ignore"):
        q_near = np.einsum("ij,ij->i", planes[near], m)
        q_far = np.einsum("ij,ij->i", planes[far], m)
        r_near = np.where(np.isfinite(q_near) & (q_near > 0), 1 / q_near, r[near])
        r_far = np.where(np.isfinite(q_far) & (q_far > 0), 1 / q_far, r[far])
    span = hi - lo
    r_near = np.clip(r_near, lo[near] - span[near], hi[near] + span[near])
    r_far = np.clip(r_far, lo[far] - span[far], hi[far] + span[far])
    r_near = np.minimum(r_near, r_far)
    lo, hi = r_near[:, None] * m, r_far[:, None] * m

    up = np.zeros((len(e), 3))
    up[jumps] = d[far] - d[near]
    across = up[surface.sides[band]].sum(axis=1)

    index = np.full(len(e), -1)
    index[jumps] = np.arange(len(jumps))
    crossed = np.sort(index[surface.sides[band]], axis=1)
    two = (crossed[:, 0] < 0) & (crossed[:, 1] >= 0)
    k1, k2 = crossed[two, 1], crossed[two, 2]
    quad = 0.5 * np.cross(hi[k2] - lo[k1], lo[k2] - hi[k1])
    # The view lies on the far side of an occluding face: its outward normal points back
    # across the jump, to the near side.
    flip = np.einsum("ij,ij->i", quad, across[two]) > 0
    quad[flip] *= -1
    out = vec.copy()
    out[two] = quad
    return out, across


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
