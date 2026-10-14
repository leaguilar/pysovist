"""The Isovist3D component of the DeCodingSpaces toolbox for Grasshopper, in pysovist terms.

``isovist3d`` reproduces the component's outputs for one vantage point: the
ray layout, the length and surface id of every ray, the hits per obstacle
mesh, the volume and the isovist mesh. It takes the component's own units:
angles in degrees, the view direction as a vector. The rules below reproduce
the measured outputs of DeCodingSpaces 2021.10.11 (see the documentation page
on directed volumes for the measurements and the agreement).

Ray layout
:   The horizontal step is ``360 / HPr`` degrees and the vertical step
    ``360 / VPr`` degrees, whatever the viewing angles. There are
    ``n_az = floor(HVA HPr / 360) + 1`` azimuths, from ``HVA / 2`` to the left
    of the view direction to ``HVA / 2`` to its right, so a full turn lists its
    first azimuth twice, and the component casts ``n_az n_el`` rays, with
    ``n_el = floor(VVA VPr / 360) + 1``: 52 rays for 12 x 6 over 360 x 180
    degrees, 14 rays over 180 x 90 degrees. It lists the elevations from
    ``-VVA / 2 + VAO`` upwards in steps, up to 0.01 radian (0.57 degrees)
    beyond ``VVA / 2 + VAO``, and gives the rays these elevations in turn:
    ray ``k`` has the azimuth ``k // m`` and the elevation ``k mod m`` of the
    ``m`` listed elevations. Both angles are stepped in single precision, in
    radians. A view direction with a vertical component is laid flat.

Fine vertical steps
:   When the step after the upper edge stays within 0.01 radian of it
    (vertical steps below 0.57 degrees for 180 degrees, or below 1.15 degrees
    with an odd VPr), ``m`` exceeds ``n_el`` and the rays leave the grid:
    each azimuth starts where the previous one stopped, beyond the pole. At
    1,440 x 720 rays the component lists 362 elevations, and its volume of
    the hospital plan is 3.4 % below the exact value. The measurements place
    the limit between 0.5 degrees (listed) and 0.6545 degrees (not listed).

Lengths
:   The distance to the first surface hit within the viewing range VR. A ray
    that hits nothing has the length VR, and its hit vector ends on the range
    sphere.

Surface ids
:   Obstacle meshes are numbered 0, 1, ... in input order, and ground meshes
    continue the numbering after them. A ray that hits nothing has id -1, and
    so does a ray that hits the first triangle of the scene, the first
    triangle of the first obstacle mesh, with its true length. For a box from
    Rhino that triangle is half of the floor. The first triangle of any other
    mesh keeps its id. Without obstacle meshes the rule marks the first
    triangle of the first ground mesh, a case that was not measured.

Hits per mesh (OV)
:   The number of rays with each obstacle id. Ground meshes are not counted.

Volume
:   The polyhedron through the hit points: every cell of the
    ``n_az x n_el`` grid of rays is split into two triangles, and the volumes
    of the tetrahedra from the eye to the triangles are summed, each counted
    positive. It is not the solid-angle integral ``1/3 int r^3 d omega``. On
    flat walls the two agree. Where a cell spans a near and a far surface, the
    polyhedron cuts the corner, so the volume falls short of the exact value.
    A grid with one azimuth or one elevation encloses nothing: volume 0.

Implicit ground plane (isZero)
:   The input changes no output: with isZero true, the rays of a room without
    a floor that look down still escape with the length VR. ``ground_at_zero``
    is accepted and has no effect, as in the component. A ground plane at
    z = 0 is a ground mesh.

Counts at which the component fails
:   Over 360 x 180 degrees the component stops with an index error, or
    crashes Rhino, at the 65 counts of ``FAILING_COUNTS``, 64 of them among
    the 101 counts from 1000 x 500 to 1100 x 550. No rule found in the counts predicts them.
    ``isovist3d`` raises a ``ValueError`` at these counts. At counts and angles
    that were not measured, it applies the rules above.

What is not reproduced
:   At 1003 x 501 rays the component returns one column of rays fewer than
    its layout (251,753 rays instead of 252,004), with the volume of the whole
    layout. At 1013 x 506 and 2,880 x 1,440 rays its volumes differ from the
    rules by up to 8e-4. Every other measured volume agrees to 2e-4. Several
    vantage points in one solution return 0 for some of them, different ones
    from run to run: each call takes one vantage point, as the component
    computes it alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..mesh import Mesh

DEFAULTS = {
    "view_range": 1000.0,
    "horizontal_angle": 180.0,
    "vertical_angle": 90.0,
    "horizontal_rays": 10,
    "vertical_rays": 10,
    "vertical_offset": 0.0,
    "geometry": False,
    "ground_at_zero": True,
}
"""The component's default inputs, under the keyword names of ``isovist3d``."""

_ERROR, _CRASH = "stops with the error 'Index was outside the bounds of the array'", "crashes Rhino"
FAILING_COUNTS = {
    **{c: _ERROR for c in [
        (1000, 500), (1002, 501), (1010, 505), (1019, 509), (1026, 513), (1029, 514),
        (1030, 515), (1050, 525), (1053, 526), (1055, 527), (1059, 529), (1067, 533),
        (1070, 535), (1071, 535), (1074, 537), (1078, 539), (1079, 539), (1083, 541),
        (1085, 542), (1090, 545), (1099, 549)]},
    **{c: _CRASH for c in [
        (1001, 500), (1004, 502), (1005, 502), (1008, 504), (1009, 504), (1011, 505),
        (1016, 508), (1017, 508), (1020, 510), (1021, 510), (1022, 511), (1023, 511),
        (1027, 513), (1031, 515), (1035, 517), (1039, 519), (1040, 520), (1041, 520),
        (1044, 522), (1045, 522), (1047, 523), (1048, 524), (1049, 524), (1051, 525),
        (1054, 527), (1056, 528), (1057, 528), (1060, 530), (1061, 530), (1062, 531),
        (1068, 534), (1069, 534), (1073, 536), (1076, 538), (1077, 538), (1080, 540),
        (1081, 540), (1091, 545), (1092, 546), (1093, 546), (1094, 547), (1096, 548),
        (1097, 548), (1200, 600)]},
}
"""Ray counts ``(HPr, VPr)`` at which the component fails over 360 x 180 degrees, measured.

Each maps to what the component does: it stops with an index error, or it
crashes Rhino. ``isovist3d`` and ``ray_layout`` raise a ``ValueError`` at these
counts with the full field of view (360 x 180 degrees, no vertical offset).
"""

_ELEVATION_SLACK = 0.01  # radians: elevations are listed up to this far beyond the upper edge


@dataclass(frozen=True)
class Isovist3D:
    """Outputs of the Isovist3D component for one vantage point.

    Attributes
    ----------
    origin : ndarray of shape (3,)
        Vantage point in metres.
    directions : ndarray of shape (n, 3)
        Unit direction of every ray, in the component's order.
    lengths : ndarray of shape (n,)
        Length of every ray in metres (output Len): the distance to the first
        hit, or the viewing range for a ray that hits nothing.
    surface_ids : ndarray of shape (n,)
        Surface id of every ray (output StR): the index of the mesh hit,
        obstacles first and ground meshes after them, -1 for no hit and for
        the first triangle of the scene (of the first obstacle mesh).
    volume : float
        Volume of the isovist polyhedron in cubic metres (output Vol).
    object_visibility : ndarray of shape (k,)
        Number of rays that hit each of the k obstacle meshes (output OV).
    grid_shape : tuple of int
        ``(n_az, n_el)``, the number of azimuths and of elevations of the
        grid. ``reshape(*grid_shape)`` arranges any per-ray array as the
        component triangulates it.
    view_range : float
        Viewing range in metres.
    mesh : Mesh or None
        The isovist polyhedron (output Mesh), closed by triangles to the
        vantage point, when ``geometry=True``.
    """

    origin: np.ndarray
    directions: np.ndarray
    lengths: np.ndarray
    surface_ids: np.ndarray
    volume: float
    object_visibility: np.ndarray
    grid_shape: tuple
    view_range: float
    mesh: Mesh | None = None

    @property
    def rays(self) -> np.ndarray:
        """Hit vector of every ray, from the vantage point to its end (output RG), shape (n, 3)."""
        return self.directions * self.lengths[:, None]


def ray_layout(view_direction, *, horizontal_angle: float = 180.0, vertical_angle: float = 90.0,
               horizontal_rays: int = 10, vertical_rays: int = 10,
               vertical_offset: float = 0.0) -> np.ndarray:
    """Unit directions of the component's rays, arranged on its grid.

    Parameters
    ----------
    view_direction : (x, y, z)
        View direction (input VD). Its vertical component is dropped, as the
        component does.
    horizontal_angle, vertical_angle : float
        Horizontal and vertical viewing angle in degrees (HVA, VVA), in
        (0, 360] and (0, 180]. Defaults 180 and 90.
    horizontal_rays, vertical_rays : int
        Horizontal and vertical precision (HPr, VPr), whole numbers of at
        least 1. The angular steps are ``360 / HPr`` and ``360 / VPr``
        degrees. Default 10 each.
    vertical_offset : float
        Vertical angle offset in degrees (VAO), added to every elevation.
        Default 0.

    Returns
    -------
    ndarray of shape (n_az, n_el, 3)
        The rays in the component's order: azimuths from left to right of
        the view direction, elevations from bottom to top. With fine vertical
        steps the rays leave the grid (see the module documentation).

    Raises
    ------
    ValueError
        When the view direction has no horizontal part or is not finite, an
        angle or count lies outside its domain, or the counts are among
        [`FAILING_COUNTS`][pysovist.compat.decodingspaces3d.FAILING_COUNTS]
        with the full field of view.
    """
    vd = np.asarray(view_direction, dtype=float).reshape(-1)
    if vd.shape != (3,) or not np.all(np.isfinite(vd)):
        raise ValueError("view_direction must be a finite vector (x, y, z)")
    if not math.hypot(vd[0], vd[1]) > 1e-12 * float(np.linalg.norm(vd)):
        raise ValueError("view_direction must have a horizontal part: the component lays the "
                         "view direction flat, and a vertical one has no azimuth")
    hva = _angle(horizontal_angle, 360.0, "horizontal_angle")
    vva = _angle(vertical_angle, 180.0, "vertical_angle")
    hpr = _count(horizontal_rays, "horizontal_rays")
    vpr = _count(vertical_rays, "vertical_rays")
    vao = float(vertical_offset)
    if not math.isfinite(vao):
        raise ValueError("vertical_offset must be a finite angle in degrees")
    if (hpr, vpr) in FAILING_COUNTS and hva == 360.0 and vva == 180.0 and vao == 0.0:
        raise ValueError(
            f"DeCodingSpaces Isovist3D fails at {hpr} x {vpr} rays over 360 x 180 degrees: it "
            f"{FAILING_COUNTS[(hpr, vpr)]} (measured), so it has no output to reproduce. Counts "
            "that work include 720 x 360, 1024 x 512, 1440 x 720 and 2048 x 1024. For the exact "
            "view volume, use pysovist.view_volume.")
    h_step, v_step = 360.0 / hpr, 360.0 / vpr
    n_az = math.floor(hva / h_step + 1e-9) + 1
    n_el = math.floor(vva / v_step + 1e-9) + 1
    m = math.floor((vva + math.degrees(_ELEVATION_SLACK)) / v_step + 1e-9) + 1
    start = math.degrees(math.atan2(vd[1], vd[0])) + hva / 2
    az = _steps(math.radians(start), -math.radians(h_step), n_az)
    el = _steps(math.radians(-vva / 2 + vao), math.radians(v_step), m)
    k = np.arange(n_az * n_el)
    a, e = az[k // m], el[k % m]
    d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)
    return d.reshape(n_az, n_el, 3)


def isovist3d(vantage_point, obstacles, view_direction, *, ground=(),
              view_range: float = 1000.0, horizontal_angle: float = 180.0,
              vertical_angle: float = 90.0, horizontal_rays: int = 10, vertical_rays: int = 10,
              vertical_offset: float = 0.0, geometry: bool = False,
              ground_at_zero: bool = True) -> Isovist3D:
    """Isovist3D of one vantage point, as the DeCodingSpaces component computes it.

    Parameters
    ----------
    vantage_point : (x, y, z)
        One vantage point (input VP), in metres. The component returns 0 for
        some of several vantage points solved together, so each call takes one.
    obstacles : Mesh or sequence of Mesh
        Obstacle meshes (input OS), numbered 0, 1, ... in order.
    view_direction : (x, y, z)
        View direction (input VD), required as in the component, which fails
        without one even for a full view. Its vertical component is dropped.
    ground : sequence of Mesh
        Ground meshes (input GS), numbered after the obstacles. Default none.
    view_range : float
        Viewing range in metres (VR), positive and finite. Default 1000.
    horizontal_angle, vertical_angle : float
        Viewing angles in degrees (HVA, VVA), in (0, 360] and (0, 180].
        Defaults 180 and 90.
    horizontal_rays, vertical_rays : int
        Horizontal and vertical precision (HPr, VPr), whole numbers of at
        least 1. Default 10 each. The component casts one column of rays for
        HPr = 0 and fails for negative counts.
    vertical_offset : float
        Vertical angle offset in degrees (VAO). Default 0.
    geometry : bool
        Also build the isovist mesh (isGI). Default False.
    ground_at_zero : bool
        The input isZero. Default True. It has no effect, as in the component.

    Returns
    -------
    Isovist3D
        Per-ray outputs, hits per obstacle mesh, volume and mesh.

    Raises
    ------
    ValueError
        For more than one vantage point, a coordinate that is not finite, a
        viewing range that is not positive and finite, and the inputs that
        [`ray_layout`][pysovist.compat.decodingspaces3d.ray_layout] refuses,
        among them the counts at which the component fails.
    """
    del ground_at_zero  # measured: the component's outputs do not depend on it
    obstacles = [obstacles] if isinstance(obstacles, Mesh) else list(obstacles)
    ground = [ground] if isinstance(ground, Mesh) else list(ground)
    o = np.asarray(vantage_point, dtype=float)
    if o.size != 3 or not np.all(np.isfinite(o)):
        raise ValueError("isovist3d takes one finite vantage point (x, y, z): call it once per "
                         "point, since the component returns 0 for some of several vantage "
                         "points solved together")
    o = o.reshape(3)
    R = float(view_range)
    if not (R > 0 and math.isfinite(R)):
        raise ValueError("view_range must be a positive, finite distance in metres")
    grid = ray_layout(view_direction, horizontal_angle=horizontal_angle,
                      vertical_angle=vertical_angle, horizontal_rays=horizontal_rays,
                      vertical_rays=vertical_rays, vertical_offset=vertical_offset)
    d = grid.reshape(-1, 3)

    t = np.full(len(d), np.inf)
    ids = np.full(len(d), -1, dtype=np.int64)
    meshes = obstacles + ground
    if meshes:
        scene = Mesh.combine(meshes)
        tt, tri = scene.cast_hits(o[None], d, max_distance=R)
        t, tri = tt[0], tri[0]
        hit = tri >= 0
        ids[hit] = scene.object_ids[tri[hit]]
        ids[tri == 0] = -1  # the first triangle of the scene reports no surface
    lengths = np.where(np.isfinite(t) & (t <= R), t, R)

    points = (d * lengths[:, None])
    tris = _grid_triangles(*grid.shape[:2])
    tets = np.einsum("ij,ij->i", points[tris[:, 0]],
                     np.cross(points[tris[:, 1]], points[tris[:, 2]]))
    volume = float(np.abs(tets).sum() / 6)
    ov = np.bincount(ids[(ids >= 0) & (ids < len(obstacles))], minlength=len(obstacles))
    mesh = _isovist_mesh(o, points, tris) if geometry else None
    return Isovist3D(origin=o, directions=d, lengths=lengths, surface_ids=ids, volume=volume,
                     object_visibility=ov[:len(obstacles)], grid_shape=tuple(grid.shape[:2]),
                     view_range=R, mesh=mesh)


def _angle(value, top, name) -> float:
    v = float(value)
    if not 0 < v <= top:
        raise ValueError(f"{name} must lie in (0, {top:g}] degrees")
    return v


def _count(value, name) -> int:
    v = float(value)
    if not (math.isfinite(v) and v == int(v)):
        raise ValueError(f"{name} must be a whole number")
    if v < 1:
        raise ValueError(f"{name} must be at least 1: the component casts a single column of "
                         "rays at 0, which encloses no volume, and fails below 0")
    return int(v)


def _steps(start, step, n) -> np.ndarray:
    """``start, start + step, ...`` (n angles in radians), stepped in single precision."""
    a = np.full(n, step, dtype=np.float32)
    a[0] = start
    return np.add.accumulate(a, dtype=np.float32).astype(float)


def _grid_triangles(n_az: int, n_el: int) -> np.ndarray:
    """Two triangles per grid cell, wound so that the polyhedron's normals point outwards."""
    i, j = np.meshgrid(np.arange(n_az - 1), np.arange(n_el - 1), indexing="ij")
    a = (i * n_el + j).ravel()
    b = ((i + 1) * n_el + j).ravel()
    c = ((i + 1) * n_el + j + 1).ravel()
    d = (i * n_el + j + 1).ravel()
    return np.concatenate([np.stack([a, c, b], axis=1), np.stack([a, d, c], axis=1)])


def _isovist_mesh(origin, points, tris) -> Mesh:
    """The grid surface through the hit points, closed by triangles to the vantage point."""
    n = len(points)
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    inner = {tuple(e) for e in edges}
    border = [(u, v) for u, v in edges if (v, u) not in inner]
    fan = np.array([(v, u, n) for u, v in border], dtype=np.int64).reshape(-1, 3)
    return Mesh(np.vstack([origin + points, origin]), np.concatenate([tris, fan]))
