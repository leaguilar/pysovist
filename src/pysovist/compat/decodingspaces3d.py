"""The Isovist3D component of the DeCodingSpaces toolbox for Grasshopper, in pysovist terms.

``isovist3d`` reproduces the component's outputs for one vantage point: the
ray layout, the length and surface id of every ray, the hits per obstacle
mesh, the volume and the isovist mesh. It takes the component's own units:
angles in degrees, the view direction as a vector. The rules below reproduce
every measured output of DeCodingSpaces 2021.10.11 to single precision (see
the documentation page on directed volumes for the measurements).

Ray layout
:   The horizontal step is ``360 / HPr`` degrees and the vertical step
    ``360 / VPr`` degrees, whatever the viewing angles. Azimuths run from
    ``HVA / 2`` to the left of the view direction to ``HVA / 2`` to its right,
    both ends included, so a full turn lists its first azimuth twice.
    Elevations run from ``-VVA / 2 + VAO`` up to ``VVA / 2 + VAO``, both ends
    included. The rays are listed azimuth by azimuth, from left to right,
    and from the lowest to the highest elevation within each azimuth. There
    are ``floor(HVA HPr / 360) + 1`` azimuths and ``floor(VVA VPr / 360) + 1``
    elevations: 52 rays for 12 x 6 over 360 x 180 degrees, 14 rays over
    180 x 90 degrees.

Lengths
:   The distance to the first surface hit within the viewing range VR. A ray
    that hits nothing has the length VR, and its hit vector ends on the range
    sphere.

Surface ids
:   Obstacle meshes are numbered 0, 1, ... in input order, and ground meshes
    continue the numbering after them. A ray that hits nothing has id -1, and
    so does a ray that hits the implicit ground plane. A ray that hits the
    last triangle of an obstacle mesh also reports -1, with its true length.
    In the measured box rooms that triangle is one half of the floor, which
    is why half of a closed room's floor reports -1.

Hits per mesh (OV)
:   The number of rays with each obstacle id. Ground meshes and the implicit
    plane are not counted.

Volume
:   The volume of the polyhedron through the hit points: every cell of the
    ray grid is split into two triangles, and the tetrahedra from the eye to
    the triangles are summed. It is not the solid-angle integral
    ``1/3 int r^3 d omega``. On flat walls the two agree. Where a cell spans a
    near and a far surface, the polyhedron cuts the corner, so the volume of
    a scene with many occluding edges falls short of the exact value, by less
    as the ray count grows.

Implicit ground plane (isZero)
:   With ``ground_at_zero=True`` (the component's default), the plane z = 0
    is an occluder with surface id -1, and a mesh triangle wins a tie with
    it.

What is not reproduced
:   At 1,440 x 720 rays on the extruded hospital plan the component returns
    193.23 m3, while its own layout and volume rule give 199.02 m3 (exact:
    200.01 m3). Every coarser measurement agrees to 2e-5. A view direction
    with a vertical component and viewing angles that are not multiples of
    the step were not measured, and ``isovist3d`` rejects the former.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class Isovist3D:
    """Outputs of the Isovist3D component for one vantage point.

    Attributes
    ----------
    origin : ndarray of shape (3,)
        Vantage point.
    directions : ndarray of shape (n, 3)
        Unit direction of every ray, in the component's order.
    lengths : ndarray of shape (n,)
        Length of every ray (output Len): the distance to the first hit, or
        the viewing range for a ray that hits nothing.
    surface_ids : ndarray of shape (n,)
        Surface id of every ray (output StR).
    volume : float
        Volume of the isovist polyhedron in cubic metres (output Vol).
    object_visibility : ndarray of shape (k,)
        Number of rays that hit each of the k obstacle meshes (output OV).
    grid_shape : tuple of int
        Number of azimuths and of elevations. ``reshape(*grid_shape)``
        arranges any per-ray array on the grid.
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
    """Unit directions of the component's rays on its grid.

    Parameters
    ----------
    view_direction : (x, y, z)
        View direction (input VD), a horizontal vector.
    horizontal_angle, vertical_angle : float
        Horizontal and vertical viewing angle in degrees (HVA, VVA).
    horizontal_rays, vertical_rays : int
        Horizontal and vertical precision (HPr, VPr). The angular steps are
        ``360 / HPr`` and ``360 / VPr`` degrees.
    vertical_offset : float
        Vertical angle offset in degrees (VAO), added to every elevation.

    Returns
    -------
    ndarray of shape (n_azimuth, n_elevation, 3)
        Azimuths from left to right of the view direction, elevations from
        bottom to top.
    """
    vd = np.asarray(view_direction, dtype=float).reshape(3)
    norm = np.linalg.norm(vd)
    if not norm > 0:
        raise ValueError("view_direction must be a non-zero vector")
    if abs(vd[2]) > 1e-12 * norm:
        raise ValueError("only horizontal view directions are reproduced: give z = 0")
    hva, vva = float(horizontal_angle), float(vertical_angle)
    hpr, vpr = int(horizontal_rays), int(vertical_rays)
    if hpr < 1 or vpr < 1:
        raise ValueError("horizontal_rays and vertical_rays must be at least 1")
    if not (hva > 0 and vva > 0):
        raise ValueError("the viewing angles must be positive")
    n_az = int(np.floor(hva * hpr / 360.0 + 1e-9)) + 1
    n_el = int(np.floor(vva * vpr / 360.0 + 1e-9)) + 1
    az = np.radians(np.degrees(np.arctan2(vd[1], vd[0])) + hva / 2 - 360.0 / hpr * np.arange(n_az))
    el = np.radians(-vva / 2 + float(vertical_offset) + 360.0 / vpr * np.arange(n_el))
    a, e = np.meshgrid(az, el, indexing="ij")
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def isovist3d(vantage_point, obstacles, view_direction, *, ground=(),
              view_range: float = 1000.0, horizontal_angle: float = 180.0,
              vertical_angle: float = 90.0, horizontal_rays: int = 10, vertical_rays: int = 10,
              vertical_offset: float = 0.0, geometry: bool = False,
              ground_at_zero: bool = True) -> Isovist3D:
    """Isovist3D of one vantage point, as the DeCodingSpaces component computes it.

    Parameters
    ----------
    vantage_point : (x, y, z)
        Vantage point (input VP).
    obstacles : Mesh or sequence of Mesh
        Obstacle meshes (input OS), numbered 0, 1, ... in order.
    view_direction : (x, y, z)
        View direction (input VD), required as in the component, which fails
        without one even for a full view. It must be horizontal.
    ground : sequence of Mesh
        Ground meshes (input GS), numbered after the obstacles.
    view_range : float
        Viewing range in metres (VR).
    horizontal_angle, vertical_angle : float
        Viewing angles in degrees (HVA, VVA).
    horizontal_rays, vertical_rays : int
        Horizontal and vertical precision (HPr, VPr).
    vertical_offset : float
        Vertical angle offset in degrees (VAO).
    geometry : bool
        Also build the isovist mesh (isGI).
    ground_at_zero : bool
        Treat the plane z = 0 as ground with surface id -1 (isZero).

    Returns
    -------
    Isovist3D
        Per-ray outputs, hits per obstacle mesh, volume and mesh.
    """
    obstacles = [obstacles] if isinstance(obstacles, Mesh) else list(obstacles)
    ground = [ground] if isinstance(ground, Mesh) else list(ground)
    o = np.asarray(vantage_point, dtype=float).reshape(3)
    R = float(view_range)
    if not R > 0:
        raise ValueError("view_range must be positive")
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
        ends = np.cumsum([len(m.triangles) for m in meshes]) - 1
        last = [ends[k] for k, m in enumerate(obstacles) if len(m.triangles)]
        ids[np.isin(tri, last) & hit] = -1
    if ground_at_zero:
        with np.errstate(divide="ignore", invalid="ignore"):
            tp = -o[2] / d[:, 2]
        # A mesh surface at z = 0 keeps its hits: Embree's single-precision depth may exceed
        # the plane's by rounding, so the plane must be nearer by more than that.
        plane = (tp > 0) & (tp <= R) & (tp < t * (1 - 1e-6))
        t[plane], ids[plane] = tp[plane], -1
    lengths = np.where(np.isfinite(t) & (t <= R), t, R)

    points = (d * lengths[:, None]).reshape(grid.shape)
    tris = _grid_triangles(*grid.shape[:2])
    p = points.reshape(-1, 3)
    volume = abs(float(np.einsum("ij,ij->i", p[tris[:, 0]],
                                 np.cross(p[tris[:, 1]], p[tris[:, 2]])).sum()) / 6)
    ov = np.bincount(ids[(ids >= 0) & (ids < len(obstacles))], minlength=len(obstacles))
    mesh = _isovist_mesh(o, p, tris) if geometry else None
    return Isovist3D(origin=o, directions=d, lengths=lengths, surface_ids=ids, volume=volume,
                     object_visibility=ov[:len(obstacles)], grid_shape=tuple(grid.shape[:2]),
                     view_range=R, mesh=mesh)


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
