"""Triangle meshes as occluders.

Rays are cast with Embree through open3d's ``RaycastingScene``, batched over
observers x directions. Embree works in single precision, so the mesh is
stored relative to the centre of its bounding box and every ray is shifted by
the same amount: depths keep about seven significant digits of the scene's
own extent, wherever the scene sits in a georeferenced frame. Clearances
(distance from an eye to the nearest triangle) are recomputed in double
precision on the triangle Embree finds.

A mesh is a set of two-sided surfaces. An eye closer than ``eps`` to a
surface stands on a wall (``flags.on_wall``). The inside of a closed solid
is not told apart from the outside.

open3d is imported on first use, so ``import pysovist`` works without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

_CHUNK_RAYS = 1 << 21  # rays per Embree call (48 MB of single-precision rays)
CLOSE_OPTIONS = ("hull", None)


class Mesh:
    """Triangle mesh in metres, z up.

    Ray casting, clearances and reading files need the ``mesh`` extra (open3d).

    Parameters
    ----------
    vertices : array_like of shape (n, 3)
        Vertex coordinates in metres. They are copied.
    triangles : array_like of shape (m, 3)
        Integer vertex indices of each triangle. Orientation does not matter
        for ray casting. Quads must be split first: (a, b, c, d) into
        (a, b, c) and (a, c, d).
    object_ids : array_like of shape (m,), optional
        Index of the object each triangle belongs to, a non-negative integer.
        Default: 0 for every triangle, one object. ``Mesh.combine`` sets them
        for a scene made of several meshes.

    Raises
    ------
    ValueError
        When ``vertices`` is not of shape (n, 3) or holds a coordinate that is
        not finite, ``triangles`` is not of shape (m, 3), holds indices that
        are not whole numbers or lie outside the vertices, or ``object_ids``
        does not give one non-negative integer per triangle.

    Attributes
    ----------
    vertices : ndarray of shape (n, 3)
        Vertex coordinates, read-only.
    triangles : ndarray of shape (m, 3)
        Vertex indices of each triangle, read-only.
    object_ids : ndarray of shape (m,)
        Object index of each triangle, read-only.
    block_flag : str
        The flag set for an eye closer than ``eps`` to a surface, ``"on_wall"``.
    """

    block_flag = "on_wall"

    def __init__(self, vertices, triangles, object_ids=None):
        v = np.array(vertices, dtype=float, copy=True)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if not np.all(np.isfinite(v)):
            raise ValueError("vertices must be finite: remove or repair the vertices with NaN or "
                             "infinite coordinates")
        t = _triangle_indices(triangles)
        if len(t) and (t.min() < 0 or t.max() >= len(v)):
            raise ValueError("triangle indices out of range")
        v.setflags(write=False)
        t.setflags(write=False)
        self.vertices: np.ndarray = v
        self.triangles: np.ndarray = t
        self.object_ids: np.ndarray = _object_ids(object_ids, len(t))
        self._shift = (v.min(axis=0) + v.max(axis=0)) / 2 if len(v) else np.zeros(3)
        self._scene = None

    def __repr__(self) -> str:
        return f"Mesh({len(self.vertices)} vertices, {len(self.triangles)} triangles)"

    @property
    def n_objects(self) -> int:
        """Number of objects: one more than the largest object id, 0 for no triangles."""
        return int(self.object_ids.max()) + 1 if len(self.object_ids) else 0

    # --- construction -----------------------------------------------------------------------

    @classmethod
    def combine(cls, meshes) -> Mesh:
        """One scene from several meshes, remembering which mesh each triangle came from.

        The vertices and triangles are concatenated in order, and every
        triangle of ``meshes[k]`` gets the object id ``k``. A ray cast with
        [`cast_hits`][pysovist.Mesh.cast_hits] then tells which object it hit,
        and [`view_volume`][pysovist.view_volume] with ``object_ids=True``
        reports it for every ray.

        Parameters
        ----------
        meshes : sequence of Mesh
            The objects of the scene. Their own object ids are replaced by
            their position in the sequence.

        Returns
        -------
        Mesh
            The combined scene, with ``object_ids`` set.

        Raises
        ------
        ValueError
            When ``meshes`` is empty.
        """
        meshes = list(meshes)
        if not meshes:
            raise ValueError("combine needs at least one mesh")
        offsets = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
        vertices = np.concatenate([m.vertices for m in meshes])
        triangles = np.concatenate([m.triangles + k for m, k in zip(meshes, offsets, strict=True)])
        ids = np.repeat(np.arange(len(meshes)), [len(m.triangles) for m in meshes])
        return cls(vertices, triangles, object_ids=ids)

    @classmethod
    def from_file(cls, path) -> Mesh:
        """Read any triangle mesh format open3d reads (``.ply``, ``.obj``, ``.stl``, ...).

        Parameters
        ----------
        path : str or os.PathLike
            Mesh file, in metres.

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        ValueError
            When the file holds no triangles, for example a point cloud,
            which [`PointCloud.read`][pysovist.PointCloud.read] reads.
        """
        import open3d as o3d

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no mesh file at {path}")
        m = o3d.io.read_triangle_mesh(str(path))
        if len(m.triangles) == 0:
            raise ValueError(f"no triangles read from {path}: a point cloud is read with "
                             "PointCloud.read")
        return cls(np.asarray(m.vertices), np.asarray(m.triangles))

    @staticmethod
    def check_closing(plan, polygon) -> None:
        """Raise a ValueError if a wall of ``plan`` lies mostly (more than half
        its length) outside the closing ``polygon`` (vertices in order).

        Every edge of a closing polygon becomes a wall, so a polygon that
        leaves walls outside cuts through the building. A wall may end on the
        polygon, run along it or reach past it.

        Parameters
        ----------
        plan : Plan or array_like of shape (n, 2, 2)
            Wall segments in metres.
        polygon : array_like of shape (k, 2)
            Closing polygon, vertices in order, the last joined to the first.

        Raises
        ------
        ValueError
            When a wall lies mostly outside the polygon, or the polygon has a
            coordinate that is not finite or fewer than three distinct corners
            enclosing an area.
        """
        from .plan import Plan

        if isinstance(plan, Plan):
            segs = plan.segments
        else:
            segs = np.asarray(plan, dtype=float).reshape(-1, 2, 2)
        out = _walls_outside_ring(segs, _closing_ring(polygon))
        if out.any():
            raise ValueError(
                f"{int(out.sum())} of {len(segs)} walls lie mostly outside the closing polygon. "
                "Every edge of the polygon becomes a wall, so a polygon that leaves walls outside "
                "cuts through the building. A plan that already includes its outline needs no "
                "polygon: keep the default close='hull'. Otherwise remove the walls that lie "
                "outside the polygon.")

    @classmethod
    def from_plan(cls, plan, floor: float = 0.0, ceiling: float = 2.5,
                  close="hull") -> Mesh:
        """Extrude a floor plan into a prism.

        Every wall segment becomes a vertical rectangle from ``floor`` to
        ``ceiling``. A floor and a ceiling cover the convex hull of the plan and of
        the closing polygon.
        With ``close="hull"`` vertical walls also run along the convex hull,
        so the domain is closed and no ray escapes. A polygon (for example
        the building footprint) closes the domain along its own outline
        instead, which keeps concave buildings concave.

        Parameters
        ----------
        plan : Plan or array_like of shape (n, 2, 2)
            Wall segments.
        floor : float
            Height of the floor plane in metres. Default 0.
        ceiling : float
            Height of the ceiling plane in metres, above ``floor``. Default 2.5.
        close : "hull", None or array_like of shape (k, 2)
            Close the sides along the convex hull, not at all, or along the
            given polygon (vertices in order, the last joins the first). Every
            edge of the polygon becomes a wall, the edge from the last vertex
            to the first included, so the polygon must hold the plan. A wall
            may end on it, run along it or reach past it, but a wall that lies
            mostly (more than half its length) outside it raises an error. A
            plan that already includes its outline needs no polygon: keep the
            default.

        Returns
        -------
        Mesh
            The walls, the floor and the ceiling, as one object. An eye outside
            the closed domain (beside the building, in the notch of a concave
            outline, above the ceiling) sees out of it: its rays escape, so
            ``view_volume`` flags it ``unbounded`` without a range limit and
            reports the share that escapes as ``escape_fraction``.

        Raises
        ------
        ValueError
            If a wall lies mostly outside the closing polygon, the polygon has
            fewer than three distinct corners enclosing an area, the walls lie
            on one line with no closing polygon, ``close`` is not one of its
            options, or ``floor`` and ``ceiling`` are not finite heights with
            the ceiling above the floor.
        """
        from .plan import Plan

        ring_close = None
        if not isinstance(close, str) and close is not None:
            ring_close = _closing_ring(close)
        elif close not in CLOSE_OPTIONS:
            raise ValueError(f"close must be one of {CLOSE_OPTIONS} or a polygon")
        if not (np.isfinite(floor) and np.isfinite(ceiling)):
            raise ValueError("floor and ceiling must be finite heights in metres")
        if not ceiling > floor:
            raise ValueError("ceiling must lie above floor")
        segs = (plan if isinstance(plan, Plan) else Plan(plan)).segments
        pts = segs.reshape(-1, 2)
        if ring_close is not None:
            cls.check_closing(segs, ring_close)
        if ring_close is not None:
            pts = np.vstack([pts, ring_close])
        try:
            ring = pts[ConvexHull(pts).vertices]  # counter-clockwise, floor and ceiling cover it
        except QhullError:
            raise ValueError("the walls lie on one line, so the floor and the ceiling would "
                             "cover no area: give walls that enclose an area, or a closing "
                             "polygon") from None
        walls = [segs]
        if ring_close is not None:
            walls.append(np.stack([ring_close, np.roll(ring_close, -1, axis=0)], axis=1))
        elif close == "hull":
            walls.append(np.stack([ring, np.roll(ring, -1, axis=0)], axis=1))
        walls = np.concatenate(walls)
        verts, tris = [], []
        n = 0
        for (x0, y0), (x1, y1) in walls:
            verts += [(x0, y0, floor), (x1, y1, floor), (x1, y1, ceiling), (x0, y0, ceiling)]
            tris += [(n, n + 1, n + 2), (n, n + 2, n + 3)]
            n += 4
        k = len(ring)
        for z, flip in ((floor, True), (ceiling, False)):
            verts += [(x, y, z) for x, y in ring]
            fan = [(n, n + i, n + i + 1) for i in range(1, k - 1)]
            tris += [(a, c, b) for a, b, c in fan] if flip else fan
            n += k
        return cls(np.array(verts), np.array(tris))

    @classmethod
    def from_ifc(cls, path, *, classes="walls", storey=None) -> Mesh:
        """Read the elements of a BIM model from an IFC file.

        The body geometry of every element of the chosen classes, in world
        coordinates and in metres, whatever the file's length unit. Openings
        are cut from the elements that host them. Needs the ``ifc`` extra
        (ifcopenshell).

        Parameters
        ----------
        path : str or os.PathLike
            IFC file.
        classes : str or sequence of str
            ``"walls"`` (IfcWall with its subtypes, IfcColumn and
            IfcCurtainWall), ``"furnished"`` (the walls plus IfcFurniture and
            IfcFurnishingElement), an IFC class, or a sequence of presets and
            classes. Each class includes its subtypes.
        storey : str, optional
            Name or GlobalId of an IfcBuildingStorey. Only the elements in
            that storey are read. By default all elements are read.

        Returns
        -------
        Mesh
            The bodies of the elements, as one object, in metres.

        Raises
        ------
        ValueError
            When no element of the chosen classes has a body, a class is
            neither a preset nor an IFC class of the file's schema, or no
            storey (or more than one) has the given name.
        ImportError
            When ifcopenshell is not installed.
        """
        from .ifc import read_ifc

        vertices, triangles, _ = read_ifc(path, classes=classes, storey=storey)
        return cls(vertices, triangles)

    # --- sections ---------------------------------------------------------------------------

    def section(self, z: float):
        """Horizontal section at height ``z``, as a floor plan.

        Every triangle that crosses the plane at height ``z`` gives the
        segment along which it crosses. A vertex at exactly ``z`` counts as
        below the plane, so a solid from ``z0`` to ``z1`` is cut for
        ``z0 <= z < z1``: walls standing on a floor at ``z`` are cut, and a
        table whose top lies at ``z`` is not. Triangles in the plane give no
        segment.

        Parameters
        ----------
        z : float
            Height of the plane in metres.

        Returns
        -------
        Plan
            One wall segment per crossing triangle.

        Raises
        ------
        ValueError
            When ``z`` is not finite.
        """
        from .plan import Plan

        if not np.isfinite(z):
            raise ValueError("z must be a finite height in metres")
        v, t = self.vertices, self.triangles
        above = v[:, 2] > z
        n_above = above[t].sum(axis=1)
        cut = (n_above == 1) | (n_above == 2)
        tri, side = t[cut], above[t[cut]]
        # The lone vertex is the one on its own side of the plane. Both crossing edges start there.
        lone = np.where(n_above[cut] == 1, side.argmax(axis=1), side.argmin(axis=1))
        rows = np.arange(len(tri))
        a = tri[rows, lone]
        ends = [_crossing(v, a, tri[rows, (lone + k) % 3], above, z) for k in (1, 2)]
        return Plan(np.stack(ends, axis=1) if len(tri) else np.empty((0, 2, 2)))

    # --- geometry ---------------------------------------------------------------------------

    @property
    def volume(self) -> float:
        """Signed volume enclosed by the triangles.

        The sum of tetrahedra from the centre of the bounding box.

        Meaningful for a closed, consistently oriented mesh: positive when the
        normals point outwards.
        """
        a, b, c = (self.vertices[self.triangles[:, k]] - self._shift for k in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper corner of the vertices' bounding box."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def _raycasting_scene(self):
        if self._scene is None:
            import open3d as o3d

            scene = o3d.t.geometry.RaycastingScene()
            if len(self.triangles):
                scene.add_triangles(
                    o3d.core.Tensor(np.asarray(self.vertices - self._shift, dtype=np.float32)),
                    o3d.core.Tensor(np.asarray(self.triangles, dtype=np.uint32)))
            self._scene = scene
        return self._scene

    def cast(self, origins, directions, *, max_distance: float = np.inf,
             near_clip: float = 0.0, n_threads: int | None = None) -> np.ndarray:
        """First-hit distance (m, n) of rays from ``origins`` (m, 3) along ``directions`` (n, 3).

        Parameters
        ----------
        origins : array_like of shape (m, 3)
            Ray origins in metres.
        directions : array_like of shape (n, 3)
            Ray directions, normalised internally.
        max_distance : float
            Range limit in metres. Default: none.
        near_clip : float
            Surfaces closer than this along a ray are skipped: the ray starts
            at that distance. Default 0.
        n_threads : int, optional
            Threads for Embree. Default: all.

        Returns
        -------
        ndarray of shape (m, n)
            Distance in metres to the first hit, ``inf`` when nothing is hit
            within ``max_distance``.

        Raises
        ------
        ValueError
            When an origin or a direction is not finite, a direction is zero,
            ``near_clip`` is negative or ``max_distance`` is not positive.
        """
        import open3d as o3d

        o, d = _rays(origins, directions, max_distance, near_clip)
        o = o - self._shift
        out = np.full((len(o), len(d)), np.inf)
        if len(self.triangles) == 0 or out.size == 0:
            return out
        scene = self._raycasting_scene()
        nthreads = 0 if n_threads is None else int(n_threads)
        step = max(1, _CHUNK_RAYS // len(d))
        dirs32 = np.broadcast_to(d.astype(np.float32), (step, len(d), 3))
        for i0 in range(0, len(o), step):
            oc = o[i0:i0 + step]
            starts = oc[:, None, :] + near_clip * d[None, :, :]
            rays = np.concatenate([starts.astype(np.float32), dirs32[:len(oc)]], axis=2)
            t = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)), nthreads=nthreads)
            out[i0:i0 + step] = t["t_hit"].numpy().astype(float).reshape(len(oc), len(d))
        out += near_clip
        out[out > max_distance] = np.inf
        return out

    def cast_hits(self, origins, directions, *, max_distance: float = np.inf,
                  near_clip: float = 0.0,
                  n_threads: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """First-hit distance and the triangle hit, for rays as in [`cast`][pysovist.Mesh.cast].

        The distances are those of ``cast``, to the last bit. The parameters
        and errors are those of ``cast``.

        Returns
        -------
        distances : ndarray of shape (m, n)
            ``inf`` when nothing is hit within ``max_distance``.
        triangles : ndarray of shape (m, n)
            Index of the triangle hit, ``-1`` when nothing is hit.
            ``object_ids[triangles]`` gives the object hit.
        """
        import open3d as o3d

        o, d = _rays(origins, directions, max_distance, near_clip)
        o = o - self._shift
        out = np.full((len(o), len(d)), np.inf)
        tri = np.full((len(o), len(d)), -1, dtype=np.int64)
        if len(self.triangles) == 0 or out.size == 0:
            return out, tri
        scene = self._raycasting_scene()
        nthreads = 0 if n_threads is None else int(n_threads)
        step = max(1, _CHUNK_RAYS // len(d))
        dirs32 = np.broadcast_to(d.astype(np.float32), (step, len(d), 3))
        for i0 in range(0, len(o), step):
            oc = o[i0:i0 + step]
            starts = oc[:, None, :] + near_clip * d[None, :, :]
            rays = np.concatenate([starts.astype(np.float32), dirs32[:len(oc)]], axis=2)
            res = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)), nthreads=nthreads)
            out[i0:i0 + step] = res["t_hit"].numpy().astype(float).reshape(len(oc), len(d))
            prim = res["primitive_ids"].numpy().astype(np.int64).reshape(len(oc), len(d))
            tri[i0:i0 + step] = np.where(prim < len(self.triangles), prim, -1)
        out += near_clip
        out[out > max_distance] = np.inf
        tri[~np.isfinite(out)] = -1
        return out, tri

    def clearance(self, origins) -> np.ndarray:
        """Distance in metres from each origin (m, 3) to the nearest triangle, ``inf`` for none.

        Raises
        ------
        ValueError
            When an origin is not finite.
        """
        import open3d as o3d

        o = _finite_points(origins) - self._shift
        if len(self.triangles) == 0:
            return np.full(len(o), np.inf)
        res = self._raycasting_scene().compute_closest_points(
            o3d.core.Tensor(o.astype(np.float32)))
        tri = self.triangles[res["primitive_ids"].numpy().astype(np.int64)]
        a, b, c = (self.vertices[tri[:, k]] - self._shift for k in range(3))
        return np.linalg.norm(o - _closest_on_triangles(o, a, b, c), axis=1)

    def blocked(self, origins, eps: float, near_clip: float = 0.0) -> np.ndarray:
        """True where an origin is closer than ``eps`` to a surface that ``near_clip`` keeps.

        Parameters
        ----------
        origins : array_like of shape (m, 3)
            Points in metres, finite.
        eps : float
            Clearance in metres below which a point counts as on a surface.
        near_clip : float
            Surfaces closer than this to the point are ignored, as in
            ``cast``. Default 0.

        Returns
        -------
        ndarray of shape (m,)
            True for the points on a surface.
        """
        cl = self.clearance(origins)
        return (cl < eps) & (cl >= near_clip)


def _triangle_indices(triangles) -> np.ndarray:
    """Read-only int64 triangles of shape (m, 3), refusing other shapes and fractional indices."""
    t = np.asarray(triangles)
    if t.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"triangles must have shape (m, 3), not {t.shape}: split each quad "
                         "(a, b, c, d) into (a, b, c) and (a, c, d)")
    if t.dtype.kind not in "iu":
        f = t.astype(float)
        if not np.all(np.isfinite(f) & (f == np.round(f))):
            raise ValueError("triangle indices must be whole numbers")
    return np.array(t, dtype=np.int64, copy=True)


def _finite_points(points) -> np.ndarray:
    """Points of shape (m, 3), refusing coordinates that are not finite."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(p)):
        raise ValueError("origins must be finite points (x, y, z)")
    return p


def _rays(origins, directions, max_distance, near_clip):
    """Origins (m, 3) and unit directions (n, 3) of a ray cast, after checking them."""
    o = _finite_points(origins)
    d = np.asarray(directions, dtype=float).reshape(-1, 3)
    norm = np.linalg.norm(d, axis=1)
    if not np.all(np.isfinite(norm) & (norm > 0)):
        raise ValueError("directions must be finite, non-zero vectors")
    if not near_clip >= 0:
        raise ValueError("near_clip must be a non-negative distance in metres")
    if not max_distance > 0:
        raise ValueError("max_distance must be positive")
    return o, d / norm[:, None]


def _closing_ring(polygon) -> np.ndarray:
    """A closing polygon (k, 2), refusing coordinates that are not finite and rings without area."""
    ring = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if not np.all(np.isfinite(ring)):
        raise ValueError("a closing polygon must have finite coordinates")
    span = float(np.ptp(ring, axis=0).max()) if len(ring) else 0.0
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if len(np.unique(ring, axis=0)) < 3 or not area > 1e-12 * span * span:
        raise ValueError("a closing polygon needs at least three distinct corners that enclose an "
                         "area: give the building outline, vertices in order")
    return ring


def _walls_outside_ring(segments, ring, rel_tol: float = 1e-6) -> np.ndarray:
    """True for each wall that lies mostly (more than half its length) outside
    the polygon ``ring``. Points within ``rel_tol`` times the scene's extent of
    an edge lie on the polygon, which covers coordinates rounded to six
    decimals."""
    segs = np.asarray(segments, dtype=float)
    a = np.asarray(ring, dtype=float)
    b = np.roll(a, -1, axis=0)
    tol = rel_tol * max(1.0, float(np.ptp(np.vstack([segs.reshape(-1, 2), a]), axis=0).max()))
    ends_out = _outside_ring(segs.reshape(-1, 2), a, tol).reshape(-1, 2)
    p, q = segs[:, 0], segs[:, 1]
    crosses = _segments_cross(p[:, None, :], q[:, None, :], a[None], b[None]).any(axis=1)
    out = np.zeros(len(segs), dtype=bool)
    e = b - a
    # the other walls lie inside or on the polygon
    for i in np.flatnonzero(ends_out.any(axis=1) | crosses):
        d = q[i] - p[i]
        den = d[0] * e[:, 1] - d[1] * e[:, 0]
        ok = np.abs(den) > 0
        w = a - p[i]
        t = np.where(ok, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1.0), -1.0)
        u = np.where(ok, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(ok, den, 1.0), -1.0)
        hit = ok & (t > 0) & (t < 1) & (u >= 0) & (u <= 1)
        ts = np.unique(np.concatenate([[0.0, 1.0], t[hit]]))
        mids = p[i] + ((ts[:-1] + ts[1:]) / 2)[:, None] * d
        out[i] = np.diff(ts)[_outside_ring(mids, a, tol)].sum() > 0.5
    return out


def _outside_ring(points, ring, tol: float) -> np.ndarray:
    """True for each point strictly outside the polygon ``ring`` (vertices in
    order) and farther than ``tol`` from its edges (even-odd rule)."""
    p = np.asarray(points, dtype=float)
    a = np.asarray(ring, dtype=float)
    b = np.roll(a, -1, axis=0)
    d = b - a
    dd = np.maximum((d * d).sum(axis=1), np.finfo(float).tiny)
    out = np.zeros(len(p), dtype=bool)
    for i in range(0, len(p), 4096):  # 4096 points x k edges per block
        q = p[i:i + 4096, None, :]
        t = np.clip(((q - a) * d).sum(axis=2) / dd, 0.0, 1.0)
        near = (np.linalg.norm(q - (a + t[..., None] * d), axis=2) <= tol).any(axis=1)
        y, x = q[..., 1], q[..., 0]
        spans = (a[:, 1] > y) != (b[:, 1] > y)
        xc = a[:, 0] + (y - a[:, 1]) * d[:, 0] / np.where(d[:, 1] == 0, 1.0, d[:, 1])
        inside = (spans & (x < xc)).sum(axis=1) % 2 == 1
        out[i:i + 4096] = ~near & ~inside
    return out


def _segments_cross(p, q, a, b) -> np.ndarray:
    """Whether the segment p-q properly crosses each segment a[j]-b[j]."""
    def side(o, u, v):
        return ((u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1])
                - (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0]))
    return ((side(p, q, a) * side(p, q, b) < 0) & (side(a, b, p) * side(a, b, q) < 0))


def _point_segment_distance(points, p, q) -> np.ndarray:
    """Distance of each point to the segment p-q."""
    d = q - p
    t = np.clip(((points - p) @ d) / max(float(d @ d), np.finfo(float).tiny), 0.0, 1.0)
    return np.linalg.norm(points - (p + t[:, None] * d), axis=1)


def _object_ids(ids, n: int) -> np.ndarray:
    """Read-only int64 object ids for ``n`` triangles (zeros by default)."""
    out = np.zeros(n, dtype=np.int64) if ids is None else np.array(ids, dtype=np.int64, copy=True)
    out = out.reshape(-1)
    if len(out) != n:
        raise ValueError("object_ids needs one entry per triangle")
    if n and out.min() < 0:
        raise ValueError("object ids must be non-negative")
    out.setflags(write=False)
    return out


def _crossing(v, a, b, above, z):
    """Plan point where each edge ``(a[i], b[i])`` crosses height ``z``.

    Each edge is interpolated from its vertex below the plane, so the two
    triangles that share an edge give the same point to the last bit, and a
    vertex on the plane is returned exactly.
    """
    lo = np.where(above[a], b, a)
    hi = np.where(above[a], a, b)
    s_lo, s_hi = v[lo, 2] - z, v[hi, 2] - z
    w = s_lo / (s_lo - s_hi)
    return v[lo, :2] + w[:, None] * (v[hi, :2] - v[lo, :2])


def _dot(u, v):
    return np.einsum("ij,ij->i", u, v)


def _closest_on_triangles(p, a, b, c):
    """Closest point to ``p[i]`` on triangle ``(a[i], b[i], c[i])`` (Ericson 2004, 5.1.5)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    bp = p - b
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    cp = p - c
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        out = a + ab * (vb / denom)[:, None] + ac * (vc / denom)[:, None]
        cases = [
            ((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
             b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None]),
            ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * (d2 / (d2 - d6))[:, None]),
            ((d6 >= 0) & (d5 <= d6), c),
            ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * (d1 / (d1 - d3))[:, None]),
            ((d3 >= 0) & (d4 <= d3), b),
            ((d1 <= 0) & (d2 <= 0), a),
        ]
    # Later cases take precedence, matching the order of the tests in Ericson's routine.
    for mask, q in cases:
        out[mask] = q[mask]
    return out
