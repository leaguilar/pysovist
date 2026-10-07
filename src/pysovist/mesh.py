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
from scipy.spatial import ConvexHull

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
        for ray casting.

    Attributes
    ----------
    vertices : ndarray of shape (n, 3)
        Vertex coordinates, read-only.
    triangles : ndarray of shape (m, 3)
        Vertex indices of each triangle, read-only.
    block_flag : str
        The flag set for an eye closer than ``eps`` to a surface, ``"on_wall"``.
    """

    block_flag = "on_wall"

    def __init__(self, vertices, triangles):
        v = np.array(vertices, dtype=float, copy=True)
        t = np.array(triangles, dtype=np.int64, copy=True).reshape(-1, 3)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if len(t) and (t.min() < 0 or t.max() >= len(v)):
            raise ValueError("triangle indices out of range")
        v.setflags(write=False)
        t.setflags(write=False)
        self.vertices: np.ndarray = v
        self.triangles: np.ndarray = t
        self._shift = (v.min(axis=0) + v.max(axis=0)) / 2 if len(v) else np.zeros(3)
        self._scene = None

    def __repr__(self) -> str:
        return f"Mesh({len(self.vertices)} vertices, {len(self.triangles)} triangles)"

    # --- construction -----------------------------------------------------------------------

    @classmethod
    def from_file(cls, path) -> Mesh:
        """Read any triangle mesh format open3d reads (``.ply``, ``.obj``, ``.stl``, ...)."""
        import open3d as o3d

        m = o3d.io.read_triangle_mesh(str(Path(path)))
        return cls(np.asarray(m.vertices), np.asarray(m.triangles))

    @classmethod
    def from_plan(cls, plan, floor: float = 0.0, ceiling: float = 2.5,
                  close="hull") -> Mesh:
        """Extrude a floor plan into a prism.

        Every wall segment becomes a vertical rectangle from ``floor`` to
        ``ceiling``. A floor and a ceiling cover the convex hull of the plan.
        With ``close="hull"`` vertical walls also run along the convex hull,
        so the domain is closed and no ray escapes. A polygon (for example
        the building footprint) closes the domain along its own outline
        instead, which keeps concave buildings concave.

        Parameters
        ----------
        plan : Plan or array_like of shape (n, 2, 2)
            Wall segments.
        floor : float
            Height of the floor plane in metres.
        ceiling : float
            Height of the ceiling plane in metres, above ``floor``.
        close : "hull", None or array_like of shape (k, 2)
            Close the sides along the convex hull, not at all, or along the
            given polygon (vertices in order, the last joins the first).
        """
        from .plan import Plan

        ring_close = None
        if not isinstance(close, str) and close is not None:
            ring_close = np.asarray(close, dtype=float).reshape(-1, 2)
            if len(ring_close) < 3:
                raise ValueError("a closing polygon needs at least three vertices")
        elif close not in CLOSE_OPTIONS:
            raise ValueError(f"close must be one of {CLOSE_OPTIONS} or a polygon")
        if not ceiling > floor:
            raise ValueError("ceiling must lie above floor")
        segs = (plan if isinstance(plan, Plan) else Plan(plan)).segments
        pts = segs.reshape(-1, 2)
        if ring_close is not None:
            pts = np.vstack([pts, ring_close])
        ring = pts[ConvexHull(pts).vertices]  # counter-clockwise, floor and ceiling cover it
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

    # --- geometry ---------------------------------------------------------------------------

    @property
    def volume(self) -> float:
        """Signed volume enclosed by the triangles (sum of tetrahedra from the origin).

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

        ``inf`` when nothing is hit within ``max_distance``. Surfaces closer
        than ``near_clip`` along a ray are skipped: the ray starts at that distance.
        """
        import open3d as o3d

        o = np.asarray(origins, dtype=float).reshape(-1, 3) - self._shift
        d = np.asarray(directions, dtype=float).reshape(-1, 3)
        d = d / np.linalg.norm(d, axis=1)[:, None]
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

    def clearance(self, origins) -> np.ndarray:
        """Distance from each origin (m, 3) to the nearest triangle."""
        import open3d as o3d

        o = np.asarray(origins, dtype=float).reshape(-1, 3) - self._shift
        if len(self.triangles) == 0:
            return np.full(len(o), np.inf)
        res = self._raycasting_scene().compute_closest_points(
            o3d.core.Tensor(o.astype(np.float32)))
        tri = self.triangles[res["primitive_ids"].numpy().astype(np.int64)]
        a, b, c = (self.vertices[tri[:, k]] - self._shift for k in range(3))
        return np.linalg.norm(o - _closest_on_triangles(o, a, b, c), axis=1)

    def blocked(self, origins, eps: float, near_clip: float = 0.0) -> np.ndarray:
        """True where an origin is closer than ``eps`` to a surface that ``near_clip`` keeps."""
        cl = self.clearance(origins)
        return (cl < eps) & (cl >= near_clip)


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
