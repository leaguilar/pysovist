"""Raw point clouds as occluders.

A scan is a set of surface samples with no connectivity. Each point is
treated as a solid ball (a "splat") of radius r, so that a dense enough
cloud closes into opaque surfaces. A surface sampled on a square lattice of
spacing s is sealed when r > s / sqrt(2), and the balls then erode the free
space by a depth between sqrt(r^2 - s^2 / 2) and r. ``spacing`` measures s.

All lengths are in metres and z is up. The radius belongs to the frame the
points are in: transforming a cloud keeps the radius.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SPACING_SAMPLE = 200_000


class PointCloud:
    """Points (n, 3) in metres, each standing for a ball of radius ``radius``.

    Ray casting needs the ``pointcloud`` extra (numba).

    Parameters
    ----------
    points : array_like of shape (n, 3)
        Point coordinates in metres. They are copied.
    radius : float
        Ball (splat) radius in metres.

    Attributes
    ----------
    points : ndarray of shape (n, 3)
        Point coordinates, read-only.
    radius : float
        Ball radius in metres.
    block_flag : str
        The flag set for an eye inside a ball or within ``eps`` of one,
        ``"inside_occluder"``.
    """

    def __init__(self, points, radius: float = 0.05):
        pts = np.array(points, dtype=float, copy=True).reshape(-1, 3)
        pts.setflags(write=False)
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.points: np.ndarray = pts
        self.radius: float = float(radius)
        # Search structures, shared with the radius variants made by with_radius. The k-d tree
        # depends only on the points. Ball grids are stored per radius and cell size.
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"PointCloud({len(self)} points, radius={self.radius:g})"

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper corner of the points' bounding box."""
        return self.points.min(axis=0), self.points.max(axis=0)

    def with_radius(self, radius: float) -> PointCloud:
        """The same points with another ball radius, without copying them.

        The new cloud shares the points, the k-d tree and the cache of ball
        grids with this one. A ball grid depends on the radius, so the first
        ray cast at a new radius builds a grid for it.
        """
        out = PointCloud.__new__(PointCloud)
        out.points, out.radius, out._cache = self.points, float(radius), self._cache
        if out.radius <= 0:
            raise ValueError("radius must be positive")
        return out

    # --- input ------------------------------------------------------------------------------

    @classmethod
    def read(cls, path, radius: float = 0.05) -> PointCloud:
        """Read a ``.pts``, ``.las``, ``.laz``, ``.e57``, ``.ply`` or ``.pcd`` file.

        ``.pts`` is the Leica text format: an optional first line with the
        point count, then one point per line starting with ``x y z``. E57
        files may hold several scans, each placed by its own pose.

        Each format needs one optional extra:

        - ``.pts``: none.
        - ``.las``, ``.laz``: ``pointcloud`` (laspy). ``.laz`` also needs a
          LAZ backend for laspy, such as lazrs.
        - ``.e57``: ``e57`` (pye57).
        - ``.ply``, ``.pcd``: ``mesh`` (open3d).

        Parameters
        ----------
        path : str or os.PathLike
            Point cloud file. The extension selects the reader.
        radius : float
            Ball radius in metres.

        Returns
        -------
        PointCloud
            The points of the file, as balls of radius ``radius``.
        """
        path = Path(path)
        ext = path.suffix.lower()
        readers = {".pts": _read_pts, ".las": _read_las, ".laz": _read_las, ".e57": _read_e57,
                   ".ply": _read_open3d, ".pcd": _read_open3d}
        if ext not in readers:
            raise ValueError(f"unsupported format {ext!r}, use one of {sorted(readers)}")
        return cls(readers[ext](path), radius=radius)

    # --- geometry ---------------------------------------------------------------------------

    def transform(self, matrix) -> PointCloud:
        """Apply an affine 4 x 4 matrix to the points (homogeneous, column vectors).

        The radius is kept.
        """
        m = np.asarray(matrix, dtype=float)
        if m.shape != (4, 4) or not np.allclose(m[3], [0, 0, 0, 1]):
            raise ValueError("matrix must be an affine 4 x 4 matrix")
        return PointCloud(self.points @ m[:3, :3].T + m[:3, 3], radius=self.radius)

    def crop(self, region, z=None) -> PointCloud:
        """Keep the points inside a box or a 2D polygon (boundaries included).

        Parameters
        ----------
        region : array_like
            A plan box ``(xmin, ymin, xmax, ymax)``, a 3D box
            ``(xmin, ymin, zmin, xmax, ymax, zmax)`` or a polygon ring of shape
            (k, 2) with k >= 3, applied in plan.
        z : (zmin, zmax), optional
            Extra height interval.
        """
        p = self.points
        reg = np.asarray(region, dtype=float)
        if reg.shape == (4,):
            keep = np.all((p[:, :2] >= reg[:2]) & (p[:, :2] <= reg[2:]), axis=1)
        elif reg.shape == (6,):
            keep = np.all((p >= reg[:3]) & (p <= reg[3:]), axis=1)
        elif reg.ndim == 2 and reg.shape[1] == 2 and len(reg) >= 3:
            keep = _in_polygon(p[:, :2], reg)
        else:
            raise ValueError("region must be a 2D box, a 3D box or a polygon ring (k, 2)")
        if z is not None:
            keep &= (p[:, 2] >= z[0]) & (p[:, 2] <= z[1])
        return PointCloud(p[keep], radius=self.radius)

    def _tree(self) -> cKDTree:
        if "tree" not in self._cache:
            self._cache["tree"] = cKDTree(self.points)
        return self._cache["tree"]

    def spacing(self, sample: int = SPACING_SAMPLE, seed: int = 0) -> dict:
        """Nearest-neighbour distance between points: median, 90th and 99th percentile.

        Measured for a random subsample of ``sample`` points (fixed ``seed``)
        against the whole cloud.
        """
        rng = np.random.default_rng(seed)
        n = len(self)
        idx = rng.choice(n, sample, replace=False) if n > sample else np.arange(n)
        d, _ = self._tree().query(self.points[idx], k=2)
        q = np.percentile(d[:, 1], [50, 90, 99])
        return {"median": float(q[0]), "p90": float(q[1]), "p99": float(q[2])}

    def clearance(self, origins, radius: float | None = None) -> np.ndarray:
        """Distance from each origin (m, 3) to the nearest ball surface (negative inside a ball).

        ``radius`` overrides the cloud's own radius.
        """
        o = np.asarray(origins, dtype=float).reshape(-1, 3)
        r = self.radius if radius is None else float(radius)
        if len(self) == 0:
            return np.full(len(o), np.inf)
        d, _ = self._tree().query(o)
        return d - r

    # --- occluder interface used by the view-volume integrator -------------------------------

    block_flag = "inside_occluder"

    def blocked(self, origins, eps: float, near_clip: float = 0.0) -> np.ndarray:
        """True where an origin lies inside a ball, or within ``eps`` of one.

        Balls whose centre is closer than ``near_clip`` are ignored, as in ``cast``.
        """
        o = np.asarray(origins, dtype=float).reshape(-1, 3)
        reach = self.radius + eps
        if len(self) == 0:
            return np.zeros(len(o), dtype=bool)
        d, _ = self._tree().query(o)
        out = d < reach
        if near_clip > 0:
            for i in np.flatnonzero(out & (d < near_clip)):
                near = np.asarray(self._tree().query_ball_point(o[i], reach), dtype=int)
                dist = np.linalg.norm(self.points[near] - o[i], axis=1)
                out[i] = bool(np.any(dist >= near_clip))
        return out

    def grid(self, cell_size: float | None = None):
        """Ball grid for ray casting, built once per radius and cell size."""
        from .raycast import build_ball_grid

        key = ("grid", self.radius, cell_size)
        if key not in self._cache:
            self._cache[key] = build_ball_grid(self.points, self.radius, cell_size)
        return self._cache[key]

    def cast(self, origins, directions, *, max_distance: float = np.inf,
             near_clip: float = 0.0, n_threads: int | None = None) -> np.ndarray:
        """First-hit distance (m, n) of rays from ``origins`` (m, 3) along ``directions`` (n, 3).

        ``0`` for rays that start inside a ball, ``inf`` when no ball is hit
        within ``max_distance``. Balls whose centre is closer than
        ``near_clip`` to the ray's origin are ignored.
        """
        from .raycast import cast_balls

        return cast_balls(self.grid(), origins, directions, max_distance=max_distance,
                          near_clip=near_clip, n_threads=n_threads)


# --- readers ----------------------------------------------------------------------------------

def _read_pts(path: Path) -> np.ndarray:
    import pandas as pd

    with open(path) as f:
        first = f.readline().split()
    skip = 1 if len(first) == 1 else 0
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip, usecols=[0, 1, 2],
                     dtype=np.float64, engine="c")
    return df.to_numpy()


def _read_las(path: Path) -> np.ndarray:
    import laspy

    las = laspy.read(path)
    return np.c_[np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]


def _read_e57(path: Path) -> np.ndarray:
    import pye57

    e57 = pye57.E57(str(path))
    parts = []
    for i in range(e57.scan_count):
        s = e57.read_scan(i, ignore_missing_fields=True)
        parts.append(np.c_[s["cartesianX"], s["cartesianY"], s["cartesianZ"]])
    return np.concatenate(parts) if parts else np.empty((0, 3))


def _read_open3d(path: Path) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=float)


def _in_polygon(xy: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd rule, with points on the boundary counted inside."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    on_edge = np.zeros(len(xy), dtype=bool)
    a, b = ring, np.roll(ring, -1, axis=0)
    for (x0, y0), (x1, y1) in zip(a, b, strict=True):
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < xc)
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        within = ((x >= min(x0, x1)) & (x <= max(x0, x1))
                  & (y >= min(y0, y1)) & (y <= max(y0, y1)))
        on_edge |= within & (np.abs(cross) <= 1e-12 * max(1.0, np.hypot(x1 - x0, y1 - y0)))
    return inside | on_edge
